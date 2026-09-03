"""
LocateAnything 封装器 — 兼容现有 stage2_labeler 接口的视觉定位适配器

功能：
- 目标检测 (detect)
- 短语定位 (ground_multi)
- OCR 文本检测 (detect_text)
- GUI 元素定位 (ground_gui)
- 点定位 (point)

输出格式：YOLO xywhn 格式，兼容现有数据处理流程
"""

import gc
import logging
import os
import re
from pathlib import Path
from typing import Optional, Callable

import torch
from PIL import Image

from pipeline.gpu_manager import gpu_stage, check_cancel_and_yield, get_device, CancelError

# HuggingFace 模型 ID
LOCATE_ANYTHING_MODEL_ID = "nvidia/LocateAnything-3B"

# 显存需求 (FP16)
REQUIRED_VRAM_GB = 6.0

# 日志
logger = logging.getLogger(__name__)

# 国内网络 HF Mirror 配置
_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "").strip()
if not _HF_ENDPOINT:
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class LocateAnythingSetupError(RuntimeError):
    """LocateAnything 初始化失败"""


class LocateAnythingRuntimeError(RuntimeError):
    """LocateAnything 运行时错误"""


def _is_locate_setup_error(exc: Exception) -> bool:
    """检测是否为 LocateAnything 初始化错误"""
    lower = str(exc).lower()
    indicators = (
        LOCATE_ANYTHING_MODEL_ID,
        "huggingface.co",
        "httpsconnectionpool",
        "certificate",
        "ssl",
        "connection",
        "timed out",
        "trust_remote_code",
        "locateanything",
        "moonshotai",
        "qwen",
    )
    return any(token in lower for token in indicators)


def _build_locate_setup_error(exc: Exception) -> LocateAnythingSetupError:
    return LocateAnythingSetupError(
        f"LocateAnything 初始化失败：需要下载模型 {LOCATE_ANYTHING_MODEL_ID}。"
        f" 请确保网络可访问 huggingface.co，或预先下载模型缓存。"
        f" 原始错误: {exc}"
    )


class LocateAnythingAdapter:
    """
    LocateAnything 模型封装，提供与 YOLO-World 兼容的接口
    
    特性：
    - 懒加载：首次使用时才初始化模型
    - GPU/CPU 自适应：根据环境自动选择设备
    - 显存管理：使用 gpu_stage 确保显存安全
    - 向后兼容：输出格式与 YOLO-World 一致
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "float16",
        conf_threshold: float = 0.25,
        batch_size: int = 4,
    ):
        """
        初始化 LocateAnything Adapter
        
        Args:
            device: 运行设备，"cuda"/"cpu"/"auto"，默认自动检测
            torch_dtype: 精度，"float16"/"float32"，默认 float16
            conf_threshold: 置信度阈值，默认 0.25
            batch_size: 批处理大小，默认 4
        """
        self._device = device or get_device()
        self._torch_dtype = torch_dtype
        self._conf_threshold = conf_threshold
        self._batch_size = batch_size
        self._worker = None
        self._loaded = False
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._worker is not None
    
    @property
    def model_id(self) -> str:
        return LOCATE_ANYTHING_MODEL_ID
    
    def _get_torch_dtype(self) -> torch.dtype:
        """获取 PyTorch 数据类型"""
        if self._torch_dtype == "float16" and self._device == "cuda":
            return torch.float16
        return torch.float32
    
    def load(self) -> "LocateAnythingAdapter":
        """
        加载 LocateAnything 模型
        
        Returns:
            self
        """
        if self._loaded:
            return self
        
        with gpu_stage("locate_anything_load", required_gb=REQUIRED_VRAM_GB):
            try:
                from locateanything_worker import LocateAnythingWorker
                
                logger.info(f"正在加载 LocateAnything: {LOCATE_ANYTHING_MODEL_ID}")
                logger.info(f"设备: {self._device}, 精度: {self._torch_dtype}")
                
                self._worker = LocateAnythingWorker(
                    LOCATE_ANYTHING_MODEL_ID,
                    torch_dtype=self._get_torch_dtype(),
                )
                
                self._loaded = True
                logger.info("LocateAnything 加载完成")
                
            except Exception as exc:
                if _is_locate_setup_error(exc):
                    raise _build_locate_setup_error(exc) from exc
                raise LocateAnythingSetupError(f"加载 LocateAnything 失败: {exc}") from exc
        
        return self
    
    def detect(
        self,
        image: Image.Image,
        classes: list[str],
        conf_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        目标检测
        
        Args:
            image: 输入图像 (PIL.Image)
            classes: 检测类别列表，如 ["person", "car"]
            conf_threshold: 置信度阈值，默认使用初始化时的值
        
        Returns:
            list[dict]: 检测结果列表，每项包含：
                - bbox_xywhn: [cx, cy, w, h]，归一化坐标
                - x1, y1, x2, y2: 像素坐标
                - conf: 置信度 (LocateAnything 固定为 1.0)
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        threshold = conf_threshold or self._conf_threshold
        
        try:
            result = self._worker.detect(
                image,
                classes,
            )
            boxes = self._parse_locate_output(result, image.size)
            
            # 过滤置信度 (虽然 LocateAnything 不输出置信度，保留接口一致性)
            if threshold > 0:
                boxes = [b for b in boxes if b.get("conf", 1.0) >= threshold]
            
            logger.debug(f"检测到 {len(boxes)} 个目标")
            return boxes
            
        except Exception as exc:
            raise LocateAnythingRuntimeError(f"检测失败: {exc}") from exc
    
    def detect_batch(
        self,
        images: list[tuple[Image.Image, list[str]]],
        conf_threshold: Optional[float] = None,
    ) -> list[list[dict]]:
        """
        批量目标检测
        
        Args:
            images: [(image, classes), ...] 列表
            conf_threshold: 置信度阈值
        
        Returns:
            list[list[dict]]: 每个图像的检测结果
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        results = []
        for img, classes in images:
            boxes = self.detect(img, classes, conf_threshold)
            results.append(boxes)
        
        return results
    
    def ground_multi(
        self,
        image: Image.Image,
        query: str,
    ) -> list[dict]:
        """
        短语定位 — 根据自然语言查询定位图像中的目标
        
        Args:
            image: 输入图像
            query: 自然语言查询，如 "people wearing red shirts"
        
        Returns:
            list[dict]: 定位结果列表
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        try:
            result = self._worker.ground_multi(image, query)
            boxes = self._parse_locate_output(result, image.size)
            
            logger.debug(f"短语定位 '{query}': {len(boxes)} 个结果")
            return boxes
            
        except Exception as exc:
            raise LocateAnythingRuntimeError(f"短语定位失败: {exc}") from exc
    
    def detect_text(self, image: Image.Image) -> list[dict]:
        """
        OCR 文本检测
        
        Args:
            image: 输入图像
        
        Returns:
            list[dict]: 文本区域列表
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        try:
            result = self._worker.detect_text(image)
            boxes = self._parse_locate_output(result, image.size)
            
            logger.debug(f"OCR 检测: {len(boxes)} 个文本区域")
            return boxes
            
        except Exception as exc:
            raise LocateAnythingRuntimeError(f"OCR 检测失败: {exc}") from exc
    
    def ground_gui(
        self,
        image: Image.Image,
        query: str,
        output_type: str = "box",
    ) -> list[dict]:
        """
        GUI 元素定位
        
        Args:
            image: 输入图像
            query: 元素描述，如 "the search button"
            output_type: "box" 或 "point"
        
        Returns:
            list[dict]: GUI 元素位置
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        try:
            result = self._worker.ground_gui(image, query, output_type=output_type)
            boxes = self._parse_locate_output(result, image.size)
            
            logger.debug(f"GUI 定位 '{query}': {len(boxes)} 个元素")
            return boxes
            
        except Exception as exc:
            raise LocateAnythingRuntimeError(f"GUI 定位失败: {exc}") from exc
    
    def point(
        self,
        image: Image.Image,
        query: str,
    ) -> list[dict]:
        """
        点定位
        
        Args:
            image: 输入图像
            query: 目标描述，如 "the traffic light"
        
        Returns:
            list[dict]: 点坐标列表
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        try:
            result = self._worker.point(image, query)
            boxes = self._parse_locate_output(result, image.size)
            
            logger.debug(f"点定位 '{query}': {len(boxes)} 个点")
            return boxes
            
        except Exception as exc:
            raise LocateAnythingRuntimeError(f"点定位失败: {exc}") from exc
    
    def _parse_locate_output(self, result: dict, image_size: tuple) -> list[dict]:
        """
        解析 LocateAnything 输出为统一格式
        
        LocateAnything 输出格式：
            <ref>label</ref><box><x1><y1><x2><y2></box>
        
        坐标范围：0-1000（需要除以 1000 转为相对坐标）
        
        Args:
            result: LocateAnything 返回结果
            image_size: (width, height)
        
        Returns:
            list[dict]: 统一格式的检测结果
        """
        w, h = image_size
        boxes = []
        answer = result.get("answer", "")
        
        # 解析所有边界框
        # 格式: <box><x1><y1><x2><y2></box>
        box_pattern = re.compile(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
        )
        
        for match in box_pattern.finditer(answer):
            x1, y1, x2, y2 = [int(g) for g in match.groups()]
            
            # 转换为相对坐标 (0-1)
            x1_r, y1_r = x1 / 1000, y1 / 1000
            x2_r, y2_r = x2 / 1000, y2 / 1000
            
            # 计算 xywhn 格式 (中心点 + 宽高，相对坐标)
            cx = (x1_r + x2_r) / 2
            cy = (y1_r + y2_r) / 2
            bw = x2_r - x1_r
            bh = y2_r - y1_r
            
            boxes.append({
                "x1": x1_r * w,
                "y1": y1_r * h,
                "x2": x2_r * w,
                "y2": y2_r * h,
                "bbox_xywhn": [cx, cy, bw, bh],
                "conf": 1.0,  # LocateAnything 不输出置信度
            })
        
        return boxes
    
    def unload(self):
        """
        释放模型显存
        
        严格遵守两段式设计，确保显存完全释放
        """
        if self._worker is not None:
            logger.info("正在释放 LocateAnything 模型...")
            del self._worker
            self._worker = None
            self._loaded = False
        
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("LocateAnything 已释放")


# 懒加载单例（可选，用于减少重复加载）
_instance: Optional[LocateAnythingAdapter] = None


def get_locate_adapter(
    device: Optional[str] = None,
    **kwargs,
) -> LocateAnythingAdapter:
    """
    获取 LocateAnything Adapter 单例
    
    Args:
        device: 运行设备
        **kwargs: 其他初始化参数
    
    Returns:
        LocateAnythingAdapter 实例
    """
    global _instance
    
    if _instance is None or not _instance.is_loaded:
        if _instance is not None:
            _instance.unload()
        _instance = LocateAnythingAdapter(device=device, **kwargs)
    
    return _instance


def release_locate_adapter():
    """释放 LocateAnything Adapter 单例"""
    global _instance
    
    if _instance is not None:
        _instance.unload()
        _instance = None
