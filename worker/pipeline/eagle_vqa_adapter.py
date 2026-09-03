"""
Eagle2.5 VQA 封装器 — 替代 Moondream2 的高级视觉问答质检器

功能：
- 图像质量评分（清晰度、完整性、准确性）
- 图像描述生成
- 复杂视觉问答

基于 nvidia/Eagle2.5-8B，提供更强的视觉理解能力
"""

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Callable

import torch
from PIL import Image

from pipeline.gpu_manager import gpu_stage, check_cancel_and_yield, get_device

# HuggingFace 模型 ID
EAGLE_VQA_MODEL_ID = "nvidia/Eagle2.5-8B"

# 显存需求 (FP16)
REQUIRED_VRAM_GB = 16.0

# 默认质量阈值
DEFAULT_QUALITY_THRESHOLD = 0.5

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


class EagleVqaSetupError(RuntimeError):
    """Eagle2.5 VQA 初始化失败"""


class EagleVqaRuntimeError(RuntimeError):
    """Eagle2.5 VQA 运行时错误"""


def _is_eagle_setup_error(exc: Exception) -> bool:
    """检测是否为 Eagle2.5 初始化错误"""
    lower = str(exc).lower()
    indicators = (
        EAGLE_VQA_MODEL_ID,
        "eagle2",
        "eagle-2",
        "huggingface.co",
        "httpsconnectionpool",
        "certificate",
        "ssl",
        "connection",
        "timed out",
        "trust_remote_code",
        "siglip",
        "qwen2.5",
    )
    return any(token in lower for token in indicators)


def _build_eagle_setup_error(exc: Exception) -> EagleVqaSetupError:
    return EagleVqaSetupError(
        f"Eagle2.5 VQA 初始化失败：需要下载模型 {EAGLE_VQA_MODEL_ID}。"
        f" 请确保网络可访问 huggingface.co，或预先下载模型缓存。"
        f" 模型较大（8B 参数），下载可能需要较长时间。"
        f" 原始错误: {exc}"
    )


class EagleVqaAdapter:
    """
    Eagle2.5 VQA 模型封装，提供高级图像理解能力
    
    相比 Moondream2 的优势：
    - 更强的视觉编码器 (SigLIP2 vs 原始 SigLIP)
    - 更长的上下文 (128K vs 2K)
    - 更强的多模态理解
    - 支持高分辨率图像 (4K)
    
    特性：
    - 懒加载：首次使用时才初始化模型
    - GPU/CPU 自适应
    - 显存管理：使用 gpu_stage 确保显存安全
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = "float16",
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        max_new_tokens: int = 256,
    ):
        """
        初始化 Eagle2.5 VQA Adapter
        
        Args:
            device: 运行设备，"cuda"/"cpu"/"auto"，默认自动检测
            torch_dtype: 精度，"float16"/"float32"，默认 float16
            quality_threshold: 质量阈值，默认 0.5
            max_new_tokens: 最大生成 token 数，默认 256
        """
        self._device = device or get_device()
        self._torch_dtype = torch_dtype
        self._quality_threshold = quality_threshold
        self._max_new_tokens = max_new_tokens
        self._processor = None
        self._model = None
        self._loaded = False
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._model is not None
    
    @property
    def model_id(self) -> str:
        return EAGLE_VQA_MODEL_ID
    
    def _get_torch_dtype(self) -> torch.dtype:
        """获取 PyTorch 数据类型"""
        if self._torch_dtype == "float16" and self._device == "cuda":
            return torch.float16
        return torch.float32
    
    def load(self) -> "EagleVqaAdapter":
        """
        加载 Eagle2.5 VQA 模型
        
        Returns:
            self
        """
        if self._loaded:
            return self
        
        with gpu_stage("eagle_vqa_load", required_gb=REQUIRED_VRAM_GB):
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                
                logger.info(f"正在加载 Eagle2.5 VQA: {EAGLE_VQA_MODEL_ID}")
                logger.info(f"设备: {self._device}, 精度: {self._torch_dtype}")
                
                dtype = self._get_torch_dtype()
                
                self._processor = AutoProcessor.from_pretrained(
                    EAGLE_VQA_MODEL_ID,
                    trust_remote_code=True,
                )
                self._model = AutoModelForVision2Seq.from_pretrained(
                    EAGLE_VQA_MODEL_ID,
                    torch_dtype=dtype,
                    device_map=self._device,
                    trust_remote_code=True,
                )
                
                self._loaded = True
                logger.info("Eagle2.5 VQA 加载完成")
                
            except Exception as exc:
                if _is_eagle_setup_error(exc):
                    raise _build_eagle_setup_error(exc) from exc
                raise EagleVqaSetupError(f"加载 Eagle2.5 VQA 失败: {exc}") from exc
        
        return self
    
    def quality_check(
        self,
        image: Image.Image,
        detected_objects: list[dict],
        quality_threshold: Optional[float] = None,
    ) -> dict:
        """
        VQA 质检 — 检查检测质量
        
        检查维度：
        1. 清晰度：图像是否清晰可辨
        2. 完整性：是否所有目标都被检测到
        3. 准确性：检测框是否准确
        
        Args:
            image: 输入图像
            detected_objects: 检测到的对象列表
            quality_threshold: 质量阈值，默认使用初始化时的值
        
        Returns:
            dict: {
                "passed": bool,           # 是否通过
                "scores": dict,           # 各维度评分
                "rejected": bool,         # 是否被拒绝
                "reason": str,            # 原因描述
                "details": dict           # 详细结果
            }
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        threshold = quality_threshold or self._quality_threshold
        
        prompt = self._build_quality_prompt(detected_objects)
        
        try:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._device)
            
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            
            response = self._processor.decode(output[0], skip_special_tokens=True)
            
            return self._parse_quality_response(response, threshold, detected_objects)
            
        except Exception as exc:
            raise EagleVqaRuntimeError(f"质检失败: {exc}") from exc
    
    def describe(
        self,
        image: Image.Image,
        question: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        图像描述
        
        Args:
            image: 输入图像
            question: 可选的提问，为空时生成详细描述
            max_new_tokens: 最大生成 token 数
        
        Returns:
            str: 描述文本
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        if question:
            prompt = question
        else:
            prompt = "请详细描述这张图像的内容，包括场景、对象、动作和细节。"
        
        try:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._device)
            
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            return self._processor.decode(output[0], skip_special_tokens=True)
            
        except Exception as exc:
            raise EagleVqaRuntimeError(f"描述生成失败: {exc}") from exc
    
    def ask(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        视觉问答
        
        Args:
            image: 输入图像
            question: 问题
            max_new_tokens: 最大生成 token 数
        
        Returns:
            str: 回答
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        try:
            inputs = self._processor(
                text=question,
                images=image,
                return_tensors="pt",
            ).to(self._device)
            
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            
            return self._processor.decode(output[0], skip_special_tokens=True)
            
        except Exception as exc:
            raise EagleVqaRuntimeError(f"问答失败: {exc}") from exc
    
    def verify_detection(
        self,
        image: Image.Image,
        class_name: str,
        bbox: list[float],
    ) -> dict:
        """
        验证单个检测是否准确
        
        Args:
            image: 输入图像
            class_name: 类别名称
            bbox: 边界框 [cx, cy, w, h]，归一化坐标
        
        Returns:
            dict: {
                "accurate": bool,
                "confidence": float,
                "reason": str
            }
        """
        check_cancel_and_yield()
        
        if not self.is_loaded:
            self.load()
        
        # 转换 bbox 为像素坐标用于提示
        cx, cy, w, h = bbox
        prompt = f"""请验证图像中的检测结果：
        
类别：{class_name}
边界框位置：中心点 ({cx:.2f}, {cy:.2f})，宽高 ({w:.2f}, {h:.2f})（归一化坐标）

请判断：
1. 该位置是否存在 {class_name} 对象？
2. 边界框是否准确覆盖了该对象？

请以JSON格式输出：{{"accurate": true/false, "confidence": 0.0-1.0, "reason": "简短原因"}}"""
        
        try:
            inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            ).to(self._device)
            
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                )
            
            response = self._processor.decode(output[0], skip_special_tokens=True)
            
            # 解析 JSON 响应
            match = re.search(r'\{[^}]+\}', response)
            if match:
                try:
                    result = json.loads(match.group())
                    return {
                        "accurate": result.get("accurate", False),
                        "confidence": float(result.get("confidence", 0.0)),
                        "reason": result.get("reason", ""),
                    }
                except json.JSONDecodeError:
                    pass
            
            return {
                "accurate": False,
                "confidence": 0.0,
                "reason": f"无法解析响应: {response[:100]}...",
            }
            
        except Exception as exc:
            raise EagleVqaRuntimeError(f"检测验证失败: {exc}") from exc
    
    def _build_quality_prompt(self, objects: list[dict]) -> str:
        """构建质检提示词"""
        if not objects:
            obj_list = "无检测结果"
        else:
            obj_details = []
            for i, obj in enumerate(objects[:10]):  # 最多 10 个对象
                name = obj.get("class_name", obj.get("prompt", "未知"))
                conf = obj.get("conf", 1.0)
                obj_details.append(f"{i+1}. {name} (置信度: {conf:.2f})")
            obj_list = "\n".join(obj_details)
        
        return f"""请检查图像中的检测质量。

检测到的对象：
{obj_list}

请从以下维度评分（0-1）：
1. 清晰度：图像是否清晰可辨，没有模糊或遮挡
2. 完整性：是否所有可见目标都被检测到
3. 准确性：检测框是否准确覆盖目标对象

请以JSON格式输出评分：
{{"clarity": 0.9, "completeness": 0.8, "accuracy": 0.85}}"""
    
    def _parse_quality_response(
        self,
        response: str,
        threshold: float,
        objects: list[dict],
    ) -> dict:
        """解析质检响应"""
        # 提取 JSON
        match = re.search(r'\{[^}]+\}', response)
        if match:
            try:
                scores = json.loads(match.group())
                
                # 确保所有维度都在阈值以上
                clarity = float(scores.get("clarity", 0))
                completeness = float(scores.get("completeness", 0))
                accuracy = float(scores.get("accuracy", 0))
                
                scores = {
                    "clarity": clarity,
                    "completeness": completeness,
                    "accuracy": accuracy,
                }
                
                passed = all(v >= threshold for v in scores.values())
                rejected = not passed
                reason = self._get_rejection_reason(scores, threshold)
                
                return {
                    "passed": passed,
                    "scores": scores,
                    "rejected": rejected,
                    "reason": reason,
                    "details": {
                        "threshold": threshold,
                        "object_count": len(objects),
                        "raw_response": response,
                    },
                }
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"质检响应 JSON 解析失败: {e}, response: {response}")
        
        # 解析失败时的默认处理
        return {
            "passed": False,
            "scores": {},
            "rejected": True,
            "reason": f"无法解析质检响应",
            "details": {
                "threshold": threshold,
                "object_count": len(objects),
                "raw_response": response[:200] if response else "",
            },
        }
    
    def _get_rejection_reason(self, scores: dict, threshold: float) -> str:
        """生成拒绝原因"""
        failed = [
            (k, v)
            for k, v in scores.items()
            if v < threshold
        ]
        
        if not failed:
            return "通过"
        
        reasons = [
            f"{k}={v:.2f}(要求>{threshold})"
            for k, v in failed
        ]
        return f"以下维度不达标：{', '.join(reasons)}"
    
    def unload(self):
        """
        释放模型显存
        
        严格遵守两段式设计，确保显存完全释放
        """
        if self._model is not None:
            logger.info("正在释放 Eagle2.5 VQA 模型...")
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor
            self._processor = None
        
        self._loaded = False
        
        gc.collect()
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Eagle2.5 VQA 已释放")


# 懒加载单例
_instance: Optional[EagleVqaAdapter] = None


def get_eagle_vqa_adapter(
    device: Optional[str] = None,
    **kwargs,
) -> EagleVqaAdapter:
    """
    获取 Eagle2.5 VQA Adapter 单例
    
    Args:
        device: 运行设备
        **kwargs: 其他初始化参数
    
    Returns:
        EagleVqaAdapter 实例
    """
    global _instance
    
    if _instance is None or not _instance.is_loaded:
        if _instance is not None:
            _instance.unload()
        _instance = EagleVqaAdapter(device=device, **kwargs)
    
    return _instance


def release_eagle_vqa_adapter():
    """释放 Eagle2.5 VQA Adapter 单例"""
    global _instance
    
    if _instance is not None:
        _instance.unload()
        _instance = None
