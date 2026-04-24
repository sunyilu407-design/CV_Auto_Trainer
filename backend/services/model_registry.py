"""
Model Registry — 预训练模型目录 & 已训练模型缓存。

核心职责：
1. 维护市面上可用预训练模型的完整目录（YOLO 全系列、RF-DETR 等）
2. 记录每个模型的适用场景、设备要求、速度/精度权衡
3. 管理已训练模型的缓存，避免重复训练
4. 为 VLM 算法规划提供模型选择上下文
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 枚举 & 数据类
# ---------------------------------------------------------------------------

class ModelFamily(str, Enum):
    YOLOV5 = "yolov5"
    YOLOV8 = "yolov8"
    YOLOV9 = "yolov9"
    YOLOV10 = "yolov10"
    YOLOV11 = "yolov11"
    YOLO26 = "yolo26"
    RF_DETR = "rf-detr"
    RT_DETR = "rt-detr"
    OCR = "ocr"
    CUSTOM = "custom"


class TaskType(str, Enum):
    DETECTION = "detection"
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    OBB = "obb"
    TRACKING = "tracking"
    FEATURE_MATCHING = "feature_matching"


class DeviceTier(str, Enum):
    """部署设备算力等级"""
    EDGE_LOW = "edge_low"          # Jetson Nano, RPi, 低端嵌入式
    EDGE_MID = "edge_mid"          # Jetson Xavier NX, 中端边缘
    EDGE_HIGH = "edge_high"        # Jetson AGX Orin, 高端边缘
    DESKTOP_CPU = "desktop_cpu"    # 普通台式机 CPU
    DESKTOP_GPU = "desktop_gpu"    # 消费级 GPU (RTX 3060-4090)
    SERVER_GPU = "server_gpu"      # 服务器 GPU (A100/H100/L40)
    APPLE_SILICON = "apple_silicon" # M1/M2/M3
    CLOUD_GENERIC = "cloud_generic" # 通用云端


@dataclass
class PretrainedModel:
    """一个预训练模型的完整描述"""
    model_id: str                       # 唯一标识, e.g. "yolov8s.pt"
    family: str                         # ModelFamily value
    variant: str                        # nano/small/medium/large/xlarge
    display_name: str                   # 人类可读名称
    display_name_zh: str                # 中文名称
    task_types: list[str]               # 支持的任务类型
    input_size_default: int             # 默认输入分辨率
    input_size_options: list[int]       # 可选分辨率
    params_m: float                     # 参数量 (百万)
    flops_g: float                      # 计算量 (GFLOPS)
    map50_coco: float | None            # COCO mAP50 (如有)
    map50_95_coco: float | None         # COCO mAP50-95 (如有)
    fps_gpu: float | None               # GPU 推理速度参考 (FPS)
    fps_cpu: float | None               # CPU 推理速度参考 (FPS)
    min_device_tier: str                # 最低部署设备等级
    recommended_device_tiers: list[str] # 推荐设备等级
    export_formats: list[str]           # 支持的导出格式
    training_framework: str             # ultralytics / mmdet / custom
    weight_url: str | None              # 权重下载地址
    description: str                    # 模型特点描述
    description_zh: str                 # 中文描述
    strengths: list[str]                # 优势标签
    weaknesses: list[str]               # 劣势标签
    use_cases: list[str]                # 适用场景
    requires_api: bool = False          # 是否需要第三方 API
    api_provider: str | None = None     # API 提供商
    is_available: bool = True           # 当前是否可用


@dataclass
class TrainedModelCache:
    """一个已训练模型的缓存记录"""
    cache_id: str
    source_model_id: str                # 基于哪个预训练模型
    task_id: str                        # 来自哪个任务
    classes: list[str]                  # 训练的类别列表
    class_count: int
    scenario_type: str                  # 场景类型
    map50: float | None
    map50_95: float | None
    weight_path: str                    # 权重文件路径
    export_paths: dict[str, str]        # 格式 -> 路径
    trained_at: float                   # 时间戳
    image_count: int                    # 训练图片数
    epochs_completed: int
    tags: list[str] = field(default_factory=list)
    reuse_count: int = 0                # 被复用次数


# ---------------------------------------------------------------------------
# 预训练模型目录（内置）
# ---------------------------------------------------------------------------

_BUILTIN_MODELS: list[dict] = [
    # ── YOLOv5 系列 ──
    {
        "model_id": "yolov5n.pt", "family": "yolov5", "variant": "nano",
        "display_name": "YOLOv5 Nano", "display_name_zh": "YOLOv5 极小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 1.9, "flops_g": 4.5, "map50_coco": 46.0, "map50_95_coco": 28.0,
        "fps_gpu": 450, "fps_cpu": 45,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Smallest YOLOv5, ideal for edge deployment with minimal compute",
        "description_zh": "YOLOv5 最小模型，适合极低算力的边缘设备部署",
        "strengths": ["极小体积", "超快推理", "广泛兼容"],
        "weaknesses": ["精度较低", "小目标表现一般"],
        "use_cases": ["简单计数", "大目标检测", "嵌入式部署"],
    },
    {
        "model_id": "yolov5s.pt", "family": "yolov5", "variant": "small",
        "display_name": "YOLOv5 Small", "display_name_zh": "YOLOv5 小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 7.2, "flops_g": 16.5, "map50_coco": 56.8, "map50_95_coco": 37.4,
        "fps_gpu": 370, "fps_cpu": 28,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Balanced YOLOv5 model, good speed-accuracy tradeoff",
        "description_zh": "YOLOv5 经典小型模型，速度与精度平衡，社区资源丰富",
        "strengths": ["成熟稳定", "社区资源丰富", "部署方案完善"],
        "weaknesses": ["架构较旧", "不如新版精度高"],
        "use_cases": ["通用目标检测", "工业质检", "安防监控"],
    },
    {
        "model_id": "yolov5m.pt", "family": "yolov5", "variant": "medium",
        "display_name": "YOLOv5 Medium", "display_name_zh": "YOLOv5 中型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 21.2, "flops_g": 49.0, "map50_coco": 64.1, "map50_95_coco": 45.4,
        "fps_gpu": 230, "fps_cpu": 12,
        "min_device_tier": "edge_high", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Mid-range YOLOv5 with better accuracy for complex scenes",
        "description_zh": "YOLOv5 中型模型，复杂场景精度更好",
        "strengths": ["精度较高", "资源消耗适中"],
        "weaknesses": ["边缘设备较慢"],
        "use_cases": ["多目标检测", "复杂场景", "中等密度监控"],
    },
    # ── YOLOv8 系列 ──
    {
        "model_id": "yolov8n.pt", "family": "yolov8", "variant": "nano",
        "display_name": "YOLOv8 Nano", "display_name_zh": "YOLOv8 极小型",
        "task_types": ["detection", "classification", "segmentation", "pose"],
        "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 3.2, "flops_g": 8.7, "map50_coco": 52.6, "map50_95_coco": 37.3,
        "fps_gpu": 520, "fps_cpu": 40,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Latest nano architecture with improved feature extraction",
        "description_zh": "YOLOv8 极小型模型，新架构比 v5n 精度更高",
        "strengths": ["新架构", "高速", "多任务支持"],
        "weaknesses": ["极小目标表现有限"],
        "use_cases": ["实时边缘检测", "移动端部署", "简单分类"],
    },
    {
        "model_id": "yolov8s.pt", "family": "yolov8", "variant": "small",
        "display_name": "YOLOv8 Small", "display_name_zh": "YOLOv8 小型",
        "task_types": ["detection", "classification", "segmentation", "pose"],
        "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 11.2, "flops_g": 28.6, "map50_coco": 61.8, "map50_95_coco": 44.9,
        "fps_gpu": 410, "fps_cpu": 22,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Strong small model with anchor-free design",
        "description_zh": "YOLOv8 小型模型，无锚框设计，综合性价比高",
        "strengths": ["无锚框", "训练稳定", "多任务"],
        "weaknesses": ["资源略多于 v5s"],
        "use_cases": ["通用检测", "实例分割", "人体姿态", "中等复杂度场景"],
    },
    {
        "model_id": "yolov8m.pt", "family": "yolov8", "variant": "medium",
        "display_name": "YOLOv8 Medium", "display_name_zh": "YOLOv8 中型",
        "task_types": ["detection", "classification", "segmentation", "pose"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 25.9, "flops_g": 78.9, "map50_coco": 67.2, "map50_95_coco": 50.2,
        "fps_gpu": 280, "fps_cpu": 10,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "High-accuracy YOLOv8 for demanding detection tasks",
        "description_zh": "YOLOv8 中型模型，高精度，适合复杂检测任务",
        "strengths": ["高精度", "特征提取强"],
        "weaknesses": ["需要 GPU", "边缘设备慢"],
        "use_cases": ["多类别检测", "密集场景", "精细分割"],
    },
    {
        "model_id": "yolov8l.pt", "family": "yolov8", "variant": "large",
        "display_name": "YOLOv8 Large", "display_name_zh": "YOLOv8 大型",
        "task_types": ["detection", "classification", "segmentation", "pose"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 43.7, "flops_g": 165.2, "map50_coco": 69.8, "map50_95_coco": 52.9,
        "fps_gpu": 170, "fps_cpu": 5,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Large YOLOv8 for maximum accuracy on server GPU",
        "description_zh": "YOLOv8 大型模型，服务器 GPU 部署追求最高精度",
        "strengths": ["最高精度档位之一", "小目标好"],
        "weaknesses": ["需大显存", "推理较慢"],
        "use_cases": ["高精度需求", "小目标密集", "服务器部署"],
    },
    # ── YOLOv11 系列 ──
    {
        "model_id": "yolo11n.pt", "family": "yolov11", "variant": "nano",
        "display_name": "YOLO11 Nano", "display_name_zh": "YOLO11 极小型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 2.6, "flops_g": 6.5, "map50_coco": 54.0, "map50_95_coco": 39.5,
        "fps_gpu": 580, "fps_cpu": 48,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Latest YOLO11 nano with improved efficiency",
        "description_zh": "YOLO11 最新架构极小模型，效率优于前代",
        "strengths": ["最新架构", "极高效率", "OBB 支持"],
        "weaknesses": ["社区资源尚在积累"],
        "use_cases": ["实时边缘", "旋转目标", "移动端"],
    },
    {
        "model_id": "yolo11s.pt", "family": "yolov11", "variant": "small",
        "display_name": "YOLO11 Small", "display_name_zh": "YOLO11 小型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 9.4, "flops_g": 21.5, "map50_coco": 63.0, "map50_95_coco": 47.0,
        "fps_gpu": 450, "fps_cpu": 25,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Balanced YOLO11 small model, recommended default",
        "description_zh": "YOLO11 小型模型，推荐默认选择，平衡性能与精度",
        "strengths": ["推荐默认", "训练快", "多任务"],
        "weaknesses": ["极小目标可加大分辨率"],
        "use_cases": ["通用检测", "安防", "工业质检", "OBB 检测"],
    },
    {
        "model_id": "yolo11m.pt", "family": "yolov11", "variant": "medium",
        "display_name": "YOLO11 Medium", "display_name_zh": "YOLO11 中型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 20.1, "flops_g": 68.0, "map50_coco": 68.5, "map50_95_coco": 51.5,
        "fps_gpu": 310, "fps_cpu": 11,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO11 medium for complex multi-class scenarios",
        "description_zh": "YOLO11 中型模型，适合多类别和复杂场景",
        "strengths": ["高精度", "多任务"],
        "weaknesses": ["训练时间较长"],
        "use_cases": ["复杂检测", "多目标", "精确分割"],
    },
    {
        "model_id": "yolo11l.pt", "family": "yolov11", "variant": "large",
        "display_name": "YOLO11 Large", "display_name_zh": "YOLO11 大型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 25.3, "flops_g": 86.9, "map50_coco": 70.0, "map50_95_coco": 53.4,
        "fps_gpu": 210, "fps_cpu": 6,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO11 large for maximum detection performance",
        "description_zh": "YOLO11 大型模型，追求极致检测性能",
        "strengths": ["顶级精度", "小目标优秀"],
        "weaknesses": ["大显存需求", "推理较慢"],
        "use_cases": ["高精度需求", "小目标密集", "服务器部署"],
    },
    # ── RF-DETR 系列 ──
    {
        "model_id": "rf-detr-base", "family": "rf-detr", "variant": "base",
        "display_name": "RF-DETR Base", "display_name_zh": "RF-DETR 基础型",
        "task_types": ["detection"], "input_size_default": 560, "input_size_options": [560, 640],
        "params_m": 29.0, "flops_g": 120.0, "map50_coco": 72.5, "map50_95_coco": 53.3,
        "fps_gpu": 130, "fps_cpu": 3,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "rf_detr", "weight_url": None,
        "description": "Roboflow DETR — transformer-based detector with high accuracy, good for dense/overlapping objects",
        "description_zh": "RF-DETR 基于 Transformer 的检测器，密集/重叠目标表现优异，精度非常高",
        "strengths": ["Transformer 架构", "密集目标优秀", "无 NMS", "高精度"],
        "weaknesses": ["推理较慢", "需 GPU", "导出格式有限"],
        "use_cases": ["密集目标检测", "重叠遮挡场景", "高精度需求", "仓位匹配"],
    },
    {
        "model_id": "rf-detr-large", "family": "rf-detr", "variant": "large",
        "display_name": "RF-DETR Large", "display_name_zh": "RF-DETR 大型",
        "task_types": ["detection"], "input_size_default": 560, "input_size_options": [560, 640],
        "params_m": 128.0, "flops_g": 450.0, "map50_coco": 75.0, "map50_95_coco": 56.2,
        "fps_gpu": 55, "fps_cpu": 1,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "rf_detr", "weight_url": None,
        "description": "Large RF-DETR for absolute maximum accuracy on powerful servers",
        "description_zh": "RF-DETR 大型模型，超高精度，需要高性能服务器 GPU",
        "strengths": ["最高精度", "Transformer", "密集场景"],
        "weaknesses": ["非常大", "推理慢", "需 A100 级 GPU"],
        "use_cases": ["极高精度需求", "研究级", "服务器批量处理"],
    },
    # ── RT-DETR 系列 ──
    {
        "model_id": "rtdetr-l.pt", "family": "rt-detr", "variant": "large",
        "display_name": "RT-DETR Large", "display_name_zh": "RT-DETR 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 32.0, "flops_g": 110.0, "map50_coco": 71.5, "map50_95_coco": 53.0,
        "fps_gpu": 160, "fps_cpu": 4,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Real-Time DETR, high accuracy with reasonable speed on GPU",
        "description_zh": "实时 DETR 模型，GPU 上兼顾精度与速度",
        "strengths": ["Transformer", "ultralytics 集成", "无 NMS"],
        "weaknesses": ["CPU 很慢", "边缘设备不适用"],
        "use_cases": ["GPU 实时检测", "密集场景", "精准定位"],
    },
    {
        "model_id": "rtdetr-x.pt", "family": "rt-detr", "variant": "xlarge",
        "display_name": "RT-DETR XLarge", "display_name_zh": "RT-DETR 超大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 67.0, "flops_g": 234.0, "map50_coco": 73.0, "map50_95_coco": 54.8,
        "fps_gpu": 95, "fps_cpu": 2,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Largest RT-DETR, for maximum Transformer-based detection accuracy",
        "description_zh": "RT-DETR 超大型模型，Transformer 检测最高精度档",
        "strengths": ["极高精度", "Transformer"],
        "weaknesses": ["非常大", "仅服务器"],
        "use_cases": ["最高精度需求", "服务器部署"],
    },
    # ── OCR 引擎 ──
    {
        "model_id": "easyocr",
        "family": "ocr",
        "variant": "default",
        "display_name": "EasyOCR",
        "display_name_zh": "EasyOCR 文字识别",
        "task_types": ["ocr"],
        "input_size_default": 640,
        "input_size_options": [320, 640, 1280],
        "params_m": 0,
        "flops_g": 0,
        "map50_coco": None,
        "map50_95_coco": None,
        "fps_gpu": 0,
        "fps_cpu": 0,
        "min_device_tier": "desktop_cpu",
        "recommended_device_tiers": ["desktop_cpu", "desktop_gpu", "edge_low", "edge_mid", "edge_high", "apple_silicon"],
        "export_formats": [],
        "training_framework": "custom",
        "weight_url": None,
        "description": "Pure Python OCR engine, supports 80+ languages, no C++ dependencies",
        "description_zh": "纯 Python OCR 引擎，支持 80+ 语言，零额外依赖，部署最简单",
        "strengths": ["纯 Python", "多语言支持", "部署零负担"],
        "weaknesses": ["精度略低于 PaddleOCR", "CPU 密集"],
        "use_cases": ["票据识别", "车牌识别", "文档 OCR", "工业读数"],
    },
]


# ---------------------------------------------------------------------------
# ModelRegistry 主类
# ---------------------------------------------------------------------------

_CACHE_DIR_ENV = "CV_AUTO_TRAINER_MODEL_CACHE_DIR"
_DEFAULT_CACHE_DIR = "model_cache"


class ModelRegistry:
    """模型注册表：管理预训练目录 + 训练缓存"""

    def __init__(self, cache_dir: str | None = None):
        import threading
        self._pretrained: dict[str, PretrainedModel] = {}
        self._cache: dict[str, TrainedModelCache] = {}
        self._cache_dir = Path(
            cache_dir or os.getenv(_CACHE_DIR_ENV, _DEFAULT_CACHE_DIR)
        )
        self._cache_index_path = self._cache_dir / "cache_index.json"
        self._write_lock = threading.RLock()

        self._load_builtin_models()
        self._load_cache_index()

    # ── 预训练模型 ──

    def _load_builtin_models(self):
        for raw in _BUILTIN_MODELS:
            model = PretrainedModel(**raw)
            self._pretrained[model.model_id] = model

    def register_model(self, model: PretrainedModel):
        self._pretrained[model.model_id] = model

    def get_model(self, model_id: str) -> PretrainedModel | None:
        return self._pretrained.get(model_id)

    def list_models(
        self,
        family: str | None = None,
        task_type: str | None = None,
        max_device_tier: str | None = None,
        available_only: bool = True,
    ) -> list[PretrainedModel]:
        results = list(self._pretrained.values())
        if available_only:
            results = [m for m in results if m.is_available]
        if family:
            results = [m for m in results if m.family == family]
        if task_type:
            results = [m for m in results if task_type in m.task_types]
        if max_device_tier:
            tier_order = _device_tier_order()
            max_idx = tier_order.get(max_device_tier, 999)
            results = [
                m for m in results
                if tier_order.get(m.min_device_tier, 999) <= max_idx
            ]
        return results

    def get_models_summary_for_vlm(
        self,
        device_tier: str | None = None,
        task_type: str | None = None,
    ) -> str:
        """生成一段给 VLM 的模型目录摘要文本，用于辅助 VLM 做模型选择。"""
        models = self.list_models(
            task_type=task_type,
            max_device_tier=device_tier,
        )
        if not models:
            models = self.list_models(task_type=task_type)

        lines = ["可选预训练模型列表："]
        for m in models:
            if m.family == "ocr":
                lines.append(
                    f"- [✓] {m.model_id} ({m.display_name_zh}): "
                    f"任务={','.join(m.task_types)}, "
                    f"特点: {m.description_zh}"
                )
                continue
            tier_ok = "✓" if device_tier and device_tier in m.recommended_device_tiers else "○"
            lines.append(
                f"- [{tier_ok}] {m.model_id} ({m.display_name_zh}): "
                f"参数 {m.params_m}M, mAP50={m.map50_coco}, "
                f"GPU {m.fps_gpu}fps, CPU {m.fps_cpu}fps, "
                f"最低设备={m.min_device_tier}, "
                f"任务={','.join(m.task_types)}, "
                f"特点: {m.description_zh}"
            )
        return "\n".join(lines)

    def get_models_as_dicts(
        self,
        device_tier: str | None = None,
        task_type: str | None = None,
    ) -> list[dict]:
        models = self.list_models(task_type=task_type, max_device_tier=device_tier)
        if not models:
            models = self.list_models(task_type=task_type)
        return [asdict(m) for m in models]

    # ── 训练缓存 ──

    def _load_cache_index(self):
        if not self._cache_index_path.exists():
            return
        try:
            data = json.loads(self._cache_index_path.read_text(encoding="utf-8"))
            for entry in data:
                cache = TrainedModelCache(**entry)
                self._cache[cache.cache_id] = cache
        except Exception:
            pass

    def _save_cache_index(self):
        """原子写入：先写 tmp，再 rename；配合 threading.Lock 防止并发损坏 JSON。"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        with self._write_lock:
            entries = [asdict(c) for c in self._cache.values()]
            tmp_path = self._cache_index_path.with_suffix(".json.tmp")
            tmp_path.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(tmp_path, self._cache_index_path)

    def register_trained_model(self, cache: TrainedModelCache):
        with self._write_lock:
            self._cache[cache.cache_id] = cache
        self._save_cache_index()

    def find_reusable_model(
        self,
        required_classes: list[str],
        scenario_type: str | None = None,
        min_map50: float = 0.5,
    ) -> TrainedModelCache | None:
        """查找可复用的已训练模型：类别完全覆盖 + 精度达标"""
        required_set = set(c.lower().strip() for c in required_classes)
        best: TrainedModelCache | None = None
        best_score = -1.0

        for cache in self._cache.values():
            cached_classes = set(c.lower().strip() for c in cache.classes)
            if not required_set.issubset(cached_classes):
                continue
            if cache.map50 is not None and cache.map50 < min_map50:
                continue
            score = (cache.map50 or 0.0) + (0.1 if scenario_type and cache.scenario_type == scenario_type else 0.0)
            if score > best_score:
                best_score = score
                best = cache

        return best

    def list_cached_models(self) -> list[TrainedModelCache]:
        return list(self._cache.values())

    def increment_reuse(self, cache_id: str):
        with self._write_lock:
            if cache_id in self._cache:
                self._cache[cache_id].reuse_count += 1
            else:
                return
        self._save_cache_index()


# ---------------------------------------------------------------------------
# 设备等级排序
# ---------------------------------------------------------------------------

def _device_tier_order() -> dict[str, int]:
    return {
        DeviceTier.EDGE_LOW.value: 0,
        DeviceTier.EDGE_MID.value: 1,
        DeviceTier.EDGE_HIGH.value: 2,
        DeviceTier.DESKTOP_CPU.value: 3,
        DeviceTier.APPLE_SILICON.value: 4,
        DeviceTier.DESKTOP_GPU.value: 5,
        DeviceTier.SERVER_GPU.value: 6,
        DeviceTier.CLOUD_GENERIC.value: 7,
    }


def infer_device_tier(gpu_type: str | None, platform: str | None = None) -> str:
    """从 GPU 型号字符串推断设备等级"""
    if not gpu_type:
        if platform and ("mac" in platform.lower() or "darwin" in platform.lower()):
            return DeviceTier.APPLE_SILICON.value
        return DeviceTier.DESKTOP_CPU.value

    gpu_lower = gpu_type.lower().strip()

    if any(kw in gpu_lower for kw in ("a100", "h100", "h200", "l40", "a6000", "a40", "v100")):
        return DeviceTier.SERVER_GPU.value
    if any(kw in gpu_lower for kw in ("rtx", "gtx", "rtx 3", "rtx 4", "rtx 2", "titan")):
        return DeviceTier.DESKTOP_GPU.value
    if "orin" in gpu_lower or "agx" in gpu_lower:
        return DeviceTier.EDGE_HIGH.value
    if "xavier" in gpu_lower or "nx" in gpu_lower:
        return DeviceTier.EDGE_MID.value
    if any(kw in gpu_lower for kw in ("nano", "jetson nano", "rpi", "raspberry")):
        return DeviceTier.EDGE_LOW.value
    if any(kw in gpu_lower for kw in ("m1", "m2", "m3", "m4", "apple")):
        return DeviceTier.APPLE_SILICON.value
    if "cloud" in gpu_lower or "autodl" in gpu_lower:
        return DeviceTier.CLOUD_GENERIC.value

    return DeviceTier.DESKTOP_GPU.value


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_registry_instance: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry()
    return _registry_instance
