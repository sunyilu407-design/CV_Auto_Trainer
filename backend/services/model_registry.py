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
    RT_DETR = "rt-detr"
    YOLOX = "yolox"
    EFFICIENTDET = "efficientdet"
    SAM = "sam"
    OCR = "ocr"
    CUSTOM = "custom"
    RF_DETR = "rf-detr"


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
    {
        "model_id": "yolov8x.pt", "family": "yolov8", "variant": "xlarge",
        "display_name": "YOLOv8 XLarge", "display_name_zh": "YOLOv8 超大型",
        "task_types": ["detection", "classification", "segmentation", "pose"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 68.2, "flops_g": 257.8, "map50_coco": 71.0, "map50_95_coco": 54.5,
        "fps_gpu": 100, "fps_cpu": 2,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Largest YOLOv8 for absolute maximum accuracy",
        "description_zh": "YOLOv8 超大型，v8 系列最高精度，适合追求极致精度的服务器场景",
        "strengths": ["v8 系列最高精度", "密集目标强"],
        "weaknesses": ["需高端 GPU", "推理最慢"],
        "use_cases": ["最高精度需求", "研究级", "服务器批量"],
    },
    {
        "model_id": "yolov5l.pt", "family": "yolov5", "variant": "large",
        "display_name": "YOLOv5 Large", "display_name_zh": "YOLOv5 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 46.5, "flops_g": 109.1, "map50_coco": 67.3, "map50_95_coco": 49.0,
        "fps_gpu": 140, "fps_cpu": 6,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Large YOLOv5 for high-accuracy deployment on GPU",
        "description_zh": "YOLOv5 大型模型，高精度 GPU 部署，成熟稳定",
        "strengths": ["高精度", "成熟稳定"],
        "weaknesses": ["体积大", "需 GPU 显存"],
        "use_cases": ["高精度需求", "服务器部署"],
    },
    {
        "model_id": "yolov5x.pt", "family": "yolov5", "variant": "xlarge",
        "display_name": "YOLOv5 XLarge", "display_name_zh": "YOLOv5 超大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 86.7, "flops_g": 205.7, "map50_coco": 68.7, "map50_95_coco": 50.7,
        "fps_gpu": 85, "fps_cpu": 3,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "Largest YOLOv5 for maximum accuracy",
        "description_zh": "YOLOv5 超大型，v5 系列最高精度档",
        "strengths": ["v5 系列最高精度"],
        "weaknesses": ["非常大", "推理慢"],
        "use_cases": ["最高精度", "服务器批量"],
    },

    # ══════════════════════════════════════════════════════════════
    # YOLOv9 系列 — GELAN + PGI 技术，精度提升显著
    # 适用：精度优先、愿意用较新架构、GPU 资源充足
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "yolov9n.pt", "family": "yolov9", "variant": "nano",
        "display_name": "YOLOv9 Nano", "display_name_zh": "YOLOv9 极小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 2.0, "flops_g": 3.8, "map50_coco": 46.8, "map50_95_coco": 38.3,
        "fps_gpu": 500, "fps_cpu": 50,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv9 smallest — GELAN architecture, higher accuracy than v8n at similar speed",
        "description_zh": "YOLOv9 极小型，GELAN 新架构，同等速度下精度高于 v8n",
        "strengths": ["GELAN 架构", "高效率", "精度优于 v8n"],
        "weaknesses": ["社区资源少", "生态不如 v8/v11"],
        "use_cases": ["实时边缘", "嵌入式", "对精度有要求的低功耗场景"],
    },
    {
        "model_id": "yolov9s.pt", "family": "yolov9", "variant": "small",
        "display_name": "YOLOv9 Small", "display_name_zh": "YOLOv9 小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 7.1, "flops_g": 17.5, "map50_coco": 51.4, "map50_95_coco": 40.2,
        "fps_gpu": 380, "fps_cpu": 24,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv9 small — PGI/GELAN architecture, stronger than v8s at comparable cost",
        "description_zh": "YOLOv9 小型，PGI+GELAN 技术，精度明显优于 v8s，适合作为 v8 的升级替代",
        "strengths": ["精度优于 v8s", "PGI 训练辅助", "GELAN 高效"],
        "weaknesses": ["比 v8s 稍慢", "生态较新"],
        "use_cases": ["精度优先的检测", "复杂场景", "替代 v8s 升级"],
    },
    {
        "model_id": "yolov9m.pt", "family": "yolov9", "variant": "medium",
        "display_name": "YOLOv9 Medium", "display_name_zh": "YOLOv9 中型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 20.1, "flops_g": 51.8, "map50_coco": 53.7, "map50_95_coco": 42.8,
        "fps_gpu": 250, "fps_cpu": 11,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv9 medium — significant accuracy gain over v8m",
        "description_zh": "YOLOv9 中型，相比 v8m 有明显精度提升，复杂场景首选",
        "strengths": ["精度大幅优于 v8m", "大目标检测强"],
        "weaknesses": ["边缘设备较慢"],
        "use_cases": ["复杂检测", "多目标", "精度敏感场景"],
    },
    {
        "model_id": "yolov9l.pt", "family": "yolov9", "variant": "large",
        "display_name": "YOLOv9 Large", "display_name_zh": "YOLOv9 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 37.5, "flops_g": 100.7, "map50_coco": 55.0, "map50_95_coco": 43.7,
        "fps_gpu": 160, "fps_cpu": 6,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv9 large for high-accuracy GPU deployment",
        "description_zh": "YOLOv9 大型，高精度 GPU 部署首选之一",
        "strengths": ["高精度", "大目标好"],
        "weaknesses": ["需大显存"],
        "use_cases": ["高精度需求", "服务器部署"],
    },
    {
        "model_id": "yolov9x.pt", "family": "yolov9", "variant": "xlarge",
        "display_name": "YOLOv9 XLarge", "display_name_zh": "YOLOv9 超大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 75.1, "flops_g": 185.3, "map50_coco": 56.1, "map50_95_coco": 44.9,
        "fps_gpu": 90, "fps_cpu": 2,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv9 xlarge — highest precision in YOLOv9 family",
        "description_zh": "YOLOv9 超大型，v9 系列最高精度，精度与算力需求均最高",
        "strengths": ["v9 系列最高精度"],
        "weaknesses": ["需高端 GPU"],
        "use_cases": ["研究级精度", "服务器批量"],
    },

    # ══════════════════════════════════════════════════════════════
    # YOLOv10 系列 — NMS-free 端到端，延迟最低，实时最优
    # 适用：延迟敏感场景（NMS 后处理开销大）、实时跟踪
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "yolov10n.pt", "family": "yolov10", "variant": "nano",
        "display_name": "YOLOv10 Nano", "display_name_zh": "YOLOv10 极小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 2.3, "flops_g": 6.7, "map50_coco": 53.0, "map50_95_coco": 39.8,
        "fps_gpu": 600, "fps_cpu": 55,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv10 nano — NMS-free end-to-end, highest speed at nano tier",
        "description_zh": "YOLOv10 极小型，端到端无 NMS，nano 档位速度最快",
        "strengths": ["NMS-free", "最高速度", "延迟最优"],
        "weaknesses": ["生态新"],
        "use_cases": ["实时边缘", "超低延迟需求", "跟踪任务"],
    },
    {
        "model_id": "yolov10s.pt", "family": "yolov10", "variant": "small",
        "display_name": "YOLOv10 Small", "display_name_zh": "YOLOv10 小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 7.2, "flops_g": 21.4, "map50_coco": 63.5, "map50_95_coco": 47.5,
        "fps_gpu": 480, "fps_cpu": 28,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv10 small — NMS-free, recommended default for real-time tasks",
        "description_zh": "YOLOv10 小型，推荐默认选择，NMS-free 延迟最低，实时跟踪首选",
        "strengths": ["NMS-free", "推荐默认", "速度最快", "跟踪友好"],
        "weaknesses": ["生态较新"],
        "use_cases": ["实时检测", "多目标跟踪", "延迟敏感场景"],
    },
    {
        "model_id": "yolov10m.pt", "family": "yolov10", "variant": "medium",
        "display_name": "YOLOv10 Medium", "display_name_zh": "YOLOv10 中型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 15.4, "flops_g": 59.2, "map50_coco": 67.0, "map50_95_coco": 51.0,
        "fps_gpu": 330, "fps_cpu": 13,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv10 medium — NMS-free, balanced accuracy and speed",
        "description_zh": "YOLOv10 中型，精度与速度平衡，端到端无 NMS 后处理开销",
        "strengths": ["NMS-free", "精度速度平衡"],
        "weaknesses": ["需 GPU"],
        "use_cases": ["复杂检测", "实时推理", "跟踪流水线"],
    },
    {
        "model_id": "yolov10l.pt", "family": "yolov10", "variant": "large",
        "display_name": "YOLOv10 Large", "display_name_zh": "YOLOv10 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 24.4, "flops_g": 91.3, "map50_coco": 69.5, "map50_95_coco": 52.8,
        "fps_gpu": 220, "fps_cpu": 7,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv10 large for high-accuracy real-time inference",
        "description_zh": "YOLOv10 大型，高精度实时推理，服务器 GPU 推荐",
        "strengths": ["高精度", "NMS-free"],
        "weaknesses": ["需大显存"],
        "use_cases": ["高精度实时", "服务器部署"],
    },
    {
        "model_id": "yolov10x.pt", "family": "yolov10", "variant": "xlarge",
        "display_name": "YOLOv10 XLarge", "display_name_zh": "YOLOv10 超大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 29.0, "flops_g": 137.2, "map50_coco": 71.5, "map50_95_coco": 54.5,
        "fps_gpu": 140, "fps_cpu": 3,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOv10 xlarge — highest precision in v10 family",
        "description_zh": "YOLOv10 超大型，v10 系列最高精度，适合追求极致实时精度",
        "strengths": ["v10 系列最高精度", "NMS-free"],
        "weaknesses": ["需高端 GPU"],
        "use_cases": ["最高精度实时", "研究级"],
    },

    # ══════════════════════════════════════════════════════════════
    # YOLO11 系列 — Ultralytics 最新一代，轻量高效
    # 适用：大多数场景首选，综合最优，OBB 旋转目标检测
    # ══════════════════════════════════════════════════════════════
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
        "description": "Latest YOLO11 nano with improved efficiency over all previous nano models",
        "description_zh": "YOLO11 最新架构极小模型，效率优于前代，支持 OBB 旋转目标检测",
        "strengths": ["最新架构", "极高效率", "OBB 支持", "多任务"],
        "weaknesses": ["社区资源尚在积累"],
        "use_cases": ["实时边缘", "旋转目标", "移动端", "OBB 检测"],
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
        "description": "Balanced YOLO11 small model — recommended default for most scenarios",
        "description_zh": "YOLO11 小型，推荐默认选择，平衡性能与精度，多任务覆盖",
        "strengths": ["推荐默认", "训练快", "多任务", "OBB"],
        "weaknesses": ["极小目标可加大分辨率"],
        "use_cases": ["通用检测", "安防", "工业质检", "OBB 检测", "旋转目标"],
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
        "description_zh": "YOLO11 中型，适合多类别和复杂场景，精度优秀",
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
        "description_zh": "YOLO11 大型，追求极致检测性能，小目标优秀",
        "strengths": ["高精度", "小目标优秀"],
        "weaknesses": ["大显存需求"],
        "use_cases": ["高精度需求", "小目标密集", "服务器部署"],
    },

    # ══════════════════════════════════════════════════════════════
    # YOLO26 系列 — 2025 年最新一代，极致精度与效率
    # 适用：对精度有极致要求、愿意使用最新架构
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "yolo26n.pt", "family": "yolo26", "variant": "nano",
        "display_name": "YOLO26 Nano", "display_name_zh": "YOLO26 极小型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [320, 640],
        "params_m": 2.0, "flops_g": 5.0, "map50_coco": 55.5, "map50_95_coco": 41.0,
        "fps_gpu": 620, "fps_cpu": 52,
        "min_device_tier": "edge_low", "recommended_device_tiers": ["edge_low", "edge_mid", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO26 nano — 2025 latest generation, best-in-class efficiency",
        "description_zh": "YOLO26 最新一代极小模型，各项指标全面超越前代",
        "strengths": ["最新架构", "效率最优", "多任务"],
        "weaknesses": ["最新架构，生态待完善"],
        "use_cases": ["最新边缘部署", "多任务场景"],
    },
    {
        "model_id": "yolo26s.pt", "family": "yolo26", "variant": "small",
        "display_name": "YOLO26 Small", "display_name_zh": "YOLO26 小型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [320, 640, 1280],
        "params_m": 7.2, "flops_g": 16.5, "map50_coco": 65.5, "map50_95_coco": 49.0,
        "fps_gpu": 490, "fps_cpu": 27,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO26 small — latest generation, best accuracy/speed ratio",
        "description_zh": "YOLO26 小型，最新一代，精度速度比最优，多任务支持",
        "strengths": ["最新架构", "精度速度比最优", "多任务", "OBB"],
        "weaknesses": ["新架构"],
        "use_cases": ["通用检测", "工业质检", "最新部署"],
    },
    {
        "model_id": "yolo26m.pt", "family": "yolo26", "variant": "medium",
        "display_name": "YOLO26 Medium", "display_name_zh": "YOLO26 中型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 16.0, "flops_g": 55.0, "map50_coco": 70.0, "map50_95_coco": 53.0,
        "fps_gpu": 340, "fps_cpu": 12,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu", "apple_silicon"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO26 medium — best-in-class accuracy for its size tier",
        "description_zh": "YOLO26 中型，同档位精度最高，适合复杂多类别场景",
        "strengths": ["同档位最高精度", "最新架构"],
        "weaknesses": ["需 GPU"],
        "use_cases": ["复杂检测", "高要求场景"],
    },
    {
        "model_id": "yolo26l.pt", "family": "yolo26", "variant": "large",
        "display_name": "YOLO26 Large", "display_name_zh": "YOLO26 大型",
        "task_types": ["detection", "classification", "segmentation", "pose", "obb"],
        "input_size_default": 640, "input_size_options": [640, 1280],
        "params_m": 22.0, "flops_g": 76.0, "map50_coco": 71.5, "map50_95_coco": 54.5,
        "fps_gpu": 230, "fps_cpu": 7,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLO26 large — top-tier accuracy with excellent efficiency",
        "description_zh": "YOLO26 大型，高精度高效率，服务器部署最新首选",
        "strengths": ["高精度", "高效率"],
        "weaknesses": ["需大显存"],
        "use_cases": ["高精度服务器部署", "复杂场景"],
    },

    # ══════════════════════════════════════════════════════════════
    # RT-DETR 系列 — 实时 DETR，Transformer 检测兼顾速度
    # 适用：密集/重叠目标，高精度实时，GPU 推理
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "rtdetr-s.pt", "family": "rt-detr", "variant": "small",
        "display_name": "RT-DETR Small", "display_name_zh": "RT-DETR 小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 20.0, "flops_g": 60.0, "map50_coco": 69.0, "map50_95_coco": 50.0,
        "fps_gpu": 200, "fps_cpu": 4,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "RT-DETR small — lightweight real-time DETR, great for dense objects",
        "description_zh": "RT-DETR 小型，Transformer 检测器，密集目标表现好，适合桌面 GPU",
        "strengths": ["Transformer 架构", "密集目标好", "无 NMS", "精度高"],
        "weaknesses": ["CPU 极慢", "边缘设备不适用"],
        "use_cases": ["GPU 实时检测", "密集场景", "桌面 GPU 部署"],
    },
    {
        "model_id": "rtdetr-m.pt", "family": "rt-detr", "variant": "medium",
        "display_name": "RT-DETR Medium", "display_name_zh": "RT-DETR 中型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 27.0, "flops_g": 85.0, "map50_coco": 70.5, "map50_95_coco": 51.8,
        "fps_gpu": 175, "fps_cpu": 3,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "RT-DETR medium — balanced real-time DETR for complex scenes",
        "description_zh": "RT-DETR 中型，精度与速度平衡，Transformer 检测适合复杂场景",
        "strengths": ["Transformer", "无 NMS", "精度高", "密集目标"],
        "weaknesses": ["仅 GPU"],
        "use_cases": ["复杂场景检测", "实时推理"],
    },
    {
        "model_id": "rtdetr-l.pt", "family": "rt-detr", "variant": "large",
        "display_name": "RT-DETR Large", "display_name_zh": "RT-DETR 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 32.0, "flops_g": 110.0, "map50_coco": 71.5, "map50_95_coco": 53.0,
        "fps_gpu": 160, "fps_cpu": 4,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "RT-DETR large — high-accuracy real-time DETR on GPU",
        "description_zh": "RT-DETR 大型，高精度实时检测，GPU 兼顾速度与精度",
        "strengths": ["Transformer", "高精度", "无 NMS"],
        "weaknesses": ["需大显存"],
        "use_cases": ["高精度实时", "服务器部署"],
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
        "description": "RT-DETR xlarge — maximum Transformer-based detection accuracy",
        "description_zh": "RT-DETR 超大型，Transformer 检测最高精度档，适合服务器",
        "strengths": ["最高精度 RT-DETR", "Transformer"],
        "weaknesses": ["需高端 GPU"],
        "use_cases": ["最高精度需求", "服务器"],
    },

    # ══════════════════════════════════════════════════════════════
    # YOLOX 系列 — 美团 Anchor-Free 成熟方案
    # 适用：需要 Anchor-Free（避免锚框调参）、成熟稳定框架
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "yolox-s", "family": "yolox", "variant": "small",
        "display_name": "YOLOX Small", "display_name_zh": "YOLOX 小型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 9.0, "flops_g": 26.8, "map50_coco": 60.8, "map50_95_coco": 44.3,
        "fps_gpu": 380, "fps_cpu": 20,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "edge_high", "desktop_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml", "tflite"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOX small — anchor-free YOLO by Meituan, mature and stable",
        "description_zh": "YOLOX 小型，美团开源 Anchor-Free 方案，无需锚框调参，社区成熟",
        "strengths": ["Anchor-Free", "无需锚框调参", "成熟稳定", "训练简单"],
        "weaknesses": ["速度略慢于 v8/v10"],
        "use_cases": ["避免锚框调参", "标准检测任务", "成熟框架"],
    },
    {
        "model_id": "yolox-m", "family": "yolox", "variant": "medium",
        "display_name": "YOLOX Medium", "display_name_zh": "YOLOX 中型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 25.3, "flops_g": 73.8, "map50_coco": 64.5, "map50_95_coco": 47.5,
        "fps_gpu": 240, "fps_cpu": 10,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine", "openvino", "coreml"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOX medium — anchor-free, good accuracy for complex scenes",
        "description_zh": "YOLOX 中型，Anchor-Free 架构，复杂场景精度好，训练稳定",
        "strengths": ["Anchor-Free", "精度好", "训练稳定"],
        "weaknesses": ["边缘设备慢"],
        "use_cases": ["复杂场景", "多目标检测"],
    },
    {
        "model_id": "yolox-l", "family": "yolox", "variant": "large",
        "display_name": "YOLOX Large", "display_name_zh": "YOLOX 大型",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 54.2, "flops_g": 155.7, "map50_coco": 66.7, "map50_95_coco": 49.6,
        "fps_gpu": 150, "fps_cpu": 5,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx", "engine", "openvino"],
        "training_framework": "ultralytics", "weight_url": None,
        "description": "YOLOX large for high-accuracy anchor-free detection",
        "description_zh": "YOLOX 大型，Anchor-Free 高精度，适合服务器部署",
        "strengths": ["Anchor-Free", "高精度"],
        "weaknesses": ["需 GPU"],
        "use_cases": ["高精度需求", "服务器部署"],
    },

    # ══════════════════════════════════════════════════════════════
    # EfficientDet 系列 — Google 早期检测器，精度效率平衡
    # 适用：互补方案、CPU 友好、TFLite 边缘部署（非 ultralytics）
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "efficientdet-d0", "family": "efficientdet", "variant": "d0",
        "display_name": "EfficientDet D0", "display_name_zh": "EfficientDet D0",
        "task_types": ["detection"], "input_size_default": 512, "input_size_options": [512],
        "params_m": 5.2, "flops_g": 6.4, "map50_coco": 51.5, "map50_95_coco": 33.8,
        "fps_gpu": 280, "fps_cpu": 30,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "tflite"],
        "training_framework": "custom", "weight_url": None,
        "description": "EfficientDet D0 — Google compound-scaled detector, lightweight baseline",
        "description_zh": "EfficientDet D0，Google 出品，复合缩放架构，轻量级，适合 CPU/Mobile 部署",
        "strengths": ["Google 架构", "轻量", "CPU 友好", "TFLite 导出"],
        "weaknesses": ["精度低于 YOLO", "非 ultralytics 生态"],
        "use_cases": ["CPU 部署", "移动端", "TFLite 边缘"],
    },
    {
        "model_id": "efficientdet-d1", "family": "efficientdet", "variant": "d1",
        "display_name": "EfficientDet D1", "display_name_zh": "EfficientDet D1",
        "task_types": ["detection"], "input_size_default": 640, "input_size_options": [640],
        "params_m": 6.6, "flops_g": 11.4, "map50_coco": 53.8, "map50_95_coco": 36.5,
        "fps_gpu": 220, "fps_cpu": 20,
        "min_device_tier": "edge_mid", "recommended_device_tiers": ["edge_mid", "desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "tflite"],
        "training_framework": "custom", "weight_url": None,
        "description": "EfficientDet D1 — compound scaling, better accuracy than D0",
        "description_zh": "EfficientDet D1，精度优于 D0，CPU 端友好，TFLite 移动部署首选",
        "strengths": ["精度优于 D0", "TFLite", "CPU 好"],
        "weaknesses": ["非 ultralytics"],
        "use_cases": ["CPU 检测", "移动端", "TFLite 部署"],
    },
    {
        "model_id": "efficientdet-d2", "family": "efficientdet", "variant": "d2",
        "display_name": "EfficientDet D2", "display_name_zh": "EfficientDet D2",
        "task_types": ["detection"], "input_size_default": 768, "input_size_options": [768],
        "params_m": 8.1, "flops_g": 17.9, "map50_coco": 56.2, "map50_95_coco": 39.3,
        "fps_gpu": 170, "fps_cpu": 14,
        "min_device_tier": "edge_high", "recommended_device_tiers": ["desktop_gpu", "apple_silicon"],
        "export_formats": ["onnx", "tflite"],
        "training_framework": "custom", "weight_url": None,
        "description": "EfficientDet D2 — compound scaling, higher accuracy, good for mobile GPU",
        "description_zh": "EfficientDet D2，高精度档，Apple GPU 友好，TFLite 部署质量高",
        "strengths": ["精度好", "Apple GPU", "TFLite"],
        "weaknesses": ["非 ultralytics 生态"],
        "use_cases": ["高精度 CPU/Mobile", "Apple 部署"],
    },

    # ══════════════════════════════════════════════════════════════
    # SAM-2 系列 — Meta 视觉分割基础模型
    # 适用：图像分割、抠图、交互式分割（非传统目标检测）
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "sam2-tiny", "family": "sam", "variant": "tiny",
        "display_name": "SAM-2 Tiny", "display_name_zh": "SAM-2 极小型",
        "task_types": ["segmentation"], "input_size_default": 1024, "input_size_options": [1024],
        "params_m": 39.0, "flops_g": 0, "map50_coco": None, "map50_95_coco": None,
        "fps_gpu": 40, "fps_cpu": 0,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx"],
        "training_framework": "sam2", "weight_url": None,
        "description": "SAM-2 tiny — Meta Segment Anything 2, zero-shot image segmentation foundation model",
        "description_zh": "SAM-2 极小型，Meta 分割一切二代，极轻量，适合交互式分割和抠图",
        "strengths": ["零样本分割", "交互式分割", "SAM 基础"],
        "weaknesses": ["不是检测器", "需 GPU", "非实时"],
        "use_cases": ["图像分割", "目标抠图", "交互式标注", "分割辅助打标"],
    },
    {
        "model_id": "sam2-small", "family": "sam", "variant": "small",
        "display_name": "SAM-2 Small", "display_name_zh": "SAM-2 小型",
        "task_types": ["segmentation"], "input_size_default": 1024, "input_size_options": [1024],
        "params_m": 46.0, "flops_g": 0, "map50_coco": None, "map50_95_coco": None,
        "fps_gpu": 35, "fps_cpu": 0,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx"],
        "training_framework": "sam2", "weight_url": None,
        "description": "SAM-2 small — balanced zero-shot segmentation, recommended default for SAM-2",
        "description_zh": "SAM-2 小型，推荐默认，零样本分割，质量与速度平衡",
        "strengths": ["零样本", "质量速度平衡", "分割质量高"],
        "weaknesses": ["需 GPU", "非检测"],
        "use_cases": ["图像分割", "抠图", "辅助标注"],
    },
    {
        "model_id": "sam2-base", "family": "sam", "variant": "base",
        "display_name": "SAM-2 Base", "display_name_zh": "SAM-2 基础型",
        "task_types": ["segmentation"], "input_size_default": 1024, "input_size_options": [1024],
        "params_m": 97.0, "flops_g": 0, "map50_coco": None, "map50_95_coco": None,
        "fps_gpu": 25, "fps_cpu": 0,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx"],
        "training_framework": "sam2", "weight_url": None,
        "description": "SAM-2 base — higher quality segmentation at moderate cost",
        "description_zh": "SAM-2 基础型，高质量分割，适合对分割质量有要求的场景",
        "strengths": ["高质量", "零样本"],
        "weaknesses": ["需较大显存"],
        "use_cases": ["高质量分割", "服务器批量"],
    },
    {
        "model_id": "sam2-large", "family": "sam", "variant": "large",
        "display_name": "SAM-2 Large", "display_name_zh": "SAM-2 大型",
        "task_types": ["segmentation"], "input_size_default": 1024, "input_size_options": [1024],
        "params_m": 200.0, "flops_g": 0, "map50_coco": None, "map50_95_coco": None,
        "fps_gpu": 15, "fps_cpu": 0,
        "min_device_tier": "server_gpu", "recommended_device_tiers": ["server_gpu"],
        "export_formats": ["onnx"],
        "training_framework": "sam2", "weight_url": None,
        "description": "SAM-2 large — highest quality segmentation for professional use",
        "description_zh": "SAM-2 大型，专业级分割质量，适合对精度要求最高的场景",
        "strengths": ["最高分割质量", "专业级"],
        "weaknesses": ["需高端 GPU"],
        "use_cases": ["最高质量需求", "专业标注"],
    },

    # ══════════════════════════════════════════════════════════════
    # RF-DETR 系列 — Roboflow Transformer 检测器
    # 适用：高精度需求、密集重叠目标
    # ══════════════════════════════════════════════════════════════
    {
        "model_id": "rf-detr-base", "family": "rf-detr", "variant": "base",
        "display_name": "RF-DETR Base", "display_name_zh": "RF-DETR 基础型",
        "task_types": ["detection"], "input_size_default": 560, "input_size_options": [560, 640],
        "params_m": 29.0, "flops_g": 120.0, "map50_coco": 72.5, "map50_95_coco": 53.3,
        "fps_gpu": 130, "fps_cpu": 3,
        "min_device_tier": "desktop_gpu", "recommended_device_tiers": ["desktop_gpu", "server_gpu"],
        "export_formats": ["onnx", "engine"],
        "training_framework": "rf_detr", "weight_url": None,
        "description": "Roboflow DETR — transformer-based, excellent for dense/overlapping objects",
        "description_zh": "RF-DETR 基于 Transformer，密集/重叠目标表现优异，精度非常高",
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
        "description": "Large RF-DETR — maximum accuracy on powerful servers",
        "description_zh": "RF-DETR 大型，超高精度，需要高性能服务器 GPU",
        "strengths": ["最高精度", "Transformer", "密集场景"],
        "weaknesses": ["非常大", "推理慢", "需 A100 级 GPU"],
        "use_cases": ["极高精度需求", "研究级", "服务器批量"],
    },

    # ══════════════════════════════════════════════════════════════
    # OCR 引擎
    # ══════════════════════════════════════════════════════════════
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

    # 服务器级 GPU（A100/H100/昇腾/Quadro L4/A10 等）
    if any(kw in gpu_lower for kw in ("a100", "h100", "h200", "l40", "a6000", "a40", "v100", "h800", "a10", "l4", "ascend", "npu")):
        return DeviceTier.SERVER_GPU.value

    # 消费级 GPU（RTX / GTX / Quadro）
    if any(kw in gpu_lower for kw in ("rtx", "gtx", "titan", "quadro")):
        if "4090" in gpu_lower or "3090" in gpu_lower:
            return DeviceTier.SERVER_GPU.value
        return DeviceTier.DESKTOP_GPU.value

    # NVIDIA Jetson 边缘
    if "orin" in gpu_lower:
        return DeviceTier.EDGE_HIGH.value
    if "xavier" in gpu_lower:
        return DeviceTier.EDGE_MID.value
    if "nano" in gpu_lower:
        return DeviceTier.EDGE_LOW.value

    # 其他边缘/嵌入式
    if any(kw in gpu_lower for kw in ("rpi", "raspberry", "jetson", "movidius", "edge tpu")):
        return DeviceTier.EDGE_LOW.value

    # Apple Silicon
    if any(kw in gpu_lower for kw in ("m1", "m2", "m3", "m4", "apple")):
        return DeviceTier.APPLE_SILICON.value

    # 云端
    if "cloud" in gpu_lower or "autodl" in gpu_lower:
        return DeviceTier.CLOUD_GENERIC.value

    # 未知：保守判断为消费级 GPU
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
