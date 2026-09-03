"""
Worker 配置文件

定义所有可配置的参数，包括：
- 检测引擎选择
- 模型路径和参数
- 显存管理
- VQA 质检设置
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ============== 路径配置 ==============

WORKER_ROOT = Path(__file__).resolve().parents[1]


# ============== 引擎类型枚举 ==============

class DetectionEngine(str, Enum):
    """检测引擎选项"""
    YOLO_WORLD = "yolo_world"
    GROUNDING_DINO = "grounding_dino"
    LOCATE_ANYTHING = "locate_anything"


class VqaEngine(str, Enum):
    """VQA 质检引擎选项"""
    MOONDREAM = "moondream"
    EAGLE_VQA = "eagle_vqa"


class EnginePreference(str, Enum):
    """引擎选择偏好"""
    AUTO = "auto"
    FORCE_LEGACY = "force_legacy"  # 强制使用 YOLO-World + Moondream2
    FORCE_EAGLE = "force_eagle"    # 强制使用 LocateAnything + Eagle2.5


# ============== 检测引擎配置 ==============

@dataclass
class YoloWorldConfig:
    """YOLO-World 配置"""
    weight_name: str = "yolov8s-world.pt"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    fp16: bool = True
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    batch_size: int = 4
    imgsz: int = 1280


@dataclass
class LocateAnythingConfig:
    """LocateAnything 配置"""
    model_id: str = "nvidia/LocateAnything-3B"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    torch_dtype: str = "float16"  # "float16" | "float32"
    conf_threshold: float = 0.25
    batch_size: int = 4


@dataclass
class GroundingDinoConfig:
    """Grounding DINO 配置"""
    model_id: str = "IDEA-Research/grounding-dino-tiny"
    device: str = "auto"
    box_threshold: float = 0.35
    text_threshold: float = 0.25


# ============== VQA 引擎配置 ==============

@dataclass
class MoondreamConfig:
    """Moondream2 配置"""
    model_id: str = "vikhyatk/moondream2"
    device: str = "auto"
    quality_threshold: float = 0.5
    clarity_weight: float = 0.4
    completeness_weight: float = 0.3
    accuracy_weight: float = 0.3


@dataclass
class EagleVqaConfig:
    """Eagle2.5 VQA 配置"""
    model_id: str = "nvidia/Eagle2.5-8B"
    device: str = "auto"
    torch_dtype: str = "float16"
    quality_threshold: float = 0.5
    max_new_tokens: int = 256


# ============== 显存管理配置 ==============

@dataclass
class VramConfig:
    """显存管理配置"""
    reserve_gb: float = 2.0  # 保留显存
    check_enabled: bool = True
    per_model: dict = field(default_factory=lambda: {
        "yolo_world": 3.0,
        "moondream2": 4.0,
        "grounding_dino": 4.0,
        "locate_anything": 6.0,
        "eagle_vqa": 16.0,
    })


# ============== 主配置类 ==============

@dataclass
class WorkerConfig:
    """Worker 主配置"""
    
    # 引擎偏好设置
    engine_preference: str = "auto"  # "auto" | "force_legacy" | "force_eagle"
    
    # 检测引擎配置
    yolo_world: YoloWorldConfig = field(default_factory=YoloWorldConfig)
    locate_anything: LocateAnythingConfig = field(default_factory=LocateAnythingConfig)
    grounding_dino: GroundingDinoConfig = field(default_factory=GroundingDinoConfig)
    
    # VQA 引擎配置
    moondream: MoondreamConfig = field(default_factory=MoondreamConfig)
    eagle_vqa: EagleVqaConfig = field(default_factory=EagleVqaConfig)
    
    # 显存管理
    vram: VramConfig = field(default_factory=VramConfig)
    
    # HuggingFace 配置
    hf_download_timeout: int = 300  # 秒
    hf_endpoint: str = ""  # 空则自动检测
    
    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # 引擎偏好
        config.engine_preference = os.getenv("CV_ENGINE_PREFERENCE", "auto")
        
        # YOLO-World
        config.yolo_world.conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
        config.yolo_world.batch_size = int(os.getenv("YOLO_BATCH_SIZE", "4"))
        
        # LocateAnything
        config.locate_anything.conf_threshold = float(os.getenv("LOCATE_CONF_THRESHOLD", "0.25"))
        
        # VQA
        config.moondream.quality_threshold = float(os.getenv("MOONDREAM_QUALITY_THRESHOLD", "0.5"))
        config.eagle_vqa.quality_threshold = float(os.getenv("EAGLE_QUALITY_THRESHOLD", "0.5"))
        
        # HF 配置
        timeout = os.getenv("HF_HUB_DOWNLOAD_TIMEOUT")
        if timeout:
            config.hf_download_timeout = int(timeout)
        
        hf_endpoint = os.getenv("HF_ENDPOINT")
        if hf_endpoint:
            config.hf_endpoint = hf_endpoint
        
        return config
    
    def get_detection_config(self, engine: DetectionEngine):
        """获取检测引擎配置"""
        if engine == DetectionEngine.LOCATE_ANYTHING:
            return self.locate_anything
        elif engine == DetectionEngine.GROUNDING_DINO:
            return self.grounding_dino
        else:
            return self.yolo_world
    
    def get_vqa_config(self, engine: VqaEngine):
        """获取 VQA 引擎配置"""
        if engine == VqaEngine.EAGLE_VQA:
            return self.eagle_vqa
        else:
            return self.moondream


# ============== 全局配置实例 ==============

_config: Optional[WorkerConfig] = None


def get_config() -> WorkerConfig:
    """获取全局配置实例（懒加载）"""
    global _config
    if _config is None:
        _config = WorkerConfig.from_env()
    return _config


def reload_config() -> WorkerConfig:
    """重新加载配置"""
    global _config
    _config = WorkerConfig.from_env()
    return _config


# ============== CLI 辅助函数 ==============

def print_engine_status():
    """打印引擎可用性状态"""
    from engine_router import get_engine_info
    
    info = get_engine_info()
    system = info["system"]
    
    print("=" * 60)
    print("CV Auto Trainer - 引擎状态")
    print("=" * 60)
    
    print(f"\n系统信息:")
    print(f"  CUDA 可用: {'是' if system['has_cuda'] else '否'}")
    if system['has_cuda']:
        print(f"  GPU: {system['gpu_name']}")
        print(f"  总显存: {system['total_vram_gb']:.1f} GB")
        print(f"  可用显存: {system['free_vram_gb']:.1f} GB")
    
    print(f"\n检测引擎:")
    for key, eng in info["detection_engines"].items():
        status = "可用" if eng["available"] else "显存不足"
        print(f"  {eng['name']}: {status} (需要 {eng['vram_required_gb']:.0f} GB)")
        print(f"    特性: {', '.join(eng['features'])}")
    
    print(f"\nVQA 引擎:")
    for key, eng in info["vqa_engines"].items():
        status = "可用" if eng["available"] else "显存不足"
        print(f"  {eng['name']}: {status} (需要 {eng['vram_required_gb']:.0f} GB)")
        print(f"    特性: {', '.join(eng['features'])}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_engine_status()
