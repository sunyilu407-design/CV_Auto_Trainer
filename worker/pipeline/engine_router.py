"""
v9.0 优化文档 3.1 节：打标引擎路由策略。

根据类别词集合在 YOLO-World / CLIP 强词表中的覆盖率，
决定走 yolo_world / grounding_dino / hybrid 三种模式之一。

扩展支持：LocateAnything / Eagle2.5 (NVIDIA Eagle 家族)

对外接口：
- select_engine(classes, task_type=None, user_preference="auto") -> dict
    返回 {
      "engine": "yolo_world" | "grounding_dino" | "hybrid" | "locate_anything",
      "reason": str,
      "strong_indices": list[int],  # 在 classes 列表中的原始下标
      "weak_indices": list[int],
    }
- select_vqa_engine(user_preference="auto") -> dict
    返回 {
      "engine": "moondream" | "eagle_vqa",
      "reason": str,
    }
- partition_classes(classes) -> (strong_with_idx, weak_with_idx)
- remap_raw_boxes(raw_boxes, idx_map) -> raw_boxes（把每个 box 的 class_idx 按 idx_map 重新映射到全局下标）
- merge_raw_boxes(a, b) -> dict
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Tuple, Optional

import torch


# COCO + Objects365 + 常见安全 / 工业场景中 YOLO-World 的强词集合
# 参考 v9.0 优化文档 3.1 节
YOLO_WORLD_STRONG_NOUNS = {
    # 人
    "person", "worker", "pedestrian", "cyclist", "driver",
    # 车
    "car", "sedan", "truck", "bus", "bicycle", "motorcycle", "forklift", "excavator",
    # 动物
    "dog", "cat", "bird", "cow", "horse", "sheep",
    # 日常物品
    "bottle", "cup", "chair", "table", "laptop", "phone", "book",
    "fork", "knife", "spoon", "pizza", "cake",
    # 安全 PPE（YOLO-World 在 COCO/Objects365 微调衍生模型中支持较好）
    "hard_hat", "helmet", "safety_vest", "face_mask", "glove",
    # 交通 / 基础设施
    "traffic_light", "stop_sign", "fire_hydrant",
}


def _normalize_token(s: str) -> str:
    """小写 + 去空格/连字符，用于与强词集合匹配。"""
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def _class_token_candidates(cls: Dict[str, Any]) -> List[str]:
    """从一个 class dict 抽出所有可能的识别 token（class_name + prompt_aliases）。"""
    out = []
    cn = cls.get("class_name", "")
    if cn:
        out.append(_normalize_token(cn))
    for alias in cls.get("prompt_aliases", []) or []:
        if isinstance(alias, str) and alias:
            out.append(_normalize_token(alias))
    return out


def is_strong_class(cls: Dict[str, Any]) -> bool:
    """类别是否命中 YOLO-World 强词集合（任一 token 命中即算强）。"""
    for tok in _class_token_candidates(cls):
        if tok in YOLO_WORLD_STRONG_NOUNS:
            return True
    return False


def partition_classes(
    classes: List[Dict[str, Any]],
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, Dict[str, Any]]]]:
    """
    把 classes 按 YOLO-World 强词集合命中情况分成 strong / weak 两组。
    返回值元素格式：(原始下标, class_dict)
    """
    strong: List[Tuple[int, Dict[str, Any]]] = []
    weak: List[Tuple[int, Dict[str, Any]]] = []
    for idx, cls in enumerate(classes):
        if is_strong_class(cls):
            strong.append((idx, cls))
        else:
            weak.append((idx, cls))
    return strong, weak


def select_engine(
    classes: List[Dict[str, Any]],
    task_type: str = "",
    user_preference: str = "auto",
) -> Dict[str, Any]:
    """
    选择最合适的打标引擎。

    user_preference == "yolo_world" | "grounding_dino" → 强制指定，绕过自动判断
    user_preference == "auto"（默认）→ 按覆盖率 + task_type 自动选：
      - task_type 含 industrial/medical/custom 或 coverage < 0.3 → grounding_dino
      - coverage >= 0.8 → yolo_world
      - 否则 → hybrid
    """
    if not classes:
        return {
            "engine": "yolo_world",
            "reason": "classes 为空，默认 yolo_world（不会真正命中任何类别）",
            "strong_indices": [],
            "weak_indices": [],
        }

    strong, weak = partition_classes(classes)
    strong_indices = [i for i, _ in strong]
    weak_indices = [i for i, _ in weak]
    coverage = len(strong) / len(classes)

    pref = (user_preference or "auto").lower()
    if pref in {"yolo_world", "grounding_dino"}:
        return {
            "engine": pref,
            "reason": f"用户强制指定引擎 = {pref}",
            "strong_indices": strong_indices,
            "weak_indices": weak_indices,
            "coverage": coverage,
        }

    tt = (task_type or "").lower()
    industrial_like = any(k in tt for k in ("industrial", "custom", "medical", "工业", "医疗", "自定义"))

    if industrial_like or coverage < 0.3:
        return {
            "engine": "grounding_dino",
            "reason": (
                f"覆盖率 {coverage:.0%} < 30% 或 task_type='{task_type}' 属于工业/医疗/自定义 "
                f"场景，CLIP 强词支持弱，改用 Grounding DINO"
            ),
            "strong_indices": strong_indices,
            "weak_indices": weak_indices,
            "coverage": coverage,
        }
    if coverage >= 0.8:
        return {
            "engine": "yolo_world",
            "reason": f"覆盖率 {coverage:.0%} ≥ 80%，全部或几乎全部类别属于 YOLO-World 强词集",
            "strong_indices": strong_indices,
            "weak_indices": weak_indices,
            "coverage": coverage,
        }
    return {
        "engine": "hybrid",
        "reason": (
            f"覆盖率 {coverage:.0%}：强词用 YOLO-World、弱词用 Grounding DINO，"
            f"结果合并以最大化召回"
        ),
        "strong_indices": strong_indices,
        "weak_indices": weak_indices,
        "coverage": coverage,
    }


def remap_raw_boxes(
    raw_boxes: Dict[str, List[Dict[str, Any]]],
    idx_map: List[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    把 raw_boxes 中每个 box 的 class_idx 根据 idx_map 转回全局下标。
    idx_map[sub_idx] = original_idx
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for img_path, boxes in raw_boxes.items():
        new_boxes = []
        for b in boxes:
            sub_idx = int(b.get("class_idx", 0))
            if 0 <= sub_idx < len(idx_map):
                new_b = dict(b)
                new_b["class_idx"] = idx_map[sub_idx]
                new_boxes.append(new_b)
            else:
                # 越界（不正常）- 保留但打标记
                new_b = dict(b)
                new_b["_idx_remap_failed"] = True
                new_boxes.append(new_b)
        out[img_path] = new_boxes
    return out


def merge_raw_boxes(
    a: Dict[str, List[Dict[str, Any]]],
    b: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, List[Dict[str, Any]]]:
    """简单按图片路径合并两个 raw_boxes 字典。"""
    out: Dict[str, List[Dict[str, Any]]] = {k: list(v) for k, v in a.items()}
    for img_path, boxes in b.items():
        if img_path in out:
            out[img_path] = out[img_path] + list(boxes)
        else:
            out[img_path] = list(boxes)
    return out


# ============== Eagle 家族引擎扩展 ==============

class DetectionEngineType(str, Enum):
    """检测引擎类型枚举"""
    YOLO_WORLD = "yolo_world"
    GROUNDING_DINO = "grounding_dino"
    HYBRID = "hybrid"
    LOCATE_ANYTHING = "locate_anything"


class VqaEngineType(str, Enum):
    """VQA 质检引擎类型枚举"""
    MOONDREAM = "moondream"
    EAGLE_VQA = "eagle_vqa"


def get_available_gpu_memory_gb() -> float:
    """获取可用的 GPU 显存（GB）"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        allocated = torch.cuda.memory_allocated(0) / 1e9
        return total - allocated
    return 0.0


def can_run_locate_anything() -> bool:
    """检查是否可以运行 LocateAnything"""
    if not torch.cuda.is_available():
        return False
    # LocateAnything 需要约 6GB 显存 (FP16)
    free = get_available_gpu_memory_gb()
    return free >= 6.0


def can_run_eagle_vqa() -> bool:
    """检查是否可以运行 Eagle2.5 VQA"""
    if not torch.cuda.is_available():
        return False
    # Eagle2.5 需要约 16GB 显存 (FP16)
    free = get_available_gpu_memory_gb()
    return free >= 16.0


def select_detection_engine(
    classes: List[Dict[str, Any]] = None,
    user_preference: str = "auto",
) -> Dict[str, Any]:
    """
    选择检测引擎，支持 LocateAnything
    
    Args:
        classes: 类别列表
        user_preference: 用户偏好 "auto" | "yolo_world" | "locate_anything"
    
    Returns:
        dict: {
            "engine": str,
            "reason": str,
            "available": bool,
            "vram_required_gb": float,
        }
    """
    pref = (user_preference or "auto").lower()
    
    # 用户强制指定
    if pref == "locate_anything":
        can_run = can_run_locate_anything()
        if can_run:
            return {
                "engine": DetectionEngineType.LOCATE_ANYTHING.value,
                "reason": "用户强制指定 LocateAnything",
                "available": True,
                "vram_required_gb": 6.0,
            }
        else:
            return {
                "engine": DetectionEngineType.YOLO_WORLD.value,
                "reason": f"LocateAnything 需要 ~6GB 显存，当前 GPU 显存不足，降至 YOLO-World",
                "available": False,
                "vram_required_gb": 6.0,
                "fallback": DetectionEngineType.YOLO_WORLD.value,
            }
    
    if pref == "yolo_world":
        return {
            "engine": DetectionEngineType.YOLO_WORLD.value,
            "reason": "用户强制指定 YOLO-World",
            "available": True,
            "vram_required_gb": 3.0,
        }
    
    # 自动模式：优先使用 LocateAnything（如果可用）
    if pref == "auto":
        # 检查是否适合 LocateAnything
        # LocateAnything 对所有类别都支持，不需要强词判断
        if can_run_locate_anything():
            return {
                "engine": DetectionEngineType.LOCATE_ANYTHING.value,
                "reason": "自动选择 LocateAnything：支持开放词汇检测、OCR、GUI定位，速度更快(12.7 BPS)",
                "available": True,
                "vram_required_gb": 6.0,
            }
        else:
            # 回退到原有逻辑
            result = select_engine(classes, user_preference="auto")
            return {
                "engine": result["engine"],
                "reason": f"{result['reason']}（LocateAnything 不可用：显存不足）",
                "available": True,
                "vram_required_gb": 3.0,
            }
    
    # 默认回退
    return {
        "engine": DetectionEngineType.YOLO_WORLD.value,
        "reason": "默认使用 YOLO-World",
        "available": True,
        "vram_required_gb": 3.0,
    }


def select_vqa_engine(
    user_preference: str = "auto",
) -> Dict[str, Any]:
    """
    选择 VQA 质检引擎，支持 Eagle2.5
    
    Args:
        user_preference: 用户偏好 "auto" | "moondream" | "eagle_vqa"
    
    Returns:
        dict: {
            "engine": str,
            "reason": str,
            "available": bool,
            "vram_required_gb": float,
        }
    """
    pref = (user_preference or "auto").lower()
    
    # 用户强制指定
    if pref == "eagle_vqa":
        can_run = can_run_eagle_vqa()
        if can_run:
            return {
                "engine": VqaEngineType.EAGLE_VQA.value,
                "reason": "用户强制指定 Eagle2.5 VQA",
                "available": True,
                "vram_required_gb": 16.0,
            }
        else:
            return {
                "engine": VqaEngineType.MOONDREAM.value,
                "reason": f"Eagle2.5 VQA 需要 ~16GB 显存，当前 GPU 显存不足，降至 Moondream2",
                "available": False,
                "vram_required_gb": 16.0,
                "fallback": VqaEngineType.MOONDREAM.value,
            }
    
    if pref == "moondream":
        return {
            "engine": VqaEngineType.MOONDREAM.value,
            "reason": "用户强制指定 Moondream2",
            "available": True,
            "vram_required_gb": 4.0,
        }
    
    # 自动模式：优先使用 Eagle2.5（如果可用）
    if pref == "auto":
        if can_run_eagle_vqa():
            return {
                "engine": VqaEngineType.EAGLE_VQA.value,
                "reason": "自动选择 Eagle2.5 VQA：更强的视觉理解能力，支持长上下文(128K)",
                "available": True,
                "vram_required_gb": 16.0,
            }
        else:
            return {
                "engine": VqaEngineType.MOONDREAM.value,
                "reason": "Eagle2.5 VQA 不可用（显存不足），使用 Moondream2",
                "available": can_run_eagle_vqa(),
                "vram_required_gb": 4.0,
            }
    
    # 默认回退
    return {
        "engine": VqaEngineType.MOONDREAM.value,
        "reason": "默认使用 Moondream2",
        "available": True,
        "vram_required_gb": 4.0,
    }


def get_engine_info() -> Dict[str, Any]:
    """
    获取所有引擎的可用性信息
    
    Returns:
        dict: 各引擎的可用性和显存需求
    """
    return {
        "detection_engines": {
            "yolo_world": {
                "name": "YOLO-World",
                "available": True,  # YOLO-World 支持 CPU
                "vram_required_gb": 3.0,
                "features": ["目标检测", "开放词汇"],
            },
            "grounding_dino": {
                "name": "Grounding DINO",
                "available": True,
                "vram_required_gb": 4.0,
                "features": ["目标检测", "开放词汇", "零样本"],
            },
            "locate_anything": {
                "name": "LocateAnything",
                "available": can_run_locate_anything(),
                "vram_required_gb": 6.0,
                "features": ["目标检测", "开放词汇", "OCR", "GUI定位", "点定位", "12.7 BPS"],
            },
        },
        "vqa_engines": {
            "moondream": {
                "name": "Moondream2",
                "available": True,  # Moondream2 支持 CPU
                "vram_required_gb": 4.0,
                "features": ["VQA质检", "图像描述"],
            },
            "eagle_vqa": {
                "name": "Eagle2.5 VQA",
                "available": can_run_eagle_vqa(),
                "vram_required_gb": 16.0,
                "features": ["VQA质检", "图像描述", "长上下文(128K)", "高分辨率(4K)"],
            },
        },
        "system": {
            "has_cuda": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            "free_vram_gb": get_available_gpu_memory_gb(),
        },
    }
