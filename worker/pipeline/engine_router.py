"""
v9.0 优化文档 3.1 节：打标引擎三路由策略。

根据类别词集合在 YOLO-World / CLIP 强词表中的覆盖率，
决定走 yolo_world / grounding_dino / hybrid 三种模式之一。

对外接口：
- select_engine(classes, task_type=None, user_preference="auto") -> dict
    返回 {
      "engine": "yolo_world" | "grounding_dino" | "hybrid",
      "reason": str,
      "strong_indices": list[int],  # 在 classes 列表中的原始下标
      "weak_indices": list[int],
    }
- partition_classes(classes) -> (strong_with_idx, weak_with_idx)
- remap_raw_boxes(raw_boxes, idx_map) -> raw_boxes（把每个 box 的 class_idx 按 idx_map 重新映射到全局下标）
- merge_raw_boxes(a, b) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


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
