"""
输出格式转换工具 — 统一不同模型输出的格式

支持：
- LocateAnything → YOLO xywhn 格式
- YOLO-World → 标准化格式
- Moondream2 → 质检报告格式
- Eagle2.5 → 质检报告格式
"""

import json
import logging
from pathlib import Path
from typing import Optional, Callable

import torch

logger = logging.getLogger(__name__)


# ============== 格式定义 ==============

class DetectionFormat:
    """检测结果格式定义"""
    
    # YOLO-World 原始输出
    YOLO_RAW_SCHEMA = {
        "class_idx": int,
        "class_name": str,
        "prompt": str,
        "bbox_xywhn": list,  # [cx, cy, w, h] 归一化
        "conf": float,
        "_source": str,  # "yolo_world" | "locate_anything" | "existing_label"
    }
    
    # LocateAnything 输出
    LOCATE_SCHEMA = {
        "x1": float,
        "y1": float,
        "x2": float,
        "y2": float,
        "bbox_xywhn": list,
        "conf": float,
    }
    
    # 质检报告格式
    QUALITY_REPORT_SCHEMA = {
        "passed": bool,
        "scores": dict,  # {"clarity": float, "completeness": float, "accuracy": float}
        "rejected": bool,
        "reason": str,
    }


# ============== 坐标转换 ==============

def xywhn_to_xyxy(xywhn: list[float]) -> dict:
    """
    将归一化 xywh 转换为 xyxy 像素坐标
    
    Args:
        xywhn: [cx, cy, w, h]，归一化坐标
    
    Returns:
        dict: {"x1", "y1", "x2", "y2"} 像素坐标
    """
    cx, cy, w, h = xywhn
    x1 = (cx - w / 2)
    y1 = (cy - h / 2)
    x2 = (cx + w / 2)
    y2 = (cy + h / 2)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def xyxy_to_xywhn(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """
    将 xyxy 像素坐标转换为归一化 xywh
    
    Args:
        x1, y1, x2, y2: 像素坐标
    
    Returns:
        list: [cx, cy, w, h]，归一化坐标
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def normalize_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    将像素坐标 bbox 归一化
    
    Args:
        bbox: [x1, y1, x2, y2] 或 [x, y, w, h] 像素坐标
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        list: 归一化后的 bbox
    """
    if len(bbox) == 4:
        # 假设是 xyxy 格式
        x1, y1, x2, y2 = bbox
        return [x1 / image_width, y1 / image_height, x2 / image_width, y2 / image_height]
    return bbox


def denormalize_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    将归一化 bbox 转换为像素坐标
    
    Args:
        bbox: 归一化 bbox
        image_width: 图像宽度
        image_height: 图像高度
    
    Returns:
        list: 像素坐标 bbox
    """
    return [b * (image_width if i % 2 == 0 else image_height) for i, b in enumerate(bbox)]


# ============== 格式标准化 ==============

def standardize_detection(
    bbox_xywhn: list[float],
    class_name: str,
    class_idx: Optional[int] = None,
    conf: float = 1.0,
    source: str = "unknown",
) -> dict:
    """
    创建标准化的检测结果
    
    Args:
        bbox_xywhn: 归一化边界框 [cx, cy, w, h]
        class_name: 类别名称
        class_idx: 类别索引
        conf: 置信度
        source: 来源 ("yolo_world" | "locate_anything" | "grounding_dino")
    
    Returns:
        dict: 标准化的检测结果
    """
    xyxy = xywhn_to_xyxy(bbox_xywhn)
    
    return {
        "class_idx": class_idx,
        "class_name": class_name,
        "bbox_xywhn": bbox_xywhn,
        "x1": xyxy["x1"],
        "y1": xyxy["y1"],
        "x2": xyxy["x2"],
        "y2": xyxy["y2"],
        "conf": conf,
        "_source": source,
    }


def standardize_quality_report(
    passed: bool,
    scores: dict,
    rejected: bool,
    reason: str,
    details: Optional[dict] = None,
) -> dict:
    """
    创建标准化的质检报告
    
    Args:
        passed: 是否通过
        scores: 评分 {"clarity": float, "completeness": float, "accuracy": float}
        rejected: 是否被拒绝
        reason: 原因
        details: 额外详情
    
    Returns:
        dict: 标准化的质检报告
    """
    report = {
        "passed": passed,
        "scores": {
            "clarity": scores.get("clarity", 0.0),
            "completeness": scores.get("completeness", 0.0),
            "accuracy": scores.get("accuracy", 0.0),
        },
        "rejected": rejected,
        "reason": reason,
    }
    
    if details:
        report["details"] = details
    
    return report


# ============== NMS 和去重 ==============

def compute_iou(box1: list[float], box2: list[float]) -> float:
    """
    计算两个 xywhn 边界框的 IoU
    
    Args:
        box1: [cx, cy, w, h]
        box2: [cx, cy, w, h]
    
    Returns:
        float: IoU 值 [0, 1]
    """
    # 转换为 xyxy
    def to_xyxy(b):
        cx, cy, w, h = b
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        return x1, y1, x2, y2
    
    x1_1, y1_1, x2_1, y2_1 = to_xyxy(box1)
    x1_2, y1_2, x2_2, y2_2 = to_xyxy(box2)
    
    # 计算交集
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    
    # 计算并集
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def non_max_suppression(
    boxes: list[dict],
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    非极大值抑制 (NMS)
    
    Args:
        boxes: 检测结果列表
        iou_threshold: IoU 阈值
        conf_threshold: 置信度阈值
    
    Returns:
        list[dict]: 过滤后的检测结果
    """
    if not boxes:
        return []
    
    # 按置信度排序
    sorted_boxes = sorted(boxes, key=lambda x: x.get("conf", 1.0), reverse=True)
    
    keep = []
    suppressed = set()
    
    for i, box in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        
        keep.append(box)
        
        for j in range(i + 1, len(sorted_boxes)):
            if j in suppressed:
                continue
            
            # 只对同类别的框计算 IoU
            if box.get("class_name") != sorted_boxes[j].get("class_name"):
                continue
            
            iou = compute_iou(box["bbox_xywhn"], sorted_boxes[j]["bbox_xywhn"])
            
            if iou >= iou_threshold:
                suppressed.add(j)
    
    return keep


def deduplicate_boxes(
    boxes: list[dict],
    iou_threshold: float = 0.45,
) -> list[dict]:
    """
    去除重复边界框
    
    Args:
        boxes: 检测结果列表
        iou_threshold: IoU 阈值，高于此值认为重复
    
    Returns:
        list[dict]: 去重后的检测结果
    """
    return non_max_suppression(boxes, iou_threshold=iou_threshold, conf_threshold=0.0)


# ============== 文件格式转换 ==============

def boxes_to_yolo_txt(
    boxes: list[dict],
    classes: list[dict],
    output_path: str,
) -> int:
    """
    将检测结果保存为 YOLO txt 格式
    
    YOLO txt 格式：<class_idx> <cx> <cy> <w> <h> (归一化坐标)
    
    Args:
        boxes: 检测结果列表
        classes: 类别列表
        output_path: 输出文件路径
    
    Returns:
        int: 保存的框数量
    """
    # 建立类别名称到索引的映射
    class_to_idx = {c["class_name"]: i for i, c in enumerate(classes)}
    
    lines = []
    for box in boxes:
        class_name = box.get("class_name", "")
        class_idx = class_to_idx.get(class_name, box.get("class_idx", 0))
        
        cx, cy, w, h = box["bbox_xywhn"]
        
        lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    
    return len(lines)


def load_yolo_txt(
    txt_path: str,
    classes: list[dict],
) -> list[dict]:
    """
    从 YOLO txt 格式加载检测结果
    
    Args:
        txt_path: txt 文件路径
        classes: 类别列表
    
    Returns:
        list[dict]: 检测结果列表
    """
    if not Path(txt_path).exists():
        return []
    
    # 建立类别索引到名称的映射
    idx_to_class = {i: c["class_name"] for i, c in enumerate(classes)}
    
    boxes = []
    for line in Path(txt_path).read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        class_idx = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        
        xyxy = xywhn_to_xyxy([cx, cy, w, h])
        
        boxes.append({
            "class_idx": class_idx,
            "class_name": idx_to_class.get(class_idx, f"class_{class_idx}"),
            "bbox_xywhn": [cx, cy, w, h],
            "x1": xyxy["x1"],
            "y1": xyxy["y1"],
            "x2": xyxy["x2"],
            "y2": xyxy["y2"],
            "conf": 1.0,
            "_source": "yolo_txt",
        })
    
    return boxes


def raw_boxes_to_image_results(
    raw_boxes: dict,
    image_path: str,
) -> list[dict]:
    """
    将 raw_boxes.json 中的单张图片结果提取出来
    
    Args:
        raw_boxes: raw_boxes.json 内容
        image_path: 图像路径
    
    Returns:
        list[dict]: 该图像的检测结果
    """
    return raw_boxes.get(image_path, [])


def image_results_to_raw_boxes(
    results: dict[str, list[dict]],
) -> dict:
    """
    将图像结果字典转换为 raw_boxes.json 格式
    
    Args:
        results: {image_path: [boxes]}
    
    Returns:
        dict: raw_boxes.json 格式
    """
    return results


# ============== 批量处理 ==============

def process_detection_results(
    boxes: list[dict],
    classes: list[dict],
    iou_threshold: float = 0.45,
    conf_threshold: float = 0.25,
) -> list[dict]:
    """
    处理检测结果：过滤、去重、排序
    
    Args:
        boxes: 原始检测结果
        classes: 类别列表
        iou_threshold: NMS IoU 阈值
        conf_threshold: 置信度阈值
    
    Returns:
        list[dict]: 处理后的检测结果
    """
    # 1. 过滤低置信度
    if conf_threshold > 0:
        boxes = [b for b in boxes if b.get("conf", 1.0) >= conf_threshold]
    
    # 2. 去重
    boxes = deduplicate_boxes(boxes, iou_threshold=iou_threshold)
    
    # 3. 按置信度排序
    boxes = sorted(boxes, key=lambda x: x.get("conf", 1.0), reverse=True)
    
    return boxes


def batch_save_yolo_labels(
    results_map: dict[str, list[dict]],
    output_dir: str,
    classes: list[dict],
) -> dict:
    """
    批量保存 YOLO 标签文件
    
    Args:
        results_map: {image_path: [boxes]}
        output_dir: 输出目录
        classes: 类别列表
    
    Returns:
        dict: 统计信息
    """
    stats = {
        "total_images": 0,
        "total_boxes": 0,
        "failed_images": [],
    }
    
    for image_path, boxes in results_map.items():
        stats["total_images"] += 1
        
        try:
            # 生成对应的 txt 文件名
            txt_name = Path(image_path).stem + ".txt"
            txt_path = Path(output_dir) / txt_name
            
            count = boxes_to_yolo_txt(boxes, classes, str(txt_path))
            stats["total_boxes"] += count
            
        except Exception as e:
            logger.warning(f"保存标签失败 {image_path}: {e}")
            stats["failed_images"].append({
                "path": image_path,
                "error": str(e),
            })
    
    return stats
