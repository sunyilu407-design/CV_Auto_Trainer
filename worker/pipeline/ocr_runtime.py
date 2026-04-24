"""
EasyOCR 运行时 — 为 pipeline 提供文字识别能力。

使用方式：
    result = run_ocr_on_crop(image_crop: np.ndarray, languages: list[str] = ["ch_sim", "en"])
    result = run_ocr_on_image_file(image_path: str, languages: list[str] = ["ch_sim", "en"])

OCR 结果格式：
    {
        "text": "识别出的文字内容，多行用换行分隔",
        "boxes": [[x1, y1, x2, y2], ...],          # 归一化坐标 (0-1)
        "confidences": [0.95, 0.88, ...],          # 每条结果的置信度
        "full_results": [                           # 详细结果
            {"text": "...", "confidence": 0.95, "bbox": [x1,y1,x2,y2]},
            ...
        ]
    }

依赖：
    pip install easyocr
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 全局 Reader 缓存（避免重复初始化，线程安全）
# ---------------------------------------------------------------------------

_OCR_READER_CACHE: dict[tuple[str, ...], Any] = {}
_cache_lock = threading.Lock()


def _get_reader(languages: list[str]) -> Any:
    """获取或创建 EasyOCR Reader（带全局缓存）"""
    key = tuple(sorted(set(languages)))
    if key in _OCR_READER_CACHE:
        return _OCR_READER_CACHE[key]

    import easyocr

    with _cache_lock:
        if key in _OCR_READER_CACHE:
            return _OCR_READER_CACHE[key]

        logger.info("Initializing EasyOCR reader with languages: %s", list(key))
        reader = easyocr.Reader(list(key), gpu=False, verbose=False)
        _OCR_READER_CACHE[key] = reader
        return reader


def run_ocr_on_image_file(
    image_path: str,
    languages: list[str] | None = None,
    width: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """
    对整张图片进行 OCR 识别。

    Args:
        image_path: 图片文件路径
        languages: 识别的语言列表，默认 ["ch_sim", "en"]
        width: 输出文本的最大宽度（像素），None 则不限制
        batch_size: 批次大小

    Returns:
        OCR 结果字典
    """
    if languages is None:
        languages = ["ch_sim", "en"]

    reader = _get_reader(languages)
    try:
        import cv2
        import numpy as np
        img_raw = cv2.imread(image_path)
        img_shape = img_raw.shape[:2] if img_raw is not None else None
        results = reader.readtext(
            image_path,
            batch_size=batch_size,
            width_ths=1.0 if width else 0.5,
        )
    except Exception as e:
        logger.warning("EasyOCR failed on %s: %s", image_path, e)
        return _empty_result()

    return _parse_results(results, img_shape)


def run_ocr_on_crop(
    image_crop: Any,
    languages: list[str] | None = None,
) -> dict[str, Any]:
    """
    对检测框裁剪区域进行 OCR 识别。

    Args:
        image_crop: 图片裁剪区域，numpy.ndarray 或 PIL.Image
        languages: 识别的语言列表，默认 ["ch_sim", "en"]

    Returns:
        OCR 结果字典
    """
    import numpy as np

    if languages is None:
        languages = ["ch_sim", "en"]

    reader = _get_reader(languages)

    # 处理 PIL Image 输入
    if hasattr(image_crop, "convert"):
        image_crop = np.array(image_crop.convert("RGB"))

    if not isinstance(image_crop, np.ndarray):
        image_crop = np.array(image_crop)

    if image_crop.ndim == 2:
        image_crop = np.stack([image_crop] * 3, axis=-1)

    try:
        results = reader.readtext(image_crop)
    except Exception as e:
        logger.warning("EasyOCR failed on crop: %s", e)
        return _empty_result()

    img_h, img_w = image_crop.shape[:2]
    return _parse_results(results, (img_h, img_w))


def run_ocr_on_crops(
    crops: list[Any],
    languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    批量对多个裁剪区域进行 OCR 识别。

    Args:
        crops: 图片裁剪区域列表
        languages: 识别的语言列表，默认 ["ch_sim", "en"]

    Returns:
        OCR 结果列表，每个元素对应一个 crop 的结果
    """
    if not crops:
        return []
    if languages is None:
        languages = ["ch_sim", "en"]

    return [run_ocr_on_crop(crop, languages) for crop in crops]


def _parse_results(results: list, image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
    """
    将 EasyOCR 原始结果解析为标准格式。

    Args:
        results: EasyOCR.readtext() 返回的原始结果
        image_shape: 可选，原始图片 (height, width)，用于归一化 bbox
    """
    if not results:
        return _empty_result()

    full_results: list[dict[str, Any]] = []
    texts: list[str] = []
    normalized_boxes: list[list[float]] = []
    confidences: list[float] = []

    for item in results:
        if len(item) < 2:
            continue

        bbox_raw = item[0]
        text = item[1].strip()
        confidence = float(item[2]) if len(item) > 2 else 0.0

        if not text:
            continue

        # 计算归一化 bbox: [min_x, min_y, max_x, max_y]
        import numpy as np
        all_points = np.array(bbox_raw).reshape(-1, 2)
        x1, y1 = float(all_points.min(axis=0)[0]), float(all_points.min(axis=0)[1])
        x2, y2 = float(all_points.max(axis=0)[0]), float(all_points.max(axis=0)[1])

        if image_shape:
            img_h, img_w = image_shape
            norm_bbox = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
        else:
            norm_bbox = [x1, y1, x2, y2]

        full_results.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox_raw,
            "bbox_norm": norm_bbox,
        })
        texts.append(text)
        normalized_boxes.append(norm_bbox)
        confidences.append(confidence)

    return {
        "text": "\n".join(texts),
        "boxes": normalized_boxes,
        "confidences": confidences,
        "full_results": full_results,
    }


def _empty_result() -> dict[str, Any]:
    return {
        "text": "",
        "boxes": [],
        "confidences": [],
        "full_results": [],
    }


def _run_ocr_crop_internal(
    image_crop: Any,
    languages: list[str],
) -> dict[str, Any]:
    """OCR 内部函数：接收裁剪图 + 图片 shape，返回标准化结果"""
    import numpy as np

    if hasattr(image_crop, "convert"):
        image_crop = np.array(image_crop.convert("RGB"))

    if not isinstance(image_crop, np.ndarray):
        image_crop = np.array(image_crop)

    if image_crop.ndim == 2:
        image_crop = np.stack([image_crop] * 3, axis=-1)

    reader = _get_reader(languages)
    try:
        results = reader.readtext(image_crop)
    except Exception as e:
        logger.warning("EasyOCR failed on crop: %s", e)
        return _empty_result()

    img_h, img_w = image_crop.shape[:2]
    return _parse_results(results, (img_h, img_w))


# ---------------------------------------------------------------------------
# Pipeline 集成入口
# ---------------------------------------------------------------------------

def process_ocr_for_detections(
    detections: list[dict[str, Any]],
    image: Any,
    languages: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    对检测框列表中的每个框执行 OCR，返回附加了 ocr_text 的检测结果。

    Args:
        detections: 检测结果列表，每个元素包含 bbox_xywhn 或 bbox_xywh
        image: 原始图片（numpy.ndarray 或 PIL.Image）
        languages: OCR 语言列表

    Returns:
        detections 列表，每个元素新增 "ocr_result" 字段
    """
    if not detections:
        return detections

    import numpy as np

    if languages is None:
        languages = ["ch_sim", "en"]

    # 处理 PIL Image
    if hasattr(image, "convert"):
        image = np.array(image.convert("RGB"))

    if not isinstance(image, np.ndarray):
        image = np.array(image)

    img_h, img_w = image.shape[:2]

    enriched = []
    for det in detections:
        det = dict(det)
        ocr_result = {"text": "", "boxes": [], "confidences": [], "full_results": []}

        # 支持归一化和像素 bbox
        bbox = det.get("bbox_xywhn") or det.get("bbox_xywh", [])
        if len(bbox) >= 4:
            x_norm, y_norm, w_norm, h_norm = bbox[:4]
            if all(0 <= v <= 1.0 for v in bbox):
                x1 = int(x_norm * img_w)
                y1 = int(y_norm * img_h)
                w = int(w_norm * img_w)
                h = int(h_norm * img_h)
            else:
                x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            x2, y2 = x1 + max(w, 1), y1 + max(h, 1)
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                # 传入 crop shape 用于归一化 bbox
                ocr_result = _run_ocr_crop_internal(crop, languages)

        det["ocr_result"] = ocr_result
        enriched.append(det)

    return enriched
