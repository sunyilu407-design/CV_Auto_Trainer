import torch
import gc
import json
import os
import re
from pathlib import Path
from urllib.error import URLError
from typing import Optional, Callable
from pipeline.gpu_manager import gpu_stage, check_cancel_and_yield, CancelError, get_device
from utils.image_files import list_image_files

CLIP_CACHE_DISPLAY_PATH = "~/.cache/clip/ViT-B-32.pt"
HF_CACHE_DISPLAY_PATH = "~/.cache/huggingface/hub"
YOLO_WORLD_WEIGHT_NAME = "yolov8s-world.pt"
MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
# HuggingFace Hub 下载超时（秒），国内网络建议设高一些
HF_HUB_DOWNLOAD_TIMEOUT = int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "300"))

# 国内网络优先使用 HF Mirror，避免 huggingface.co 超时
_HF_ENDPOINT = os.getenv("HF_ENDPOINT", "").strip()
if not _HF_ENDPOINT:
    # 自动检测并设置 hf-mirror.com（国内镜像）
    import socket
    try:
        socket.create_connection(("huggingface.co", 443), timeout=5).close()
    except OSError:
        # huggingface.co 无法直连，切换到国内镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class DetectionSetupError(RuntimeError):
    """检测阶段初始化失败，通常是模型权重或网络前置条件未满足。"""


def _is_clip_setup_error(exc: Exception) -> bool:
    lower = str(exc).lower()
    indicators = (
        "openaipublic.azureedge.net",
        "vit-b-32.pt",
        "eof occurred in violation of protocol",
        "ssl_error_syscall",
        "ssl",
        "urlopen error",
        "operation not permitted",
    )
    return isinstance(exc, URLError) or any(token in lower for token in indicators)


def _build_clip_setup_error(exc: Exception) -> DetectionSetupError:
    return DetectionSetupError(
        "CLIP 权重初始化失败：YOLO-World 在设置类别时需要读取/下载 CLIP 权重 ViT-B-32.pt。"
        f" 请先将权重文件放到 {CLIP_CACHE_DISPLAY_PATH}，"
        "或切换到可访问 openaipublic.azureedge.net 的网络后重试。"
        f" 原始错误: {exc}"
    )


def _is_yolo_world_setup_error(exc: Exception) -> bool:
    lower = str(exc).lower()
    indicators = (
        YOLO_WORLD_WEIGHT_NAME,
        "github.com/ultralytics/assets",
        "urlopen error",
        "ssl",
        "connection",
        "timed out",
    )
    return isinstance(exc, URLError) or any(token in lower for token in indicators)


def _build_yolo_world_setup_error(exc: Exception) -> DetectionSetupError:
    return DetectionSetupError(
        f"YOLO-World 权重初始化失败：Worker 在启动检测阶段时需要读取/下载 {YOLO_WORLD_WEIGHT_NAME}。"
        f" 请先将该权重文件放到 worker 目录，或切换到可访问 GitHub/Ultralytics 资源的网络后重试。"
        f" 原始错误: {exc}"
    )


def _is_moondream_setup_error(exc: Exception) -> bool:
    lower = str(exc).lower()
    indicators = (
        MOONDREAM_MODEL_ID,
        "huggingface.co",
        "httpsconnectionpool",
        "certificate",
        "ssl",
        "connection",
        "timed out",
        "eof occurred in violation of protocol",
        "trust_remote_code",
    )
    return isinstance(exc, URLError) or any(token in lower for token in indicators)


def _build_moondream_setup_error(exc: Exception) -> DetectionSetupError:
    return DetectionSetupError(
        f"Moondream2 初始化失败：第二段质检需要读取/下载模型 {MOONDREAM_MODEL_ID}。"
        f" 请确保当前网络可访问 huggingface.co，或预先将模型缓存到 {HF_CACHE_DISPLAY_PATH} 后重试。"
        f" 原始错误: {exc}"
    )


def run_detection(
    image_dir: str,
    classes: list[dict],
    output_raw_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 4,
    progress_callback: Optional[Callable] = None,
    use_existing_labels: bool = False,
) -> dict:
    """
    第一段：使用 YOLO-World 对全量图片进行目标检测，输出原始框 JSON。

    参数 use_existing_labels: 若为 True，则跳过 YOLO-World 推理，
    直接从 image_dir 中的 YOLO .txt 标注文件读取检测框，
    适用于用户已用 LabelImg/roLabelImg 等工具预先标注好的数据集。
    """
    # 检测预标注数据：查找与图片同名的 .txt 文件
    if use_existing_labels:
        from utils.yolo_io import load_yolo_labels

        image_paths = list_image_files(image_dir)
        if not image_paths:
            raise ValueError(f"图片目录为空: {image_dir}")

        results_map: dict = {}
        label_dir = Path(image_dir)
        for img_path in image_paths:
            label_path = label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                results_map[str(img_path)] = []
                continue
            bboxes, labels = load_yolo_labels(str(label_path))
            # 转换：class_idx, class_name, bbox_xywhn, conf(=1.0)
            mapped = []
            for cls_idx, bbox in zip(labels, bboxes):
                mapped.append({
                    "class_idx": int(cls_idx),
                    "class_name": classes[int(cls_idx)]["class_name"] if int(cls_idx) < len(classes) else f"class_{cls_idx}",
                    "prompt": classes[int(cls_idx)]["prompt"] if int(cls_idx) < len(classes) else "",
                    "bbox_xywhn": [float(x) for x in bbox],
                    "conf": 1.0,
                    "_source": "existing_label",
                })
            results_map[str(img_path)] = mapped

        os.makedirs(output_raw_dir, exist_ok=True)
        with open(f"{output_raw_dir}/raw_boxes.json", "w", encoding="utf-8") as f:
            json.dump(results_map, f, ensure_ascii=False)
        return results_map

    model = None
    try:
        with gpu_stage("yolo_detection", required_gb=3.0):
            from ultralytics import YOLOWorld

            device = get_device()
            try:
                model = YOLOWorld(YOLO_WORLD_WEIGHT_NAME)
            except Exception as exc:
                if _is_yolo_world_setup_error(exc):
                    raise _build_yolo_world_setup_error(exc) from exc
                raise
            if device == "cuda":
                model.half()  # FP16 半精度 — 仅 CUDA 支持
            if device in ("cuda", "mps"):
                model.to(device)
            try:
                model.set_classes([c["prompt"] for c in classes])
            except Exception as exc:
                if _is_clip_setup_error(exc):
                    raise _build_clip_setup_error(exc) from exc
                raise

            image_paths = list_image_files(image_dir)
            if not image_paths:
                raise ValueError(
                    f"图片目录为空或未找到支持格式图片: {image_dir} "
                    "(支持 .jpg/.jpeg/.png，大小写均可)"
                )

            effective_batch_size = 1 if device == "mps" else max(1, batch_size)
            results_map: dict = {}

            if progress_callback:
                progress_callback(0, len(image_paths), "detection")

            for i in range(0, len(image_paths), effective_batch_size):
                check_cancel_and_yield()

                batch = [str(p) for p in image_paths[i:i + effective_batch_size]]
                results = model.predict(
                    batch,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False,
                )
                for img_path, result in zip(batch, results):
                    boxes = []
                    for box in result.boxes:
                        cls_idx = int(box.cls[0])
                        boxes.append({
                            "class_idx": cls_idx,
                            "class_name": classes[cls_idx]["class_name"],
                            "prompt": classes[cls_idx]["prompt"],
                            "bbox_xywhn": box.xywhn[0].tolist(),
                            "conf": float(box.conf[0]),
                        })
                    results_map[str(img_path)] = boxes

                if progress_callback:
                    progress_callback(
                        min(i + effective_batch_size, len(image_paths)),
                        len(image_paths),
                        "detection",
                    )

            os.makedirs(output_raw_dir, exist_ok=True)
            with open(f"{output_raw_dir}/raw_boxes.json", "w", encoding="utf-8") as f:
                json.dump(results_map, f, ensure_ascii=False)

            return results_map

    finally:
        if model is not None:
            del model


def run_quality_check(
    raw_boxes_path: str,
    min_confidence: float = 0.5,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    第二段：使用 Moondream2 对每个裁剪框进行三维度 VQA 质检。
    三维度：清晰度 + 完整性 + 目标一致性
    任一维度 < 0.4 则丢弃该框。
    """
    model = None
    try:
        with gpu_stage("moondream_qa", required_gb=2.0):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import cv2

            device = get_device()
            dtype = torch.float16 if device == "cuda" else torch.float32

            if progress_callback:
                progress_callback(0, 1, "loading_moondream")

            hf_endpoint = os.environ.get("HF_ENDPOINT", "")
            if hf_endpoint:
                print(f"[Moondream2] Using HuggingFace endpoint: {hf_endpoint}")
            print(f"[Moondream2] Loading model {MOONDREAM_MODEL_ID} (first time may take a few minutes)...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MOONDREAM_MODEL_ID,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    timeout=HF_HUB_DOWNLOAD_TIMEOUT,
                ).to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    MOONDREAM_MODEL_ID,
                    trust_remote_code=True,
                    timeout=HF_HUB_DOWNLOAD_TIMEOUT,
                )
            except Exception as exc:
                if _is_moondream_setup_error(exc):
                    raise _build_moondream_setup_error(exc) from exc
                raise

            with open(raw_boxes_path, encoding="utf-8") as f:
                raw_boxes: dict = json.load(f)

            passed_boxes: dict = {}
            total = sum(len(v) for v in raw_boxes.values())
            processed = 0

            for img_path, boxes in raw_boxes.items():
                check_cancel_and_yield()

                img = cv2.imread(img_path)
                if img is None:
                    processed += len(boxes)
                    continue
                h, w = img.shape[:2]
                passed = []

                for box in boxes:
                    cx, cy, bw, bh = box["bbox_xywhn"]
                    x1 = max(0, int((cx - bw / 2) * w))
                    y1 = max(0, int((cy - bh / 2) * h))
                    x2 = min(w, int((cx + bw / 2) * w))
                    y2 = min(h, int((cy + bh / 2) * h))

                    if (x2 - x1) < 10 or (y2 - y1) < 10:
                        processed += 1
                        continue

                    crop = img[y1:y2, x1:x2]
                    scores = _multi_dim_vqa(model, tokenizer, crop, box["prompt"])
                    avg_score = sum(scores) / len(scores)

                    if avg_score >= min_confidence and all(s >= 0.4 for s in scores):
                        box["qa_score"] = avg_score
                        box["qa_dimensions"] = {
                            "clarity": scores[0],
                            "completeness": scores[1],
                            "match": scores[2],
                        }
                        passed.append(box)

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total, "quality_check")

                passed_boxes[img_path] = passed

            return passed_boxes

    finally:
        if model is not None:
            del model


def _multi_dim_vqa(model, tokenizer, crop, prompt_text: str) -> list[float]:
    questions = [
        (
            "Is this image region clear and in focus, not blurry or severely distorted? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is perfectly clear.",
            "clarity",
        ),
        (
            "Does this cropped image show a complete or mostly complete object "
            "(not severely cropped or truncated)? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is complete.",
            "completeness",
        ),
        (
            f"Does this image clearly show: {prompt_text}? "
            f"Answer with a number from 0.0 to 1.0, where 1.0 is a clear match.",
            "match",
        ),
    ]

    scores = []
    enc_img = model.encode_image(crop)

    for question, _ in questions:
        answer = model.answer_question(enc_img, question, tokenizer)
        score = _parse_confidence(answer)
        scores.append(score)

    return scores


def _parse_confidence(answer: str) -> float:
    answer_clean = answer.strip()

    # Try to find a decimal number between 0 and 1 (e.g., "0.85", "0.7", "1.0")
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0?)?)\b", answer_clean)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val

    # Try percentage (e.g., "85%", "70 %")
    match = re.search(r"(\d{1,3})\s*%", answer_clean)
    if match:
        return max(0.0, min(1.0, float(match.group(1)) / 100.0))

    # Try to find any standalone number and interpret it
    match = re.search(r"(\d+\.?\d*)", answer_clean)
    if match:
        val = float(match.group(1))
        if 0.0 <= val <= 1.0:
            return val
        if 1.0 < val <= 100.0:
            return max(0.0, min(1.0, val / 100.0))

    # Keyword-based fallback
    lower = answer_clean.lower()
    positive_keywords = ("yes", "clear", "complete", "match", "good", "high", "perfect")
    negative_keywords = ("no", "blur", "unclear", "incomplete", "mismatch", "bad", "low", "poor")
    if any(kw in lower for kw in positive_keywords):
        return 0.8
    if any(kw in lower for kw in negative_keywords):
        return 0.1

    return 0.5
