import torch
import gc
import json
import os
from pathlib import Path
from typing import Optional, Callable
from pipeline.gpu_manager import gpu_stage, check_cancel_and_yield, CancelError


def run_detection(
    image_dir: str,
    classes: list[dict],
    output_raw_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 4,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """
    第一段：使用 YOLO-World 对全量图片进行目标检测，输出原始框 JSON。
    """
    model = None
    try:
        with gpu_stage("yolo_detection", required_gb=3.0):
            from ultralytics import YOLOWorld

            model = YOLOWorld("yolov8s-world.pt")
            model.half()  # FP16 半精度
            model.set_classes([c["prompt"] for c in classes])

            image_paths = (
                list(Path(image_dir).glob("*.jpg")) +
                list(Path(image_dir).glob("*.png"))
            )
            results_map: dict = {}

            for i in range(0, len(image_paths), batch_size):
                check_cancel_and_yield()

                batch = [str(p) for p in image_paths[i:i + batch_size]]
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
                        min(i + batch_size, len(image_paths)),
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

            model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                "vikhyatk/moondream2",
                trust_remote_code=True,
            )

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
    import re

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
    import re

    answer_clean = answer.strip()
    match = re.search(r"(0(?:\.\d+|\.0)|1(?:\.0)?)", answer_clean)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d{1,3})%", answer_clean)
    if match:
        return float(match.group(1)) / 100.0

    lower = answer_clean.lower()
    if lower.startswith("yes"):
        num_match = re.search(r"[\d.]+", answer_clean[len("yes"):])
        if num_match:
            return float(num_match.group())
        return 0.8
    if lower.startswith("no"):
        num_match = re.search(r"[\d.]+", answer_clean[len("no"):])
        if num_match:
            return float(num_match.group())
        return 0.1

    return 0.5
