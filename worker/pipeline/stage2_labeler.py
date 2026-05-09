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
CLIP_HF_MODEL_ID = "openai/clip-vit-base-patch32"
HF_CACHE_DISPLAY_PATH = "~/.cache/huggingface/hub"
YOLO_WORLD_WEIGHT_NAME = "yolov8s-world.pt"
MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
WORKER_ROOT = Path(__file__).resolve().parents[1]
# HuggingFace Hub 下载超时（秒），国内网络建议设高一些
HF_HUB_DOWNLOAD_TIMEOUT = int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "300"))
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT))

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


def _resolve_yolo_world_weight() -> str:
    worker_weight = WORKER_ROOT / YOLO_WORLD_WEIGHT_NAME
    if worker_weight.exists():
        return str(worker_weight)
    return YOLO_WORLD_WEIGHT_NAME


def _get_hf_hub_cache_dir() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _path_summary(path: Path) -> dict:
    exists = path.exists()
    result = {
        "path": str(path),
        "exists": exists,
        "size_bytes": 0,
        "file_count": 0,
    }
    if not exists:
        return result
    if path.is_file():
        result["size_bytes"] = path.stat().st_size
        result["file_count"] = 1
        return result
    total = 0
    file_count = 0
    for item in path.rglob("*"):
        if item.is_file():
            file_count += 1
            total += item.stat().st_size
    result["size_bytes"] = total
    result["file_count"] = file_count
    return result


def _get_moondream_cache_details(moondream_dir: Path) -> dict:
    incomplete_files = []
    complete_snapshots = []
    snapshot_count = 0

    if moondream_dir.exists():
        incomplete_files = [
            str(item.relative_to(moondream_dir))
            for item in moondream_dir.rglob("*")
            if item.name.endswith(".incomplete") or item.name.endswith(".lock")
        ]

        snapshots_dir = moondream_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if not snapshot.is_dir():
                    continue
                snapshot_count += 1
                has_remote_config = (snapshot / "configuration_moondream.py").exists()
                has_model_config = (snapshot / "config.json").exists()
                has_weights = any(
                    item.is_file() and item.suffix in {".bin", ".safetensors"}
                    for item in snapshot.rglob("*")
                )
                if has_remote_config and has_model_config and has_weights:
                    complete_snapshots.append(str(snapshot))

    return {
        "installed": bool(complete_snapshots) and not incomplete_files,
        "incomplete_files": incomplete_files,
        "complete_snapshots": complete_snapshots,
        "snapshot_count": snapshot_count,
    }


def get_model_cache_status() -> dict:
    yolo_worker_path = WORKER_ROOT / YOLO_WORLD_WEIGHT_NAME
    yolo_cwd_path = Path.cwd() / YOLO_WORLD_WEIGHT_NAME
    clip_path = Path(CLIP_CACHE_DISPLAY_PATH).expanduser()
    # YOLO-World uses HF transformers CLIP, not the OpenAI .pt file
    clip_hf_dir = _get_hf_hub_cache_dir() / f"models--{CLIP_HF_MODEL_ID.replace('/', '--')}"
    moondream_dir = _get_hf_hub_cache_dir() / "models--vikhyatk--moondream2"
    moondream_summary = _path_summary(moondream_dir)
    moondream_details = _get_moondream_cache_details(moondream_dir)
    return {
        "yolo_world": {
            "model": YOLO_WORLD_WEIGHT_NAME,
            "selected_path": _resolve_yolo_world_weight(),
            "worker_path": _path_summary(yolo_worker_path),
            "cwd_path": _path_summary(yolo_cwd_path),
            "installed": yolo_worker_path.exists() or yolo_cwd_path.exists(),
        },
        "clip": {
            "model": CLIP_HF_MODEL_ID,
            "cache": _path_summary(clip_hf_dir) if clip_hf_dir.exists() else _path_summary(clip_path),
            "installed": clip_hf_dir.exists() or clip_path.exists(),
            "hf_cache": _path_summary(clip_hf_dir),
            "legacy_cache": _path_summary(clip_path),
        },
        "moondream": {
            "model": MOONDREAM_MODEL_ID,
            "cache": moondream_summary,
            "installed": moondream_details["installed"],
            "incomplete_files": moondream_details["incomplete_files"],
            "complete_snapshots": moondream_details["complete_snapshots"],
            "snapshot_count": moondream_details["snapshot_count"],
        },
    }


def prepare_yolo_world_cache(progress_callback: Optional[Callable] = None) -> dict:
    model = None
    try:
        if progress_callback:
            progress_callback("yolo_world", "running", "正在准备 YOLO-World 权重")
        from ultralytics import YOLOWorld

        model = YOLOWorld(_resolve_yolo_world_weight())
        if progress_callback:
            progress_callback("clip", "running", "正在准备 CLIP 类别编码权重")
        model.set_classes(["object"])
        if progress_callback:
            progress_callback("clip", "complete", "YOLO-World 和 CLIP 已准备完成")
        return get_model_cache_status()
    finally:
        if model is not None:
            del model


def prepare_moondream_cache(progress_callback: Optional[Callable] = None) -> dict:
    if progress_callback:
        progress_callback("moondream", "running", f"正在下载 {MOONDREAM_MODEL_ID}")
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MOONDREAM_MODEL_ID,
            resume_download=True,
        )
    except Exception as exc:
        if _is_moondream_setup_error(exc):
            raise _build_moondream_setup_error(exc) from exc
        raise
    if progress_callback:
        progress_callback("moondream", "complete", "Moondream2 已准备完成")
    return get_model_cache_status()


def prepare_labeling_model_cache(include_moondream: bool = False, progress_callback: Optional[Callable] = None) -> dict:
    prepare_yolo_world_cache(progress_callback)
    if include_moondream:
        prepare_moondream_cache(progress_callback)
    return get_model_cache_status()


def _unique_prompts(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        value = str(item or "").strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


MAX_ALIASES_PER_CLASS = 8  # 过多近义 CLIP 词条会导致 softmax 稀释、置信度极低


def _expand_detection_classes(classes: list[dict]) -> list[dict]:
    expanded = []
    for class_idx, class_item in enumerate(classes):
        prompts = _unique_prompts([
            *(class_item.get("prompt_aliases") or []),
            class_item.get("prompt", ""),
        ])
        if not prompts:
            prompts = [class_item.get("class_name", f"class_{class_idx}")]
        # Cap aliases to avoid CLIP softmax dilution
        if len(prompts) > MAX_ALIASES_PER_CLASS:
            prompts = prompts[:MAX_ALIASES_PER_CLASS]
        for prompt in prompts:
            expanded.append({
                **class_item,
                "prompt": prompt,
                "_source_class_idx": class_idx,
            })
    return expanded


def _box_iou_xywhn(a: list[float], b: list[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _dedupe_canonical_boxes(boxes: list[dict], iou_threshold: float) -> list[dict]:
    kept = []
    for box in sorted(boxes, key=lambda item: item.get("conf", 0), reverse=True):
        duplicate = False
        for existing in kept:
            if existing["class_idx"] != box["class_idx"]:
                continue
            if _box_iou_xywhn(existing["bbox_xywhn"], box["bbox_xywhn"]) >= iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


def run_detection(
    image_dir: str,
    classes: list[dict],
    output_raw_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 4,
    imgsz: int = 1280,
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
                model = YOLOWorld(_resolve_yolo_world_weight())
            except Exception as exc:
                if _is_yolo_world_setup_error(exc):
                    raise _build_yolo_world_setup_error(exc) from exc
                raise
            if device == "cuda":
                model.half()  # FP16 半精度 — 仅 CUDA 支持
            if device in ("cuda", "mps"):
                model.to(device)
            detection_classes = _expand_detection_classes(classes)
            try:
                model.set_classes([c["prompt"] for c in detection_classes])
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
                effective_imgsz = max(640, int(imgsz or 1280))
                results = model.predict(
                    batch,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    imgsz=effective_imgsz,
                    verbose=False,
                )
                for img_path, result in zip(batch, results):
                    boxes = []
                    for box in result.boxes:
                        cls_idx = int(box.cls[0])
                        detected_class = detection_classes[cls_idx]
                        source_cls_idx = int(detected_class.get("_source_class_idx", cls_idx))
                        source_class = classes[source_cls_idx]
                        boxes.append({
                            "class_idx": source_cls_idx,
                            "class_name": source_class["class_name"],
                            "prompt": detected_class["prompt"],
                            "bbox_xywhn": box.xywhn[0].tolist(),
                            "conf": float(box.conf[0]),
                        })
                    results_map[str(img_path)] = _dedupe_canonical_boxes(boxes, iou_threshold)

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


# 五维质检阈值。category_match 与文档 v9.0 一致——错框混入训练集的最大风险点，单独最高优先级。
QC_DIM_THRESHOLDS = {
    "category_match": 0.5,    # 框内容是否真的是该类别（最严，错配直接丢）
    "tightness": 0.35,        # 框松紧——太松会让模型学到大量背景
    "completeness": 0.4,      # 是否完整目标（未被严重截断）
    "clarity": 0.35,          # 清晰度
    "no_occlusion_error": 0.35,  # 是否被其它物体严重遮挡
}

QC_REJECT_REASONS = {
    "category_match": "category_mismatch",
    "tightness": "box_too_loose",
    "completeness": "object_cut_off",
    "clarity": "too_blurry",
    "no_occlusion_error": "occluded",
}


def _should_keep_box(scores: dict, min_confidence: float, thresholds: dict) -> tuple[bool, str]:
    """五维度判定 + 结构化拒绝原因。category_match 最严，其余按 thresholds。"""
    for dim in ("category_match", "tightness", "completeness", "clarity", "no_occlusion_error"):
        threshold = thresholds.get(dim, QC_DIM_THRESHOLDS[dim])
        if scores.get(dim, 0.0) < threshold:
            return False, QC_REJECT_REASONS[dim]
    avg = sum(scores.values()) / max(1, len(scores))
    if avg < min_confidence:
        return False, "low_overall"
    return True, "pass"


def run_quality_check(
    raw_boxes_path: str,
    min_confidence: float = 0.5,
    progress_callback: Optional[Callable] = None,
    dim_thresholds: Optional[dict] = None,
) -> tuple[dict, dict]:
    """
    第二段：使用 Moondream2 对每个裁剪框进行五维度 VQA 质检。
    五维度：clarity + completeness + category_match + tightness + no_occlusion_error
    任一维度低于阈值（见 QC_DIM_THRESHOLDS）则丢弃该框，并记录结构化拒绝原因。

    Returns: (passed_boxes_map, stats_dict)
    stats_dict 含 reject_reasons 计数（category_mismatch / box_too_loose / ...），
    用于前端展示「为什么这么多框被过滤」。
    """
    thresholds = {**QC_DIM_THRESHOLDS, **(dim_thresholds or {})}
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
                ).to(device)
                tokenizer = AutoTokenizer.from_pretrained(
                    MOONDREAM_MODEL_ID,
                    trust_remote_code=True,
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
            stats = {
                "total_boxes": total,
                "passed_boxes": 0,
                "rejected_boxes": 0,
                "rejected_too_small": 0,
                "reject_reasons": {
                    "category_mismatch": 0,
                    "box_too_loose": 0,
                    "object_cut_off": 0,
                    "too_blurry": 0,
                    "occluded": 0,
                    "low_overall": 0,
                },
                "thresholds": dict(thresholds),
                "min_confidence": min_confidence,
            }

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
                        stats["rejected_boxes"] += 1
                        stats["rejected_too_small"] += 1
                        continue

                    # 紧裁剪用于 clarity / completeness / category_match
                    tight_crop = img[y1:y2, x1:x2]
                    # 带 25% padding 的扩展裁剪用于 tightness / occlusion 评估
                    pad_x = int((x2 - x1) * 0.25)
                    pad_y = int((y2 - y1) * 0.25)
                    px1 = max(0, x1 - pad_x)
                    py1 = max(0, y1 - pad_y)
                    px2 = min(w, x2 + pad_x)
                    py2 = min(h, y2 + pad_y)
                    padded_crop = img[py1:py2, px1:px2]

                    scores = _multi_dim_vqa(
                        model, tokenizer, tight_crop, padded_crop, box["prompt"]
                    )
                    keep, reason = _should_keep_box(scores, min_confidence, thresholds)
                    avg_score = sum(scores.values()) / max(1, len(scores))

                    box["qa_score"] = avg_score
                    box["qa_dimensions"] = {
                        **scores,
                        # 保留旧字段名 'match' 以兼容历史消费者
                        "match": scores["category_match"],
                    }
                    box["qa_reject_reason"] = reason

                    if keep:
                        passed.append(box)
                        stats["passed_boxes"] += 1
                    else:
                        stats["rejected_boxes"] += 1
                        stats["reject_reasons"][reason] = stats["reject_reasons"].get(reason, 0) + 1

                    processed += 1
                    if progress_callback:
                        progress_callback(processed, total, "quality_check")

                passed_boxes[img_path] = passed

            return passed_boxes, stats

    finally:
        if model is not None:
            del model


def _multi_dim_vqa(model, tokenizer, tight_crop, padded_crop, prompt_text: str) -> dict:
    """
    五维度质检：
    - clarity / completeness / category_match  使用 tight_crop（贴近目标的裁剪）
    - tightness / no_occlusion_error           使用 padded_crop（含周边背景，更易判断框松紧/遮挡）
    返回 dict: {clarity, completeness, category_match, tightness, no_occlusion_error}
    """
    tight_questions = [
        (
            "clarity",
            "Is this image region clear and in focus, not blurry or severely distorted? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is perfectly clear.",
        ),
        (
            "completeness",
            "Does this cropped image show a complete or mostly complete object "
            "(not severely cropped or truncated at the edges)? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is fully complete.",
        ),
        (
            "category_match",
            f"Does this image clearly show a {prompt_text}? "
            f"Answer with a number from 0.0 to 1.0, where 1.0 means definitely yes "
            f"and 0.0 means definitely not.",
        ),
    ]
    padded_questions = [
        (
            "tightness",
            f"Imagine the central region of this image is the bounding box for a {prompt_text}. "
            f"Does the {prompt_text} fill most of the central region without too much empty background? "
            "Answer with a number from 0.0 to 1.0, where 1.0 means the box is tight and well-fitted, "
            "and 0.0 means the box contains mostly background.",
        ),
        (
            "no_occlusion_error",
            f"Is the {prompt_text} in this image clearly visible without being heavily occluded "
            "by other objects or going significantly off-frame? "
            "Answer with a number from 0.0 to 1.0, where 1.0 means fully visible "
            "and 0.0 means mostly hidden or occluded.",
        ),
    ]

    scores: dict = {}
    enc_tight = model.encode_image(tight_crop)
    for key, question in tight_questions:
        answer = model.answer_question(enc_tight, question, tokenizer)
        scores[key] = _parse_confidence(answer)

    enc_padded = model.encode_image(padded_crop)
    for key, question in padded_questions:
        answer = model.answer_question(enc_padded, question, tokenizer)
        scores[key] = _parse_confidence(answer)

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


# ---------------------------------------------------------------------------
# Negotiate Preview: 单张图片快速检测（供需求确认对话预览使用）
# ---------------------------------------------------------------------------

MAX_ALIASES_PER_CLASS = 5


def run_detection_preview(
    image_path: str,
    detection_classes: list,
    conf_threshold: float = 0.15,
    imgsz: int = 1280,
) -> list:
    """
    对单张图片用 YOLO-World 做快速检测，返回检测结果列表。

    Args:
        image_path: 图片绝对路径
        detection_classes: [{"class_name": "...", "prompt": "...", "prompt_aliases": [...]}]
        conf_threshold: 置信度阈值
        imgsz: 输入图像尺寸

    Returns:
        [{"class_name": "...", "confidence": 0.xx, "bbox": [x1, y1, x2, y2]}]
    """
    from ultralytics import YOLO

    # 加载模型
    weight_path = WORKER_ROOT / YOLO_WORLD_WEIGHT_NAME
    if not weight_path.exists():
        weight_path = Path(YOLO_WORLD_WEIGHT_NAME)
    model = YOLO(str(weight_path))

    # 构建 classes 列表（展开 aliases）
    # 每个 class → [primary] + aliases，但限制数量
    expanded_classes = []
    class_map = {}  # expanded_index → canonical class_name

    for cls_def in detection_classes:
        canonical = cls_def["class_name"]
        primary = cls_def.get("prompt", canonical)
        aliases = cls_def.get("prompt_aliases", [])[:MAX_ALIASES_PER_CLASS]

        all_prompts = [primary] + [a for a in aliases if a != primary]
        for prompt in all_prompts:
            class_map[len(expanded_classes)] = canonical
            expanded_classes.append(prompt)

    model.set_classes(expanded_classes)

    # 推理
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        imgsz=imgsz,
        verbose=False,
    )

    # 解析结果
    detections = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()
                canonical_name = class_map.get(cls_idx, expanded_classes[cls_idx] if cls_idx < len(expanded_classes) else "unknown")
                detections.append({
                    "class_name": canonical_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(v, 1) for v in xyxy],
                })

    # 去重：同一 canonical class 的重叠框按 IoU 合并
    detections = _deduplicate_detections(detections, iou_threshold=0.45)

    # 释放显存
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return detections


def _deduplicate_detections(detections: list, iou_threshold: float = 0.45) -> list:
    """对同类别检测框按 IoU 去重，保留置信度最高的。"""
    if not detections:
        return detections

    # 按置信度降序
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in sorted_dets:
        is_dup = False
        for k in kept:
            if k["class_name"] == det["class_name"]:
                if _compute_iou(k["bbox"], det["bbox"]) > iou_threshold:
                    is_dup = True
                    break
        if not is_dup:
            kept.append(det)

    return kept


def _compute_iou(box1: list, box2: list) -> float:
    """计算两个 [x1,y1,x2,y2] 框的 IoU。"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
