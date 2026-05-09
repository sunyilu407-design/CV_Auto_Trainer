"""
Grounding DINO 备胎检测器（v9.0 P1-B）。

定位：YOLO-World 完全没召回时的 fallback，而**不是**自定义类别（例如工业小目标）的银弹。
对自定义类别仍建议走「人工种子框 + 专用模型微调」的路线。

设计要点：
- 模型默认：IDEA-Research/grounding-dino-tiny（~700MB），平衡效果 / 显存
- 仅在 worker 收到 fallback 触发时才加载，初次会下载
- 输出与 stage2_labeler.run_detection 完全相同的 canonical box dict 格式，便于无缝接入下游
- 可通过环境变量 `GROUNDING_DINO_MODEL` 切换为 `grounding-dino-base`
- 可通过环境变量 `ENABLE_GROUNDING_DINO=1` 默认启用 fallback

注意：本模块不导入 transformers / torch 至模块顶部，以免普通用户的 worker 启动失败。
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")


# ---------------------------------------------------------------------------
# 缓存状态
# ---------------------------------------------------------------------------


def _get_hf_hub_cache_dir() -> Path:
    if os.getenv("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"]).expanduser()
    hf_home = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "hub"


def _model_cache_dir() -> Path:
    safe = DEFAULT_MODEL_ID.replace("/", "--")
    return _get_hf_hub_cache_dir() / f"models--{safe}"


def get_grounding_dino_status() -> Dict[str, Any]:
    cache_dir = _model_cache_dir()
    snapshots = cache_dir / "snapshots"
    has_snapshot = snapshots.exists() and any(p.is_dir() for p in snapshots.iterdir())
    return {
        "model": DEFAULT_MODEL_ID,
        "cache_dir": str(cache_dir),
        "installed": bool(has_snapshot),
    }


def is_grounding_dino_available() -> bool:
    """是否可以立即使用：模型已缓存 或 显式启用了下载。"""
    if os.getenv("ENABLE_GROUNDING_DINO", "").strip() == "1":
        return True
    return get_grounding_dino_status()["installed"]


# ---------------------------------------------------------------------------
# 检测主流程
# ---------------------------------------------------------------------------


def _build_text_query(classes: List[Dict[str, Any]]) -> str:
    """
    Grounding DINO 接受 'a cat. a dog.' 这种以句号分隔的低粒度短语。
    把每个类的 prompt + aliases 合并去重后用 '. ' 连接。
    """
    items: List[str] = []
    seen: set = set()
    for c in classes:
        prompts = []
        for v in [c.get("prompt", ""), *(c.get("prompt_aliases") or [])]:
            v = (v or "").strip().lower()
            if v and v not in seen:
                seen.add(v)
                prompts.append(v)
        items.extend(prompts)
    return ". ".join(items) + "."


def run_grounding_dino_detection(
    image_dir: str,
    classes: List[Dict[str, Any]],
    output_raw_dir: str,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    用 Grounding DINO 全量推理，输出 canonical box dict（与 run_detection 一致）。
    """
    from utils.image_files import list_image_files
    from pipeline.gpu_manager import gpu_stage, check_cancel_and_yield, get_device

    image_paths = list_image_files(image_dir)
    if not image_paths:
        raise ValueError(f"图片目录为空: {image_dir}")

    text_query = _build_text_query(classes)
    if not text_query.strip(". "):
        raise ValueError("Grounding DINO 至少需要一个 prompt")

    # 把 prompt 反查 class_idx：用 lowercase 匹配
    prompt_to_class_idx: Dict[str, int] = {}
    for idx, c in enumerate(classes):
        for v in [c.get("prompt", ""), *(c.get("prompt_aliases") or [])]:
            v = (v or "").strip().lower()
            if v:
                prompt_to_class_idx.setdefault(v, idx)

    results_map: Dict[str, List[Dict[str, Any]]] = {}

    with gpu_stage("grounding_dino", required_gb=2.5):
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        device = get_device()
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("[GroundingDINO] Loading %s on %s", DEFAULT_MODEL_ID, device)
        processor = AutoProcessor.from_pretrained(DEFAULT_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            DEFAULT_MODEL_ID, torch_dtype=dtype
        ).to(device)
        model.eval()

        if progress_callback:
            progress_callback(0, len(image_paths), "detection")

        try:
            for i, img_path in enumerate(image_paths):
                check_cancel_and_yield()
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as exc:
                    logger.warning("[GroundingDINO] cannot read %s: %s", img_path, exc)
                    results_map[str(img_path)] = []
                    continue

                inputs = processor(images=image, text=text_query, return_tensors="pt").to(device)
                if dtype == torch.float16:
                    # processor 输出 long tensors，不能强转；只对 model 推理使用半精度
                    pass
                with torch.no_grad():
                    outputs = model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
                post = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_sizes,
                )[0]

                W, H = image.size
                boxes = []
                # transformers 4.51+ 用 text_labels；旧版本用 labels
                labels = post.get("text_labels") or post.get("labels") or []
                for box, score, label in zip(
                    post["boxes"].tolist(), post["scores"].tolist(), labels
                ):
                    x1, y1, x2, y2 = box
                    cx = (x1 + x2) / 2.0 / W
                    cy = (y1 + y2) / 2.0 / H
                    bw = max(0.0, (x2 - x1)) / W
                    bh = max(0.0, (y2 - y1)) / H
                    label_text = (label if isinstance(label, str) else "").strip().lower()
                    cls_idx = prompt_to_class_idx.get(label_text)
                    if cls_idx is None:
                        # 尝试匹配前缀
                        for k, v in prompt_to_class_idx.items():
                            if label_text and (label_text in k or k in label_text):
                                cls_idx = v
                                break
                    if cls_idx is None:
                        continue
                    boxes.append({
                        "class_idx": cls_idx,
                        "class_name": classes[cls_idx]["class_name"],
                        "prompt": classes[cls_idx].get("prompt", ""),
                        "bbox_xywhn": [cx, cy, bw, bh],
                        "conf": float(score),
                        "_engine": "grounding_dino",
                    })
                results_map[str(img_path)] = boxes

                if progress_callback:
                    progress_callback(i + 1, len(image_paths), "detection")
        finally:
            del model
            del processor
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception:
                pass

    os.makedirs(output_raw_dir, exist_ok=True)
    with open(f"{output_raw_dir}/raw_boxes.json", "w", encoding="utf-8") as f:
        json.dump(results_map, f, ensure_ascii=False)
    return results_map
