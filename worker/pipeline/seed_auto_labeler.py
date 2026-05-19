"""
Seed Auto Labeler — 用种子模型对未标注图片进行自动推理打标。
置信度分层：高 → 自动采纳，低 → 标记待审核，极低 → 丢弃。
"""

import os
from pathlib import Path
from typing import Optional, Callable

from utils.image_files import list_image_files

HIGH_CONF_THRESHOLD = 0.5
LOW_CONF_THRESHOLD = 0.25


def run_seed_auto_labeling(
    seed_model_path: str,
    task_dir: str,
    class_names: list[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
    high_conf: float = HIGH_CONF_THRESHOLD,
    low_conf: float = LOW_CONF_THRESHOLD,
    batch_size: int = 16,
) -> dict:
    """
    用种子模型推理未标注的图片，按置信度分层保存标注。

    Returns: {
        "auto_accepted": int,    # >= high_conf 的框数
        "needs_review": int,     # low_conf ~ high_conf 的框数
        "no_detection": int,     # 没有任何检测的图片数
        "total_processed": int,
        "avg_confidence": float,
        "output_label_dir": str, # 自动标注结果目录
    }
    """
    from ultralytics import YOLO

    task_path = Path(task_dir)
    seed_label_dir = task_path / "seed_labels"

    # Find image directory
    image_dir: Optional[Path] = None
    for subdir in ["images", "video_frames"]:
        candidate = task_path / subdir
        if candidate.exists() and any(candidate.iterdir()):
            image_dir = candidate
            break

    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in: {task_path}")

    # Get already-annotated stems (skip these)
    annotated_stems: set[str] = set()
    if seed_label_dir.exists():
        annotated_stems = {p.stem for p in seed_label_dir.glob("*.txt") if p.stat().st_size > 0}

    # Collect unannotated images
    all_images = list_image_files(str(image_dir))
    unannotated = [p for p in all_images if p.stem not in annotated_stems]

    if not unannotated:
        return {
            "auto_accepted": 0,
            "needs_review": 0,
            "no_detection": 0,
            "total_processed": 0,
            "avg_confidence": 0.0,
            "output_label_dir": "",
        }

    # Load model
    model = YOLO(seed_model_path)

    # Prepare output directories
    auto_label_dir = task_path / "auto_labels"
    review_label_dir = task_path / "review_labels"
    os.makedirs(auto_label_dir, exist_ok=True)
    os.makedirs(review_label_dir, exist_ok=True)

    stats = {
        "auto_accepted": 0,
        "needs_review": 0,
        "no_detection": 0,
        "total_processed": 0,
        "sum_confidence": 0.0,
        "total_boxes": 0,
    }

    total = len(unannotated)

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch = unannotated[batch_start:batch_start + batch_size]
        batch_paths = [str(p) for p in batch]

        results = model.predict(
            source=batch_paths,
            conf=low_conf,
            verbose=False,
            stream=False,
        )

        for img_path, result in zip(batch, results):
            stats["total_processed"] += 1
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                stats["no_detection"] += 1
                if progress_callback:
                    progress_callback({
                        "type": "seed_auto_label_progress",
                        "current": stats["total_processed"],
                        "total": total,
                    })
                continue

            high_lines = []
            review_lines = []

            for box in boxes:
                conf = float(box.conf[0])
                cls_idx = int(box.cls[0])
                xywhn = box.xywhn[0]  # normalized cx, cy, w, h
                cx, cy, w, h = float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])

                line = f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                stats["sum_confidence"] += conf
                stats["total_boxes"] += 1

                if conf >= high_conf:
                    high_lines.append(line)
                    stats["auto_accepted"] += 1
                else:
                    review_lines.append(line)
                    stats["needs_review"] += 1

            # Save high-confidence labels as "accepted"
            if high_lines:
                label_path = auto_label_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(high_lines) + "\n")

            # Save low-confidence labels for review
            if review_lines:
                label_path = review_label_dir / f"{img_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(review_lines) + "\n")

            if progress_callback:
                progress_callback({
                    "type": "seed_auto_label_progress",
                    "current": stats["total_processed"],
                    "total": total,
                })

    avg_conf = stats["sum_confidence"] / max(stats["total_boxes"], 1)

    return {
        "auto_accepted": stats["auto_accepted"],
        "needs_review": stats["needs_review"],
        "no_detection": stats["no_detection"],
        "total_processed": stats["total_processed"],
        "avg_confidence": round(avg_conf, 4),
        "output_label_dir": str(auto_label_dir),
    }


def merge_seed_and_auto_labels(
    task_dir: str,
    class_names: list[str],
) -> dict:
    """
    合并三路标注：手动种子标注 + 自动采纳标注 + 审核后采纳标注，
    输出为最终 labeled_images + labels 目录。

    优先级：manual_seed > review_accepted > auto_accepted

    Returns: { "total_labeled": int, "image_dir": str, "label_dir": str }
    """
    import shutil

    task_path = Path(task_dir)
    seed_label_dir = task_path / "seed_labels"
    auto_label_dir = task_path / "auto_labels"
    review_label_dir = task_path / "review_labels"

    # Find image source
    image_dir: Optional[Path] = None
    for subdir in ["images", "video_frames"]:
        candidate = task_path / subdir
        if candidate.exists() and any(candidate.iterdir()):
            image_dir = candidate
            break

    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in: {task_path}")

    # Output directories that downstream pipeline expects
    out_image_dir = task_path / "labeled_images"
    out_label_dir = task_path / "labels"
    os.makedirs(out_image_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    merged_stems: set[str] = set()

    # Priority 1: manual seed annotations (highest trust)
    if seed_label_dir.exists():
        for lbl in seed_label_dir.glob("*.txt"):
            if lbl.stat().st_size == 0:
                continue
            stem = lbl.stem
            if stem in merged_stems:
                continue
            img_file = _find_image(image_dir, stem)
            if img_file is None:
                continue
            shutil.copy(lbl, out_label_dir / lbl.name)
            shutil.copy(img_file, out_image_dir / img_file.name)
            merged_stems.add(stem)

    # Priority 2: review-accepted labels (user explicitly confirmed)
    # ReviewAutoLabels saves accepted boxes back to seed_labels/ — already handled above.
    # Explicitly copy non-empty review labels only if not already merged.
    if review_label_dir.exists():
        for lbl in review_label_dir.glob("*.txt"):
            if lbl.stat().st_size == 0:
                continue
            stem = lbl.stem
            if stem in merged_stems:
                continue
            img_file = _find_image(image_dir, stem)
            if img_file is None:
                continue
            shutil.copy(lbl, out_label_dir / lbl.name)
            shutil.copy(img_file, out_image_dir / img_file.name)
            merged_stems.add(stem)

    # Priority 3: auto-accepted labels (highest conf from seed model)
    if auto_label_dir.exists():
        for lbl in auto_label_dir.glob("*.txt"):
            if lbl.stat().st_size == 0:
                continue
            stem = lbl.stem
            if stem in merged_stems:
                continue  # manual or review takes priority
            img_file = _find_image(image_dir, stem)
            if img_file is None:
                continue
            shutil.copy(lbl, out_label_dir / lbl.name)
            shutil.copy(img_file, out_image_dir / img_file.name)
            merged_stems.add(stem)

    return {
        "total_labeled": len(merged_stems),
        "image_dir": str(out_image_dir),
        "label_dir": str(out_label_dir),
    }


def _find_image(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None
