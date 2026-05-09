"""
Incremental Data Merger — 将新增数据与旧训练集合并,去重,重新分割。
用于增量训练场景: 用户追加 badcase 图片后基于 best.pt fine-tune。
"""

import hashlib
import os
import shutil
import yaml
from pathlib import Path
from typing import Optional

from utils.dataset_splitter import split_dataset
from utils.image_files import SUPPORTED_IMAGE_EXTENSIONS


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """计算文件 MD5 用于去重"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def merge_incremental_data(
    task_dir: str,
    new_image_dir: str,
    new_label_dir: Optional[str] = None,
    base_model_path: Optional[str] = None,
    auto_label_new: bool = False,
    class_names: Optional[list[str]] = None,
) -> dict:
    """
    合并旧训练数据 + 新增数据，去重后重新分割。

    Args:
        task_dir: 任务根目录 (backend/uploads/{task_id})
        new_image_dir: 新增图片目录
        new_label_dir: 新增标注目录 (可选,如果用户已手动标注)
        base_model_path: 上次训练的 best.pt (用于辅助预标注)
        auto_label_new: 是否用旧模型对新图自动预标注
        class_names: 类别名列表

    Returns: {
        "merged_image_dir": str,
        "merged_label_dir": str,
        "total_images": int,
        "old_images": int,
        "new_images": int,
        "duplicates_removed": int,
        "auto_labeled": int,
    }
    """
    task_path = Path(task_dir)
    new_img_path = Path(new_image_dir)

    # Collect existing labeled data (from labeled_images + labels)
    old_image_dir = task_path / "labeled_images"
    old_label_dir = task_path / "labels"

    # Also check for previously merged data
    prev_merged_img = task_path / "merged_images"
    prev_merged_lbl = task_path / "merged_labels"
    if prev_merged_img.exists() and prev_merged_lbl.exists():
        old_image_dir = prev_merged_img
        old_label_dir = prev_merged_lbl

    # Prepare merge output directories
    merged_image_dir = task_path / "incremental_merged" / "images"
    merged_label_dir = task_path / "incremental_merged" / "labels"

    # Clean previous merge
    merge_root = task_path / "incremental_merged"
    if merge_root.exists():
        shutil.rmtree(merge_root)
    os.makedirs(merged_image_dir, exist_ok=True)
    os.makedirs(merged_label_dir, exist_ok=True)

    # Hash index for deduplication
    seen_hashes: set[str] = set()
    old_count = 0
    new_count = 0
    duplicates = 0

    # Copy old data
    if old_image_dir.exists():
        for img_file in sorted(old_image_dir.iterdir()):
            if not img_file.is_file() or img_file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            h = file_hash(img_file)
            if h in seen_hashes:
                duplicates += 1
                continue
            seen_hashes.add(h)

            shutil.copy2(img_file, merged_image_dir / img_file.name)
            # Copy corresponding label
            lbl_file = old_label_dir / f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, merged_label_dir / f"{img_file.stem}.txt")
            old_count += 1

    # Copy new data (with dedup)
    new_label_path = Path(new_label_dir) if new_label_dir else None
    auto_labeled_count = 0

    new_images_needing_labels: list[Path] = []

    for img_file in sorted(new_img_path.iterdir()):
        if not img_file.is_file() or img_file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        h = file_hash(img_file)
        if h in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(h)

        # Avoid filename collision with old data
        dest_name = img_file.name
        if (merged_image_dir / dest_name).exists():
            dest_name = f"new_{img_file.stem}{img_file.suffix}"

        shutil.copy2(img_file, merged_image_dir / dest_name)
        dest_stem = Path(dest_name).stem

        # Check if label exists
        has_label = False
        if new_label_path and (new_label_path / f"{img_file.stem}.txt").exists():
            lbl_src = new_label_path / f"{img_file.stem}.txt"
            if lbl_src.stat().st_size > 0:
                shutil.copy2(lbl_src, merged_label_dir / f"{dest_stem}.txt")
                has_label = True

        if not has_label:
            new_images_needing_labels.append(merged_image_dir / dest_name)

        new_count += 1

    # Auto-label new images without annotations using base model
    if auto_label_new and base_model_path and new_images_needing_labels:
        auto_labeled_count = _auto_label_with_model(
            base_model_path, new_images_needing_labels, merged_label_dir
        )

    return {
        "merged_image_dir": str(merged_image_dir),
        "merged_label_dir": str(merged_label_dir),
        "total_images": old_count + new_count,
        "old_images": old_count,
        "new_images": new_count,
        "duplicates_removed": duplicates,
        "auto_labeled": auto_labeled_count,
    }


def prepare_incremental_dataset(
    task_dir: str,
    class_names: list[str],
    merged_image_dir: str,
    merged_label_dir: str,
) -> dict:
    """
    将合并后的数据做 80/20 split 并生成 data.yaml。

    Returns: {
        "dataset_dir": str,
        "data_yaml": str,
        "train_count": int,
        "val_count": int,
    }
    """
    task_path = Path(task_dir)
    dataset_dir = task_path / "incremental_dataset"

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    stats = split_dataset(
        image_dir=merged_image_dir,
        label_dir=merged_label_dir,
        output_root=str(dataset_dir),
        ratios=(0.8, 0.2, 0.0),
        seed=42,
    )

    # Generate data.yaml
    data_yaml_path = dataset_dir / "data.yaml"
    data_config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, default_flow_style=False)

    return {
        "dataset_dir": str(dataset_dir),
        "data_yaml": str(data_yaml_path),
        "train_count": stats.get("train", 0),
        "val_count": stats.get("val", 0),
    }


def archive_training_version(
    task_dir: str,
    version: int,
    best_pt_path: str,
    data_yaml_path: str,
    stats: dict,
) -> str:
    """
    将训练产物归档到 training_history/v{N}/ 目录。

    Returns: 归档目录路径
    """
    import json

    task_path = Path(task_dir)
    history_dir = task_path / "training_history" / f"v{version}"
    os.makedirs(history_dir, exist_ok=True)

    # Copy best.pt
    src_pt = Path(best_pt_path)
    if src_pt.exists():
        shutil.copy2(src_pt, history_dir / "best.pt")

    # Copy data.yaml
    src_yaml = Path(data_yaml_path)
    if src_yaml.exists():
        shutil.copy2(src_yaml, history_dir / "data.yaml")

    # Save stats
    stats_file = history_dir / "stats.json"
    stats["version"] = version
    stats["archived_at"] = _now_iso()
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return str(history_dir)


def get_training_history(task_dir: str) -> list[dict]:
    """
    获取该任务的所有训练版本历史。
    Returns: [{ version, images, map50, new_images, archived_at, ... }]
    """
    import json

    task_path = Path(task_dir)
    history_root = task_path / "training_history"
    if not history_root.exists():
        return []

    versions = []
    for vdir in sorted(history_root.iterdir()):
        if not vdir.is_dir() or not vdir.name.startswith("v"):
            continue
        stats_file = vdir / "stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, encoding="utf-8") as f:
                    stats = json.load(f)
                stats["has_model"] = (vdir / "best.pt").exists()
                versions.append(stats)
            except Exception:
                versions.append({
                    "version": int(vdir.name[1:]),
                    "has_model": (vdir / "best.pt").exists(),
                })
        else:
            versions.append({
                "version": int(vdir.name[1:]),
                "has_model": (vdir / "best.pt").exists(),
            })

    return sorted(versions, key=lambda v: v.get("version", 0))


def get_latest_version(task_dir: str) -> int:
    """获取最新版本号, 无历史则返回 0"""
    history = get_training_history(task_dir)
    if not history:
        return 0
    return max(v.get("version", 0) for v in history)


def get_latest_model_path(task_dir: str) -> Optional[str]:
    """获取最新版本的 best.pt 路径"""
    task_path = Path(task_dir)
    latest_ver = get_latest_version(task_dir)
    if latest_ver == 0:
        # Check default training output
        default_best = task_path / "local_training_output" / "exp" / "weights" / "best.pt"
        if default_best.exists():
            return str(default_best)
        # Also check seed training
        seed_best = task_path / "seed_training_output" / "exp" / "weights" / "best.pt"
        if seed_best.exists():
            return str(seed_best)
        return None

    model_path = task_path / "training_history" / f"v{latest_ver}" / "best.pt"
    return str(model_path) if model_path.exists() else None


def _auto_label_with_model(
    model_path: str, image_paths: list[Path], output_label_dir: Path, conf: float = 0.4
) -> int:
    """用已有模型对新图自动打标"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception:
        return 0

    labeled = 0
    for img_path in image_paths:
        try:
            results = model.predict(source=str(img_path), conf=conf, verbose=False)
            if results and results[0].boxes and len(results[0].boxes) > 0:
                lines = []
                for box in results[0].boxes:
                    cls_idx = int(box.cls[0])
                    xywhn = box.xywhn[0]
                    cx, cy, w, h = float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                lbl_path = output_label_dir / f"{img_path.stem}.txt"
                with open(lbl_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
                labeled += 1
        except Exception:
            continue

    return labeled


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
