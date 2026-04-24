"""
数据集打包器。

职责：
1. 将 labeled_images + labels 按分层抽样分割为 train/val/test
2. 生成 Ultralytics 所需的 data.yaml
3. 返回分割统计和绝对路径，供前端和 Worker 使用
"""

from __future__ import annotations

import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

import yaml


# ─── 分割 ────────────────────────────────────────────────────────────────

def split_dataset_stratified(
    image_dir: str,
    label_dir: str,
    output_root: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, int]:
    """
    分层抽样分割：确保每个类别在 train/val/test 中的比例一致。

    输出目录结构：
        output_root/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
    返回：{train: count, val: count, test: count}
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "分割比例之和必须为 1"
    random.seed(seed)

    def _dominant_class(lbl_path: Path) -> int:
        try:
            with open(lbl_path, encoding="utf-8") as f:
                classes = [int(line.strip().split()[0]) for line in f if line.strip().split()]
            return Counter(classes).most_common(1)[0][0] if classes else -1
        except (ValueError, IndexError, OSError):
            return -1

    def _find_image(image_dir: Path, stem: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = image_dir / f"{stem}{ext}"
            if p.is_file():
                return p
        return None

    class_groups: dict[int, list[tuple[Path, Path]]] = {}
    for lbl_path in Path(label_dir).glob("*.txt"):
        img_path = _find_image(Path(image_dir), lbl_path.stem)
        if img_path is None:
            continue
        cls = _dominant_class(lbl_path)
        class_groups.setdefault(cls, []).append((img_path, lbl_path))

    splits: dict[str, list[tuple[Path, Path]]] = {"train": [], "val": [], "test": []}

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for split_name, items in splits.items():
        img_out = Path(output_root) / "images" / split_name
        lbl_out = Path(output_root) / "labels" / split_name
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, lbl_path in items:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

    return {k: len(v) for k, v in splits.items()}


# ─── YAML 生成 ────────────────────────────────────────────────────────────

def generate_data_yaml(
    dataset_root: str,
    class_names: list[str],
    output_path: str,
) -> str:
    """
    生成 Ultralytics/YOLO 训练所需的 data.yaml 配置文件。
    返回写入的绝对路径。
    """
    config = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    return str(out_path.resolve())


# ─── 质量报告 ─────────────────────────────────────────────────────────────

def compute_quality_report(
    label_dir: str,
    class_names: list[str],
) -> dict:
    """
    基于 label_dir 中的标注文件，生成数据质量报告。
    """
    from pathlib import Path

    label_path = Path(label_dir)
    if not label_path.exists():
        return {
            "total_images": 0,
            "class_distribution": [],
            "avg_boxes_per_image": 0.0,
            "warnings": ["标注目录不存在"],
        }

    label_files = list(label_path.glob("*.txt"))
    total_boxes = 0
    class_counts: dict[int, int] = Counter()
    images_with_boxes = 0

    for lbl_file in label_files:
        try:
            with open(lbl_file, encoding="utf-8") as f:
                lines = f.readlines()
            boxes = [line.strip().split() for line in lines if line.strip().split()]
            if boxes:
                images_with_boxes += 1
                total_boxes += len(boxes)
                for parts in boxes:
                    try:
                        class_counts[int(parts[0])] += 1
                    except (ValueError, IndexError):
                        pass
        except OSError:
            continue

    total_images = len(label_files)
    avg_boxes = total_boxes / total_images if total_images > 0 else 0.0

    class_distribution = []
    for idx, name in enumerate(class_names):
        count = class_counts.get(idx, 0)
        class_distribution.append({
            "class_name": name,
            "box_count": count,
            "avg_boxes_per_image": round(count / total_images, 2) if total_images > 0 else 0.0,
        })

    warnings: list[str] = []
    for dist in class_distribution:
        if dist["box_count"] == 0:
            warnings.append(f"类别「{dist['class_name']}」没有标注数据，请补充")
        elif dist["box_count"] < 5:
            warnings.append(f"类别「{dist['class_name']}」数据过少（{dist['box_count']} 张），建议补充更多样本")
    if total_images < 20:
        warnings.append(f"总样本数仅 {total_images} 张，建议至少 50 张以获得较好训练效果")
    if avg_boxes < 0.5:
        warnings.append("平均每张图片的标注框数量过低，请检查标注质量")

    return {
        "total_images": total_images,
        "class_distribution": class_distribution,
        "avg_boxes_per_image": round(avg_boxes, 2),
        "warnings": warnings,
    }


# ─── 一站式准备函数 ──────────────────────────────────────────────────────

def prepare_full_dataset(
    labeled_images_dir: str,
    labels_dir: str,
    output_root: str,
    class_names: list[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    """
    执行完整数据集准备流程：
        1. 分层分割 train/val/test
        2. 生成 data.yaml
        3. 计算质量报告

    返回：
        {
            "dataset_root": str,          # 数据集根目录绝对路径
            "data_yaml_path": str,        # data.yaml 绝对路径
            "split_stats": {train, val, test},
            "quality_report": {...},
            "image_dir": str,
            "label_dir": str,
        }
    """
    split_stats = split_dataset_stratified(
        image_dir=labeled_images_dir,
        label_dir=labels_dir,
        output_root=output_root,
        ratios=ratios,
        seed=seed,
    )
    data_yaml_path = generate_data_yaml(
        dataset_root=output_root,
        class_names=class_names,
        output_path=str(Path(output_root) / "data.yaml"),
    )
    quality_report = compute_quality_report(
        label_dir=labels_dir,
        class_names=class_names,
    )
    return {
        "dataset_root": str(Path(output_root).resolve()),
        "data_yaml_path": data_yaml_path,
        "split_stats": split_stats,
        "quality_report": quality_report,
        "image_dir": str(Path(output_root) / "images"),
        "label_dir": str(Path(output_root) / "labels"),
    }
