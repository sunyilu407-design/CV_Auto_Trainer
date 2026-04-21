import random
import shutil
import os
from pathlib import Path
from collections import defaultdict, Counter
from utils.image_files import find_image_for_stem


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_root: str,
    ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict:
    """
    分层抽样分割：确保每个类别在 train/val/test 中的比例一致。
    输出目录结构：
        output_root/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
    返回：{split: count} 统计
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "比例之和必须为 1"
    random.seed(seed)

    class_groups: dict = defaultdict(list)
    for lbl_path in Path(label_dir).glob("*.txt"):
        img_path = find_image_for_stem(image_dir, lbl_path.stem)
        if img_path is None:
            continue
        dominant_class = _get_dominant_class(lbl_path)
        class_groups[dominant_class].append((img_path, lbl_path))

    splits: dict = {"train": [], "val": [], "test": []}

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    for split, items in splits.items():
        img_out = Path(output_root) / "images" / split
        lbl_out = Path(output_root) / "labels" / split
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, lbl_path in items:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

    return {k: len(v) for k, v in splits.items()}


def _get_dominant_class(label_path: Path) -> int:
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(int(parts[0]))
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]
