import os
import shutil
from pathlib import Path


def save_yolo_labels(
    passed_boxes: dict,
    output_label_dir: str,
    output_image_dir: str,
):
    """
    将质检通过的框保存为 YOLO .txt 格式。
    每行格式：class_idx cx cy w h
    无有效框的图片不写入数据集。
    """
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    for img_path, boxes in passed_boxes.items():
        if not boxes:
            continue
        stem = Path(img_path).stem
        label_path = f"{output_label_dir}/{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                cx, cy, w, h = box["bbox_xywhn"]
                f.write(
                    f"{box['class_idx']} {cx:.6f} {cy:.6f} "
                    f"{w:.6f} {h:.6f}\n"
                )
        shutil.copy(img_path, f"{output_image_dir}/{Path(img_path).name}")


def load_yolo_labels(label_path: str) -> tuple[list, list]:
    """读取 YOLO .txt 标注文件，返回 (bboxes, labels)"""
    bboxes, labels = [], []
    if not os.path.exists(label_path):
        return bboxes, labels
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, labels
