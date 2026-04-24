import albumentations as A
import cv2
import math
import os
import shutil
from pathlib import Path
from typing import Optional, Callable
from utils.image_files import list_image_files


def build_pipeline(
    strength: str = "medium",
    enabled: Optional[dict] = None,
) -> A.Compose:
    if enabled is None:
        enabled = {k: True for k in ["geometric", "color", "noise", "weather", "occlusion"]}

    transforms = []

    if enabled.get("geometric", True):
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.3,
                rotate_limit=15 if strength != "light" else 5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ]

    if enabled.get("color", True):
        transforms += [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30,
                val_shift_limit=20, p=0.4,
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ]
        if strength in ("medium", "heavy"):
            transforms.append(A.CLAHE(clip_limit=4.0, p=0.2))

    if enabled.get("noise", True) and strength != "light":
        transforms += [
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),
        ]

    if enabled.get("weather", False) and strength == "heavy":
        transforms += [
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        ]

    if enabled.get("occlusion", False) and strength == "heavy":
        transforms.append(
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                fill_value=0, p=0.3,
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["labels"],
            min_visibility=0.3,
            clip=True,
        ),
    )


def augment_dataset(
    src_image_dir: str,
    src_label_dir: str,
    output_image_dir: str,
    output_label_dir: str,
    target_count: int,
    strength: str = "medium",
    enabled: Optional[dict] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    src_images = list_image_files(src_image_dir)
    if not src_images:
        raise ValueError(f"源目录无图片: {src_image_dir}")

    pipeline = build_pipeline(strength, enabled)

    # 复制原始数据
    for img_path in src_images:
        label_path = Path(src_label_dir) / f"{img_path.stem}.txt"
        shutil.copy(img_path, output_image_dir / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, output_label_dir / label_path.name)

    existing = len(src_images)
    needed = max(0, target_count - existing)
    per_image = math.ceil(needed / existing) if needed > 0 else 0
    generated = 0

    for img_path in src_images:
        if generated >= needed:
            break

        label_path = Path(src_label_dir) / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        bboxes, labels = _load_yolo_label(label_path)

        if img is None:
            continue

        for aug_idx in range(per_image):
            if generated >= needed:
                break
            try:
                result = pipeline(image=img, bboxes=bboxes, labels=labels)
                result_bboxes = result.get("bboxes")
                if not result_bboxes:
                    continue

                result_labels = result.get("labels", [])
                out_stem = f"{img_path.stem}_aug{generated:05d}"
                cv2.imwrite(f"{output_image_dir}/{out_stem}.jpg", result["image"])
                _save_yolo_label(
                    f"{output_label_dir}/{out_stem}.txt",
                    result_bboxes,
                    result_labels,
                )
                generated += 1

                if progress_callback:
                    progress_callback(generated, needed, "augmentation")

            except Exception:
                continue

    return {"existing": existing, "generated": generated, "total": existing + generated}


def _load_yolo_label(label_path: Path) -> tuple[list, list]:
    bboxes, labels = [], []
    if not label_path.exists():
        return bboxes, labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, labels


def _save_yolo_label(path: str, bboxes: list, labels: list):
    with open(path, "w", encoding="utf-8") as f:
        for label, bbox in zip(labels, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
