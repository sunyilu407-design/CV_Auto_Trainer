"""
Seed Trainer — 用少量手动标注图片训练一个轻量种子模型 (yolov8n)。
复用 LocalTrainer 的子进程隔离方式，但参数更轻量。

注意：SEED_MODEL 固定为 yolov8n.pt，
这是有意为之 — 种子训练的目标是"快速验证"，不是"最优精度"。
yolov8n.pt 权重极小、下载快、训练快，适合 5~50 张图的场景。
用户实际训练的模型由 TrainConfig 页面选择，不受此处影响。
"""

import os
import sys
import time
import subprocess
import threading
import yaml
from pathlib import Path
from typing import Optional, Callable

from utils.dataset_splitter import split_dataset
from utils.image_files import list_image_files, find_image_for_stem


SEED_MODEL = "yolov8n.pt"
SEED_EPOCHS = 50
SEED_IMGSZ = 640
SEED_BATCH = 8
SEED_LR0 = 0.01
SEED_PATIENCE = 15


def prepare_seed_dataset(
    task_dir: str,
    class_names: list[str],
) -> dict:
    """
    从 seed_labels 和 images/video_frames 目录组装训练数据集。
    返回: { "dataset_dir": str, "data_yaml": str, "train_count": int, "val_count": int }
    """
    task_path = Path(task_dir)
    seed_label_dir = task_path / "seed_labels"

    if not seed_label_dir.exists():
        raise FileNotFoundError(f"Seed label directory not found: {seed_label_dir}")

    # Find the image source directory
    image_dir: Optional[Path] = None
    for subdir in ["images", "video_frames"]:
        candidate = task_path / subdir
        if candidate.exists() and any(candidate.iterdir()):
            image_dir = candidate
            break

    if image_dir is None:
        raise FileNotFoundError(f"No image directory found in task: {task_path}")

    # Count annotated images
    label_files = list(seed_label_dir.glob("*.txt"))
    valid_labels = [lf for lf in label_files if lf.stat().st_size > 0]

    if len(valid_labels) < 5:
        raise ValueError(f"Too few seed annotations ({len(valid_labels)}). Need at least 5.")

    # Prepare output directory
    dataset_dir = task_path / "seed_dataset"
    if dataset_dir.exists():
        import shutil
        shutil.rmtree(dataset_dir)

    # Split using existing utility (80/20, no test for seed)
    stats = split_dataset(
        image_dir=str(image_dir),
        label_dir=str(seed_label_dir),
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


def run_seed_training(
    dataset_dir: str,
    class_names: list[str],
    progress_callback: Optional[Callable[[dict], None]] = None,
    epochs: int = SEED_EPOCHS,
    device: int = 0,
) -> dict:
    """
    运行种子模型训练。
    返回: { "seed_model_path": str, "best_map": float, "training_time_seconds": float }
    """
    data_yaml = Path(dataset_dir) / "data.yaml"
    output_dir = Path(dataset_dir).parent / "seed_training_output"
    os.makedirs(output_dir, exist_ok=True)

    project_dir = str(output_dir)

    cmd = [
        sys.executable, "-c",
        f"from ultralytics import YOLO; "
        f"model = YOLO('{SEED_MODEL}'); "
        f"model.train(data='{data_yaml}', "
        f"epochs={epochs}, imgsz={SEED_IMGSZ}, lr0={SEED_LR0}, "
        f"batch={SEED_BATCH}, patience={SEED_PATIENCE}, "
        f"project='{project_dir}', name='exp', exist_ok=True, "
        f"device={device})",
    ]

    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )

    # Poll training progress in background
    exp_dir = output_dir / "exp"
    stop_flag = threading.Event()

    def poll_progress():
        last_epoch = 0
        while not stop_flag.is_set():
            time.sleep(3)
            # Check results.csv for progress
            results_csv = exp_dir / "results.csv"
            if results_csv.exists():
                try:
                    with open(results_csv) as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        current_epoch = len(lines) - 1  # header row
                        if current_epoch > last_epoch:
                            last_epoch = current_epoch
                            # Parse last line for mAP
                            parts = lines[-1].strip().split(",")
                            map50 = 0.0
                            if len(parts) > 6:
                                try:
                                    map50 = float(parts[6].strip())
                                except (ValueError, IndexError):
                                    pass
                            if progress_callback:
                                progress_callback({
                                    "type": "seed_training_progress",
                                    "currentEpoch": current_epoch,
                                    "totalEpochs": epochs,
                                    "currentMap": map50,
                                })
                except Exception:
                    pass

    poller = threading.Thread(target=poll_progress, daemon=True)
    poller.start()

    returncode = process.wait()
    stop_flag.set()
    poller.join(timeout=5)

    training_time = time.time() - start_time

    if returncode != 0:
        stderr = process.stderr.read().decode() if process.stderr else ""
        raise RuntimeError(f"Seed training failed (code {returncode}): {stderr[:500]}")

    # Locate best.pt
    best_pt = exp_dir / "weights" / "best.pt"
    if not best_pt.exists():
        # Fallback to last.pt
        best_pt = exp_dir / "weights" / "last.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Training completed but no weight file found in {exp_dir / 'weights'}")

    # Get final mAP from results
    best_map = 0.0
    results_csv = exp_dir / "results.csv"
    if results_csv.exists():
        try:
            with open(results_csv) as f:
                lines = f.readlines()
            if len(lines) > 1:
                # Find max mAP50 (column 6)
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) > 6:
                        try:
                            val = float(parts[6].strip())
                            best_map = max(best_map, val)
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass

    return {
        "seed_model_path": str(best_pt),
        "best_map": best_map,
        "training_time_seconds": round(training_time, 1),
    }
