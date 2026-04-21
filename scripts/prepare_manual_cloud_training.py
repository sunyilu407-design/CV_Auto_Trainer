#!/usr/bin/env python3
"""
Prepare a manual cloud training bundle for cases where the system cannot connect
to the user's rented cloud GPU instance directly.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CLOUD_SCRIPTS_DIR = ROOT_DIR / "cloud_scripts"


def _pack_dataset(dataset_dir: Path, output_zip: Path) -> None:
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in dataset_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(dataset_dir))


def _copy_cloud_scripts(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in CLOUD_SCRIPTS_DIR.glob("*.py"):
        shutil.copyfile(path, target_dir / path.name)


def _build_readme(
    *,
    remote_work_dir: str,
    model: str,
    epochs: int,
    imgsz: int,
    lr0: float,
    patience: int,
    export_formats: list[str],
) -> str:
    export_lines = export_formats or ["onnx"]
    export_commands = "\n".join(
        f"python cloud_scripts/export.py --weights {remote_work_dir}/training_output/exp/weights/best.pt --format {fmt}"
        for fmt in export_lines
    )
    return "\n".join(
        [
            "# Manual Cloud Training",
            "",
            "Use this bundle when the local system cannot connect to the rented cloud GPU environment directly.",
            "",
            "## 1. Upload files to the cloud machine",
            "",
            f"`scp dataset.zip <user>@<host>:{remote_work_dir}/`",
            f"`scp -r cloud_scripts <user>@<host>:{remote_work_dir}/`",
            "",
            "## 2. Start training on the cloud machine",
            "",
            f"`cd {remote_work_dir}`",
            "`unzip -q dataset.zip -d dataset`",
            (
                f"`python cloud_scripts/train.py --data {remote_work_dir}/dataset/data.yaml "
                f"--model {model} --epochs {epochs} --imgsz {imgsz} --lr0 {lr0} "
                f"--patience {patience} --project {remote_work_dir}/training_output`"
            ),
            "",
            "Recommended: run it inside `screen` so the session survives disconnects.",
            "",
            "Example:",
            (
                f"`screen -dmS train bash -c \"cd {remote_work_dir} && "
                f"python cloud_scripts/train.py --data {remote_work_dir}/dataset/data.yaml "
                f"--model {model} --epochs {epochs} --imgsz {imgsz} --lr0 {lr0} "
                f"--patience {patience} --project {remote_work_dir}/training_output\"`"
            ),
            "",
            "## 3. Export deployment formats after training",
            "",
            export_commands,
            "",
            "## 4. Download artifacts back to the local machine",
            "",
            f"`scp <user>@<host>:{remote_work_dir}/training_output/exp/weights/best.pt ./best.pt`",
            f"`scp <user>@<host>:{remote_work_dir}/training_output/exp/results.csv ./results.csv`",
            "",
            "## 5. Verify training progress on the cloud machine",
            "",
            f"`python cloud_scripts/health_check.py`",
            "",
            "Expected outputs after success:",
            "- `best.pt`",
            "- `last.pt`",
            "- `results.csv`",
            "- exported model files such as `best.onnx`",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a manual cloud training bundle")
    parser.add_argument("--dataset-dir", required=True, help="path to prepared YOLO dataset directory")
    parser.add_argument("--output-dir", required=True, help="directory to write the manual bundle into")
    parser.add_argument("--remote-work-dir", default="/root/workspace", help="target work directory on the cloud machine")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--export-format", action="append", dest="export_formats", default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "data.yaml").exists():
        raise FileNotFoundError(f"Missing data.yaml under dataset directory: {dataset_dir}")

    output_dir = Path(args.output_dir).resolve()
    bundle_dir = output_dir / "manual_cloud_training"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dataset_zip = bundle_dir / "dataset.zip"
    _pack_dataset(dataset_dir, dataset_zip)

    script_dir = bundle_dir / "cloud_scripts"
    _copy_cloud_scripts(script_dir)

    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        _build_readme(
            remote_work_dir=args.remote_work_dir,
            model=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            lr0=args.lr0,
            patience=args.patience,
            export_formats=args.export_formats or ["onnx"],
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Manual cloud training bundle ready: {bundle_dir}")
    print(f"Dataset archive: {dataset_zip}")
    print(f"Instructions: {readme_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
