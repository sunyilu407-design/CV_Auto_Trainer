#!/usr/bin/env python3
"""
云端训练入口脚本。
上传至 AutoDL/云服务器实例后执行。
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="CV Auto Trainer Cloud Training")
    parser.add_argument("--data", type=str, required=True, help="data.yaml path")
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--project", type=str, default="/root/training_output")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)

    resume_str = f", resume='{args.resume}'" if args.resume else ""

    print(f"Starting training: data={args.data}, epochs={args.epochs}, imgsz={args.imgsz}")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name="exp",
        exist_ok=True,
        device=0,
    )
    print(f"Training complete. Best map: {results.best_map}")


if __name__ == "__main__":
    main()
