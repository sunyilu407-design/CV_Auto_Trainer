#!/usr/bin/env python3
"""
训练健康检查脚本。
定期执行，检测训练是否正常，输出状态到指定文件。
"""
import sys
import time
from pathlib import Path


def check_training_health(output_dir: str = "/root/training_output/exp") -> dict:
    exp_dir = Path(output_dir)
    weights_dir = exp_dir / "weights"

    status = {
        "running": True,
        "best_exists": False,
        "last_exists": False,
        "current_epoch": 0,
        "total_epochs": 100,
        "error": None,
    }

    # Check best.pt
    best_pt = weights_dir / "best.pt"
    last_pt = weights_dir / "last.pt"
    results_csv = exp_dir / "results.csv"
    train_log = exp_dir / "train.log"

    status["best_exists"] = best_pt.exists()
    status["last_exists"] = last_pt.exists()

    # Read last line of results.csv
    if results_csv.exists():
        try:
            with open(results_csv) as f:
                lines = f.readlines()
            if lines:
                last = lines[-1].strip().split(",")
                if len(last) > 0:
                    try:
                        status["current_epoch"] = int(last[0].strip())
                    except ValueError:
                        pass
        except Exception:
            pass

    # Check for errors
    if train_log.exists():
        try:
            with open(train_log) as f:
                content = f.read()
            if "error" in content.lower() or "exception" in content.lower():
                status["error"] = "Training error detected in log"
        except Exception:
            pass

    if status["best_exists"] or status["last_exists"]:
        status["running"] = False

    return status


if __name__ == "__main__":
    import json
    result = check_training_health()
    print(json.dumps(result))
