import os
import sys
import time
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Callable


class LocalTrainer:
    """
    本地 GPU 训练器。
    通过子进程运行 Ultralytics 训练，显存与 Worker 主进程完全隔离。
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stop_flag = False
        self._output_dir = None

    def train(
        self,
        dataset_dir: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None,
        data_yaml: str | None = None,
    ) -> dict:
        self._output_dir = (
            Path(dataset_dir).parent / "local_training_output" / "exp"
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # 增量模式：使用合并后的 data.yaml；首次训练：自动生成
        yaml_path = Path(data_yaml) if data_yaml else (Path(dataset_dir) / "data.yaml")
        cmd = self._build_command(train_config, yaml_path)

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
        )

        poller = threading.Thread(
            target=self._poll_progress,
            args=(progress_callback, train_config),
            daemon=True,
        )
        poller.start()

        returncode = self._process.wait()

        artifacts = self._collect_artifacts()

        # 从 results.csv 读取最终 best mAP，写入 artifacts
        best_map = self._read_final_map()
        if best_map is not None:
            artifacts["best_map"] = best_map

        if returncode != 0 and not self._stop_flag:
            stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"本地训练子进程异常退出（code {returncode}）: {stderr}")

        return artifacts

    def _build_command(self, cfg: dict, data_yaml: Path) -> list[str]:
        model = cfg.get("model", "yolov8s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = str(Path(self._output_dir).parent)
        device = cfg.get("device", 0)
        resume_str = (
            ", resume='" + str(self._output_dir / 'weights' / 'last.pt') + "'"
            if cfg.get("resume_last", False) else ""
        )

        cmd = [
            sys.executable, "-c",
            f"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='{data_yaml}', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device={device}{resume_str})",
        ]
        return cmd

    def _poll_progress(
        self,
        progress_callback: Optional[Callable],
        cfg: dict,
    ):
        results_csv = self._output_dir / "results.csv"
        while True:
            time.sleep(10)
            if self._stop_flag:
                break
            if not results_csv.exists():
                continue
            try:
                with open(results_csv, "r") as f:
                    lines = f.readlines()
                if not lines:
                    continue
                # Find last non-header data line
                last_line = ""
                for candidate in reversed(lines):
                    stripped = candidate.strip()
                    if stripped and not stripped.startswith("epoch"):
                        last_line = stripped
                        break
                if not last_line:
                    continue
                parts = last_line.split(",")
                if len(parts) > 6:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[6].strip())
                    if progress_callback:
                        progress_callback({
                            "current_epoch": current_epoch,
                            "total_epochs": cfg.get("epochs", 100),
                            "current_map": current_map,
                            "done": current_epoch >= cfg.get("epochs", 100),
                        })
            except (ValueError, IndexError, FileNotFoundError):
                continue

    def _collect_artifacts(self) -> dict:
        weights_dir = self._output_dir / "weights"
        artifacts = {}
        for fname in ["best.pt", "last.pt"]:
            fpath = weights_dir / fname
            if fpath.exists():
                artifacts[fname] = str(fpath)
        results_csv = self._output_dir / "results.csv"
        if results_csv.exists():
            artifacts["results.csv"] = str(results_csv)
        return artifacts

    def _read_final_map(self) -> float | None:
        """
        从 results.csv 最后一行读取 best mAP50 值。
        Ultralytics results.csv 列（YOLO11, approximate order）：
          0=epoch, 1=train/box_loss, 2=train/cls_loss, 3=train/dfl_loss,
          4=metrics/precision(B), 5=metrics/recall(B), 6=metrics/mAP50(B),
          7=metrics/mAP50-95(B)
        改用 header 查找而非固定 index，避免版本间列顺序差异导致读错。
        """
        results_csv = self._output_dir / "results.csv"
        if not results_csv.exists():
            return None
        try:
            with open(results_csv, encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                return None
            header = lines[0].strip().split(",")
            try:
                map50_col = header.index("metrics/mAP50(B)")
            except ValueError:
                return None
            for line in reversed(lines[1:]):
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split(",")
                if len(parts) > map50_col:
                    try:
                        return float(parts[map50_col].strip())
                    except ValueError:
                        continue
        except OSError:
            pass
        return None

    def cancel(self):
        self._stop_flag = True
        if self._process is None:
            return
        try:
            if os.name == "nt":
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
