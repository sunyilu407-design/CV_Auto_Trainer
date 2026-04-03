import os
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
    ) -> dict:
        self._output_dir = (
            Path(dataset_dir).parent / "local_training_output" / "exp"
        )
        os.makedirs(self._output_dir, exist_ok=True)

        data_yaml = Path(dataset_dir) / "data.yaml"
        cmd = self._build_command(train_config, data_yaml)

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

        if returncode != 0 and not self._stop_flag:
            stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"本地训练子进程异常退出（code {returncode}）: {stderr}")

        return self._collect_artifacts()

    def _build_command(self, cfg: dict, data_yaml: Path) -> list[str]:
        model = cfg.get("model", "yolov8s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = str(Path(self._output_dir).parent)
        resume_str = (
            [f"--resume={self._output_dir / 'weights' / 'last.pt'}"]
            if cfg.get("resume_last", False) else []
        )

        cmd = [
            "python", "-c",
            f"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='{data_yaml}', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device=0)",
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
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 3:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
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
