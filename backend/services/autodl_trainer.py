import time
import os
import requests
import paramiko
from typing import Optional, Callable
from .cloud_trainer import CloudTrainer, CloudTrainState


class AutoDLCloudTrainer(CloudTrainer):
    """
    AutoDL 专用云端训练器。
    用户只需提供 token，系统自动通过 AutoDL API 创建和销毁 GPU 实例。
    """

    def __init__(self, config: dict):
        self.token = config["autodl_token"]
        self.api_base = "https://www.autodl.com/api/v1"
        self.gpu_type = config.get("gpu_type", "RTX 4090")
        self.remote_work_dir = "/root"
        self.gpu_device = "0"
        self.state = CloudTrainState.IDLE
        self._instance_id: Optional[str] = None
        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._alert_manager = None

    def connect(self):
        self.state = CloudTrainState.CONNECTING
        self._instance_id = self._create_instance()
        self._wait_for_running()
        self._ssh = self._get_ssh()
        self._sftp = self._ssh.open_sftp()

    def _create_instance(self) -> str:
        resp = requests.post(
            f"{self.api_base}/instance/create",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"gpu_type": self.gpu_type, "image": "pytorch:2.1.0-cuda11.8"},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"]["instance_id"]

    def _wait_for_running(self, timeout: int = 300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{self.api_base}/instance/status/{self._instance_id}",
                headers={"Authorization": f"Bearer {self.token}"},
            )
            if resp.json()["data"]["status"] == "running":
                return
            time.sleep(5)
        raise TimeoutError(f"AutoDL 实例 {self._instance_id} 启动超时（{timeout}s）")

    def _get_ssh(self) -> paramiko.SSHClient:
        info = self._get_instance_info()
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            info["host"], port=info["port"],
            username="root", password=info["password"], timeout=30,
        )
        return ssh

    def _get_instance_info(self) -> dict:
        resp = requests.get(
            f"{self.api_base}/instance/info/{self._instance_id}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def upload_dataset(self, zip_path: str):
        self.state = CloudTrainState.UPLOADING
        self._sftp.put(zip_path, "/root/dataset.zip")
        self._ssh.exec_command("cd /root && unzip -q dataset.zip -d dataset")
        time.sleep(5)

    def run_training(self, train_config: dict, progress_callback: Optional[Callable] = None):
        self.state = CloudTrainState.TRAINING
        train_cmd = self._build_train_command(train_config)
        self._ssh.exec_command(
            f"cd /root && screen -dmS train bash -c '{train_cmd}'"
        )
        while True:
            time.sleep(30)
            status = self._check_training_status(train_config)
            if progress_callback:
                progress_callback(status)
            if status.get("done"):
                break
            if status.get("error"):
                raise RuntimeError(f"训练失败: {status.get('error_msg')}")

    def _build_train_command(self, cfg: dict) -> str:
        model = cfg.get("model", "yolo11s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = "/root/training_output"
        resume_str = (
            f", resume='/root/training_output/exp/weights/last.pt'"
            if cfg.get("resume_last", False) else ""
        )
        return (
            f"python -c \"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='/root/dataset/data.yaml', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device={self.gpu_device}{resume_str})\""
        )

    def _check_training_status(self, cfg: dict) -> dict:
        _, stdout, _ = self._ssh.exec_command(
            "tail -1 /root/training_output/exp/results.csv 2>/dev/null"
        )
        line = stdout.read().decode().strip()
        _, stdout2, _ = self._ssh.exec_command(
            "[ -f /root/training_output/exp/weights/best.pt ] && echo done"
        )
        is_done = "done" in stdout2.read().decode()
        _, stdout3, _ = self._ssh.exec_command(
            "tail -5 /root/training_output/exp/train.log 2>/dev/null | grep -i error"
        )
        error_line = stdout3.read().decode().strip()
        current_epoch, current_map = 0, 0.0
        if line:
            parts = line.split(",")
            if len(parts) > 3:
                try:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
                except (ValueError, IndexError):
                    pass
        return {
            "done": is_done,
            "error": bool(error_line),
            "error_msg": error_line,
            "current_epoch": current_epoch,
            "total_epochs": cfg.get("epochs", 100),
            "current_map": current_map,
        }

    def pull_artifacts(self, cfg: dict) -> dict:
        self.state = CloudTrainState.PULLING
        local_dir = f"/tmp/artifacts/{self._instance_id}"
        os.makedirs(local_dir, exist_ok=True)
        artifacts = {}
        files = [
            "/root/training_output/exp/weights/best.pt",
            "/root/training_output/exp/weights/last.pt",
            "/root/training_output/exp/results.csv",
            "/root/training_output/exp/confusion_matrix.png",
            "/root/training_output/exp/PR_curve.png",
            "/root/training_output/exp/F1_curve.png",
            "/root/training_output/exp/results.png",
        ]
        for remote_path in files:
            fname = os.path.basename(remote_path)
            local_path = f"{local_dir}/{fname}"
            try:
                self._sftp.get(remote_path, local_path)
                artifacts[fname] = local_path
            except FileNotFoundError:
                pass
        for fmt in cfg.get("export_formats", []):
            export_local = self._export_model(fmt, local_dir)
            if export_local:
                artifacts[f"model.{fmt}"] = export_local
        return artifacts

    def _export_model(self, fmt: str, local_dir: str) -> Optional[str]:
        weights = "/root/training_output/exp/weights/best.pt"
        export_cmd = (
            f"python -c "
            f"\"from ultralytics import YOLO; "
            f"YOLO('{weights}').export(format='{fmt}')\""
        )
        self._ssh.exec_command(f"cd /root && {export_cmd}")
        time.sleep(60)
        remote = f"/root/training_output/exp/weights/best.{fmt}"
        local = f"{local_dir}/best.{fmt}"
        try:
            self._sftp.get(remote, local)
            return local
        except FileNotFoundError:
            return None

    def shutdown(self):
        self.state = CloudTrainState.SHUTTING_DOWN
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.api_base}/instance/shutdown",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"instance_id": self._instance_id},
                    timeout=30,
                )
                resp.raise_for_status()
                return
            except Exception:
                time.sleep(5 * (attempt + 1))
        self._alert(
            f"[CRITICAL] AutoDL 实例 {self._instance_id} 关机失败",
            "请立即登录 AutoDL 控制台手动关闭实例以防继续扣费",
        )

    def _alert(self, title: str, detail: str):
        if self._alert_manager:
            self._alert_manager.send(title, detail)

    def cancel(self):
        if self._ssh:
            self._ssh.exec_command(
                "screen -S train -X quit; killall -SIGTERM python; true"
            )
