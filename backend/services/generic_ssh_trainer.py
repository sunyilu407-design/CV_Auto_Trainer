import time
import os
import io
import paramiko
from pathlib import Path
from typing import Optional, Callable
from .cloud_trainer import CloudTrainer, CloudTrainState, build_train_command


class GenericSSHCloudTrainer(CloudTrainer):
    """
    通用 SSH 云端训练器。
    适用于：阿里云、腾讯云、AWS、Google Cloud、AutoDL（SSH 直连）、
            学员自有 GPU 服务器、实验室服务器等任意提供 SSH 访问的机器。
    """

    def __init__(self, config: dict):
        self.host = config["ssh_host"]
        self.port = config.get("ssh_port", 22)
        self.username = config["ssh_username"]
        self.password = config.get("ssh_password")
        self.private_key_path = config.get("ssh_private_key_path")
        self.remote_work_dir = config.get("remote_work_dir", "/root/workspace")
        self.gpu_device = config.get("gpu_device", "0")
        self.state = CloudTrainState.IDLE
        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._alert_manager = None

    def connect(self):
        self.state = CloudTrainState.CONNECTING
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if self.private_key_path:
            pkey = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            self._ssh.connect(
                self.host, port=self.port,
                username=self.username, pkey=pkey, timeout=30,
            )
        else:
            self._ssh.connect(
                self.host, port=self.port,
                username=self.username, password=self.password, timeout=30,
            )
        self._sftp = self._ssh.open_sftp()

    def _exec_wait(self, cmd: str, timeout: int = 300) -> tuple[str, str]:
        """Execute a remote command and wait for it to finish."""
        _, stdout, stderr = self._ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err

    def upload_dataset(self, zip_path: str):
        self.state = CloudTrainState.UPLOADING
        remote_zip = f"{self.remote_work_dir}/dataset.zip"
        self._sftp.put(zip_path, remote_zip)
        self._exec_wait(
            f"cd {self.remote_work_dir} && unzip -oq dataset.zip -d dataset",
            timeout=600,
        )

    def run_training(self, train_config: dict, progress_callback: Optional[Callable] = None):
        self.state = CloudTrainState.TRAINING
        py_code = build_train_command(
            train_config,
            data_yaml_path=f"{self.remote_work_dir}/dataset/data.yaml",
            project_dir=f"{self.remote_work_dir}/training_output",
            device=self.gpu_device,
        )
        train_cmd = f'python -c "{py_code}"'
        self._last_train_command = train_cmd
        self._ssh.exec_command(
            f"cd {self.remote_work_dir} && screen -dmS train bash -c '{train_cmd}'"
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

        # 训练完成后在云端导出（利用云端 GPU 架构）
        self._run_cloud_export(train_config)

    def _check_training_status(self, cfg: dict) -> dict:
        # Check if the training process is still running
        proc_out, _ = self._exec_wait(
            "screen -ls train 2>/dev/null | grep -c train || echo 0",
            timeout=10,
        )
        process_alive = proc_out.strip() not in ("", "0")

        # Read last data line from results.csv (skip header)
        csv_out, _ = self._exec_wait(
            f"tail -1 {self.remote_work_dir}/training_output/exp/results.csv 2>/dev/null",
            timeout=10,
        )
        line = csv_out.strip()

        # Check for errors in log
        err_out, _ = self._exec_wait(
            f"tail -5 {self.remote_work_dir}/training_output/exp/train.log 2>/dev/null | grep -i error",
            timeout=10,
        )
        error_line = err_out.strip()

        current_epoch, current_map = 0, 0.0
        if line and not line.startswith("epoch"):
            parts = line.split(",")
            if len(parts) > 3:
                try:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
                except (ValueError, IndexError):
                    pass

        # Training is done when process has exited AND best.pt exists
        best_out, _ = self._exec_wait(
            f"[ -f {self.remote_work_dir}/training_output/exp/weights/best.pt ] && echo done",
            timeout=10,
        )
        has_best = "done" in best_out
        is_done = (not process_alive) and has_best

        return {
            "done": is_done,
            "error": bool(error_line) and not process_alive,
            "error_msg": error_line,
            "current_epoch": current_epoch,
            "total_epochs": cfg.get("epochs", 100),
            "current_map": current_map,
        }

    def pull_artifacts(self, cfg: dict) -> dict:
        self.state = CloudTrainState.PULLING
        local_dir = Path("/tmp/cloud_artifacts") / f"{self.host}_{int(time.time())}"
        local_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}
        files_to_pull = [
            f"{self.remote_work_dir}/training_output/exp/weights/best.pt",
            f"{self.remote_work_dir}/training_output/exp/weights/last.pt",
            f"{self.remote_work_dir}/training_output/exp/results.csv",
            f"{self.remote_work_dir}/training_output/exp/confusion_matrix.png",
            f"{self.remote_work_dir}/training_output/exp/PR_curve.png",
            f"{self.remote_work_dir}/training_output/exp/F1_curve.png",
            f"{self.remote_work_dir}/training_output/exp/results.png",
        ]

        for remote_path in files_to_pull:
            fname = Path(remote_path).name
            local_path = local_dir / fname
            try:
                self._sftp.get(remote_path, str(local_path))
                artifacts[fname] = str(local_path)
            except FileNotFoundError:
                pass

        # 拉取在云端导出的模型文件
        for fmt in cfg.get("export_formats", []):
            cloud_export_path = f"{self.remote_work_dir}/training_output/exp/weights/best.{fmt}"
            local_export_path = local_dir / f"model.{fmt}"
            try:
                self._sftp.get(cloud_export_path, str(local_export_path))
                artifacts[f"model.{fmt}"] = str(local_export_path)
            except FileNotFoundError:
                # 云端导出失败时退化为本地导出
                fallback = self._export_model(fmt, local_dir, cfg)
                if fallback:
                    artifacts[f"model.{fmt}"] = fallback

        return artifacts

    def _run_cloud_export(self, cfg: dict) -> None:
        """在云端导出模型格式（利用云端 GPU 架构，导出的模型更高效）"""
        weights = f"{self.remote_work_dir}/training_output/exp/weights/best.pt"
        script_path = f"{self.remote_work_dir}/_cloud_export.py"
        lines = [
            f"from ultralytics import YOLO",
            f"m = YOLO('{weights}')",
        ]
        for fmt in cfg.get("export_formats", []):
            lines.append(f"m.export(format='{fmt}')")
        # 写入脚本文件（避免命令行引号嵌套问题）
        with self._sftp.open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self._exec_wait(
            f"cd {self.remote_work_dir} && python _cloud_export.py",
            timeout=600,
        )
        # 清理临时脚本
        self._exec_wait(f"rm -f {script_path}", timeout=10)

    def _export_model(self, fmt: str, local_dir: Path, cfg: dict) -> Optional[str]:
        weights = f"{self.remote_work_dir}/training_output/exp/weights/best.pt"
        script_path = f"{self.remote_work_dir}/_export_{fmt}.py"
        script_body = (
            f"from ultralytics import YOLO\n"
            f"YOLO('{weights}').export(format='{fmt}')"
        )
        with self._sftp.open(script_path, "w") as f:
            f.write(script_body + "\n")
        self._exec_wait(
            f"cd {self.remote_work_dir} && python _export_{fmt}.py",
            timeout=600,
        )
        remote = f"{self.remote_work_dir}/training_output/exp/weights/best.{fmt}"
        local_path = local_dir / f"best.{fmt}"
        try:
            self._sftp.get(remote, str(local_path))
            return str(local_path)
        except FileNotFoundError:
            return None
        finally:
            self._exec_wait(f"rm -f {script_path}", timeout=10)

    def shutdown(self):
        self.state = CloudTrainState.SHUTTING_DOWN
        # 安全关机：只关闭训练 screen，不影响其他服务
        # 等待训练进程自行结束（最多 120 秒），再强制杀掉
        self._exec_wait(
            "screen -S train -X quit 2>/dev/null; "
            "sleep 2; "
            "pkill -TERM -f 'yolo|train' 2>/dev/null; "
            "sleep 10; "
            "pkill -KILL -f 'yolo|train' 2>/dev/null; true",
            timeout=120,
        )
        if self._ssh:
            self._ssh.close()
            self._ssh = None

    def _alert(self, title: str, detail: str):
        if self._alert_manager:
            self._alert_manager.send(title, detail)

    def cancel(self):
        if self._ssh:
            self._ssh.exec_command(
                "screen -S train -X quit 2>/dev/null; "
                "killall -SIGTERM python 2>/dev/null; true"
            )
