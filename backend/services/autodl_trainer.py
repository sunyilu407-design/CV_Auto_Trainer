import time
import os
import httpx
import paramiko
from typing import Optional, Callable
from .cloud_trainer import CloudTrainer, CloudTrainState, build_train_command


class AutoDLTrainingError(RuntimeError):
    """
    AutoDL 训练失败异常。携带手动恢复所需的全部信息（SSH、数据集路径、训练命令等），
    前端可据此展示手动操作教程，防止用户租用的 GPU 浪费。
    """

    def __init__(self, message: str, recovery_info: Optional[dict] = None):
        super().__init__(message)
        self.recovery_info = recovery_info or {}


class AutoDLCloudTrainer(CloudTrainer):
    """
    AutoDL 专用云端训练器（原型实现）。

    ⚠️  当前状态：原型 / 手动模式
    AutoDL 官方不提供公开 REST API。当前实现仅记录训练参数和路径，
    实际训练需要用户手动通过 SSH 接入 AutoDL 实例执行。

    使用方式：
    1. 在 AutoDL 控制台手动开启 GPU 实例
    2. 获取 SSH 信息
    3. 在"云端训练"页面选择"SSH 手动模式"，填入 SSH 信息
    4. 系统会准备好训练包（dataset.zip + cloud_scripts/），按提示上传即可

    如需自动化 AutoDL，请参考：
    https://www.autodl.com/docs/
    """

    _PROTOTYPE_WARNING = True

    def __init__(self, config: dict):
        self.token = config.get("autodl_token", "")
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
        # AutoDL 需要先手动创建实例，无法自动化
        self._instance_id = self._create_instance()
        self._wait_for_running()
        self._ssh = self._get_ssh()
        self._sftp = self._ssh.open_sftp()

    def _exec_wait(self, cmd: str, timeout: int = 300) -> tuple[str, str]:
        """Execute a remote command and wait for it to finish."""
        _, stdout, stderr = self._ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        return out, err

    def _create_instance(self) -> str:
        # ⚠️ AutoDL 官方不提供公开 REST API，无法自动创建实例
        # 请在 AutoDL 控制台手动开启实例后，使用 GenericSSH trainer（SSH 手动模式）
        raise AutoDLTrainingError(
            message=(
                "AutoDL 原型实现不支持自动创建实例。"
                "请在 AutoDL 控制台 (https://www.autodl.com/console/instance) "
                "手动开启 GPU 实例，然后使用「SSH 手动模式」接入。"
            ),
            recovery_info={
                "autodl_guide_url": "https://www.autodl.com/docs/",
                "console_url": "https://www.autodl.com/console/instance",
                "manual_mode_required": True,
                "hint": "选择 GenericSSH trainer（SSH 手动模式），填入 AutoDL 实例的 SSH 信息即可",
            },
        )

    def _wait_for_running(self, timeout: int = 300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = httpx.get(
                f"{self.api_base}/instance/status/{self._instance_id}",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=30,
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
        resp = httpx.get(
            f"{self.api_base}/instance/info/{self._instance_id}",
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def upload_dataset(self, zip_path: str):
        self.state = CloudTrainState.UPLOADING
        self._sftp.put(zip_path, "/root/dataset.zip")
        self._exec_wait(
            "cd /root && unzip -oq dataset.zip -d dataset",
            timeout=600,
        )

    def run_training(self, train_config: dict, progress_callback: Optional[Callable] = None):
        self.state = CloudTrainState.TRAINING
        py_code = build_train_command(
            train_config,
            data_yaml_path="/root/dataset/data.yaml",
            project_dir="/root/training_output",
            device=self.gpu_device,
        )
        train_cmd = f'python -c "{py_code}"'
        self._last_train_command = train_cmd
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
                raise AutoDLTrainingError(
                    f"训练失败: {status.get('error_msg')}",
                    recovery_info=self.get_recovery_info(train_config, error_msg=status.get("error_msg", "")),
                )

        # 训练完成后在云端导出（利用云端 GPU 架构）
        self._run_cloud_export(train_config)

    def get_recovery_info(self, train_config: dict, error_msg: str = "") -> dict:
        """
        返回手动恢复所需的全部信息：SSH、数据集位置、训练命令。
        用户可据此手动 SSH 到实例继续训练，避免已租用的 GPU 浪费。
        """
        info = {
            "instance_id": self._instance_id,
            "error_msg": error_msg,
            "train_command": getattr(self, "_last_train_command", ""),
            "data_yaml_path": "/root/dataset/data.yaml",
            "project_dir": "/root/training_output",
            "weights_path": "/root/training_output/exp/weights/best.pt",
        }
        try:
            ssh_info = self._get_instance_info()
            raw_pwd = ssh_info.get("password", "")
            masked_pwd = (raw_pwd[:2] + "*" * max(0, len(raw_pwd) - 4) + raw_pwd[-2:]) if len(raw_pwd) > 6 else "*" * len(raw_pwd)
            info.update({
                "ssh_host": ssh_info.get("host"),
                "ssh_port": ssh_info.get("port"),
                "ssh_username": "root",
                "ssh_password_masked": masked_pwd,
                "autodl_console_url": f"https://www.autodl.com/console/instance/{self._instance_id}",
            })
        except Exception:
            # 如果获取 SSH 信息也失败，至少给出 instance_id
            info["ssh_retrieval_failed"] = True
        return info

    def _check_training_status(self, cfg: dict) -> dict:
        # Check if the training process is still running
        proc_out, _ = self._exec_wait(
            "screen -ls train 2>/dev/null | grep -c train || echo 0",
            timeout=10,
        )
        process_alive = proc_out.strip() not in ("", "0")

        # Read last data line from results.csv (skip header)
        csv_out, _ = self._exec_wait(
            "tail -1 /root/training_output/exp/results.csv 2>/dev/null",
            timeout=10,
        )
        line = csv_out.strip()

        # Check for errors in log
        err_out, _ = self._exec_wait(
            "tail -5 /root/training_output/exp/train.log 2>/dev/null | grep -i error",
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
            "[ -f /root/training_output/exp/weights/best.pt ] && echo done",
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

    def _run_cloud_export(self, cfg: dict) -> None:
        """在云端导出模型格式"""
        weights = "/root/training_output/exp/weights/best.pt"
        script_path = "/root/_cloud_export.py"
        lines = [
            f"from ultralytics import YOLO",
            f"m = YOLO('{weights}')",
        ]
        for fmt in cfg.get("export_formats", []):
            lines.append(f"m.export(format='{fmt}')")
        with self._sftp.open(script_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        self._exec_wait(
            "cd /root && python _cloud_export.py",
            timeout=600,
        )
        self._exec_wait("rm -f /root/_cloud_export.py", timeout=10)

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
        # 拉取在云端导出的模型文件
        for fmt in cfg.get("export_formats", []):
            cloud_export_path = f"/root/training_output/exp/weights/best.{fmt}"
            local_export_path = f"{local_dir}/model.{fmt}"
            try:
                self._sftp.get(cloud_export_path, local_export_path)
                artifacts[f"model.{fmt}"] = local_export_path
            except FileNotFoundError:
                export_local = self._export_model(fmt, local_dir)
                if export_local:
                    artifacts[f"model.{fmt}"] = export_local
        return artifacts

    def _export_model(self, fmt: str, local_dir: str) -> Optional[str]:
        weights = "/root/training_output/exp/weights/best.pt"
        script_path = "/root/_export_model.py"
        script_body = (
            f"from ultralytics import YOLO\n"
            f"YOLO('{weights}').export(format='{fmt}')"
        )
        with self._sftp.open(script_path, "w") as f:
            f.write(script_body + "\n")
        self._exec_wait(
            f"cd /root && python _export_model.py",
            timeout=600,
        )
        remote = f"/root/training_output/exp/weights/best.{fmt}"
        local = f"{local_dir}/best.{fmt}"
        try:
            self._sftp.get(remote, local)
            return local
        except FileNotFoundError:
            return None
        finally:
            self._exec_wait("rm -f /root/_export_model.py", timeout=10)

    def shutdown(self):
        self.state = CloudTrainState.SHUTTING_DOWN
        for attempt in range(3):
            try:
                resp = httpx.post(
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
