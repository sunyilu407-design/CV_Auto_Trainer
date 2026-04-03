from enum import Enum
from sqlalchemy.orm import Session
from models.db import Task, UserSettings
from services.settings_manager import get_settings, decrypt_value
from typing import Optional, Callable
import httpx
import time
import os
import paramiko
from pathlib import Path


class TrainMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class TrainDispatcher:
    """
    统一训练分发器。根据 mode 路由到本地或云端。
    """

    def __init__(self, db: Session):
        self.db = db
        self._active_trainer = None

    def dispatch(
        self,
        mode: TrainMode,
        task_id: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        task = self.db.query(Task).filter(Task.id == task_id).first()
        if not task:
            raise ValueError(f"Task {task_id} not found")

        if mode == TrainMode.LOCAL:
            return self._run_local(task, train_config, progress_callback)
        else:
            return self._run_cloud(task, train_config, progress_callback)

    def _run_local(self, task: Task, train_config: dict, progress_callback: Optional[Callable]):
        """本地训练：通过 HTTP 请求 Worker 接口"""
        # Worker 本地训练通过 WebSocket 已在前端启动
        # 这里只更新数据库状态
        task.status = "training_local"
        self.db.commit()
        return {"status": "local_training_started"}

    def _run_cloud(self, task: Task, train_config: dict, progress_callback: Optional[Callable]):
        """云端训练"""
        settings = get_settings(self.db)
        provider = settings.cloud_provider

        if provider == "autodl":
            return self._run_autodl(task, train_config, settings, progress_callback)
        else:
            return self._run_generic_ssh(task, train_config, settings, progress_callback)

    def _run_autodl(self, task: Task, train_config: dict, settings: UserSettings, progress_callback: Optional[Callable]) -> dict:
        from services.autodl_trainer import AutoDLCloudTrainer

        token = decrypt_value(settings.autodl_token_encrypted) if settings.autodl_token_encrypted else ""
        config = {
            "autodl_token": token,
            "gpu_type": train_config.get("gpu_type", "RTX 4090"),
        }
        trainer = AutoDLCloudTrainer(config)
        task.status = "training_cloud"
        task.cloud_instance_id = trainer._instance_id if hasattr(trainer, '_instance_id') else None
        self.db.commit()

        try:
            result = trainer.train(
                dataset_dir=task.dataset_dir or "",
                train_config=train_config,
                progress_callback=progress_callback,
            )
            task.status = "done"
            task.artifact_paths = result
            self.db.commit()
            return result
        except Exception as e:
            task.status = "error"
            task.error_message = str(e)
            self.db.commit()
            raise

    def _run_generic_ssh(self, task: Task, train_config: dict, settings: UserSettings, progress_callback: Optional[Callable]) -> dict:
        from services.generic_ssh_trainer import GenericSSHCloudTrainer

        config = {
            "ssh_host": settings.ssh_host or "",
            "ssh_port": settings.ssh_port or 22,
            "ssh_username": settings.ssh_username or "root",
            "ssh_password": decrypt_value(settings.ssh_password_encrypted) if settings.ssh_password_encrypted else "",
            "ssh_private_key_path": settings.ssh_private_key_path or "",
            "remote_work_dir": settings.remote_work_dir or "/root/workspace",
            "gpu_device": "0",
        }
        trainer = GenericSSHCloudTrainer(config)
        task.status = "training_cloud"
        self.db.commit()

        try:
            result = trainer.train(
                dataset_dir=task.dataset_dir or "",
                train_config=train_config,
                progress_callback=progress_callback,
            )
            task.status = "done"
            task.artifact_paths = result
            self.db.commit()
            return result
        except Exception as e:
            task.status = "error"
            task.error_message = str(e)
            self.db.commit()
            raise

    def cancel_training(self, task_id: str):
        task = self.db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return
        # Cancel logic depends on implementation
        # For cloud training, call trainer.cancel()
        task.status = "cancelled"
        self.db.commit()
