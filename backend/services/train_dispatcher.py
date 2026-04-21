from enum import Enum
from sqlalchemy.orm import Session
from models.db import Task, UserSettings
from services.settings_manager import get_settings, decrypt_value
from services.model_registry import get_model_registry, TrainedModelCache
from typing import Optional, Callable
import httpx
import time
import os
import logging
import paramiko
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainMode(Enum):
    LOCAL = "local"
    CLOUD = "cloud"


class TrainDispatcher:
    """
    统一训练分发器。根据 mode 路由到本地或云端。
    """

    _active_trainers: dict[str, object] = {}

    def __init__(self, db: Session):
        self.db = db

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
        settings = get_settings(self.db, task.owner_user_id)
        provider = settings.cloud_provider

        if provider == "autodl":
            return self._run_autodl(task, train_config, settings, progress_callback)
        else:
            return self._run_generic_ssh(task, train_config, settings, progress_callback)

    def _run_autodl(self, task: Task, train_config: dict, settings: UserSettings, progress_callback: Optional[Callable]) -> dict:
        from services.autodl_trainer import AutoDLCloudTrainer, AutoDLTrainingError

        token = decrypt_value(settings.autodl_token_encrypted) if settings.autodl_token_encrypted else ""
        config = {
            "autodl_token": token,
            "gpu_type": train_config.get("gpu_type", "RTX 4090"),
        }
        trainer = AutoDLCloudTrainer(config)
        TrainDispatcher._active_trainers[task.id] = trainer
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
            self._register_trained_model_cache(task, train_config, result)
            return result
        except AutoDLTrainingError as e:
            # 训练失败但实例可能还活着 —— 保留恢复信息，不自动关机
            task.status = "error"
            task.error_message = str(e)
            task.cloud_instance_id = e.recovery_info.get("instance_id") or task.cloud_instance_id
            # 将恢复信息存入 artifact_paths 的特殊键，供前端读取
            existing = dict(task.artifact_paths or {})
            existing["__autodl_recovery__"] = e.recovery_info
            task.artifact_paths = existing
            self.db.commit()
            logger.warning(
                "AutoDL training failed but instance %s may still be alive; recovery info saved",
                e.recovery_info.get("instance_id"),
            )
            raise
        except Exception as e:
            # 非训练阶段的异常（连接失败、实例创建失败等）—— 尝试尽力而为获取恢复信息
            logger.exception("AutoDL non-training failure for task %s", task.id)
            task.status = "error"
            task.error_message = str(e)
            try:
                recovery = trainer.get_recovery_info(train_config, error_msg=str(e)) if hasattr(trainer, "get_recovery_info") else None
                if recovery and recovery.get("instance_id"):
                    existing = dict(task.artifact_paths or {})
                    existing["__autodl_recovery__"] = recovery
                    task.artifact_paths = existing
            except Exception:
                logger.exception("Failed to capture recovery info")
            self.db.commit()
            raise
        finally:
            TrainDispatcher._active_trainers.pop(task.id, None)

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
        TrainDispatcher._active_trainers[task.id] = trainer
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
            self._register_trained_model_cache(task, train_config, result)
            return result
        except Exception as e:
            task.status = "error"
            task.error_message = str(e)
            self.db.commit()
            raise
        finally:
            TrainDispatcher._active_trainers.pop(task.id, None)

    def _register_trained_model_cache(self, task: Task, train_config: dict, result: dict):
        """训练完成后自动注册模型缓存，供后续任务复用"""
        # 若是多模型 orchestrator 调用，由 orchestrator 自己注册 (带 step_id)，避免 cache_id 冲突
        if train_config.get("__orchestrated__"):
            return
        try:
            algorithm_plan = task.algorithm_plan or {}
            targets = algorithm_plan.get("targets", [])
            classes = [t.get("class_name", "") for t in targets if t.get("class_name")]
            if not classes:
                return

            best_pt = result.get("best.pt") or result.get("best_weight")
            if not best_pt:
                return

            cache = TrainedModelCache(
                cache_id=f"{task.id}_{train_config.get('model', 'unknown')}",
                source_model_id=train_config.get("model", "unknown"),
                task_id=task.id,
                classes=classes,
                class_count=len(classes),
                scenario_type=algorithm_plan.get("scenario_type", "unknown"),
                map50=task.best_map50,
                map50_95=task.best_map50_95,
                weight_path=str(best_pt),
                export_paths={k: v for k, v in result.items() if k != "best.pt" and isinstance(v, str)},
                trained_at=time.time(),
                image_count=task.total_image_count or 0,
                epochs_completed=train_config.get("epochs", 0),
                tags=[algorithm_plan.get("scenario_type", "")],
            )
            registry = get_model_registry()
            registry.register_trained_model(cache)
            logger.info("Registered trained model cache: %s", cache.cache_id)
        except Exception as e:
            logger.warning("Failed to register model cache: %s", e)

    def cancel_training(self, task_id: str):
        task = self.db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return
        trainer = TrainDispatcher._active_trainers.pop(task_id, None)
        if trainer and hasattr(trainer, 'cancel'):
            try:
                trainer.cancel()
            except Exception:
                pass
        task.status = "cancelled"
        self.db.commit()
