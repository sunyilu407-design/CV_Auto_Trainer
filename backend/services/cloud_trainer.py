from abc import ABC, abstractmethod
from typing import Optional, Callable
from enum import Enum
import zipfile
from pathlib import Path


class CloudTrainState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    UPLOADING = "uploading"
    TRAINING = "training"
    PULLING = "pulling"
    SHUTTING_DOWN = "shutting_down"
    DONE = "done"
    ERROR = "error"


class CloudTrainer(ABC):
    """
    云端训练抽象基类。
    支持任意可通过 SSH 连接 GPU 服务器（AutoDL / 阿里云 / 腾讯云 / AWS / 自有服务器）。
    """

    @staticmethod
    def for_provider(provider: str, config: dict) -> "CloudTrainer":
        """工厂方法：根据 provider 类型返回对应实现"""
        if provider == "autodl":
            from .autodl_trainer import AutoDLCloudTrainer
            return AutoDLCloudTrainer(config)
        else:
            from .generic_ssh_trainer import GenericSSHCloudTrainer
            return GenericSSHCloudTrainer(config)

    @abstractmethod
    def connect(self) -> None:
        """建立 SSH 连接"""
        pass

    @abstractmethod
    def upload_dataset(self, zip_path: str) -> None:
        """上传数据集 zip 到远程服务器"""
        pass

    @abstractmethod
    def run_training(
        self,
        train_config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> None:
        """在远程服务器上执行训练命令"""
        pass

    @abstractmethod
    def pull_artifacts(self, train_config: dict) -> dict:
        """从远程服务器拉取训练产物"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """关闭远程服务器（关机/退订实例）"""
        pass

    @abstractmethod
    def cancel(self) -> None:
        """取消训练"""
        pass

    def train(
        self,
        dataset_dir: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """
        完整云端训练流水线。finally 块保证无论如何都执行关机。
        返回：{best_pt_path, last_pt_path, metrics, export_paths}
        """
        try:
            self.connect()
            zip_path = self._pack_dataset(dataset_dir)
            self.upload_dataset(zip_path)
            self.run_training(train_config, progress_callback)
            artifacts = self.pull_artifacts(train_config)
            return artifacts
        except Exception:
            raise
        finally:
            self.shutdown()

    def _pack_dataset(self, dataset_dir: str) -> str:
        """打包数据集为 zip"""
        zip_path = Path(dataset_dir).parent / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in Path(dataset_dir).rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(dataset_dir))
        return str(zip_path)
