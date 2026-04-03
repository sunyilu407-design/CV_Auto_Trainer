from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import uuid as uuid_lib
from datetime import datetime

Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid_lib.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    status = Column(String, default="created")

    # 阶段一
    vlm_result = Column(JSON)

    # 阶段二统计
    raw_image_count = Column(Integer, default=0)
    labeled_image_count = Column(Integer, default=0)

    # 阶段二点五
    augment_config = Column(JSON)

    # 阶段三统计
    total_image_count = Column(Integer, default=0)
    train_split_count = Column(Integer, default=0)
    val_split_count = Column(Integer, default=0)
    test_split_count = Column(Integer, default=0)

    # 训练配置
    train_config = Column(JSON)

    # 图片清理策略
    delete_original_images = Column(Boolean, default=False)

    # 阶段四结果
    cloud_instance_id = Column(String)
    best_map50 = Column(Float)
    best_map50_95 = Column(Float)
    artifact_paths = Column(JSON)
    error_message = Column(Text)

    # 文件路径
    image_dir = Column(String)
    label_dir = Column(String)
    dataset_dir = Column(String)


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, default=1)

    vlm_provider = Column(String, default="openai")
    vlm_base_url = Column(String, default="https://api.openai.com/v1")
    vlm_api_key_encrypted = Column(String)

    # 云端训练通用配置
    cloud_provider = Column(String, default="generic")
    ssh_host = Column(String)
    ssh_port = Column(Integer, default=22)
    ssh_username = Column(String, default="root")
    ssh_password_encrypted = Column(String)
    ssh_private_key_path = Column(String)
    remote_work_dir = Column(String, default="/root/workspace")

    # AutoDL 专用
    autodl_token_encrypted = Column(String)

    # 全局
    default_model = Column(String, default="yolo11s.pt")
    default_augment_strength = Column(String, default="medium")
    default_delete_original = Column(Boolean, default=False)
    default_gpu_type = Column(String, default="RTX 4090")
    default_train_mode = Column(String, default="local")
