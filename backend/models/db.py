from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import DeclarativeBase
import uuid as uuid_lib
from datetime import datetime, timezone


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid_lib.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    status = Column(String, default="created")
    owner_user_id = Column(Integer, nullable=True)

    # 阶段一
    vlm_result = Column(JSON)
    algorithm_plan = Column(JSON)
    algorithm_plan_status = Column(String, default="draft")
    pipeline_config = Column(JSON)
    negotiation_summary = Column(JSON)
    offline_evaluation = Column(JSON)
    training_plan = Column(JSON)
    delivery_package = Column(JSON)

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
    training_state = Column(String, default="idle")
    training_progress = Column(JSON)
    training_started_at = Column(DateTime)
    training_finished_at = Column(DateTime)
    error_message = Column(Text)

    # 文件路径
    image_dir = Column(String)
    label_dir = Column(String)
    dataset_dir = Column(String)
    # 增量训练：追加图片存放目录
    incremental_image_dir = Column(String)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user")  # admin | user
    token_version = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, default=1)
    user_id = Column(Integer, nullable=True)

    vlm_provider = Column(String, default="openai")
    vlm_base_url = Column(String, default="https://api.openai.com/v1")
    vlm_api_key_encrypted = Column(String)
    vlm_api_format = Column(String, default="openai")  # openai | anthropic | gemini
    vlm_model = Column(String)  # 自定义模型名（custom/claude 时使用）
    vlm_temperature = Column(Float, default=0.7)
    vlm_top_p = Column(Float, default=0.7)
    vlm_stop = Column(String)  # JSON 数组字符串，如 "[\"###\"]"

    # 推理模型（v9.0 P1-A 决策层）配置
    reasoning_enabled = Column(Boolean, default=True)
    reasoning_provider = Column(String, default="deepseek")  # deepseek | openai | kimi | qwen | zhipu | custom
    reasoning_base_url = Column(String, default="https://api.deepseek.com/v1")
    reasoning_api_key_encrypted = Column(String)
    reasoning_model = Column(String, default="deepseek-reasoner")

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

    # Eagle 引擎配置
    detection_engine = Column(String, default="auto")  # auto | yolo_world | locate_anything
    vqa_engine = Column(String, default="auto")  # auto | moondream | eagle_vqa
    locate_anything_enabled = Column(Boolean, default=False)
    eagle_vqa_enabled = Column(Boolean, default=False)


class NegotiationConversation(Base):
    """多智能体需求确认对话记录"""
    __tablename__ = "negotiation_conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid_lib.uuid4()))
    task_id = Column(String, ForeignKey("tasks.id"), nullable=False, index=True)
    messages = Column(JSON, default=list)
    current_config = Column(JSON)
    algorithm_hints = Column(JSON)
    confirmed = Column(Boolean, default=False)
    preview_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
