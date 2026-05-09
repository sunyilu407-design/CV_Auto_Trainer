import os

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from models.db import Base

DATABASE_URL = os.getenv("CV_AUTO_TRAINER_DB_URL", "sqlite:///cv_auto_trainer.db")
ENGINE_CONNECT_ARGS = (
    {"check_same_thread": False, "timeout": 30}
    if DATABASE_URL.startswith("sqlite") else {}
)

engine = create_engine(
    DATABASE_URL,
    connect_args=ENGINE_CONNECT_ARGS,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)

# SQLite WAL 模式：允许读写并发，大幅减少 "database is locked" 错误
if DATABASE_URL.startswith("sqlite"):
    with engine.connect() as _conn:
        _conn.execute(text("PRAGMA journal_mode=WAL"))
        _conn.execute(text("PRAGMA busy_timeout=30000"))
        _conn.commit()


def _get_table_columns(table_name: str) -> set[str]:
    return {column["name"] for column in inspect(engine).get_columns(table_name)}


def _ensure_table_columns(table_name: str, required_columns: dict[str, str]) -> None:
    inspector = inspect(engine)
    if table_name not in inspector.get_table_names():
        return

    existing_columns = _get_table_columns(table_name)
    for column_name, statement in required_columns.items():
        if column_name in existing_columns:
            continue
        try:
            with engine.begin() as connection:
                connection.execute(text(statement))
        except SQLAlchemyError:
            existing_columns = _get_table_columns(table_name)
            if column_name not in existing_columns:
                raise
        else:
            existing_columns.add(column_name)


def _ensure_task_columns():
    required_columns = {
        "algorithm_plan": "ALTER TABLE tasks ADD COLUMN algorithm_plan JSON",
        "algorithm_plan_status": "ALTER TABLE tasks ADD COLUMN algorithm_plan_status VARCHAR",
        "pipeline_config": "ALTER TABLE tasks ADD COLUMN pipeline_config JSON",
        "negotiation_summary": "ALTER TABLE tasks ADD COLUMN negotiation_summary JSON",
        "offline_evaluation": "ALTER TABLE tasks ADD COLUMN offline_evaluation JSON",
        "training_plan": "ALTER TABLE tasks ADD COLUMN training_plan JSON",
        "delivery_package": "ALTER TABLE tasks ADD COLUMN delivery_package JSON",
        "owner_user_id": "ALTER TABLE tasks ADD COLUMN owner_user_id INTEGER",
        "training_state": "ALTER TABLE tasks ADD COLUMN training_state VARCHAR",
        "training_progress": "ALTER TABLE tasks ADD COLUMN training_progress JSON",
        "training_started_at": "ALTER TABLE tasks ADD COLUMN training_started_at DATETIME",
        "training_finished_at": "ALTER TABLE tasks ADD COLUMN training_finished_at DATETIME",
        "split_stats": "ALTER TABLE tasks ADD COLUMN split_stats JSON",
        "quality_report": "ALTER TABLE tasks ADD COLUMN quality_report JSON",
        "artifact_paths": "ALTER TABLE tasks ADD COLUMN artifact_paths JSON",
        "error_message": "ALTER TABLE tasks ADD COLUMN error_message TEXT",
    }
    _ensure_table_columns("tasks", required_columns)


def _ensure_user_columns():
    _ensure_table_columns(
        "users",
        {"token_version": "ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 0"},
    )
    _ensure_table_columns(
        "user_settings",
        {
            "user_id": "ALTER TABLE user_settings ADD COLUMN user_id INTEGER",
            # v9.0 P1-A 推理模型配置
            "reasoning_enabled": "ALTER TABLE user_settings ADD COLUMN reasoning_enabled BOOLEAN DEFAULT 1",
            "reasoning_provider": "ALTER TABLE user_settings ADD COLUMN reasoning_provider VARCHAR DEFAULT 'deepseek'",
            "reasoning_base_url": "ALTER TABLE user_settings ADD COLUMN reasoning_base_url VARCHAR DEFAULT 'https://api.deepseek.com/v1'",
            "reasoning_api_key_encrypted": "ALTER TABLE user_settings ADD COLUMN reasoning_api_key_encrypted VARCHAR",
            "reasoning_model": "ALTER TABLE user_settings ADD COLUMN reasoning_model VARCHAR DEFAULT 'deepseek-reasoner'",
        },
    )


_ensure_task_columns()
_ensure_user_columns()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
