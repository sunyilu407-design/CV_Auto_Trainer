"""
需求确认对话 API 路由

提供:
- POST /api/negotiate/chat — 发送对话消息
- POST /api/negotiate/confirm — 确认需求，进入下一阶段
- GET  /api/negotiate/conversation/{task_id} — 获取/恢复对话历史
"""

import logging
from types import SimpleNamespace as _SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from models.database import get_db
from routers.auth import require_auth
from services.vlm_adapter import VLMAdapter
from services.settings_manager import get_settings, decrypt_value
from services.task_access import get_task_for_user
from services.negotiation_orchestrator import NegotiationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/negotiate", tags=["negotiate"])


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class NegotiateChatRequest(BaseModel):
    task_id: str
    message: str
    conversation_id: Optional[str] = None
    preview_stats: Optional[dict] = None
    include_initial: bool = False  # 是否传入 VLM parse 初始结果


class NegotiateConfirmRequest(BaseModel):
    task_id: str
    conversation_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_vlm_adapter(settings) -> VLMAdapter:
    """从用户设置构建 VLM adapter"""
    return VLMAdapter(
        provider=settings.vlm_provider,
        base_url=settings.vlm_base_url,
        api_key=decrypt_value(settings.vlm_api_key_encrypted) if settings.vlm_api_key_encrypted else "",
        api_format=settings.vlm_api_format,
        model=settings.vlm_model,
        temperature=settings.vlm_temperature,
        top_p=settings.vlm_top_p,
    )


def _build_reasoning_settings_ns(settings_row) -> _SimpleNamespace:
    """把 DB UserSettings 行打包成 reasoning_adapter 工厂期望的对象（API key 已解密）。"""
    enabled = getattr(settings_row, "reasoning_enabled", None)
    if enabled is None:
        enabled = True
    return _SimpleNamespace(
        reasoning_enabled=bool(enabled),
        reasoning_provider=(getattr(settings_row, "reasoning_provider", "") or "").strip() or "deepseek",
        reasoning_base_url=getattr(settings_row, "reasoning_base_url", "") or "",
        reasoning_api_key=decrypt_value(getattr(settings_row, "reasoning_api_key_encrypted", "") or ""),
        reasoning_model=getattr(settings_row, "reasoning_model", "") or "",
    )


def _build_orchestrator(settings, vlm_adapter, db: Session) -> NegotiationOrchestrator:
    """构建编排器实例"""
    reasoning_ns = _build_reasoning_settings_ns(settings)
    return NegotiationOrchestrator(
        vlm_adapter=vlm_adapter,
        settings_obj=reasoning_ns,
        db=db,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat")
def negotiate_chat(
    payload: NegotiateChatRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    发送对话消息，返回 AI 回复 + 可能的配置更新。
    """
    # 验证任务归属
    task = get_task_for_user(db, payload.task_id, current_user)

    # 检查 VLM 配置
    settings = get_settings(db, current_user["user_id"])
    if not settings.vlm_provider or not settings.vlm_api_key_encrypted:
        raise HTTPException(
            status_code=400,
            detail="请先在设置中配置 VLM 模型才能开始需求确认",
        )

    vlm_adapter = _build_vlm_adapter(settings)
    orchestrator = _build_orchestrator(settings, vlm_adapter, db)

    # 获取初始理解（每轮都传，确保 Agent A 始终知道用户的检测目标）
    initial_understanding = None
    if task.vlm_result:
        initial_understanding = (
            task.vlm_result if isinstance(task.vlm_result, dict)
            else {"classes": task.vlm_result}
        )

    # 如果是 __INIT__ 信号，替换为任务的原始需求描述
    effective_message = payload.message
    if payload.message == "__INIT__":
        effective_message = getattr(task, "user_description", "") or "请帮我分析需求"

    logger.info(
        "negotiate_chat: task=%s, is_init=%s, include_initial=%s, "
        "has_vlm_result=%s, effective_message=%s",
        payload.task_id,
        payload.message == "__INIT__",
        payload.include_initial,
        bool(task.vlm_result),
        effective_message[:80] if effective_message else "(empty)",
    )
    if initial_understanding:
        classes = initial_understanding.get("classes", [])
        logger.info(
            "initial_understanding: %d classes -> %s",
            len(classes),
            [c.get("display_name_zh") or c.get("class_name", "") for c in classes[:5]],
        )

    # 执行对话
    result = orchestrator.handle_chat(
        task_id=payload.task_id,
        message=effective_message,
        conversation_id=payload.conversation_id,
        preview_stats=payload.preview_stats,
        initial_understanding=initial_understanding,
        user_description=getattr(task, "user_description", "") or "",
    )

    return {"code": 0, "msg": "ok", "data": result.to_dict()}


@router.delete("/reset/{task_id}")
def negotiate_reset(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    删除该任务的所有未确认对话，允许重新开始。
    """
    from models.db import NegotiationConversation

    task = get_task_for_user(db, task_id, current_user)
    deleted = (
        db.query(NegotiationConversation)
        .filter_by(task_id=task_id, confirmed=False)
        .delete()
    )
    db.commit()
    logger.info("Reset negotiation for task %s: deleted %d conversations", task_id, deleted)
    return {"code": 0, "msg": "ok", "data": {"deleted": deleted}}


@router.post("/confirm")
def negotiate_confirm(
    payload: NegotiateConfirmRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    确认需求，标记对话完成。
    前端收到后写入 taskStore，然后 setStage('algorithm_plan')。
    """
    task = get_task_for_user(db, payload.task_id, current_user)

    settings = get_settings(db, current_user["user_id"])
    vlm_adapter = _build_vlm_adapter(settings)
    orchestrator = _build_orchestrator(settings, vlm_adapter, db)

    try:
        result = orchestrator.confirm(payload.task_id, payload.conversation_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # 同步更新 task 的 vlm_result（用确认后的 classes 覆盖）
    finalized = result.get("finalized_config", {})
    if finalized and finalized.get("classes"):
        task.vlm_result = finalized["classes"]
        db.commit()

    return {"code": 0, "msg": "ok", "data": result}


@router.get("/conversation/{task_id}")
def get_conversation(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    获取任务的对话历史（用于页面恢复）。
    """
    task = get_task_for_user(db, task_id, current_user)

    settings = get_settings(db, current_user["user_id"])
    vlm_adapter = _build_vlm_adapter(settings)
    orchestrator = _build_orchestrator(settings, vlm_adapter, db)

    data = orchestrator.get_conversation(task_id)
    return {"code": 0, "msg": "ok", "data": data}


# ---------------------------------------------------------------------------
# Preview endpoint
# ---------------------------------------------------------------------------

class NegotiatePreviewRequest(BaseModel):
    task_id: str
    conversation_id: Optional[str] = None
    vocab: Optional[dict] = None
    detection_rules: Optional[dict] = None
    max_images: int = 3


@router.post("/preview")
def negotiate_preview(
    payload: NegotiatePreviewRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    快速预览：用当前 vocab 配置对 2-3 张样张进行 YOLO-World 检测。
    返回检测结果（命中/误检统计 + 标注图 base64）。
    """
    import base64
    import random
    from pathlib import Path

    task = get_task_for_user(db, payload.task_id, current_user)

    # 获取当前配置
    settings = get_settings(db, current_user["user_id"])
    vlm_adapter = _build_vlm_adapter(settings)
    orchestrator = _build_orchestrator(settings, vlm_adapter, db)

    # 从对话中获取配置（如未直接传入）
    vocab = payload.vocab
    rules = payload.detection_rules
    if not vocab:
        conv_data = orchestrator.get_conversation(payload.task_id)
        if conv_data and conv_data.get("current_config"):
            vocab = conv_data["current_config"].get("vocab")
            rules = conv_data["current_config"].get("detection_rules")

    if not vocab:
        raise HTTPException(status_code=400, detail="无可用的检测配置，请先完成需求确认")

    # 收集样张路径
    task_dir = Path("uploads") / payload.task_id
    images_dir = task_dir / "images"
    if not images_dir.exists():
        raise HTTPException(status_code=400, detail="未找到上传的图片")

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.webp')]
    if not image_files:
        raise HTTPException(status_code=400, detail="图片目录为空")

    # 随机选 max_images 张
    sample_files = random.sample(image_files, min(payload.max_images, len(image_files)))

    # 构建 detection_classes（从 vocab 展开）
    detection_classes = []
    for class_name, entry in vocab.items():
        primary = entry.get("primary", class_name)
        aliases = entry.get("aliases", [])
        # 限制 aliases 避免 CLIP 退化
        aliases = aliases[:5]
        detection_classes.append({
            "class_name": class_name,
            "prompt": primary,
            "prompt_aliases": aliases,
        })

    # 置信度阈值
    conf_threshold = 0.15
    if rules and rules.get("conf_threshold"):
        conf_threshold = rules["conf_threshold"]

    # 调用 YOLO-World 检测（轻量模式）
    results = []
    try:
        from worker.pipeline.stage2_labeler import run_detection_preview
        for img_path in sample_files:
            detections = run_detection_preview(
                image_path=str(img_path),
                detection_classes=detection_classes,
                conf_threshold=conf_threshold,
            )
            # 读取图片 base64
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            results.append({
                "image_name": img_path.name,
                "image_base64": img_b64,
                "detections": detections,
                "detection_count": len(detections),
            })
    except ImportError:
        # Worker 不在同一进程，返回占位结果
        for img_path in sample_files:
            with open(img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            results.append({
                "image_name": img_path.name,
                "image_base64": img_b64,
                "detections": [],
                "detection_count": 0,
                "note": "预览需要 Worker 进程支持",
            })

    # 更新预览计数
    if payload.conversation_id:
        orchestrator.increment_preview_count(payload.task_id, payload.conversation_id)

    total_detections = sum(r["detection_count"] for r in results)
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "results": results,
            "total_images": len(results),
            "total_detections": total_detections,
            "conf_threshold": conf_threshold,
        },
    }
