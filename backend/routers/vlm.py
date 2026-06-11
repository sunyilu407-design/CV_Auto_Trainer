from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import json
from models.database import get_db
from routers.auth import require_auth
from services.vlm_adapter import VLMAdapter
from services.settings_manager import get_settings, decrypt_value
from services.task_access import get_task_for_user

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


class VLMParseRequest(BaseModel):
    images_base64: list[str]
    user_text: str
    sample_boxes: Optional[list] = []
    video_info: Optional[dict] = None


class VLMClass(BaseModel):
    class_name: str
    prompt: str
    negative_prompt: str = ""
    color_hint: Optional[str] = None
    display_name_zh: Optional[str] = None
    display_prompt_zh: Optional[str] = None
    display_negative_prompt_zh: Optional[str] = None
    display_color_hint_zh: Optional[str] = None


class VLMParseResponse(BaseModel):
    status: str
    message: str = ""
    retryable: bool = False
    classes: list[VLMClass] = []
    raw_vlm_response: str = ""
    confidence: Optional[float] = None


def _parse_stop(stop_str: Optional[str]) -> Optional[list[str]]:
    if not stop_str:
        return None
    try:
        return json.loads(stop_str)
    except json.JSONDecodeError:
        return None


@router.post("/parse")
def parse_intent(
    payload: VLMParseRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    settings = get_settings(db, current_user["user_id"])
    adapter = VLMAdapter(
        provider=settings.vlm_provider,
        base_url=settings.vlm_base_url,
        api_key=decrypt_value(settings.vlm_api_key_encrypted) if settings.vlm_api_key_encrypted else "",
        api_format=settings.vlm_api_format,
        model=settings.vlm_model,
        temperature=settings.vlm_temperature,
        top_p=settings.vlm_top_p,
        stop=_parse_stop(settings.vlm_stop),
    )

    result = adapter.parse_intent(
        images_base64=payload.images_base64,
        user_text=payload.user_text,
        sample_boxes=payload.sample_boxes or [],
        video_info=payload.video_info or None,
    )
    return {"code": 0, "msg": "ok", "data": result}


@router.put("/result/{task_id}")
def update_vlm_result(
    task_id: str,
    payload: dict,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    task.vlm_result = payload.get("data", {}).get("classes") if isinstance(payload.get("data"), dict) else payload.get("classes")

    # VLM 结果更新 → 自动清理旧的未确认对话，确保协商阶段重新开始
    from models.db import NegotiationConversation
    db.query(NegotiationConversation).filter_by(
        task_id=task_id, confirmed=False
    ).delete()

    db.commit()
    return {"code": 0, "msg": "ok", "data": None}


class VLMTestRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str
    api_format: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[str] = None  # JSON 数组字符串


@router.post("/test")
def test_connection(payload: VLMTestRequest):
    stop_list = None
    if payload.stop:
        import json
        try:
            stop_list = json.loads(payload.stop)
        except json.JSONDecodeError:
            pass
    adapter = VLMAdapter(
        provider=payload.provider,
        base_url=payload.base_url,
        api_key=payload.api_key,
        api_format=payload.api_format,
        model=payload.model,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stop=stop_list,
    )
    result = adapter.test_connection()
    return {"code": 0, "msg": "ok", "data": result}
