from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db
from models.db import Task
from services.vlm_adapter import VLMAdapter
from services.settings_manager import get_settings

router = APIRouter(prefix="/api/vlm", tags=["vlm"])


class VLMParseRequest(BaseModel):
    images_base64: list[str]
    user_text: str
    sample_boxes: Optional[list] = []


class VLMClass(BaseModel):
    class_name: str
    prompt: str
    negative_prompt: str = ""
    color_hint: Optional[str] = None


class VLMParseResponse(BaseModel):
    classes: list[VLMClass]
    raw_vlm_response: str = ""
    confidence: float


@router.post("/parse")
def parse_intent(payload: VLMParseRequest, db: Session = Depends(get_db)):
    settings = get_settings(db)
    adapter = VLMAdapter(
        provider=settings.vlm_provider,
        base_url=settings.vlm_base_url,
        api_key=settings.vlm_api_key_encrypted or "",
    )

    try:
        result = adapter.parse_intent(
            images_base64=payload.images_base64,
            user_text=payload.user_text,
            sample_boxes=payload.sample_boxes or [],
        )
        return {"code": 0, "msg": "ok", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/result/{task_id}")
def update_vlm_result(task_id: str, payload: dict, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.vlm_result = payload.get("classes")
    db.commit()
    return {"code": 0, "msg": "ok", "data": None}


class VLMTestRequest(BaseModel):
    provider: str
    base_url: str
    api_key: str
    api_format: Optional[str] = None
    model: Optional[str] = None


@router.post("/test")
def test_connection(payload: VLMTestRequest):
    adapter = VLMAdapter(
        provider=payload.provider,
        base_url=payload.base_url,
        api_key=payload.api_key,
        api_format=payload.api_format,
        model=payload.model,
    )
    result = adapter.test_connection()
    return {"code": 0, "msg": "ok", "data": result}
