"""推理模型（v9.0 P1-A 决策层）路由：测试连接 + 探活。"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from services.reasoning_adapter import OpenAICompatibleReasoner, REASONING_PROVIDER_CONFIG

router = APIRouter(prefix="/api/reasoning", tags=["reasoning"])


class ReasoningTestRequest(BaseModel):
    provider: str
    base_url: str = ""
    api_key: str
    model: str = ""


@router.post("/test")
def test_connection(payload: ReasoningTestRequest):
    provider = payload.provider if payload.provider in REASONING_PROVIDER_CONFIG else "custom"
    adapter = OpenAICompatibleReasoner(
        provider=provider,
        api_key=payload.api_key,
        base_url=payload.base_url,
        model=payload.model,
    )
    result = adapter.test_connection()
    return {"code": 0, "msg": "ok", "data": result}


@router.get("/providers")
def list_providers():
    """返回支持的推理模型 provider 默认配置（前端 settings 页面用）。"""
    return {
        "code": 0,
        "msg": "ok",
        "data": [
            {
                "id": pid,
                "base_url": cfg["base_url"],
                "model": cfg["model"],
                "supports_json_format": cfg["supports_json_format"],
            }
            for pid, cfg in REASONING_PROVIDER_CONFIG.items()
        ],
    }
