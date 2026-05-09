from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db
from services.settings_manager import SettingsManager, encrypt_value, decrypt_value
from routers.auth import require_auth

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _mask_secret(value: str) -> str:
    """对敏感值做脱敏处理，只保留首尾各 2 个字符"""
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


@router.get("")
def get_settings_api(current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    mgr = SettingsManager(db, current_user["user_id"])
    settings = mgr.get_settings()
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "vlm_provider": settings.vlm_provider,
            "vlm_base_url": settings.vlm_base_url,
            "vlm_api_key": _mask_secret(decrypt_value(settings.vlm_api_key_encrypted)) if settings.vlm_api_key_encrypted else "",
            "vlm_api_key_set": bool(settings.vlm_api_key_encrypted),
            "vlm_api_format": settings.vlm_api_format or "openai",
            "vlm_model": settings.vlm_model or "",
            "vlm_temperature": settings.vlm_temperature or 0.7,
            "vlm_top_p": settings.vlm_top_p or 0.7,
            "vlm_stop": settings.vlm_stop or "",
            "reasoning_enabled": bool(settings.reasoning_enabled) if settings.reasoning_enabled is not None else True,
            "reasoning_provider": settings.reasoning_provider or "deepseek",
            "reasoning_base_url": settings.reasoning_base_url or "https://api.deepseek.com/v1",
            "reasoning_api_key": _mask_secret(decrypt_value(settings.reasoning_api_key_encrypted)) if settings.reasoning_api_key_encrypted else "",
            "reasoning_api_key_set": bool(settings.reasoning_api_key_encrypted),
            "reasoning_model": settings.reasoning_model or "deepseek-reasoner",
            "cloud_provider": settings.cloud_provider,
            "ssh_host": settings.ssh_host or "",
            "ssh_port": settings.ssh_port,
            "ssh_username": settings.ssh_username,
            "ssh_password": _mask_secret(decrypt_value(settings.ssh_password_encrypted)) if settings.ssh_password_encrypted else "",
            "ssh_password_set": bool(settings.ssh_password_encrypted),
            "ssh_private_key_path": settings.ssh_private_key_path or "",
            "remote_work_dir": settings.remote_work_dir or "/root/workspace",
            "autodl_token": _mask_secret(decrypt_value(settings.autodl_token_encrypted)) if settings.autodl_token_encrypted else "",
            "autodl_token_set": bool(settings.autodl_token_encrypted),
            "default_model": settings.default_model,
            "default_augment_strength": settings.default_augment_strength,
            "default_delete_original": settings.default_delete_original,
            "default_gpu_type": settings.default_gpu_type,
            "default_train_mode": settings.default_train_mode,
        },
    }


class SettingsUpdateRequest(BaseModel):
    vlm_provider: Optional[str] = None
    vlm_base_url: Optional[str] = None
    vlm_api_key: Optional[str] = None
    vlm_api_format: Optional[str] = None
    vlm_model: Optional[str] = None
    vlm_temperature: Optional[float] = None
    vlm_top_p: Optional[float] = None
    vlm_stop: Optional[str] = None
    reasoning_enabled: Optional[bool] = None
    reasoning_provider: Optional[str] = None
    reasoning_base_url: Optional[str] = None
    reasoning_api_key: Optional[str] = None
    reasoning_model: Optional[str] = None
    cloud_provider: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    ssh_username: Optional[str] = None
    ssh_password: Optional[str] = None
    ssh_private_key_path: Optional[str] = None
    remote_work_dir: Optional[str] = None
    autodl_token: Optional[str] = None
    default_model: Optional[str] = None
    default_augment_strength: Optional[str] = None
    default_delete_original: Optional[bool] = None
    default_gpu_type: Optional[str] = None
    default_train_mode: Optional[str] = None


@router.put("")
def update_settings(payload: SettingsUpdateRequest, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    mgr = SettingsManager(db, current_user["user_id"])
    data = payload.model_dump(exclude_unset=True)

    # Encrypt sensitive fields; skip if value is masked placeholder or unchanged
    def _is_masked(v: str | None) -> bool:
        return bool(v and "*" in v)

    if "vlm_api_key" in data:
        raw = data.pop("vlm_api_key") or ""
        if raw and not _is_masked(raw):
            data["vlm_api_key_encrypted"] = encrypt_value(raw)
    if "reasoning_api_key" in data:
        raw = data.pop("reasoning_api_key") or ""
        if raw and not _is_masked(raw):
            data["reasoning_api_key_encrypted"] = encrypt_value(raw)
    if "ssh_password" in data:
        raw = data.pop("ssh_password") or ""
        if raw and not _is_masked(raw):
            data["ssh_password_encrypted"] = encrypt_value(raw)
    if "autodl_token" in data:
        raw = data.pop("autodl_token") or ""
        if raw and not _is_masked(raw):
            data["autodl_token_encrypted"] = encrypt_value(raw)

    mgr.update_settings(**data)
    return {"code": 0, "msg": "ok", "data": None}
