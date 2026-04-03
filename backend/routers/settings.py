from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db
from models.db import UserSettings
from services.settings_manager import SettingsManager, encrypt_value, decrypt_value

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("")
def get_settings_api(db: Session = Depends(get_db)):
    mgr = SettingsManager(db)
    settings = mgr.get_settings()
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "vlm_provider": settings.vlm_provider,
            "vlm_base_url": settings.vlm_base_url,
            "vlm_api_key": decrypt_value(settings.vlm_api_key_encrypted) if settings.vlm_api_key_encrypted else "",
            "cloud_provider": settings.cloud_provider,
            "ssh_host": settings.ssh_host or "",
            "ssh_port": settings.ssh_port,
            "ssh_username": settings.ssh_username,
            "ssh_password": decrypt_value(settings.ssh_password_encrypted) if settings.ssh_password_encrypted else "",
            "ssh_private_key_path": settings.ssh_private_key_path or "",
            "remote_work_dir": settings.remote_work_dir or "/root/workspace",
            "autodl_token": decrypt_value(settings.autodl_token_encrypted) if settings.autodl_token_encrypted else "",
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
def update_settings(payload: SettingsUpdateRequest, db: Session = Depends(get_db)):
    mgr = SettingsManager(db)
    data = payload.model_dump(exclude_unset=True)

    # Encrypt sensitive fields
    if "vlm_api_key" in data and data["vlm_api_key"]:
        data["vlm_api_key_encrypted"] = encrypt_value(data.pop("vlm_api_key"))
    if "ssh_password" in data and data["ssh_password"]:
        data["ssh_password_encrypted"] = encrypt_value(data.pop("ssh_password"))
    if "autodl_token" in data and data["autodl_token"]:
        data["autodl_token_encrypted"] = encrypt_value(data.pop("autodl_token"))

    mgr.update_settings(**data)
    return {"code": 0, "msg": "ok", "data": None}
