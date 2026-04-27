import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from models.db import UserSettings

logger = logging.getLogger(__name__)

_APP_SECRET = os.getenv("CV_AUTO_TRAINER_SECRET_KEY") or "cv-auto-trainer-dev-secret-key-v1-change-in-prod"
_KEY = base64.urlsafe_b64encode(hashlib.sha256(_APP_SECRET.encode("utf-8")).digest())
_CIPHER = Fernet(_KEY)


def encrypt_value(value: str) -> str:
    if not value:
        return ""
    return _CIPHER.encrypt(value.encode("utf-8")).decode("utf-8")


def decrypt_value(encrypted: str) -> str:
    if not encrypted:
        return ""
    try:
        return _CIPHER.decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("无法解密已存储的凭据（加密密钥可能已变更），请在设置页面重新输入 API Key")
        return ""


def get_settings(db: Session, user_id: int) -> UserSettings:
    settings = db.query(UserSettings).filter(UserSettings.user_id == user_id).first()
    if settings:
        return settings

    orphan_settings = db.query(UserSettings).filter(UserSettings.user_id.is_(None)).order_by(UserSettings.id.asc()).first()
    if orphan_settings:
        orphan_settings.user_id = user_id
        db.commit()
        db.refresh(orphan_settings)
        return orphan_settings

    settings = UserSettings(user_id=user_id)
    db.add(settings)
    db.commit()
    db.refresh(settings)
    return settings


class SettingsManager:
    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id

    def get_settings(self) -> UserSettings:
        return get_settings(self.db, self.user_id)

    def update_settings(self, **kwargs):
        settings = self.get_settings()
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.db.commit()
