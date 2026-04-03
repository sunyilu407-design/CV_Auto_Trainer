from sqlalchemy.orm import Session
from models.db import UserSettings
from cryptography.fernet import Fernet
import base64
import hashlib

# Generate a key from a fixed seed (in production, store this securely)
_KEY_SEED = b"cv-auto-trainer-secret-key-change-in-production"
_KEY = base64.urlsafe_b64encode(hashlib.sha256(_KEY_SEED).digest())
_cipher = Fernet(_KEY)


def encrypt_value(value: str) -> str:
    if not value:
        return ""
    return _cipher.encrypt(value.encode()).decode()


def decrypt_value(encrypted: str) -> str:
    if not encrypted:
        return ""
    return _cipher.decrypt(encrypted.encode()).decode()


def get_settings(db: Session) -> UserSettings:
    settings = db.query(UserSettings).filter(UserSettings.id == 1).first()
    if not settings:
        settings = UserSettings(id=1)
        db.add(settings)
        db.commit()
        db.refresh(settings)
    return settings


class SettingsManager:
    def __init__(self, db: Session):
        self.db = db

    def get_settings(self) -> UserSettings:
        return get_settings(self.db)

    def update_settings(self, **kwargs):
        settings = self.get_settings()
        for key, value in kwargs.items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        self.db.commit()
