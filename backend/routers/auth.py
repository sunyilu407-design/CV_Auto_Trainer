import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from models.database import get_db
from models.db import User

router = APIRouter(prefix="/api/auth", tags=["auth"])

_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
_PBKDF2_ITERATIONS = 600_000
_APP_SECRET = os.getenv("CV_AUTO_TRAINER_SECRET_KEY") or "cv-auto-trainer-dev-secret-key-v1-change-in-prod"


def _app_secret_bytes() -> bytes:
    return _APP_SECRET.encode("utf-8")


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${_b64url_encode(digest)}"


def verify_password(password: str, password_hash: str) -> bool:
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iteration_str, salt, expected = password_hash.split("$", 3)
            iterations = int(iteration_str)
        except ValueError:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            iterations,
        )
        return hmac.compare_digest(_b64url_encode(digest), expected)

    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(legacy_hash, password_hash)


def _sign_token_payload(payload: str) -> str:
    signature = hmac.new(_app_secret_bytes(), payload.encode("utf-8"), hashlib.sha256).digest()
    return _b64url_encode(signature)


def _decode_token(token: str) -> Optional[dict]:
    try:
        encoded_payload, signature = token.split(".", 1)
    except ValueError:
        return None

    expected_signature = _sign_token_payload(encoded_payload)
    if not hmac.compare_digest(signature, expected_signature):
        return None

    try:
        payload = json.loads(_b64url_decode(encoded_payload).decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return None

    if payload.get("exp", 0) < int(time.time()):
        return None

    return payload


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    code: int
    msg: str
    data: Optional[dict]


def create_token(user: User) -> str:
    payload = {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "token_version": user.token_version or 0,
        "nonce": secrets.token_urlsafe(8),
        "exp": int(time.time()) + _TOKEN_TTL_SECONDS,
    }
    encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = _sign_token_payload(encoded_payload)
    return f"{encoded_payload}.{signature}"


def get_current_user(token: str, db: Optional[Session] = None) -> Optional[dict]:
    payload = _decode_token(token)
    if not payload:
        return None

    user_info = {
        "user_id": payload.get("user_id"),
        "username": payload.get("username"),
        "role": payload.get("role"),
        "token_version": payload.get("token_version", 0),
    }
    if db is None:
        return user_info

    user = db.query(User).filter(User.id == user_info["user_id"]).first()
    if not user or (user.token_version or 0) != user_info["token_version"]:
        return None

    return {
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "token_version": user.token_version or 0,
    }


def require_auth(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录")

    token = authorization[7:]
    user = get_current_user(token, db)
    if not user:
        raise HTTPException(status_code=401, detail="未登录")
    return user


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        return {"code": 401, "msg": "用户名或密码错误", "data": None}

    token = create_token(user)
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "token": token,
            "username": user.username,
            "role": user.role,
        },
    }


@router.post("/logout")
def logout(current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == current_user["user_id"]).first()
    if user:
        user.token_version = (user.token_version or 0) + 1
        db.commit()
    return {"code": 0, "msg": "ok"}


@router.get("/me")
def me(current_user: dict = Depends(require_auth)):
    return {"code": 0, "msg": "ok", "data": current_user}


def seed_admin_user(db: Session):
    username = os.getenv("CV_AUTO_TRAINER_ADMIN_USERNAME", "admin")
    password = os.getenv("CV_AUTO_TRAINER_ADMIN_PASSWORD", "admin123")

    admin = db.query(User).filter(User.username == username).first()
    if not admin:
        admin = User(
            username=username,
            password_hash=hash_password(password),
            role="admin",
            token_version=0,
        )
        db.add(admin)
        db.commit()
