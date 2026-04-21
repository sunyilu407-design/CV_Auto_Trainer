from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import shutil
from pathlib import Path
from typing import Optional
from models.database import get_db
from routers.auth import require_auth
from services.task_access import get_task_for_user

router = APIRouter(prefix="/api/files", tags=["files"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _safe_relative_path(raw_path: Optional[str]) -> Path:
    if not raw_path:
        return Path()

    candidate = Path(raw_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


def _safe_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return safe_name


def _ensure_within(base_dir: Path, target: Path) -> Path:
    resolved_base = base_dir.resolve()
    resolved_target = target.resolve()
    if not resolved_target.is_relative_to(resolved_base):
        raise HTTPException(status_code=400, detail="Invalid path")
    return resolved_target


@router.post("/upload")
def upload_file(
    task_id: str,
    subdir: Optional[str] = None,
    file: UploadFile = File(...),
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)

    task_dir = UPLOAD_DIR / task_id
    task_dir = task_dir / _safe_relative_path(subdir)
    task_dir.mkdir(exist_ok=True, parents=True)
    task_dir = _ensure_within(UPLOAD_DIR / task_id, task_dir)

    file_path = _ensure_within(task_dir, task_dir / _safe_filename(file.filename))
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if subdir == "images":
        task.image_dir = str(task_dir.resolve())
        db.commit()

    return {"code": 0, "msg": "ok", "data": {"path": str(file_path)}}


@router.post("/upload-video")
async def upload_video(
    task_id: str,
    video: UploadFile = File(...),
    frame_mode: str = "interval",
    interval_seconds: float = 1.0,
    max_frames: int = 200,
    purpose: str = "training",
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    上传视频并自动拆帧。
    purpose: "training" (拆帧用于训练打标) / "vlm_analysis" (抽帧给 VLM 分析) / "validation" (离线验证)
    """
    task = get_task_for_user(db, task_id, current_user)

    task_dir = UPLOAD_DIR / task_id / "videos"
    task_dir.mkdir(exist_ok=True, parents=True)

    video_filename = _safe_filename(video.filename or "upload.mp4")
    video_path = task_dir / video_filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    from services.video_processor import extract_frames, extract_frames_for_vlm, get_video_info

    video_info = get_video_info(str(video_path))

    if purpose == "vlm_analysis":
        frames_b64 = extract_frames_for_vlm(str(video_path), max_frames=8)
        return {
            "code": 0, "msg": "ok",
            "data": {
                "video_path": str(video_path),
                "video_info": video_info,
                "frames_base64": frames_b64,
                "frame_count": len(frames_b64),
            },
        }

    frames_dir = UPLOAD_DIR / task_id / "video_frames"
    result = extract_frames(
        video_path=str(video_path),
        output_dir=str(frames_dir),
        mode=frame_mode,
        interval_seconds=interval_seconds,
        max_frames=max_frames,
    )

    if purpose == "training":
        task.image_dir = str(frames_dir.resolve())
        db.commit()

    return {
        "code": 0, "msg": "ok",
        "data": {
            "video_path": str(video_path),
            "video_info": video_info,
            "frame_count": result["frame_count"],
            "frames_dir": str(frames_dir),
        },
    }


@router.get("/{task_id}/artifacts")
def list_artifacts(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)

    files = []
    seen = set()

    for name, raw_path in (task.artifact_paths or {}).items():
        if not raw_path:
            continue
        path = Path(raw_path)
        if not path.exists():
            continue
        normalized_name = name or path.name
        if normalized_name in seen:
            continue
        seen.add(normalized_name)
        files.append({
            "name": normalized_name,
            "path": str(path),
            "size": path.stat().st_size,
        })

    artifacts_dir = Path("./artifacts") / task_id
    if artifacts_dir.exists():
        for f in artifacts_dir.iterdir():
            if f.is_file() and f.name not in seen:
                seen.add(f.name)
                files.append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                })
    return {"code": 0, "msg": "ok", "data": files}


@router.get("/{task_id}/artifacts/{filename}")
def download_artifact(task_id: str, filename: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)

    if task.artifact_paths and filename in task.artifact_paths:
        artifact_path = Path(task.artifact_paths[filename])
        if artifact_path.exists():
            return FileResponse(path=artifact_path, filename=filename)

    artifacts_dir = Path("./artifacts") / task_id
    file_path = _ensure_within(artifacts_dir, artifacts_dir / _safe_filename(filename))
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)
