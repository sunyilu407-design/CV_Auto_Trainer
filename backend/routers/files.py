from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
import shutil
from pathlib import Path
from models.database import get_db
from models.db import Task

router = APIRouter(prefix="/api/files", tags=["files"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/upload")
def upload_file(task_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(exist_ok=True, parents=True)

    file_path = task_dir / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return {"code": 0, "msg": "ok", "data": {"path": str(file_path)}}


@router.get("/{task_id}/artifacts")
def list_artifacts(task_id: str, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    artifacts_dir = Path("./artifacts") / task_id
    if not artifacts_dir.exists():
        return {"code": 0, "msg": "ok", "data": []}

    files = []
    for f in artifacts_dir.iterdir():
        if f.is_file():
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
            })
    return {"code": 0, "msg": "ok", "data": files}


@router.get("/{task_id}/artifacts/{filename}")
def download_artifact(task_id: str, filename: str):
    file_path = Path("./artifacts") / task_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, filename=filename)
