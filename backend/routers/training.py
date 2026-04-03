from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
from models.database import get_db
from models.db import Task
from services.train_dispatcher import TrainDispatcher, TrainMode
import threading

router = APIRouter(prefix="/api/training", tags=["training"])

# In-memory training state (in production, use Redis or DB)
_training_states: dict[str, dict] = {}
_dispatcher_lock = threading.Lock()


class TrainStartRequest(BaseModel):
    task_id: str
    model: str
    epochs: int
    imgsz: int
    lr0: float
    patience: int
    conf: float
    iou: float
    export_formats: list[str]
    train_mode: str  # "local" | "cloud"
    gpu_type: str
    resume_last: bool = False


@router.post("/start")
def start_training(payload: TrainStartRequest, db: Session = Depends(get_db)):
    task = db.query(Task).filter(Task.id == payload.task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    mode = TrainMode.LOCAL if payload.train_mode == "local" else TrainMode.CLOUD
    dispatcher = TrainDispatcher(db)

    def run_in_thread():
        with _dispatcher_lock:
            _training_states[payload.task_id] = {"state": "running", "progress": {}}

        def progress_callback(status: dict):
            if payload.task_id in _training_states:
                _training_states[payload.task_id]["progress"] = status

        try:
            result = dispatcher.dispatch(
                mode=mode,
                task_id=payload.task_id,
                train_config=payload.model_dump(),
                progress_callback=progress_callback,
            )
            with _dispatcher_lock:
                _training_states[payload.task_id] = {"state": "done", "result": result}
        except Exception as e:
            with _dispatcher_lock:
                _training_states[payload.task_id] = {"state": "error", "error": str(e)}

    threading.Thread(target=run_in_thread, daemon=True).start()

    return {
        "code": 0,
        "msg": "ok",
        "data": {"instance_id": payload.task_id},
    }


@router.get("/{task_id}/status")
def get_training_status(task_id: str):
    with _dispatcher_lock:
        state = _training_states.get(task_id, {"state": "unknown", "progress": {}})

    progress = state.get("progress", {})
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "state": state.get("state", "unknown"),
            "current_epoch": progress.get("current_epoch", 0),
            "total_epochs": progress.get("total_epochs", 100),
            "current_map": progress.get("current_map", 0.0),
            "done": progress.get("done", False),
            "error": state.get("error"),
        },
    }


@router.post("/{task_id}/cancel")
def cancel_training(task_id: str, db: Session = Depends(get_db)):
    dispatcher = TrainDispatcher(db)
    dispatcher.cancel_training(task_id)
    return {"code": 0, "msg": "ok", "data": None}
