import asyncio
import os
import uvicorn
import torch
import gc
import httpx
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pipeline.gpu_manager import cancel_current_stage, get_device, get_free_memory_gb, CancelError


def _resolve_upload_root() -> Path:
    """
    解析后端上传目录的根路径。

    优先级：
    1. 环境变量 CV_AUTO_TRAINER_UPLOAD_ROOT（绝对路径）
    2. 检测 Worker 启动位置，自动推导相对路径

    例如：Worker 从项目根目录启动 -> UPLOAD_ROOT = backend/uploads
          Worker 从 worker/ 目录启动 -> UPLOAD_ROOT = ../backend/uploads
    """
    env_root = os.getenv("CV_AUTO_TRAINER_UPLOAD_ROOT", "").strip()
    if env_root:
        p = Path(env_root)
        return p if p.is_absolute() else (Path.cwd() / p)

    # 自动检测：Worker 的 __file__ 是 worker/main.py
    # project_root = worker/../ = 项目根目录
    worker_file = Path(__file__).resolve()
    project_root = worker_file.parent.parent
    backend_dir = project_root / "backend"
    uploads = backend_dir / "uploads"

    # 检查 uploads 是否在预期位置，以确认项目结构
    if uploads.exists():
        return uploads
    # 回退：假设 Worker 从 worker/ 启动，backend 是同级的兄弟目录
    return Path("../backend/uploads")


# 共享的配置
WORKER_ALLOWED_ORIGINS = _parse_allowed_origins()
WORKER_HOST = os.getenv("CV_AUTO_TRAINER_WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("CV_AUTO_TRAINER_WORKER_PORT", "7860"))

# 后端回调配置
_WORKER_CALLBACK_SECRET = os.getenv("CV_AUTO_TRAINER_WORKER_CALLBACK_SECRET", "worker-secret-change-me")
_BACKEND_BASE_URL = os.getenv("CV_AUTO_TRAINER_BACKEND_URL", "http://localhost:8000")

# 上传文件根目录（绝对路径）
UPLOAD_ROOT = _resolve_upload_root()


def _parse_allowed_origins() -> list[str]:
    raw = os.getenv(
        "CV_AUTO_TRAINER_WORKER_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    )
    return [item.strip() for item in raw.split(",") if item.strip()]


WORKER_ALLOWED_ORIGINS = _parse_allowed_origins()
WORKER_HOST = os.getenv("CV_AUTO_TRAINER_WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.getenv("CV_AUTO_TRAINER_WORKER_PORT", "7860"))

app = FastAPI(title="CV Auto Trainer Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=WORKER_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            try:
                await _handle_command(ws, data)
            except Exception as e:
                await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        pass


async def _handle_command(ws: WebSocket, data: dict):
    cmd = data.get("type")
    payload = data.get("payload", {})
    loop = asyncio.get_running_loop()

    if cmd == "start_detection":
        from pipeline.stage2_labeler import run_detection, run_quality_check
        from utils.yolo_io import save_yolo_labels
        from utils.image_files import list_image_files

        def make_progress(current, total, phase):
            asyncio.run_coroutine_threadsafe(ws.send_json(_build_gpu_info_msg()), loop)
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "progress",
                    "stage": phase,
                    "current": current,
                    "total": total,
                }),
                loop,
            )

        image_total = len(list_image_files(payload["image_dir"]))
        await ws.send_json(_build_gpu_info_msg())
        await ws.send_json({
            "type": "progress",
            "stage": "detection",
            "current": 0,
            "total": image_total,
        })

        try:
            use_existing = payload.get("use_existing_labels", False)

            # 第一段
            raw_boxes = await asyncio.to_thread(
                run_detection,
                image_dir=payload["image_dir"],
                classes=payload["classes"],
                output_raw_dir=payload["output_raw_dir"],
                conf_threshold=payload.get("conf_threshold", 0.25),
                iou_threshold=payload.get("iou_threshold", 0.45),
                batch_size=payload.get("batch_size", 4),
                progress_callback=make_progress,
                use_existing_labels=use_existing,
            )

            if use_existing:
                # 预标注数据：跳过 Moondream VQA 质检，直接使用已有标注
                import json as _json

                await ws.send_json(_build_gpu_info_msg())

                with open(f"{payload['output_raw_dir']}/raw_boxes.json", encoding="utf-8") as _f:
                    passed = _json.load(_f)

                # 发送质检跳过消息
                await ws.send_json({
                    "type": "progress",
                    "stage": "quality_check_skipped",
                    "current": image_total,
                    "total": image_total,
                })
            else:
                # 第二段：Moondream VQA 质检
                passed = await asyncio.to_thread(
                    run_quality_check,
                    raw_boxes_path=f"{payload['output_raw_dir']}/raw_boxes.json",
                    min_confidence=payload.get("qa_threshold", 0.5),
                    progress_callback=make_progress,
                )

            await asyncio.to_thread(
                save_yolo_labels,
                passed,
                output_label_dir=payload["output_label_dir"],
                output_image_dir=payload["output_image_dir"],
            )

            await ws.send_json({
                "type": "stage_complete",
                "stage": "labeling",
                "result": {"labeled_count": sum(len(v) > 0 for v in passed.values())},
            })
        except CancelError:
            await ws.send_json({"type": "cancel_ack", "cancelled": True, "stage": "labeling"})

    elif cmd == "start_augmentation":
        from pipeline.stage25_augmentor import augment_dataset

        def aug_progress(current, total, _phase):
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "progress",
                    "stage": "augmentation",
                    "current": current,
                    "total": total,
                }),
                loop,
            )

        result = await asyncio.to_thread(
            augment_dataset,
            src_image_dir=payload["src_image_dir"],
            src_label_dir=payload["src_label_dir"],
            output_image_dir=payload["output_image_dir"],
            output_label_dir=payload["output_label_dir"],
            target_count=payload["target_count"],
            strength=payload.get("strength", "medium"),
            enabled=payload.get("enabled"),
            progress_callback=aug_progress,
        )

        await ws.send_json({
            "type": "stage_complete",
            "stage": "augmentation",
            "result": result,
        })

    elif cmd == "start_local_training":
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        from pipeline.local_trainer import LocalTrainer

        trainer = LocalTrainer()

        def training_progress_cb(status: dict):
            asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "training_progress",
                    "currentEpoch": status.get("current_epoch", 0),
                    "totalEpochs": status.get("total_epochs", 100),
                    "currentMap": status.get("current_map", 0.0),
                    "done": status.get("done", False),
                }),
                loop,
            )

        task_id = payload.get("task_id", "")
        train_config = payload.get("train_config", {})

        # 优先使用 payload 中的绝对路径，否则退化为相对路径（相对 Worker 工作目录）
        dataset_dir = payload.get("dataset_dir", "dataset")

        try:
            artifacts = await asyncio.to_thread(
                trainer.train,
                dataset_dir=dataset_dir,
                train_config=train_config,
                progress_callback=training_progress_cb,
            )

            await ws.send_json({
                "type": "training_complete",
                "mode": "local",
                "task_id": task_id,
                "artifacts": artifacts,
                "metrics": {
                    "current_epoch": train_config.get("epochs", 100),
                    "total_epochs": train_config.get("epochs", 100),
                    "current_map": artifacts.get("best_map", 0.0),
                    "best_map": artifacts.get("best_map", 0.0),
                    "last_map": artifacts.get("last_map", 0.0),
                },
            })

            # 通知后端写入数据库（即使前端 WebSocket 断连也能保证闭环）
            if task_id:
                _notify_backend_complete(task_id, artifacts, train_config)

        except Exception as e:
            await ws.send_json({
                "type": "training_error",
                "message": str(e),
                "recoverable": train_config.get("resume_last", False),
            })
            if task_id:
                _notify_backend_error(task_id, str(e))

    elif cmd == "cancel":
        cancel_current_stage()
        await ws.send_json({"type": "cancel_ack", "cancelled": True})

    elif cmd == "ping":
        await ws.send_json({"type": "pong"})


def _notify_backend_complete(task_id: str, artifacts: dict, train_config: dict):
    """训练完成后通知后端写入 artifact_paths 和 mAP"""
    try:
        payload = {
            "task_id": task_id,
            "status": "complete",
            "artifacts": artifacts,
            "metrics": {
                "current_epoch": train_config.get("epochs", 100),
                "total_epochs": train_config.get("epochs", 100),
                "current_map": artifacts.get("best_map", 0.0),
            },
        }
        httpx.post(
            f"{_BACKEND_BASE_URL}/api/training/worker-callback",
            json=payload,
            timeout=15,
        )
    except Exception:
        pass  # 后端回调失败不影响前端体验


def _notify_backend_error(task_id: str, error_msg: str):
    """训练出错后通知后端记录错误"""
    try:
        httpx.post(
            f"{_BACKEND_BASE_URL}/api/training/worker-callback",
            json={"task_id": task_id, "status": "error", "error_message": error_msg},
            timeout=15,
        )
    except Exception:
        pass


def _build_gpu_info_msg() -> dict:
    device = get_device()

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        used = torch.cuda.memory_allocated(0) / 1e9
        return {
            "type": "gpu_info",
            "available": True,
            "device": "cuda",
            "name": torch.cuda.get_device_name(0),
            "totalMemoryGB": round(total, 1),
            "usedMemoryGB": round(used, 1),
            "freeMemoryGB": round(total - used, 1),
        }

    if device == "mps":
        free = get_free_memory_gb()
        return {
            "type": "gpu_info",
            "available": True,
            "device": "mps",
            "name": "Apple Silicon (MPS)",
            "totalMemoryGB": round(free, 1),
            "usedMemoryGB": 0.0,
            "freeMemoryGB": round(free, 1),
        }
        
    return {"type": "gpu_info", "available": False, "device": "cpu"}


@app.get("/gpu-info")
def get_gpu_info():
    return _build_gpu_info_msg()


@app.post("/check-health")
def check_health():
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)
