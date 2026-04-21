import asyncio
import os
import uvicorn
import torch
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pipeline.gpu_manager import cancel_current_stage, get_device, get_free_memory_gb, CancelError


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
            )

            # 第二段
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

        try:
            artifacts = await asyncio.to_thread(
                trainer.train,
                dataset_dir=payload["dataset_dir"],
                train_config=payload["train_config"],
                progress_callback=training_progress_cb,
            )
            await ws.send_json({
                "type": "training_complete",
                "mode": "local",
                "artifacts": artifacts,
                "metrics": {
                    "bestMap": artifacts.get("best_map", 0.0),
                    "lastMap": artifacts.get("last_map", 0.0),
                },
            })
        except Exception as e:
            await ws.send_json({
                "type": "training_error",
                "message": str(e),
                "recoverable": payload.get("train_config", {}).get("resume_last", False),
            })

    elif cmd == "cancel":
        cancel_current_stage()
        await ws.send_json({"type": "cancel_ack", "cancelled": True})

    elif cmd == "ping":
        await ws.send_json({"type": "pong"})


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
