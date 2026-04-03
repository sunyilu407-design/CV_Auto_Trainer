import uvicorn
import torch
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pipeline.gpu_manager import gpu_stage, is_cancelled, cancel_current_stage

app = FastAPI(title="CV Auto Trainer Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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
            await _handle_command(ws, data)
    except WebSocketDisconnect:
        pass


async def _handle_command(ws: WebSocket, data: dict):
    cmd = data.get("type")
    payload = data.get("payload", {})

    if cmd == "start_detection":
        from pipeline.stage2_labeler import run_detection, run_quality_check
        from utils.yolo_io import save_yolo_labels

        def make_progress(current, total, phase):
            ws.send_json(_build_gpu_info_msg())
            ws.send_json({
                "type": "progress",
                "stage": phase,
                "current": current,
                "total": total,
            })

        # 第一段
        raw_boxes = run_detection(
            image_dir=payload["image_dir"],
            classes=payload["classes"],
            output_raw_dir=payload["output_raw_dir"],
            conf_threshold=payload.get("conf_threshold", 0.25),
            iou_threshold=payload.get("iou_threshold", 0.45),
            batch_size=payload.get("batch_size", 4),
            progress_callback=make_progress,
        )

        # 第二段
        passed = run_quality_check(
            raw_boxes_path=f"{payload['output_raw_dir']}/raw_boxes.json",
            min_confidence=payload.get("qa_threshold", 0.5),
            progress_callback=make_progress,
        )

        save_yolo_labels(
            passed,
            output_label_dir=payload["output_label_dir"],
            output_image_dir=payload["output_image_dir"],
        )

        ws.send_json({
            "type": "stage_complete",
            "stage": "labeling",
            "result": {"labeled_count": sum(len(v) > 0 for v in passed.values())},
        })

    elif cmd == "start_augmentation":
        from pipeline.stage25_augmentor import augment_dataset

        def aug_progress(current, total, _phase):
            ws.send_json({
                "type": "progress",
                "stage": "augmentation",
                "current": current,
                "total": total,
            })

        result = augment_dataset(
            src_image_dir=payload["src_image_dir"],
            src_label_dir=payload["src_label_dir"],
            output_image_dir=payload["output_image_dir"],
            output_label_dir=payload["output_label_dir"],
            target_count=payload["target_count"],
            strength=payload.get("strength", "medium"),
            enabled=payload.get("enabled"),
            progress_callback=aug_progress,
        )

        ws.send_json({
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
            ws.send_json({
                "type": "training_progress",
                "currentEpoch": status.get("current_epoch", 0),
                "totalEpochs": status.get("total_epochs", 100),
                "currentMap": status.get("current_map", 0.0),
                "done": status.get("done", False),
            })

        try:
            artifacts = trainer.train(
                dataset_dir=payload["dataset_dir"],
                train_config=payload["train_config"],
                progress_callback=training_progress_cb,
            )
            ws.send_json({
                "type": "training_complete",
                "mode": "local",
                "artifacts": artifacts,
                "metrics": {
                    "bestMap": artifacts.get("best_map", 0.0),
                    "lastMap": artifacts.get("last_map", 0.0),
                },
            })
        except Exception as e:
            ws.send_json({
                "type": "training_error",
                "message": str(e),
                "recoverable": payload.get("train_config", {}).get("resume_last", False),
            })

    elif cmd == "cancel":
        cancel_current_stage()
        ws.send_json({"type": "cancel_ack", "cancelled": True})

    elif cmd == "ping":
        ws.send_json({"type": "pong"})


def _build_gpu_info_msg() -> dict:
    if not torch.cuda.is_available():
        return {"type": "gpu_info", "available": False}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    used = torch.cuda.memory_allocated(0) / 1e9
    return {
        "type": "gpu_info",
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "totalMemoryGB": round(total, 1),
        "usedMemoryGB": round(used, 1),
        "freeMemoryGB": round(total - used, 1),
    }


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
    uvicorn.run(app, host="127.0.0.1", port=7860)
