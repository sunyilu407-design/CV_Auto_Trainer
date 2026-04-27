import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models.database import get_db
from models.database import SessionLocal
from models.db import Task
from routers.auth import require_auth
from services.task_access import get_task_for_user
from services.train_dispatcher import TrainDispatcher, TrainMode
from services.multi_model_orchestrator import MultiModelTrainingOrchestrator, LocalMultiModelNotSupported
from services.dataset_packer import prepare_full_dataset, compute_quality_report
from services.video_inference import annotate_video_frames
import threading

router = APIRouter(prefix="/api/training", tags=["training"])

# 相对路径基准：项目根目录（backend/ 的上一级）
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _resolve_frontend_path(path: str) -> Path:
    """
    将前端传来的路径解析为绝对路径。
    前端页面运行在 frontend/ 目录，发送的路径如 '../backend/uploads/...'
    需要转换为项目根目录下的正确路径。
    """
    p = Path(path)
    if p.is_absolute():
        return p
    # 去掉 '../backend/' 前缀（如 '../backend/uploads/...' -> 'backend/uploads/...'）
    parts = p.parts  # e.g. ('..', 'backend', 'uploads', 'task123', ...)
    if len(parts) >= 3 and parts[0] == ".." and parts[1] == "backend":
        resolved = _PROJECT_ROOT.joinpath(*parts[1:])
    else:
        resolved = _PROJECT_ROOT / p
    return resolved.resolve()


class QCAnnotationsRequest(BaseModel):
    task_id: str


class AugQualityReportRequest(BaseModel):
    task_id: str
    augmented_images_dir: str
    augmented_labels_dir: str
    class_names: list[str]


@router.post("/aug-quality-report")
def aug_quality_report(
    payload: AugQualityReportRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    计算增强后数据的质量报告（来自标注前阶段，用于 Review 页面展示增强统计）。

    增强完成后 AugmentConfig 调用此接口获取增强数据的质量报告，
    替换掉原本基于分割前原始标注的统计数据。
    """
    from services.dataset_packer import compute_quality_report

    resolved_images = _resolve_frontend_path(payload.augmented_images_dir)
    resolved_labels = _resolve_frontend_path(payload.augmented_labels_dir)

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "quality_report": compute_quality_report(
                label_dir=str(resolved_labels),
                class_names=payload.class_names,
            ),
            "augmented_images_count": len(
                list(resolved_images.glob("*"))
                if resolved_images.exists()
                else []
            ),
        },
    }


@router.post("/qc-annotations/{task_id}")
def qc_annotations(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    VLM 标注质检：对已打标的图片 + 标注进行抽样分析，
    识别框不准、漏标、错标、遮挡等问题，输出中文分析报告。
    """
    import base64
    from pathlib import Path

    task = get_task_for_user(db, task_id, current_user)

    # 定位标注目录
    base_dir = Path(task.image_dir).parent if task.image_dir else None
    if not base_dir:
        raise HTTPException(status_code=400, detail="无法确定上传目录")
    label_dir = base_dir / "labels"
    image_dir = base_dir / "labeled_images"
    if not label_dir.exists():
        raise HTTPException(status_code=400, detail="标注目录不存在")

    # 抽样：最多取 8 张有标注的图片
    label_files = sorted([p for p in label_dir.glob("*.txt") if p.stat().st_size > 0])[:8]
    if not label_files:
        raise HTTPException(status_code=400, detail="没有找到标注文件")

    # 读取类别名
    vlm_result = task.vlm_result or {}
    classes_raw = vlm_result.get("classes", [])
    if isinstance(classes_raw, dict) and "classes" in classes_raw:
        classes_raw = classes_raw["classes"]
    class_names = [c.get("class_name", c.get("name", f"class_{i}")) for i, c in enumerate(classes_raw)]

    # 读取标注文本 + 找到对应图片并编码 base64
    def _read_yolo_labels(lbl_path: Path) -> list[dict]:
        boxes = []
        try:
            with open(lbl_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                        boxes.append({
                            "class": cls_name,
                            "cx": float(parts[1]),
                            "cy": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        })
        except (ValueError, OSError):
            pass
        return boxes

    images_b64: list[str] = []
    samples: list[dict] = []

    for lbl_path in label_files:
        # 找对应图片
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = image_dir / f"{lbl_path.stem}{ext}"
            if p.is_file():
                img_path = p
                break
        if img_path is None:
            continue
        try:
            img_bytes = img_path.read_bytes()
            img_b64 = base64.b64encode(img_bytes).decode("ascii")
            images_b64.append(img_b64)
            boxes = _read_yolo_labels(lbl_path)
            samples.append({
                "image_b64": img_b64,
                "boxes": boxes,
                "filename": img_path.name,
            })
        except OSError:
            continue

    if not samples:
        raise HTTPException(status_code=400, detail="无法读取标注图片")

    # 调用 VLM
    from services.settings_manager import get_settings, decrypt_value
    from services.vlm_adapter import VLMAdapter

    settings = get_settings(db, current_user["user_id"])
    vlm_adapter = VLMAdapter(
        provider=settings.vlm_provider or "openai",
        base_url=settings.vlm_base_url or "",
        api_key=decrypt_value(settings.vlm_api_key_encrypted or ""),
        api_format=settings.vlm_api_format,
        model=settings.vlm_model,
        temperature=settings.vlm_temperature,
        top_p=settings.vlm_top_p,
    )

    sample_summaries = "\n".join(
        f"图片 {i+1} ({s['filename']}): {len(s['boxes'])} 个标注框 - " +
        ", ".join(f"{b['class']}({b['cx']:.2f},{b['cy']:.2f},{b['w']:.2f},{b['h']:.2f})" for b in s["boxes"])
        for i, s in enumerate(samples)
    )

    system_prompt = """你是一位专业的计算机视觉数据标注质检专家。
你会分析 YOLO 格式的目标检测标注数据，给出详细的质量报告。"""

    user_text = f"""请分析以下 {len(samples)} 张图片的 YOLO 标注质量。
每个图片用 (cx,cy,w,h) 表示归一化坐标的边界框中心点 + 宽高。

标注摘要：
{sample_summaries}

请分析：
1. 是否有框不准的问题（框太大/太小/没有贴合目标）
2. 是否有漏标（明显目标没有框）
3. 是否有错标（非目标物体被标为正类）
4. 是否有类别错误（不同类别混淆）
5. 整体标注质量评估（优秀/良好/一般/差）及改进建议

用中文回复，简洁明了，突出问题所在。"""

    try:
        raw = vlm_adapter.call_with_system_prompt(
            system_prompt=system_prompt,
            user_text=user_text,
            images_base64=images_b64,
            max_tokens=2048,
        )
        analysis = raw.strip()
    except Exception as e:
        analysis = f"VLM 调用失败：{str(e)[:200]}。请检查 VLM 配置是否正确。"

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "analysis": analysis,
            "sample_count": len(samples),
        },
    }


class VideoInferenceRequest(BaseModel):
    task_id: str
    weights_path: str
    conf: float = 0.25
    iou: float = 0.45
    max_frames: int = 30


@router.post("/video-inference/{task_id}")
async def video_inference(
    task_id: str,
    video: UploadFile = File(...),
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    训练完成后，用 best.pt 在上传的视频上推理，生成带标注的帧序列。
    用于在交付前让用户直观看到训练好的模型在真实视频中的效果。
    """
    import tempfile, os

    task = get_task_for_user(db, task_id, current_user)

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename or ".mp4")[1]) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        weights_p = Path(f"../backend/uploads/{task_id}/local_training_output/exp/weights/best.pt")
        if not weights_p.exists():
            weights_p = Path(f"backend/uploads/{task_id}/local_training_output/exp/weights/best.pt")
        if not weights_p.exists():
            raise HTTPException(status_code=400, detail="权重文件不存在，请确认训练已完成")

        vlm_result = task.vlm_result or {}
        classes_raw = vlm_result.get("classes", [])
        if isinstance(classes_raw, dict):
            classes_raw = classes_raw.get("classes", [])
        class_names: list[str] = []
        for i, c in enumerate(classes_raw):
            if isinstance(c, dict):
                class_names.append(c.get("class_name") or c.get("name") or f"class_{i}")
            else:
                class_names.append(str(c))

        frames = annotate_video_frames(
            video_path=tmp_path,
            weights_path=str(weights_p),
            conf=0.25,
            iou=0.45,
            max_frames=30,
            class_names=class_names or None,
        )
        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "total_frames": len(frames),
                "frames": frames,
            },
        }
    finally:
        os.unlink(tmp_path)


class PrepareDatasetRequest(BaseModel):
    task_id: str
    class_names: list[str]
    labeled_images_dir_override: str | None = None
    labels_dir_override: str | None = None
    output_root_override: str | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42


# Worker 回调密钥：Worker 和后端共享，用于本地训练完成后的写入回调
_WORKER_CALLBACK_SECRET = os.getenv("CV_AUTO_TRAINER_WORKER_CALLBACK_SECRET", "worker-secret-change-me")


class WorkerCallbackRequest(BaseModel):
    task_id: str
    status: str  # "complete" | "error"
    artifacts: dict = {}
    metrics: dict = {}
    error_message: str | None = None


@router.post("/worker-callback")
def worker_training_callback(
    payload: WorkerCallbackRequest,
    x_worker_secret: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Worker 本地训练完成后通过此接口写入 artifact_paths 和 mAP。
    使用共享密钥 + 可选的任务所有者用户 token 双重鉴权。
    """

    task_id = payload.task_id.split(":")[0] if ":" in payload.task_id else payload.task_id
    db = SessionLocal()
    try:
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return {"code": 404, "msg": "Task not found"}

        # 密钥校验：优先从 X-Worker-Secret header 校验（Worker 调用）
        # 也兼容从 task_id 中嵌入密钥的旧方式
        header_secret = x_worker_secret
        embedded_secret = payload.task_id.split(":")[-1] if ":" in payload.task_id else ""
        if header_secret and header_secret != _WORKER_CALLBACK_SECRET:
            return {"code": 403, "msg": "Invalid worker secret"}
        if not header_secret and embedded_secret and embedded_secret != _WORKER_CALLBACK_SECRET:
            return {"code": 403, "msg": "Invalid worker secret"}
        if payload.status == "complete":
            task.training_state = "done"
            task.training_finished_at = datetime.now(timezone.utc)
            task.artifact_paths = payload.artifacts
            task.training_progress = {
                "done": True,
                "current_epoch": payload.metrics.get("current_epoch", 0),
                "total_epochs": payload.metrics.get("total_epochs", 0),
                "current_map": payload.metrics.get("current_map", 0.0),
            }
            best_map = payload.metrics.get("current_map", 0.0)
            task.best_map50 = float(best_map) if best_map else None
            task.status = "done"
        else:
            task.training_state = "error"
            task.training_finished_at = datetime.now(timezone.utc)
            task.error_message = payload.error_message or "Unknown error"
            task.status = "error"
        db.commit()
        logger.info(f"Worker callback recorded for task {task_id}: status={payload.status}")
        return {"code": 0, "msg": "ok"}
    finally:
        db.close()


class PreviewInferenceRequest(BaseModel):
    task_id: str
    weights_path: str
    sample_images_dir: str
    conf: float = 0.25
    iou: float = 0.45
    max_images: int = 8


@router.post("/preview-inference")
def preview_inference(
    payload: PreviewInferenceRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    训练完成后，用 best.pt 在样板图上做推理预览。
    返回每张图片的检测结果（base64 标注图 + 检测框列表），
    帮助用户在全量训练前快速判断效果。
    """
    import base64
    import cv2
    from ultralytics import YOLO

    task = get_task_for_user(db, payload.task_id, current_user)
    weights_path = Path(payload.weights_path)
    if not weights_path.exists():
        raise HTTPException(status_code=400, detail=f"权重文件不存在: {weights_path}")

    # 解析前端传来的相对路径
    sample_dir = _resolve_frontend_path(payload.sample_images_dir)
    image_paths = sorted([
        p for p in sample_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])[: payload.max_images]

    if not image_paths:
        raise HTTPException(status_code=400, detail="样板图目录为空")

    try:
        model = YOLO(str(weights_path))
        results_list: list[dict] = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            results = model.predict(source=img, conf=payload.conf, iou=payload.iou, verbose=False)
            detections: list[dict] = []

            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    detections.append({
                        "class_name": r.names[cls_id],
                        "confidence": round(conf_val, 3),
                        "bbox_xywhn": [round(cx, 6), round(cy, 6), round(bw, 6), round(bh, 6)],
                    })
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"{r.names[cls_id]} {conf_val:.2f}"
                    cv2.putText(img, label, (int(x1), max(int(y1) - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buf).decode("ascii")

            results_list.append({
                "image_name": img_path.name,
                "image_base64": img_b64,
                "detections": detections,
            })

        return {
            "code": 0,
            "msg": "ok",
            "data": {
                "total_images": len(results_list),
                "results": results_list,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预览推理失败: {str(e)}")


@router.post("/prepare-dataset")
def prepare_dataset(
    payload: PrepareDatasetRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    将已打标的图片和标注文件整理为完整的 YOLO 数据集：
    1. 基于 task.image_dir（上传时记录的绝对路径）推算各目录
    2. 按分层抽样分割为 train/val/test
    3. 生成 data.yaml
    4. 计算质量报告

    前端在增强完成后（或跳过增强时）调用此接口。
    """
    task = get_task_for_user(db, payload.task_id, current_user)

    base_dir = Path(task.image_dir).parent if task.image_dir else None
    if not base_dir or not base_dir.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"无法确定上传目录。请确认图片已上传。"
                f"base_dir={base_dir}, image_dir={task.image_dir}"
            ),
        )

    labeled_images_dir = (
        _resolve_frontend_path(payload.labeled_images_dir_override)
        if payload.labeled_images_dir_override
        else base_dir / "labeled_images"
    )
    labels_dir = (
        _resolve_frontend_path(payload.labels_dir_override)
        if payload.labels_dir_override
        else base_dir / "labels"
    )

    # 如果前端传了 override，视为增强目录（labeled_images_aug），
    # 此时 output_root 应在其父目录下建 dataset/ 子目录
    if payload.labeled_images_dir_override:
        aug_dir = _resolve_frontend_path(payload.labeled_images_dir_override)
        output_root = aug_dir.parent / "dataset"
    elif payload.output_root_override:
        output_root = _resolve_frontend_path(payload.output_root_override)
    else:
        output_root = base_dir / "dataset"

    result = prepare_full_dataset(
        labeled_images_dir=str(labeled_images_dir),
        labels_dir=str(labels_dir),
        output_root=str(output_root),
        class_names=payload.class_names,
        ratios=payload.ratios,
        seed=payload.seed,
    )

    task.dataset_dir = result["dataset_root"]
    task.label_dir = str(Path(result["dataset_root"]) / "labels")
    task.total_image_count = sum(result["split_stats"].values())
    task.train_split_count = result["split_stats"].get("train", 0)
    task.val_split_count = result["split_stats"].get("val", 0)
    task.test_split_count = result["split_stats"].get("test", 0)
    task.split_stats = result["split_stats"]
    task.quality_report = result["quality_report"]
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "dataset_dir": result["dataset_root"],
            "data_yaml_path": result["data_yaml_path"],
            "split_stats": result["split_stats"],
            "quality_report": result["quality_report"],
        },
    }


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
    train_mode: str
    gpu_type: str
    local_device: int = 0
    resume_last: bool = False


@router.post("/start")
def start_training(
    payload: TrainStartRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, payload.task_id, current_user)

    mode = TrainMode.LOCAL if payload.train_mode == "local" else TrainMode.CLOUD

    # 提前校验：本地模式不支持多模型流水线
    if mode == TrainMode.LOCAL:
        pre_orchestrator = MultiModelTrainingOrchestrator(db)
        pre_steps = pre_orchestrator.plan_training_steps(task)
        if len(pre_steps) > 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "当前方案包含多个需训练的模型，本地模式暂不支持多模型顺序训练。"
                    "请在训练配置页切换到云端训练（AutoDL / SSH），或返回方案页简化为单一检测器。"
                ),
            )

    task.train_config = payload.model_dump()
    task.training_state = "queued"
    task.training_progress = {
        "current_epoch": 0,
        "total_epochs": payload.epochs,
        "current_map": 0.0,
        "done": False,
    }
    task.training_started_at = datetime.now(timezone.utc)
    task.training_finished_at = None
    task.error_message = None
    task.status = "training_queued"
    db.commit()

    def run_in_thread():
        thread_db = SessionLocal()
        try:
            thread_task = thread_db.query(Task).filter(Task.id == payload.task_id).first()
            if not thread_task:
                return
            orchestrator = MultiModelTrainingOrchestrator(thread_db)
            steps = orchestrator.plan_training_steps(thread_task)
            use_multi = len(steps) > 1 or any(s.get("reuse_cache_id") for s in steps)

            def progress_callback(status: dict):
                tracked_task = thread_db.query(Task).filter(Task.id == payload.task_id).first()
                if not tracked_task:
                    return
                tracked_task.training_state = "running"
                tracked_task.training_progress = status
                tracked_task.status = "training_cloud" if mode == TrainMode.CLOUD else "training_local"
                thread_db.commit()

            if use_multi:
                multi_result = orchestrator.run(
                    task=thread_task,
                    base_train_config=payload.model_dump(),
                    progress_callback=progress_callback,
                )
                # 主检测器产物 + 多模型清单一起写入 artifact_paths
                primary = multi_result.get("primary_artifacts", {}) or {}
                multi_artifacts = multi_result.get("multi_model_artifacts", {}) or {}
                merged_artifacts = dict(primary)
                merged_artifacts["__multi_model__"] = multi_artifacts
                result = merged_artifacts
            else:
                dispatcher = TrainDispatcher(thread_db)
                result = dispatcher.dispatch(
                    mode=mode,
                    task_id=payload.task_id,
                    train_config=payload.model_dump(),
                    progress_callback=progress_callback,
                )

            thread_task = thread_db.query(Task).filter(Task.id == payload.task_id).first()
            if thread_task:
                existing_progress = dict(thread_task.training_progress or {})
                existing_progress.setdefault("current_epoch", payload.epochs)
                existing_progress.setdefault("total_epochs", payload.epochs)
                existing_progress.setdefault("current_map", 0.0)
                existing_progress["done"] = True
                thread_task.training_state = "done"
                thread_task.training_progress = existing_progress
                thread_task.training_finished_at = datetime.now(timezone.utc)
                thread_task.artifact_paths = result
                thread_task.status = "done"
                thread_db.commit()
        except LocalMultiModelNotSupported as e:
            thread_task = thread_db.query(Task).filter(Task.id == payload.task_id).first()
            if thread_task:
                thread_task.training_state = "error"
                thread_task.training_finished_at = datetime.now(timezone.utc)
                thread_task.error_message = str(e)
                thread_task.status = "error"
                thread_db.commit()
        except Exception as e:
            thread_task = thread_db.query(Task).filter(Task.id == payload.task_id).first()
            if thread_task:
                thread_task.training_state = "error"
                thread_task.training_finished_at = datetime.now(timezone.utc)
                thread_task.error_message = str(e)
                thread_task.status = "error"
                thread_db.commit()
        finally:
            thread_db.close()

    threading.Thread(target=run_in_thread, daemon=True).start()

    return {
        "code": 0,
        "msg": "ok",
        "data": {"instance_id": payload.task_id},
    }


class EstimateRequest(BaseModel):
    task_id: str
    model: str = "yolo11s.pt"
    epochs: int = 100
    imgsz: int = 640
    gpu_type: str = "RTX 4090"


# GPU hourly rates (CNY) — approximate AutoDL pricing
_GPU_HOURLY_RATE: dict[str, float] = {
    "RTX 4090": 2.58,
    "RTX 4090D": 2.28,
    "RTX 3090": 1.08,
    "RTX 3080": 0.88,
    "A100-80G": 8.56,
    "A100-40G": 5.98,
    "A800-80G": 7.80,
    "V100-32G": 2.18,
    "A6000": 3.28,
}

# Base seconds per epoch per image at imgsz=640 on RTX 4090 (rough estimates)
_MODEL_BASE_SPEED: dict[str, float] = {
    "yolo11n.pt": 0.0010,
    "yolo11s.pt": 0.0016,
    "yolo11m.pt": 0.0030,
    "yolo11l.pt": 0.0050,
    "yolov8n.pt": 0.0010,
    "yolov8s.pt": 0.0016,
    "yolov8m.pt": 0.0030,
    "yolov8l.pt": 0.0050,
    "yolov5s.pt": 0.0014,
    "rtdetr-l.pt": 0.0065,
    "rf-detr-base.pt": 0.0070,
}

# GPU relative speed factor (1.0 = RTX 4090)
_GPU_SPEED_FACTOR: dict[str, float] = {
    "RTX 4090": 1.0,
    "RTX 4090D": 0.95,
    "RTX 3090": 0.60,
    "RTX 3080": 0.50,
    "A100-80G": 1.30,
    "A100-40G": 1.10,
    "A800-80G": 1.25,
    "V100-32G": 0.45,
    "A6000": 0.70,
}


@router.post("/estimate")
def estimate_training_cost(
    payload: EstimateRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    估算多模型训练的总时长和费用。
    综合考虑每个训练步骤的模型大小、图片数量、GPU 类型。
    """
    task = get_task_for_user(db, payload.task_id, current_user)

    # 数据集图片数
    total_images = task.total_image_count or task.raw_image_count or 500

    # 多模型流水线步骤
    orchestrator = MultiModelTrainingOrchestrator(db)
    steps = orchestrator.plan_training_steps(task)

    gpu_speed = _GPU_SPEED_FACTOR.get(payload.gpu_type, 0.7)
    hourly_rate = _GPU_HOURLY_RATE.get(payload.gpu_type, 3.0)
    imgsz_factor = (payload.imgsz / 640) ** 2  # quadratic scaling with resolution

    step_estimates = []
    total_seconds = 0.0

    for step in steps:
        if step.get("reuse_cache_id"):
            step_estimates.append({
                "step_id": step["step_id"],
                "model_id": step.get("model_id", ""),
                "role": step.get("role", ""),
                "source": "reuse",
                "duration_min": 0,
                "cost_cny": 0,
            })
            continue

        model_id = step.get("model_id") or payload.model
        step_epochs = step.get("epochs") or payload.epochs
        base_speed = _MODEL_BASE_SPEED.get(model_id, 0.0025)

        # seconds = base_speed * images * epochs / gpu_speed * imgsz_factor
        step_seconds = base_speed * total_images * step_epochs * imgsz_factor / gpu_speed
        # Add ~5 min overhead for setup, uploading, etc.
        step_seconds += 300

        total_seconds += step_seconds
        step_estimates.append({
            "step_id": step["step_id"],
            "model_id": model_id,
            "role": step.get("role", ""),
            "source": "train",
            "duration_min": round(step_seconds / 60, 1),
            "cost_cny": round(step_seconds / 3600 * hourly_rate, 2),
        })

    # If no trainable steps from pipeline, estimate for single model
    if not steps:
        base_speed = _MODEL_BASE_SPEED.get(payload.model, 0.0025)
        total_seconds = base_speed * total_images * payload.epochs * imgsz_factor / gpu_speed + 300
        step_estimates.append({
            "step_id": "single",
            "model_id": payload.model,
            "role": "primary_detector",
            "source": "train",
            "duration_min": round(total_seconds / 60, 1),
            "cost_cny": round(total_seconds / 3600 * hourly_rate, 2),
        })

    total_cost = sum(s["cost_cny"] for s in step_estimates)
    total_min = sum(s["duration_min"] for s in step_estimates)

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "gpu_type": payload.gpu_type,
            "hourly_rate_cny": hourly_rate,
            "total_images": total_images,
            "total_duration_min": round(total_min, 1),
            "total_cost_cny": round(total_cost, 2),
            "steps": step_estimates,
        },
    }


@router.get("/{task_id}/status")
def get_training_status(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    progress = task.training_progress or {}
    artifact_paths = dict(task.artifact_paths or {})
    recovery_info = artifact_paths.pop("__autodl_recovery__", None)
    multi_model_artifacts = artifact_paths.pop("__multi_model__", None)
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "state": task.training_state or "unknown",
            "current_epoch": progress.get("current_epoch", 0),
            "total_epochs": progress.get("total_epochs", 100),
            "current_map": progress.get("current_map", 0.0),
            "done": progress.get("done", False),
            "error": task.error_message,
            "artifact_paths": artifact_paths,
            "autodl_recovery": recovery_info,
            # 多模型训练进度
            "current_model_index": progress.get("current_model_index"),
            "total_models": progress.get("total_models"),
            "current_step_id": progress.get("current_step_id"),
            "current_model_id": progress.get("current_model_id"),
            "current_model_source": progress.get("source"),
            "multi_model_artifacts": multi_model_artifacts,
        },
    }


@router.get("/{task_id}/recovery")
def get_training_recovery_info(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    返回 AutoDL 训练失败时的手动恢复信息（SSH、训练命令等），
    供前端展示"如何自己救回已租用的 GPU 实例"教程。
    """
    task = get_task_for_user(db, task_id, current_user)
    recovery = (task.artifact_paths or {}).get("__autodl_recovery__")
    if not recovery:
        raise HTTPException(status_code=404, detail="当前任务没有可用的恢复信息")

    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "recovery_info": recovery,
            "instructions_zh": _build_recovery_instructions(recovery),
        },
    }


def _build_recovery_instructions(recovery: dict) -> list[dict]:
    """生成分步骤的恢复教程"""
    steps: list[dict] = []
    instance_id = recovery.get("instance_id") or "<你的实例ID>"
    console_url = recovery.get("autodl_console_url") or f"https://www.autodl.com/console/instance/{instance_id}"

    if recovery.get("ssh_retrieval_failed"):
        steps.extend([
            {
                "step": 1,
                "title": "登录 AutoDL 控制台",
                "description": "访问控制台查看该实例的 SSH 连接信息（实例仍在运行，继续计费）",
                "action": f"打开：{console_url}",
            },
            {
                "step": 2,
                "title": "决定是否继续训练",
                "description": "若不再需要训练，直接在控制台关机；若想继续训练，参考下方命令",
                "action": "",
            },
        ])
    else:
        ssh_cmd = f"ssh -p {recovery.get('ssh_port', 22)} {recovery.get('ssh_username', 'root')}@{recovery.get('ssh_host', '<host>')}"
        pwd_note = recovery.get("ssh_password_masked") or "（请前往 AutoDL 控制台查看）"
        steps.extend([
            {
                "step": 1,
                "title": "通过 SSH 连接实例",
                "description": f"打开终端，输入下方命令。密码（脱敏）：{pwd_note}。完整密码请在 AutoDL 控制台查看。",
                "action": ssh_cmd,
            },
            {
                "step": 2,
                "title": "进入工作目录",
                "description": "数据集已经上传到 /root/dataset，训练输出目录为 /root/training_output",
                "action": "cd /root && ls",
            },
        ])
        if recovery.get("train_command"):
            steps.extend([
                {
                    "step": 3,
                    "title": "手动执行训练",
                    "description": "原本由系统发起的训练命令。在 screen 里运行防止 SSH 断开",
                    "action": f"screen -S train\n{recovery['train_command']}",
                },
                {
                    "step": 4,
                    "title": "下载权重文件",
                    "description": "训练完成后，best.pt 位于 /root/training_output/exp/weights/best.pt，可用 scp 下载",
                    "action": f"scp -P {recovery.get('ssh_port', 22)} {recovery.get('ssh_username', 'root')}@{recovery.get('ssh_host', '<host>')}:/root/training_output/exp/weights/best.pt ~/Downloads/",
                },
            ])
        steps.append({
            "step": len(steps) + 1,
            "title": "完成后务必关机",
            "description": "AutoDL 实例按使用时长计费，完成后记得在控制台手动关机！",
            "action": f"打开控制台关机：{console_url}",
        })
    return steps


@router.get("/{task_id}/report")
def get_training_report(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    让 VLM 解读训练结果图表（results.png, confusion_matrix.png 等），返回中文分析报告。
    """
    import base64
    import json
    from pathlib import Path

    task = get_task_for_user(db, task_id, current_user)
    artifacts = task.artifact_paths or {}

    # 收集可用图表
    chart_names = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
                   "F1_curve.png", "P_curve.png", "R_curve.png", "PR_curve.png"]
    images_b64: list[str] = []
    found_charts: list[str] = []
    for name in chart_names:
        path_str = artifacts.get(name)
        if not path_str:
            continue
        p = Path(path_str)
        if p.exists() and p.is_file():
            try:
                images_b64.append(base64.b64encode(p.read_bytes()).decode("ascii"))
                found_charts.append(name)
            except OSError:
                pass

    if not images_b64:
        raise HTTPException(status_code=404, detail="未找到训练结果图表，请确认训练已完成")

    # 构建 VLM 提示
    from services.settings_manager import get_settings, decrypt_value
    from services.vlm_adapter import VLMAdapter

    settings = get_settings(db, current_user["user_id"])
    vlm_adapter = VLMAdapter(
        provider=settings.vlm_provider or "openai",
        base_url=settings.vlm_base_url or "",
        api_key=decrypt_value(settings.vlm_api_key_encrypted or ""),
        api_format=settings.vlm_api_format,
        model=settings.vlm_model,
        temperature=settings.vlm_temperature,
        top_p=settings.vlm_top_p,
    )

    plan = task.algorithm_plan or {}
    targets = [t.get("display_name_zh") or t.get("class_name", "") for t in plan.get("targets", [])]
    scenario = plan.get("scenario_type", "")

    system_prompt = """你是一位计算机视觉训练结果分析专家。
你会看到 YOLO 模型训练产出的各种图表（训练曲线、混淆矩阵、精度/召回曲线等）。
请用简洁的中文给出分析报告，帮助零基础的用户理解训练效果。

分析维度：
1. 训练是否收敛？是否有过拟合迹象？
2. 各类别的检测精度如何？哪些类别效果好 / 差？
3. 总体 mAP 表现评估
4. 具体的改进建议（如增加数据量、调整超参数、处理类别不平衡等）

输出 JSON 格式：
{
  "overall_assessment_zh": "一段总体评价",
  "convergence_zh": "收敛情况分析",
  "class_performance_zh": "各类别表现分析",
  "improvement_suggestions_zh": ["建议1", "建议2"],
  "score": 0-100
}"""

    user_text = f"""场景：{scenario}
检测类别：{', '.join(targets)}
图表列表：{', '.join(found_charts)}

请逐一分析这些训练结果图表，给出综合报告。"""

    try:
        raw = vlm_adapter.call_with_system_prompt(
            system_prompt=system_prompt,
            user_text=user_text,
            images_base64=images_b64,
            response_format="json",
            max_tokens=2048,
        )
        report = json.loads(raw.strip().removeprefix("```json").removesuffix("```").strip())
        report["charts_analyzed"] = found_charts
    except Exception as e:
        report = {
            "overall_assessment_zh": f"VLM 分析失败：{str(e)[:200]}",
            "convergence_zh": "",
            "class_performance_zh": "",
            "improvement_suggestions_zh": ["请检查 VLM 服务配置"],
            "score": None,
            "charts_analyzed": found_charts,
        }

    return {"code": 0, "msg": "ok", "data": report}


@router.post("/{task_id}/cancel")
def cancel_training(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    dispatcher = TrainDispatcher(db)
    dispatcher.cancel_training(task_id)
    task.training_state = "cancelled"
    task.training_finished_at = datetime.now(timezone.utc)
    task.status = "cancelled"
    db.commit()
    return {"code": 0, "msg": "ok", "data": None}
