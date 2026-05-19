from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import shutil
import json
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
    elif subdir == "incremental_images":
        task.incremental_image_dir = str(task_dir.resolve())
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
        frames_b64, frame_meta = extract_frames_for_vlm(str(video_path), max_frames=8)
        return {
            "code": 0, "msg": "ok",
            "data": {
                "video_path": str(video_path),
                "video_info": video_info,
                "frames_base64": frames_b64,
                "frame_count": len(frames_b64),
                "frame_meta": frame_meta,
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


@router.get("/{task_id}/check-annotations")
def check_existing_annotations(
    task_id: str,
    subdir: str = "images",
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    检查任务目录下是否存在预标注的 YOLO .txt 文件，
    用于判断用户是否已经使用 LabelImg/roLabelImg 等工具预先标注好了数据。
    返回找到的标注文件数量和覆盖的图片数。
    """
    task = get_task_for_user(db, task_id, current_user)

    img_dir = UPLOAD_DIR / task_id / subdir
    if not img_dir.exists():
        return {
            "code": 0, "msg": "ok",
            "data": {
                "has_annotations": False,
                "total_images": 0,
                "annotated_images": 0,
                "total_boxes": 0,
                "detected_classes": [],
                "message": "目录不存在或为空",
            },
        }

    from pathlib import Path

    exts = {".jpg", ".jpeg", ".png"}
    image_paths = sorted(
        [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=lambda p: p.name.lower(),
    )

    if not image_paths:
        return {
            "code": 0, "msg": "ok",
            "data": {
                "has_annotations": False,
                "total_images": len(image_paths),
                "annotated_images": 0,
                "total_boxes": 0,
                "detected_classes": [],
                "message": "目录中无图片文件",
            },
        }

    annotated_images = 0
    total_boxes = 0
    class_sets: set[int] = set()

    for img_path in image_paths:
        label_path = img_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        annotated_images += 1
        try:
            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        total_boxes += 1
                        try:
                            class_sets.add(int(parts[0]))
                        except ValueError:
                            pass
        except Exception:
            pass

    detected_classes = sorted(class_sets)

    return {
        "code": 0, "msg": "ok",
        "data": {
            "has_annotations": annotated_images > 0,
            "total_images": len(image_paths),
            "annotated_images": annotated_images,
            "total_boxes": total_boxes,
            "detected_classes": detected_classes,
            "message": (
                f"发现 {annotated_images}/{len(image_paths)} 张图片已标注，共 {total_boxes} 个框，类别索引 {detected_classes}"
                if annotated_images > 0
                else "未发现 YOLO 标注文件"
            ),
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


@router.get("/{task_id}/image/{image_name}")
def serve_dataset_image(
    task_id: str,
    image_name: str,
    token: Optional[str] = None,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    safe_name = _safe_filename(image_name)
    for subdir_name in ["images", "video_frames"]:
        candidate = UPLOAD_DIR / task_id / subdir_name / safe_name
        if candidate.exists():
            _ensure_within(UPLOAD_DIR / task_id, candidate)
            return FileResponse(path=candidate)
    raise HTTPException(status_code=404, detail="Image not found")


# ── Seed Annotations (Snowball Labeling) ──

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


class SeedAnnotationBox(BaseModel):
    class_index: int
    cx: float
    cy: float
    w: float
    h: float


class SaveSeedAnnotationRequest(BaseModel):
    image_name: str
    boxes: list[SeedAnnotationBox]


@router.post("/{task_id}/seed-annotations")
def save_seed_annotation(
    task_id: str,
    payload: SaveSeedAnnotationRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)

    seed_dir = UPLOAD_DIR / task_id / "seed_labels"
    seed_dir.mkdir(exist_ok=True, parents=True)

    stem = Path(_safe_filename(payload.image_name)).stem
    label_path = _ensure_within(seed_dir, seed_dir / f"{stem}.txt")

    if not payload.boxes:
        label_path.unlink(missing_ok=True)
    else:
        with open(label_path, "w", encoding="utf-8") as f:
            for box in payload.boxes:
                f.write(f"{box.class_index} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}\n")

    return {"code": 0, "msg": "ok", "data": {"saved": len(payload.boxes)}}


@router.get("/{task_id}/seed-annotations")
def get_seed_annotations(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)

    seed_dir = UPLOAD_DIR / task_id / "seed_labels"
    annotations: dict[str, list[dict]] = {}
    annotated_count = 0

    if seed_dir.exists():
        for txt_file in sorted(seed_dir.glob("*.txt")):
            boxes = []
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        boxes.append({
                            "class_index": int(parts[0]),
                            "cx": float(parts[1]),
                            "cy": float(parts[2]),
                            "w": float(parts[3]),
                            "h": float(parts[4]),
                        })
            if boxes:
                annotations[txt_file.stem] = boxes
                annotated_count += 1

    # Count total dataset images
    total_count = 0
    for subdir_name in ["images", "video_frames"]:
        img_dir = UPLOAD_DIR / task_id / subdir_name
        if img_dir.exists():
            total_count += sum(
                1 for p in img_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )

    return {
        "code": 0, "msg": "ok",
        "data": {
            "annotations": annotations,
            "annotated_count": annotated_count,
            "total_count": total_count,
        },
    }


@router.delete("/{task_id}/seed-annotations/{image_name}")
def delete_seed_annotation(
    task_id: str,
    image_name: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)

    seed_dir = UPLOAD_DIR / task_id / "seed_labels"
    stem = Path(_safe_filename(image_name)).stem
    label_path = seed_dir / f"{stem}.txt"
    if label_path.exists():
        _ensure_within(seed_dir, label_path)
        label_path.unlink()

    return {"code": 0, "msg": "ok"}


@router.get("/{task_id}/dataset-images")
def list_dataset_images(
    task_id: str,
    page: int = 1,
    size: int = 50,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)

    # Collect images from both possible directories
    all_images: list[str] = []
    for subdir_name in ["images", "video_frames"]:
        img_dir = UPLOAD_DIR / task_id / subdir_name
        if img_dir.exists():
            all_images.extend(
                p.name for p in sorted(img_dir.iterdir(), key=lambda p: p.name.lower())
                if p.is_file() and p.suffix.lower() in IMG_EXTS
            )

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for name in all_images:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    all_images = unique

    # Check which ones have seed annotations
    seed_dir = UPLOAD_DIR / task_id / "seed_labels"
    annotated_stems: set[str] = set()
    if seed_dir.exists():
        annotated_stems = {p.stem for p in seed_dir.glob("*.txt") if p.stat().st_size > 0}

    total = len(all_images)
    start = (page - 1) * size
    page_images = all_images[start:start + size]

    items = []
    for name in page_images:
        stem = Path(name).stem
        items.append({
            "name": name,
            "has_annotation": stem in annotated_stems,
        })

    return {
        "code": 0, "msg": "ok",
        "data": {
            "images": items,
            "total": total,
            "annotated": len(annotated_stems),
            "page": page,
            "size": size,
        },
    }


# ── Review Labels (低置信框审核) ──

@router.get("/{task_id}/review-labels")
def list_review_labels(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """列出 review_labels/ 目录中有待审核标注的图片名"""
    task = get_task_for_user(db, task_id, current_user)
    review_dir = UPLOAD_DIR / task_id / "review_labels"
    if not review_dir.exists():
        return {"code": 0, "msg": "ok", "data": []}

    # Find image names that correspond to review labels
    image_dir = None
    for sub in ["images", "video_frames"]:
        candidate = UPLOAD_DIR / task_id / sub
        if candidate.exists():
            image_dir = candidate
            break

    if image_dir is None:
        return {"code": 0, "msg": "ok", "data": []}

    names = []
    for lbl in sorted(review_dir.iterdir()):
        if not lbl.is_file() or lbl.suffix != ".txt" or lbl.stat().st_size == 0:
            continue
        stem = lbl.stem
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            img_path = image_dir / f"{stem}{ext}"
            if img_path.exists():
                names.append(img_path.name)
                break

    return {"code": 0, "msg": "ok", "data": names}


@router.get("/{task_id}/review-labels/{image_stem}")
def get_review_label(
    task_id: str,
    image_stem: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """获取单张图片的待审核标注 (YOLO 格式)"""
    task = get_task_for_user(db, task_id, current_user)
    safe_stem = _safe_filename(image_stem.replace(".txt", ""))
    label_path = UPLOAD_DIR / task_id / "review_labels" / f"{safe_stem}.txt"

    if not label_path.exists():
        return {"code": 0, "msg": "ok", "data": []}

    _ensure_within(UPLOAD_DIR / task_id, label_path)

    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            cls, cx, cy, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})

    return {"code": 0, "msg": "ok", "data": boxes}


@router.post("/{task_id}/merge-labels")
def merge_labels_after_review(
    task_id: str,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """
    审核完成后重新合并三路标注（manual / review / auto），
    将结果写入 labeled_images/ 和 labels/ 目录，
    供后续 augment / training 流程使用。

    Fix for: review_labels 被丢弃的 bug
    """
    task = get_task_for_user(db, task_id, current_user)

    from worker.pipeline.seed_auto_labeler import merge_seed_and_auto_labels
    from services.settings_manager import get_settings

    settings = get_settings(db, current_user["user_id"])
    task_dir = Path("./uploads") / task_id

    class_names = [
        cls.get("class_name", f"class_{i}")
        for i, cls in enumerate(task.vlm_result or [])
    ]

    result = merge_seed_and_auto_labels(str(task_dir), class_names)

    return {"code": 0, "msg": "ok", "data": result}
