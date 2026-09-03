import asyncio
import base64
import os
import tempfile
import uvicorn
import torch
import gc
import httpx
from pathlib import Path
from threading import Lock
from typing import Optional

# 关闭 ultralytics 内部的 auto-install：
# ultralytics 8.4+ 在 set_classes() 时如果检测到缺 clip 包，会自动 `uv pip install`，
# 但该行为可能撞 Windows/macOS 写权限问题。我们的启动脚本已经主动 `pip install --user`
# 预装好 clip 包；如果此处漏装，宁可 worker 报 ModuleNotFoundError 让用户明确感知，
# 也不要让 ultralytics 静默修改用户系统。
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")
os.environ.setdefault("ULTRALYTICS_AUTOINSTALL", "0")

# 相对路径基准：项目根目录
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def _resolve_path(p: str) -> Path:
    """将前端传来的相对路径解析为绝对路径，与后端 _resolve_frontend_path 保持一致。"""
    path = Path(p)
    if path.is_absolute():
        return path
    # 去掉 '../backend/' 前缀（如 '../backend/uploads/...' -> 'backend/uploads/...'）
    parts = path.parts
    if len(parts) >= 3 and parts[0] == ".." and parts[1] == "backend":
        resolved = _PROJECT_ROOT.joinpath(*parts[1:])
    else:
        resolved = _PROJECT_ROOT / path
    return resolved.resolve()
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


_MODEL_PREP_LOCK = Lock()
_MODEL_PREP_STATE = {
    "running": False,
    "include_moondream": False,
    "steps": {
        "yolo_world": {"status": "pending", "message": "等待检查 YOLO-World"},
        "clip": {"status": "pending", "message": "等待检查 CLIP"},
        "moondream": {"status": "optional", "message": "Moondream VQA 可选"},
    },
    "error": None,
    "status": None,
}

# 是否在 Worker 启动时自动安装所需模型（YOLO-World + CLIP）
_AUTO_PREPARE_ON_STARTUP = os.getenv("CV_AUTO_TRAINER_WORKER_AUTO_PREPARE", "on").strip().lower() in ("1", "true", "on", "yes")
# 是否在启动时同时预装 Moondream2（第二段质检模型，会增加下载时间和显存占用）
_PREPARE_MOONDREAM_ON_STARTUP = os.getenv("CV_AUTO_TRAINER_WORKER_PREPARE_MOONDREAM", "off").strip().lower() in ("1", "true", "on", "yes")


def _set_model_prep_step(name: str, status: str, message: str):
    with _MODEL_PREP_LOCK:
        _MODEL_PREP_STATE["steps"][name] = {"status": status, "message": message}


def _set_model_prep_status(status: dict):
    with _MODEL_PREP_LOCK:
        _MODEL_PREP_STATE["status"] = status
        if status["yolo_world"]["installed"]:
            _MODEL_PREP_STATE["steps"]["yolo_world"]["status"] = "complete"
            _MODEL_PREP_STATE["steps"]["yolo_world"]["message"] = "YOLO-World 已安装"
        if status["clip"]["installed"]:
            _MODEL_PREP_STATE["steps"]["clip"]["status"] = "complete"
            _MODEL_PREP_STATE["steps"]["clip"]["message"] = "CLIP 已安装"
        if status["moondream"]["installed"]:
            _MODEL_PREP_STATE["steps"]["moondream"]["status"] = "complete"
            _MODEL_PREP_STATE["steps"]["moondream"]["message"] = "Moondream2 已安装"


def _model_prep_progress(name: str, status: str, message: str):
    _set_model_prep_step(name, status, message)


def _run_model_prepare(
    include_moondream: bool = False,
    include_locate_anything: bool = False,
    include_eagle_vqa: bool = False,
):
    import logging
    logger = logging.getLogger("worker.startup")

    def _log_progress(name: str, status: str, message: str):
        logger.info(f"[模型准备] {name} -> {status}: {message}")
        _model_prep_progress(name, status, message)

    try:
        from pipeline.stage2_labeler import prepare_labeling_model_cache

        status = prepare_labeling_model_cache(
            include_moondream=include_moondream,
            include_locate_anything=include_locate_anything,
            include_eagle_vqa=include_eagle_vqa,
            progress_callback=_log_progress,
        )
        _set_model_prep_status(status)
        logger.info(f"[模型准备] 完成: {status}")
        with _MODEL_PREP_LOCK:
            _MODEL_PREP_STATE["error"] = None
    except Exception as exc:
        logger.error(f"[模型准备] 失败: {exc}")
        with _MODEL_PREP_LOCK:
            _MODEL_PREP_STATE["error"] = str(exc)
    finally:
        with _MODEL_PREP_LOCK:
            _MODEL_PREP_STATE["running"] = False


def _refresh_model_status_only():
    from pipeline.stage2_labeler import get_model_cache_status

    status = get_model_cache_status()
    _set_model_prep_status(status)
    return status


@app.on_event("startup")
async def startup_model_status_check():
    import logging
    logger = logging.getLogger("worker.startup")

    # 首次启动时检查模型缓存状态（快速，非阻塞）
    await asyncio.to_thread(_refresh_model_status_only)

    if not _AUTO_PREPARE_ON_STARTUP:
        logger.info("[启动] 模型自动预装已禁用 (CV_AUTO_TRAINER_WORKER_AUTO_PREPARE=off)，跳过启动时安装。")
        return

    status = _refresh_model_status_only()
    yolo_ok = status.get("yolo_world", {}).get("installed", False)
    clip_ok = status.get("clip", {}).get("installed", False)
    moon_ok = status.get("moondream", {}).get("installed", False)
    locate_ok = status.get("locate_anything", {}).get("installed", False)
    eagle_vqa_ok = status.get("eagle_vqa", {}).get("installed", False)

    need_yolo = not yolo_ok
    need_clip = not clip_ok
    need_moon = _PREPARE_MOONDREAM_ON_STARTUP and not moon_ok
    # Eagle 引擎默认不自动下载，除非显式启用
    need_locate = os.getenv("CV_PREPARE_LOCATE_ANYTHING", "0") == "1" and not locate_ok
    need_eagle = os.getenv("CV_PREPARE_EAGLE_VQA", "0") == "1" and not eagle_vqa_ok

    if not (need_yolo or need_clip or need_moon or need_locate or need_eagle):
        logger.info("[启动] 所有必需模型（YOLO-World + CLIP）已就绪，跳过预装。")
        return

    logger.info(
        f"[启动] 检测到未安装的模型，正在后台预装... "
        f"需要 YOLO-World={need_yolo}, CLIP={need_clip}, "
        f"Moondream2={need_moon}, LocateAnything={need_locate}, Eagle2.5={need_eagle}"
    )

    include_moondream = _PREPARE_MOONDREAM_ON_STARTUP
    include_locate = need_locate
    include_eagle = need_eagle

    with _MODEL_PREP_LOCK:
        if _MODEL_PREP_STATE["running"]:
            logger.info("[启动] 模型预装任务已在运行中，跳过重复启动。")
            return
        _MODEL_PREP_STATE["running"] = True
        _MODEL_PREP_STATE["include_moondream"] = include_moondream
        _MODEL_PREP_STATE["include_locate_anything"] = include_locate
        _MODEL_PREP_STATE["include_eagle_vqa"] = include_eagle
        _MODEL_PREP_STATE["error"] = None
        if need_yolo:
            _MODEL_PREP_STATE["steps"]["yolo_world"] = {"status": "running", "message": "启动中准备 YOLO-World"}
        if need_clip:
            _MODEL_PREP_STATE["steps"]["clip"] = {"status": "pending", "message": "等待 YOLO-World 类别编码"}
        if include_moondream and need_moon:
            _MODEL_PREP_STATE["steps"]["moondream"] = {"status": "pending", "message": "等待下载 Moondream2"}
        elif not include_moondream:
            _MODEL_PREP_STATE["steps"]["moondream"] = {"status": "optional", "message": "已选择跳过 Moondream VQA"}
        if need_locate:
            _MODEL_PREP_STATE["steps"]["locate_anything"] = {"status": "pending", "message": "等待下载 LocateAnything-3B"}
        else:
            _MODEL_PREP_STATE["steps"]["locate_anything"] = {"status": "optional", "message": "可选：更快的检测速度"}
        if need_eagle:
            _MODEL_PREP_STATE["steps"]["eagle_vqa"] = {"status": "pending", "message": "等待下载 Eagle2.5-8B"}
        else:
            _MODEL_PREP_STATE["steps"]["eagle_vqa"] = {"status": "optional", "message": "可选：更强的质检能力"}

    # 在后台线程执行，不阻塞 uvicorn 启动
    asyncio.create_task(
        asyncio.to_thread(_run_model_prepare, include_moondream, include_locate, include_eagle)
    )


async def _safe_ws_send(ws: WebSocket, data: dict) -> bool:
    """Send JSON on WebSocket, return False if connection is already closed."""
    try:
        await ws.send_json(data)
        return True
    except (RuntimeError, WebSocketDisconnect):
        return False


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            try:
                await _handle_command(ws, data)
            except (RuntimeError, WebSocketDisconnect):
                break
            except Exception as e:
                if not await _safe_ws_send(ws, {"type": "error", "message": str(e)}):
                    break
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

        # 解析前端传来的相对路径
        resolved_image_dir = str(_resolve_path(payload["image_dir"]))
        resolved_output_raw_dir = str(_resolve_path(payload["output_raw_dir"]))
        resolved_output_label_dir = str(_resolve_path(payload["output_label_dir"]))
        resolved_output_image_dir = str(_resolve_path(payload["output_image_dir"]))

        def _send(msg: dict):
            asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, msg), loop)

        def make_progress(current, total, phase):
            _send(_build_gpu_info_msg())
            _send({
                "type": "progress",
                "stage": phase,
                "current": current,
                "total": total,
            })

        def count_boxes(box_map: dict) -> int:
            return sum(len(boxes) for boxes in box_map.values())

        def count_images_with_boxes(box_map: dict) -> int:
            return sum(1 for boxes in box_map.values() if boxes)

        def compute_class_balance(passed: dict, classes: list) -> dict:
            """
            统计每类标注框数量，给出训练前类别平衡预警。
            errors（阻断）：单类样本框 < 50
            warnings（提示）：单类比例 < 30% / < 10%
            """
            class_names = [c.get("class_name") or f"class_{i}" for i, c in enumerate(classes)]
            counts = {name: 0 for name in class_names}
            for boxes in passed.values():
                for box in boxes:
                    idx = box.get("class_idx")
                    if idx is not None and 0 <= idx < len(class_names):
                        counts[class_names[idx]] += 1
            warnings: list = []
            errors: list = []
            if counts:
                max_count = max(counts.values()) if counts else 0
                for name, cnt in counts.items():
                    ratio = (cnt / max_count) if max_count > 0 else 0.0
                    if cnt == 0:
                        errors.append({
                            "class": name, "count": cnt, "level": "error",
                            "message": f"[{name}] 没有任何有效标注框，无法训练该类别",
                        })
                    elif cnt < 50:
                        errors.append({
                            "class": name, "count": cnt, "level": "error",
                            "message": f"[{name}] 样本严重不足: {cnt} 个框（建议 ≥ 100）",
                        })
                    elif ratio < 0.1:
                        warnings.append({
                            "class": name, "count": cnt, "level": "warning",
                            "message": f"[{name}] 严重不平衡: 只占最大类的 {int(ratio*100)}%",
                        })
                    elif ratio < 0.3:
                        warnings.append({
                            "class": name, "count": cnt, "level": "warning",
                            "message": f"[{name}] 偏少: 只占最大类的 {int(ratio*100)}%",
                        })
            return {
                "counts": counts,
                "warnings": warnings,
                "errors": errors,
                "blocked": bool(errors),
            }

        def build_labeling_result(
            raw_boxes: dict,
            passed: dict,
            use_existing: bool,
            skip_quality_check: bool,
            qc_stats: Optional[dict] = None,
            class_balance: Optional[dict] = None,
        ) -> dict:
            raw_box_count = count_boxes(raw_boxes)
            passed_box_count = count_boxes(passed)
            labeled_count = count_images_with_boxes(passed)
            detected_image_count = count_images_with_boxes(raw_boxes)
            quality_filtered_box_count = 0 if use_existing or skip_quality_check else max(0, raw_box_count - passed_box_count)

            if image_total <= 0:
                message = "上传目录中没有可处理的图片。"
                suggestions = ["重新上传 jpg、jpeg 或 png 图片。", "确认任务上传目录没有被清空或移动。"]
            elif use_existing and passed_box_count <= 0:
                message = "未读取到有效的 YOLO 预标注文件。"
                suggestions = ["确认每张图片旁边有同名 .txt 标注文件。", "确认 .txt 每行格式为 class cx cy w h。", "取消“我已用 LabelImg 标注好数据”后让系统自动打标。"]
            elif raw_box_count <= 0:
                message = "YOLO-World 没有检测到任何候选框。"
                suggestions = ["检查需求确认页里的监测对象是否是图片中真实可见的物体。", "把提示词写得更具体，例如 person、helmet、forklift、car，而不是抽象事件。", "降低检测置信度后重试，或上传更清晰、目标更大的样本。"]
            elif passed_box_count <= 0 and not skip_quality_check:
                # 根据五维度拒绝原因 Top-1 给出针对性建议
                top_reason = None
                if qc_stats and qc_stats.get("reject_reasons"):
                    reasons = qc_stats["reject_reasons"]
                    top_reason = max(reasons, key=lambda k: reasons.get(k, 0)) if any(reasons.values()) else None
                if top_reason == "category_mismatch":
                    message = "YOLO-World 检出了候选框，但 Moondream 判定大部分框内容与类别不匹配（category_match 过低）。"
                    suggestions = ["返回需求确认页修改提示词，使其更接近图片中真实可见的物体。", "在意图确认页提供 5-10 张人工种子框示例，再训练专用模型。", "如确信类别正确，可临时降低 VQA 阈值或跳过 Moondream 质检。"]
                elif top_reason == "box_too_loose":
                    message = "候选框普遍过松，包含大量背景（tightness 过低）。"
                    suggestions = ["提高检测置信度阈值，过滤低置信度的大框。", "上传目标更大、构图更紧凑的样本。", "考虑后续用 SAM2 点提示精修边界。"]
                elif top_reason == "object_cut_off":
                    message = "候选框中目标普遍被截断（completeness 过低）。"
                    suggestions = ["上传目标完整出现在画面中的样本。", "调整图像尺寸 imgsz 让目标更易被完整识别。"]
                elif top_reason == "too_blurry":
                    message = "候选框普遍模糊（clarity 过低）。"
                    suggestions = ["上传更清晰、对焦良好的样本。", "避免运动模糊场景；改为静止帧。"]
                elif top_reason == "occluded":
                    message = "候选框中目标普遍被严重遮挡（no_occlusion_error 过低）。"
                    suggestions = ["上传目标可见度更高、未被遮挡的样本。"]
                else:
                    message = "YOLO-World 检出了候选框，但全部被 Moondream 五维度质检过滤。"
                    suggestions = ["先在环境准备页选择跳过 Moondream 质检，仅使用 YOLO-World 初筛。", "降低 VQA 质检阈值后重试。", "检查目标框是否过小、模糊、遮挡严重，或提示词与目标外观不匹配。"]
            elif labeled_count <= 0:
                message = "检测结果没有形成可训练的标注图片。"
                suggestions = ["调整监测对象提示词或上传已有 YOLO 标注后重试。"]
            else:
                message = "打标完成。"
                suggestions = []

            return {
                "image_count": image_total,
                "raw_box_count": raw_box_count,
                "detected_image_count": detected_image_count,
                "passed_box_count": passed_box_count,
                "quality_filtered_box_count": quality_filtered_box_count,
                "labeled_count": labeled_count,
                "mode": "existing_labels" if use_existing else "yolo_only" if skip_quality_check else "yolo_vqa",
                "message": message,
                "suggestions": suggestions,
                "qc_stats": qc_stats or None,
                "class_balance": class_balance or None,
            }

        image_total = len(list_image_files(resolved_image_dir))
        await _safe_ws_send(ws, _build_gpu_info_msg())
        await _safe_ws_send(ws, {
            "type": "progress",
            "stage": "detection",
            "current": 0,
            "total": image_total,
        })

        try:
            use_existing = payload.get("use_existing_labels", False)
            skip_quality_check = payload.get("skip_quality_check", False)
            full_classes = payload["classes"]

            # ── v9.0 P1：三路由引擎选择（在 run_detection 之前决定） ──
            from pipeline.engine_router import (
                select_engine,
                select_detection_engine,
                remap_raw_boxes,
                merge_raw_boxes,
            )

            # 新增：Eagle 引擎选择 (LocateAnything / YOLO-World)
            engine_preference = payload.get("engine_preference", "auto")
            eagle_engine_decision = select_detection_engine(
                full_classes,
                user_preference=engine_preference,
            )

            # 优先检查 Eagle 引擎（如果用户指定或自动选择）
            if eagle_engine_decision["engine"] == "locate_anything" and eagle_engine_decision["available"]:
                chosen_engine = "locate_anything"
            else:
                # 回退到原有三路由逻辑
                engine_decision = select_engine(
                    full_classes,
                    task_type=payload.get("task_type", ""),
                    user_preference=payload.get("engine_preference", "auto"),
                )
                chosen_engine = engine_decision["engine"] if not use_existing else "existing_labels"

            if not use_existing:
                await _safe_ws_send(ws, {
                    "type": "progress",
                    "stage": "detection",
                    "current": 0,
                    "total": image_total,
                    "engine": chosen_engine,
                    "message": f"打标引擎：{chosen_engine}（{engine_decision['reason']}）",
                })

            def _run_yolo_world(classes_subset):
                return run_detection(
                    image_dir=str(resolved_image_dir),
                    classes=classes_subset,
                    output_raw_dir=str(resolved_output_raw_dir),
                    conf_threshold=payload.get("conf_threshold", 0.25),
                    iou_threshold=payload.get("iou_threshold", 0.45),
                    batch_size=payload.get("batch_size", 4),
                    imgsz=payload.get("imgsz", 1280),
                    progress_callback=make_progress,
                    use_existing_labels=use_existing,
                )

            def _run_locate_anything(classes_subset):
                from pipeline.stage2_labeler import run_locate_anything_detection
                return run_locate_anything_detection(
                    image_dir=str(resolved_image_dir),
                    classes=classes_subset,
                    output_raw_dir=str(resolved_output_raw_dir),
                    conf_threshold=payload.get("conf_threshold", 0.25),
                    batch_size=payload.get("batch_size", 4),
                    progress_callback=make_progress,
                )

            def _run_gdino(classes_subset):
                from pipeline.grounding_dino_detector import run_grounding_dino_detection
                return run_grounding_dino_detection(
                    image_dir=str(resolved_image_dir),
                    classes=classes_subset,
                    output_raw_dir=str(resolved_output_raw_dir),
                    box_threshold=payload.get("conf_threshold", 0.25),
                    text_threshold=0.25,
                    progress_callback=make_progress,
                )

            # 根据选择的引擎执行检测
            if use_existing:
                raw_boxes = await asyncio.to_thread(_run_yolo_world, full_classes)
                engine_used = "existing_labels"
            elif chosen_engine == "locate_anything":
                raw_boxes = await asyncio.to_thread(_run_locate_anything, full_classes)
                engine_used = "locate_anything"
            elif chosen_engine == "grounding_dino":
                from pipeline.grounding_dino_detector import is_grounding_dino_available
                if is_grounding_dino_available():
                    raw_boxes = await asyncio.to_thread(_run_gdino, full_classes)
                    engine_used = "grounding_dino"
                else:
                    await _safe_ws_send(ws, {
                        "type": "progress",
                        "stage": "detection",
                        "current": 0,
                        "total": image_total,
                        "engine": "grounding_dino_unavailable",
                        "message": "Grounding DINO 不可用（未缓存或未启用 ENABLE_GROUNDING_DINO），回退 YOLO-World",
                    })
                    raw_boxes = await asyncio.to_thread(_run_yolo_world, full_classes)
                    engine_used = "yolo_world"
            else:  # hybrid
                from pipeline.grounding_dino_detector import is_grounding_dino_available

                strong_pairs = [(i, full_classes[i]) for i in engine_decision["strong_indices"]]
                weak_pairs = [(i, full_classes[i]) for i in engine_decision["weak_indices"]]

                raw_boxes = {}
                engine_used_parts = []

                if strong_pairs:
                    strong_idx_map = [i for i, _ in strong_pairs]
                    strong_classes = [c for _, c in strong_pairs]
                    sub_raw = await asyncio.to_thread(_run_yolo_world, strong_classes)
                    raw_boxes = merge_raw_boxes(raw_boxes, remap_raw_boxes(sub_raw, strong_idx_map))
                    engine_used_parts.append("yolo_world")

                if weak_pairs and is_grounding_dino_available():
                    weak_idx_map = [i for i, _ in weak_pairs]
                    weak_classes = [c for _, c in weak_pairs]
                    try:
                        sub_raw = await asyncio.to_thread(_run_gdino, weak_classes)
                        raw_boxes = merge_raw_boxes(raw_boxes, remap_raw_boxes(sub_raw, weak_idx_map))
                        engine_used_parts.append("grounding_dino")
                    except Exception as gd_exc:
                        await _safe_ws_send(ws, {
                            "type": "progress",
                            "stage": "detection",
                            "current": image_total,
                            "total": image_total,
                            "engine": "grounding_dino_failed",
                            "message": f"Hybrid 模式下 Grounding DINO 调用失败：{gd_exc}",
                        })
                elif weak_pairs:
                    await _safe_ws_send(ws, {
                        "type": "progress",
                        "stage": "detection",
                        "current": 0,
                        "total": image_total,
                        "engine": "grounding_dino_unavailable",
                        "message": (
                            f"Hybrid 模式发现 {len(weak_pairs)} 个 CLIP 弱词需要 Grounding DINO，"
                            "但未启用；这些类别可能召回为 0"
                        ),
                    })
                    # 仍然把弱词跑一遍 YOLO-World，作为兜底
                    weak_idx_map = [i for i, _ in weak_pairs]
                    weak_classes = [c for _, c in weak_pairs]
                    sub_raw = await asyncio.to_thread(_run_yolo_world, weak_classes)
                    raw_boxes = merge_raw_boxes(raw_boxes, remap_raw_boxes(sub_raw, weak_idx_map))

                engine_used = "hybrid(" + "+".join(engine_used_parts or ["yolo_world"]) + ")"

            # 安全网：任何情况下召回为 0 且 GDINO 可用 → 再尝试一次 GDINO（文档 P1-B 行为保留）
            fallback_engine = payload.get("fallback_engine", True)
            if (
                fallback_engine
                and not use_existing
                and engine_used == "yolo_world"
                and sum(len(b) for b in raw_boxes.values()) == 0
            ):
                from pipeline.grounding_dino_detector import is_grounding_dino_available
                if is_grounding_dino_available():
                    await _safe_ws_send(ws, {
                        "type": "progress",
                        "stage": "detection",
                        "current": 0,
                        "total": image_total,
                        "engine": "grounding_dino",
                        "message": "YOLO-World 召回 0，自动切换 Grounding DINO 备胎",
                    })
                    try:
                        raw_boxes = await asyncio.to_thread(_run_gdino, full_classes)
                        engine_used = "grounding_dino"
                    except Exception as gd_exc:
                        await _safe_ws_send(ws, {
                            "type": "progress",
                            "stage": "detection",
                            "current": image_total,
                            "total": image_total,
                            "engine": "grounding_dino_failed",
                            "message": f"Grounding DINO 调用失败：{gd_exc}",
                        })

            qc_stats: Optional[dict] = None
            if use_existing or skip_quality_check:
                # 预标注数据：跳过 Moondream VQA 质检，直接使用已有标注
                import json as _json

                await _safe_ws_send(ws, _build_gpu_info_msg())

                with open(f"{resolved_output_raw_dir}/raw_boxes.json", encoding="utf-8") as _f:
                    passed = _json.load(_f)

                # 发送质检跳过消息
                await _safe_ws_send(ws, {
                    "type": "progress",
                    "stage": "quality_check_skipped",
                    "current": image_total,
                    "total": image_total,
                })
            else:
                # 第二段：VQA 质检
                # 选择 VQA 引擎
                vqa_engine_pref = payload.get("vqa_engine_preference", "auto")
                from pipeline.engine_router import select_vqa_engine
                vqa_engine_decision = select_vqa_engine(user_preference=vqa_engine_pref)
                
                if vqa_engine_decision["engine"] == "eagle_vqa" and vqa_engine_decision["available"]:
                    from pipeline.stage2_labeler import run_eagle_vqa_quality_check
                    await _safe_ws_send(ws, {
                        "type": "progress",
                        "stage": "quality_check",
                        "current": 0,
                        "total": 1,
                        "engine": "eagle_vqa",
                        "message": f"VQA 引擎：Eagle2.5（{vqa_engine_decision['reason']}）",
                    })
                    passed, qc_stats = await asyncio.to_thread(
                        run_eagle_vqa_quality_check,
                        raw_boxes_path=f"{resolved_output_raw_dir}/raw_boxes.json",
                        min_confidence=payload.get("qa_threshold", 0.5),
                        progress_callback=make_progress,
                    )
                else:
                    await _safe_ws_send(ws, {
                        "type": "progress",
                        "stage": "quality_check",
                        "current": 0,
                        "total": 1,
                        "engine": "moondream",
                        "message": f"VQA 引擎：Moondream2（{vqa_engine_decision.get('reason', '自动选择')}）",
                    })
                    passed, qc_stats = await asyncio.to_thread(
                        run_quality_check,
                        raw_boxes_path=f"{resolved_output_raw_dir}/raw_boxes.json",
                        min_confidence=payload.get("qa_threshold", 0.5),
                        progress_callback=make_progress,
                        dim_thresholds=payload.get("qa_dim_thresholds"),
                    )

            await asyncio.to_thread(
                save_yolo_labels,
                passed,
                output_label_dir=resolved_output_label_dir,
                output_image_dir=resolved_output_image_dir,
            )

            class_balance = compute_class_balance(passed, payload.get("classes", []))
            result_payload = build_labeling_result(
                raw_boxes, passed, use_existing, skip_quality_check, qc_stats, class_balance
            )
            result_payload["engine_used"] = engine_used
            await _safe_ws_send(ws, {
                "type": "stage_complete",
                "stage": "labeling",
                "result": result_payload,
            })
        except CancelError:
            await _safe_ws_send(ws, {"type": "cancel_ack", "cancelled": True, "stage": "labeling"})

    elif cmd == "start_augmentation":
        from pipeline.stage25_augmentor import augment_dataset
        from utils.image_files import list_image_files

        def _send(msg: dict):
            asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, msg), loop)

        def aug_progress(current, total, _phase):
            _send({
                "type": "progress",
                "stage": "augmentation",
                "current": current,
                "total": total,
            })

        src_image_dir = _resolve_path(payload["src_image_dir"])
        src_label_dir = _resolve_path(payload["src_label_dir"])
        output_image_dir = _resolve_path(payload["output_image_dir"])
        output_label_dir = _resolve_path(payload["output_label_dir"])

        if not list_image_files(src_image_dir):
            await _safe_ws_send(ws, {
                "type": "error",
                "stage": "augmentation",
                "message": (
                    f"增强无法开始：上一阶段没有生成有效标注图片，源目录为空：{src_image_dir}。"
                    "请返回需求确认页调整监测对象提示词，或上传已有 YOLO 标注后重新打标。"
                ),
            })
            return

        result = await asyncio.to_thread(
            augment_dataset,
            src_image_dir=str(src_image_dir),
            src_label_dir=str(src_label_dir),
            output_image_dir=str(output_image_dir),
            output_label_dir=str(output_label_dir),
            target_count=payload["target_count"],
            strength=payload.get("strength", "medium"),
            enabled=payload.get("enabled"),
            progress_callback=aug_progress,
            delete_original=payload.get("delete_original", False),
            min_visibility=payload.get("min_visibility", 0.1),
            max_per_image=payload.get("max_per_image", 10),
        )

        await _safe_ws_send(ws, {
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

        def _send(msg: dict):
            asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, msg), loop)

        def training_progress_cb(status: dict):
            _send({
                "type": "training_progress",
                "currentEpoch": status.get("current_epoch", 0),
                "totalEpochs": status.get("total_epochs", 100),
                "currentMap": status.get("current_map", 0.0),
                "done": status.get("done", False),
            })

        task_id = payload.get("task_id", "")
        train_config = payload.get("train_config", {})

        # 解析前端传来的相对路径（如 '../backend/uploads/...' -> 项目根目录下的绝对路径）
        raw_dataset_dir = payload.get("dataset_dir", "dataset")
        dataset_dir = str(_resolve_path(raw_dataset_dir))
        # 增量训练时使用合并后的 data.yaml（由 startIncremental API 生成）
        raw_data_yaml = payload.get("data_yaml")
        data_yaml = str(_resolve_path(raw_data_yaml)) if raw_data_yaml else None

        try:
            artifacts = await asyncio.to_thread(
                trainer.train,
                dataset_dir=dataset_dir,
                train_config=train_config,
                progress_callback=training_progress_cb,
                data_yaml=data_yaml,
            )

            await _safe_ws_send(ws, {
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
            await _safe_ws_send(ws, {
                "type": "training_error",
                "message": str(e),
                "recoverable": train_config.get("resume_last", False),
            })
            if task_id:
                _notify_backend_error(task_id, str(e))

    elif cmd == "start_seed_training":
        from pipeline.seed_trainer import prepare_seed_dataset, run_seed_training
        from pipeline.seed_auto_labeler import run_seed_auto_labeling, merge_seed_and_auto_labels

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        task_id = payload.get("task_id", "")
        task_dir = str(_resolve_upload_root() / task_id) if task_id else ""
        class_names = payload.get("class_names", [])

        if not task_dir or not Path(task_dir).exists():
            await _safe_ws_send(ws, {"type": "error", "message": f"Task directory not found: {task_dir}"})
            return

        def _send_seed(msg: dict):
            asyncio.run_coroutine_threadsafe(_safe_ws_send(ws, msg), loop)

        try:
            # Phase 1: Prepare dataset
            await _safe_ws_send(ws, {"type": "seed_training_started"})
            prep = await loop.run_in_executor(
                None,
                lambda: prepare_seed_dataset(task_dir, class_names),
            )

            # Phase 2: Train seed model
            def _train_progress(msg):
                _send_seed(msg)

            result = await loop.run_in_executor(
                None,
                lambda: run_seed_training(
                    dataset_dir=prep["dataset_dir"],
                    class_names=class_names,
                    progress_callback=_train_progress,
                    device=get_device(),
                ),
            )

            await _safe_ws_send(ws, {
                "type": "seed_training_complete",
                "seed_model_path": result["seed_model_path"],
                "best_map": result["best_map"],
                "training_time_seconds": result["training_time_seconds"],
            })

            # Phase 3: Auto-label remaining images
            def _auto_label_progress(msg):
                _send_seed(msg)

            auto_result = await loop.run_in_executor(
                None,
                lambda: run_seed_auto_labeling(
                    seed_model_path=result["seed_model_path"],
                    task_dir=task_dir,
                    class_names=class_names,
                    progress_callback=_auto_label_progress,
                    high_conf=payload.get("high_conf", 0.5),
                    low_conf=payload.get("low_conf", 0.25),
                ),
            )

            # Phase 4: Merge labels
            merge_result = await loop.run_in_executor(
                None,
                lambda: merge_seed_and_auto_labels(task_dir, class_names),
            )

            await _safe_ws_send(ws, {
                "type": "seed_auto_label_complete",
                "auto_accepted": auto_result["auto_accepted"],
                "needs_review": auto_result["needs_review"],
                "no_detection": auto_result["no_detection"],
                "avg_confidence": auto_result["avg_confidence"],
                "total_labeled": merge_result["total_labeled"],
            })

        except Exception as e:
            await _safe_ws_send(ws, {"type": "error", "message": f"Seed training failed: {str(e)}"})

    elif cmd == "cancel":
        cancel_current_stage()
        await _safe_ws_send(ws, {"type": "cancel_ack", "cancelled": True})

    elif cmd == "ping":
        await _safe_ws_send(ws, {"type": "pong"})


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
            headers={"X-Worker-Secret": _WORKER_CALLBACK_SECRET},
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
            headers={"X-Worker-Secret": _WORKER_CALLBACK_SECRET},
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


@app.get("/model-status")
def get_model_status():
    status = _refresh_model_status_only()

    with _MODEL_PREP_LOCK:
        _MODEL_PREP_STATE["status"] = status
        return dict(_MODEL_PREP_STATE)


@app.post("/prepare-models")
async def prepare_models(payload: dict):
    include_moondream = bool(payload.get("include_moondream", False))
    include_locate_anything = bool(payload.get("include_locate_anything", False))
    include_eagle_vqa = bool(payload.get("include_eagle_vqa", False))

    with _MODEL_PREP_LOCK:
        if _MODEL_PREP_STATE["running"]:
            return dict(_MODEL_PREP_STATE)
        _MODEL_PREP_STATE["running"] = True
        _MODEL_PREP_STATE["include_moondream"] = include_moondream
        _MODEL_PREP_STATE["include_locate_anything"] = include_locate_anything
        _MODEL_PREP_STATE["include_eagle_vqa"] = include_eagle_vqa
        _MODEL_PREP_STATE["error"] = None
        _MODEL_PREP_STATE["steps"]["yolo_world"] = {"status": "running", "message": "准备 YOLO-World"}
        _MODEL_PREP_STATE["steps"]["clip"] = {"status": "pending", "message": "等待 YOLO-World 类别编码"}
        if include_moondream:
            _MODEL_PREP_STATE["steps"]["moondream"] = {"status": "pending", "message": "等待下载 Moondream2"}
        else:
            _MODEL_PREP_STATE["steps"]["moondream"] = {"status": "optional", "message": "已选择跳过 Moondream VQA"}
        if include_locate_anything:
            _MODEL_PREP_STATE["steps"]["locate_anything"] = {"status": "pending", "message": "等待下载 LocateAnything-3B"}
        else:
            _MODEL_PREP_STATE["steps"]["locate_anything"] = {"status": "optional", "message": "可选：更快的检测速度"}
        if include_eagle_vqa:
            _MODEL_PREP_STATE["steps"]["eagle_vqa"] = {"status": "pending", "message": "等待下载 Eagle2.5-8B"}
        else:
            _MODEL_PREP_STATE["steps"]["eagle_vqa"] = {"status": "optional", "message": "可选：更强的质检能力"}

    asyncio.create_task(asyncio.to_thread(
        _run_model_prepare, include_moondream, include_locate_anything, include_eagle_vqa
    ))

    with _MODEL_PREP_LOCK:
        return dict(_MODEL_PREP_STATE)


@app.post("/detection-preview")
async def detection_preview(payload: dict):
    from pipeline.stage2_labeler import run_detection
    from utils.image_files import list_image_files
    import cv2

    image_dir = _resolve_path(payload["image_dir"])
    classes = payload.get("classes", [])
    max_images = max(1, min(int(payload.get("max_images", 2)), 4))
    accepted_conf_threshold = float(payload.get("conf_threshold", 0.12))
    diagnostic_conf_threshold = float(payload.get("diagnostic_conf_threshold", 0.03))
    iou_threshold = float(payload.get("iou_threshold", 0.45))
    imgsz = int(payload.get("imgsz", 1280))
    prompts_used = []
    for class_item in classes:
        for prompt in [*(class_item.get("prompt_aliases") or []), class_item.get("prompt", "")]:
            value = str(prompt or "").strip()
            if value and value not in prompts_used:
                prompts_used.append(value)

    image_paths = list_image_files(image_dir)[:max_images]
    if not image_paths:
        return {
            "total_images": 0,
            "raw_box_count": 0,
            "results": [],
            "message": f"预览目录中没有可处理图片: {image_dir}",
        }

    with tempfile.TemporaryDirectory(prefix="cv_preview_") as tmp:
        preview_dir = Path(tmp) / "images"
        raw_dir = Path(tmp) / "raw"
        preview_dir.mkdir(parents=True, exist_ok=True)
        for src in image_paths:
            dst = preview_dir / src.name
            try:
                dst.symlink_to(src)
            except Exception:
                import shutil
                shutil.copy(src, dst)

        raw_boxes = await asyncio.to_thread(
            run_detection,
            image_dir=str(preview_dir),
            classes=classes,
            output_raw_dir=str(raw_dir),
            conf_threshold=diagnostic_conf_threshold,
            iou_threshold=iou_threshold,
            batch_size=1,
            imgsz=imgsz,
            progress_callback=None,
            use_existing_labels=False,
        )

        results = []
        for img_path, boxes in raw_boxes.items():
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            for box in boxes:
                accepted = float(box.get("conf", 0)) >= accepted_conf_threshold
                cx, cy, bw, bh = box["bbox_xywhn"]
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w - 1, int((cx + bw / 2) * w))
                y2 = min(h - 1, int((cy + bh / 2) * h))
                color = (45, 114, 239) if accepted else (0, 154, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{box.get('class_name', 'target')} {float(box.get('conf', 0)):.2f}"
                cv2.putText(img, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            ok, encoded = cv2.imencode(".jpg", img)
            if not ok:
                continue
            results.append({
                "image_name": Path(img_path).name,
                "image_base64": base64.b64encode(encoded.tobytes()).decode("utf-8"),
                "detections": [
                    {
                        "class_name": box.get("class_name", ""),
                        "prompt": box.get("prompt", ""),
                        "confidence": float(box.get("conf", 0)),
                        "accepted": float(box.get("conf", 0)) >= accepted_conf_threshold,
                        "bbox_xywhn": box.get("bbox_xywhn", []),
                    }
                    for box in boxes
                ],
            })

        candidate_box_count = sum(len(boxes) for boxes in raw_boxes.values())
        accepted_box_count = sum(
            1
            for boxes in raw_boxes.values()
            for box in boxes
            if float(box.get("conf", 0)) >= accepted_conf_threshold
        )
        if accepted_box_count > 0:
            message = "预览完成，但请确认框的位置和目标语义是否符合需求"
            suggestions = [
                "如果框到了车轮、车身、水印或其他物体上，说明开集检测词与真实目标仍不匹配，不能直接进入训练。",
                "三角木/轮挡通常是小目标，建议至少人工确认或修正 5-10 张种子框后再训练专用模型。",
            ]
        elif candidate_box_count > 0:
            message = "YOLO-World 看到了低分候选，但还没达到正式打标阈值"
            suggestions = [
                "当前目标可能太小、被遮挡或只占画面很小区域。",
                "低分框只适合作为诊断线索，不能直接当作训练标签。",
                "请确认英文提示词是否准确描述目标外观，例如 wheel chock、wedge block behind wheel。",
            ]
        else:
            message = "YOLO-World 在低阈值下也没有找到候选框"
            suggestions = [
                "请把检测对象改成图片里清晰可见的英文物体短词。",
                "如果目标很小，建议上传更近距离或目标更大的样图。",
                "也可以先人工标 5-10 张作为种子数据，再训练专用模型。",
            ]
        return {
            "total_images": len(image_paths),
            "raw_box_count": accepted_box_count,
            "accepted_box_count": accepted_box_count,
            "candidate_box_count": candidate_box_count,
            "diagnostic_conf_threshold": diagnostic_conf_threshold,
            "accepted_conf_threshold": accepted_conf_threshold,
            "imgsz": imgsz,
            "prompts_used": prompts_used,
            "results": results,
            "message": message,
            "suggestions": suggestions,
        }


@app.post("/check-health")
def check_health():
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT)
