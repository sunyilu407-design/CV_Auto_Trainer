from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models.database import get_db
from models.database import SessionLocal
from models.db import Task
from routers.auth import require_auth
from services.task_access import get_task_for_user
from services.train_dispatcher import TrainDispatcher, TrainMode
from services.multi_model_orchestrator import MultiModelTrainingOrchestrator, LocalMultiModelNotSupported
import threading

router = APIRouter(prefix="/api/training", tags=["training"])


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
    steps = []
    instance_id = recovery.get("instance_id") or "<你的实例ID>"
    console_url = recovery.get("autodl_console_url") or f"https://www.autodl.com/console/instance/{instance_id}"

    if recovery.get("ssh_retrieval_failed"):
        steps.append({
            "step": 1,
            "title": "登录 AutoDL 控制台",
            "description": "访问控制台查看该实例的 SSH 连接信息（实例仍在运行，继续计费）",
            "action": f"打开：{console_url}",
        })
        steps.append({
            "step": 2,
            "title": "决定是否继续训练",
            "description": "若不再需要训练，直接在控制台关机；若想继续训练，参考下方命令",
            "action": "",
        })
    else:
        ssh_cmd = f"ssh -p {recovery.get('ssh_port', 22)} {recovery.get('ssh_username', 'root')}@{recovery.get('ssh_host', '<host>')}"
        pwd_note = recovery.get("ssh_password_masked") or "（请前往 AutoDL 控制台查看）"
        steps.append({
            "step": 1,
            "title": "通过 SSH 连接实例",
            "description": f"打开终端，输入下方命令。密码（脱敏）：{pwd_note}。完整密码请在 AutoDL 控制台查看。",
            "action": ssh_cmd,
        })
        steps.append({
            "step": 2,
            "title": "进入工作目录",
            "description": "数据集已经上传到 /root/dataset，训练输出目录为 /root/training_output",
            "action": "cd /root && ls",
        })
        if recovery.get("train_command"):
            steps.append({
                "step": 3,
                "title": "手动执行训练",
                "description": "原本由系统发起的训练命令。在 screen 里运行防止 SSH 断开",
                "action": f"screen -S train\n{recovery['train_command']}",
            })
            steps.append({
                "step": 4,
                "title": "下载权重文件",
                "description": "训练完成后，best.pt 位于 /root/training_output/exp/weights/best.pt，可用 scp 下载",
                "action": f"scp -P {recovery.get('ssh_port', 22)} {recovery.get('ssh_username', 'root')}@{recovery.get('ssh_host', '<host>')}:/root/training_output/exp/weights/best.pt ~/Downloads/",
            })
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
