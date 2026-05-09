import copy
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional

from models.database import get_db
from routers.auth import require_auth
from services.algorithm_package_service import export_task_algorithm_package
from services.algorithm_planner import build_algorithm_plan
from services.algorithm_preview_service import preview_algorithm_events
from services.model_registry import get_model_registry, infer_device_tier
from services.pipeline_compiler import compile_algorithm_pipeline
from services.settings_manager import get_settings, decrypt_value as _decrypt_value
from services.task_access import get_task_for_user
from services.training_recommendation_service import enrich_pipeline_with_training_recommendation
from services.vlm_algorithm_planner import build_vlm_algorithm_plan, revise_vlm_algorithm_plan
from types import SimpleNamespace as _SimpleNamespace

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/algorithm", tags=["algorithm"])


def _build_reasoning_settings_ns(settings_row) -> _SimpleNamespace:
    """把 DB UserSettings 行打包成 reasoning_adapter 工厂期望的对象（API key 已解密）。"""
    enabled = getattr(settings_row, "reasoning_enabled", None)
    if enabled is None:
        enabled = True
    return _SimpleNamespace(
        reasoning_enabled=bool(enabled),
        reasoning_provider=(getattr(settings_row, "reasoning_provider", "") or "").strip() or "deepseek",
        reasoning_base_url=getattr(settings_row, "reasoning_base_url", "") or "",
        reasoning_api_key=_decrypt_value(getattr(settings_row, "reasoning_api_key_encrypted", "") or ""),
        reasoning_model=getattr(settings_row, "reasoning_model", "") or "",
    )


class AlgorithmPlanRequest(BaseModel):
    task_id: str
    user_description: str
    vlm_result: Optional[dict] = None
    runtime_capability: Optional[dict] = None
    images_base64: Optional[list[str]] = None
    gpu_type: Optional[str] = None
    platform: Optional[str] = None
    device_description: Optional[str] = None
    image_count: int = 0
    use_vlm_planner: bool = True
    algorithm_hints: Optional[dict] = None


class AlgorithmPlanResponse(BaseModel):
    task_id: str
    status: str
    algorithm_plan: dict
    pipeline_config: Optional[dict] = None


class AlgorithmPlanConfirmRequest(BaseModel):
    region_overrides: list[dict] = []
    runtime_capability: Optional[dict] = None


class AlgorithmPlanNegotiateRequest(BaseModel):
    negotiation_summary: dict
    offline_evaluation: Optional[dict] = None


class AlgorithmPlanNegotiateResponse(BaseModel):
    task_id: str
    negotiation_summary: dict
    offline_evaluation: Optional[dict] = None


class AlgorithmPlanReviseRequest(BaseModel):
    user_feedback: str
    runtime_capability: Optional[dict] = None
    gpu_type: Optional[str] = None
    platform: Optional[str] = None
    device_description: Optional[str] = None


class AlgorithmPreviewRequest(BaseModel):
    region_overrides: list[dict] = []
    sample_boxes: list[dict] = []
    observation_frames: list[dict] = []


def _load_sample_images_for_vlm(task, max_images: int = 3) -> list[str]:
    """从 task.image_dir 读取最多 N 张样本图并返回 base64 列表，用于方案协商时给 VLM 提供视觉上下文。"""
    import base64
    from pathlib import Path

    image_dir = getattr(task, "image_dir", None)
    if not image_dir or not Path(image_dir).exists():
        return []

    exts = {".jpg", ".jpeg", ".png"}
    try:
        files = sorted(
            (p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in exts),
            key=lambda p: p.name,
        )[:max_images]
        result = []
        for f in files:
            try:
                data = f.read_bytes()
                result.append(base64.b64encode(data).decode("ascii"))
            except OSError:
                continue
        return result
    except Exception:
        logger.exception("Failed to load sample images from %s", image_dir)
        return []


def _recommended_pipeline_config(
    *,
    algorithm_plan: dict,
    db: Session,
    current_user: dict,
    runtime_capability: Optional[dict] = None,
    base_pipeline_config: Optional[dict] = None,
) -> dict:
    settings = get_settings(db, current_user["user_id"])
    pipeline_config = base_pipeline_config or compile_algorithm_pipeline(algorithm_plan)
    return enrich_pipeline_with_training_recommendation(
        algorithm_plan=algorithm_plan,
        pipeline_config=pipeline_config,
        settings=settings,
        runtime_capability=runtime_capability,
    )


@router.post("/plan")
def generate_algorithm_plan(
    payload: AlgorithmPlanRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, payload.task_id, current_user)
    settings = get_settings(db, current_user["user_id"])

    plan: dict
    if payload.use_vlm_planner:
        try:
            from services.vlm_adapter import VLMAdapter
            from services.settings_manager import decrypt_value

            vlm_adapter = VLMAdapter(
                provider=settings.vlm_provider or "openai",
                base_url=settings.vlm_base_url or "",
                api_key=decrypt_value(settings.vlm_api_key_encrypted or ""),
                api_format=settings.vlm_api_format,
                model=settings.vlm_model,
                temperature=settings.vlm_temperature,
                top_p=settings.vlm_top_p,
            )
            gpu_type = payload.gpu_type or getattr(settings, "default_gpu_type", None)
            plan = build_vlm_algorithm_plan(
                user_description=payload.user_description,
                vlm_result=payload.vlm_result,
                vlm_adapter=vlm_adapter,
                images_base64=payload.images_base64,
                gpu_type=gpu_type,
                platform=payload.platform,
                device_description=payload.device_description or "",
                image_count=payload.image_count,
                reasoning_settings=_build_reasoning_settings_ns(settings),
                algorithm_hints=payload.algorithm_hints,
            )
        except Exception as e:
            logger.warning("VLM planner failed, falling back to rule-based: %s", e)
            plan = build_algorithm_plan(
                user_description=payload.user_description,
                vlm_result=payload.vlm_result,
            )
    else:
        plan = build_algorithm_plan(
            user_description=payload.user_description,
            vlm_result=payload.vlm_result,
        )

    preview_pipeline = _recommended_pipeline_config(
        algorithm_plan=plan,
        db=db,
        current_user=current_user,
        runtime_capability=payload.runtime_capability,
    )
    task.algorithm_plan = plan
    task.algorithm_plan_status = "draft"
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanResponse(
            task_id=task.id,
            status=task.algorithm_plan_status,
            algorithm_plan=plan,
            pipeline_config=preview_pipeline,
        ).model_dump(),
    }


@router.get("/plan/{task_id}")
def get_algorithm_plan(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        return {"code": 0, "msg": "ok", "data": None}

    preview_pipeline = _recommended_pipeline_config(
        algorithm_plan=task.algorithm_plan,
        db=db,
        current_user=current_user,
        base_pipeline_config=task.pipeline_config,
    )

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanResponse(
            task_id=task.id,
            status=task.algorithm_plan_status or "draft",
            algorithm_plan=task.algorithm_plan,
            pipeline_config=preview_pipeline,
        ).model_dump(),
    }


@router.post("/plan/{task_id}/confirm")
def confirm_algorithm_plan(
    task_id: str,
    payload: Optional[AlgorithmPlanConfirmRequest] = None,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=404, detail="Algorithm plan not found")

    region_overrides = payload.region_overrides if payload else []
    runtime_capability = payload.runtime_capability if payload else None
    algorithm_plan = copy.deepcopy(task.algorithm_plan)

    if region_overrides:
        overrides_by_id = {
            item.get("region_id"): item
            for item in region_overrides
            if item.get("region_id")
        }
        merged_regions = []
        for region in algorithm_plan.get("regions", []):
            override = overrides_by_id.get(region.get("region_id"))
            if override:
                merged_region = dict(region)
                merged_region.update(override)
                merged_regions.append(merged_region)
            else:
                merged_regions.append(region)
        algorithm_plan["regions"] = merged_regions

    task.algorithm_plan = algorithm_plan
    task.algorithm_plan_status = "confirmed"
    task.pipeline_config = _recommended_pipeline_config(
        algorithm_plan=algorithm_plan,
        db=db,
        current_user=current_user,
        runtime_capability=runtime_capability,
    )
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanResponse(
            task_id=task.id,
            status=task.algorithm_plan_status,
            algorithm_plan=task.algorithm_plan,
            pipeline_config=task.pipeline_config,
        ).model_dump(),
    }


@router.post("/plan/{task_id}/revise")
def revise_algorithm_plan(
    task_id: str,
    payload: AlgorithmPlanReviseRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """根据用户反馈让 VLM 重新生成算法方案（方案协商对话）"""
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=404, detail="Algorithm plan not found")

    settings = get_settings(db, current_user["user_id"])
    existing_plan = copy.deepcopy(task.algorithm_plan)
    revision_history = list(existing_plan.get("revision_history", []))

    # ── 保存修订前快照，供后续回滚 ──
    snapshots: list = list(existing_plan.get("revision_snapshots", []))
    snapshot_entry = {
        "version": len(snapshots) + 1,
        "summary_zh": existing_plan.get("summary_zh") or existing_plan.get("summary", ""),
        "timestamp": __import__("time").time(),
        "plan": {k: v for k, v in existing_plan.items() if k not in ("revision_snapshots",)},
    }
    snapshots.append(snapshot_entry)

    from services.vlm_adapter import VLMAdapter
    from services.settings_manager import decrypt_value

    # 加载原始样本图片供 VLM 参考（最多 3 张，从 task.image_dir）
    sample_images_b64 = _load_sample_images_for_vlm(task)

    try:
        vlm_adapter = VLMAdapter(
            provider=settings.vlm_provider or "openai",
            base_url=settings.vlm_base_url or "",
            api_key=decrypt_value(settings.vlm_api_key_encrypted or ""),
            api_format=settings.vlm_api_format,
            model=settings.vlm_model,
            temperature=settings.vlm_temperature,
            top_p=settings.vlm_top_p,
        )
        gpu_type = payload.gpu_type or getattr(settings, "default_gpu_type", None)
        new_plan = revise_vlm_algorithm_plan(
            existing_plan=existing_plan,
            user_feedback=payload.user_feedback,
            vlm_adapter=vlm_adapter,
            reasoning_settings=_build_reasoning_settings_ns(settings),
            revision_history=revision_history,
            gpu_type=gpu_type,
            platform=payload.platform,
            device_description=payload.device_description or "",
            images_base64=sample_images_b64,
        )
    except Exception as e:
        logger.exception("Plan revision failed")
        raise HTTPException(status_code=500, detail=f"方案修订失败：{e}")

    # 追加本轮对话到历史
    revision_history.append({"role": "user", "content": payload.user_feedback})
    revision_history.append({
        "role": "assistant",
        "content": new_plan.get("summary_zh") or new_plan.get("summary", ""),
    })
    new_plan["revision_history"] = revision_history
    new_plan["revision_snapshots"] = snapshots

    # 重新编译 pipeline
    preview_pipeline = _recommended_pipeline_config(
        algorithm_plan=new_plan,
        db=db,
        current_user=current_user,
        runtime_capability=payload.runtime_capability,
    )

    task.algorithm_plan = new_plan
    task.algorithm_plan_status = "draft"  # 协商中，需要重新确认
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanResponse(
            task_id=task.id,
            status=task.algorithm_plan_status,
            algorithm_plan=new_plan,
            pipeline_config=preview_pipeline,
        ).model_dump(),
    }


class RollbackRequest(BaseModel):
    version: int


@router.post("/plan/{task_id}/rollback")
def rollback_algorithm_plan(
    task_id: str,
    payload: RollbackRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """回滚到指定版本的方案快照"""
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=404, detail="Algorithm plan not found")

    snapshots = task.algorithm_plan.get("revision_snapshots", [])
    target = None
    for s in snapshots:
        if s.get("version") == payload.version:
            target = s
            break
    if not target or "plan" not in target:
        raise HTTPException(status_code=404, detail=f"版本 {payload.version} 的快照不存在")

    restored = copy.deepcopy(target["plan"])
    # 保留快照历史（不丢失其他版本记录）
    restored["revision_snapshots"] = snapshots
    # 追加一条回滚记录到 revision_history
    history = list(restored.get("revision_history", []))
    history.append({
        "role": "system",
        "content": f"用户回滚到版本 {payload.version}",
    })
    restored["revision_history"] = history

    preview_pipeline = _recommended_pipeline_config(
        algorithm_plan=restored,
        db=db,
        current_user=current_user,
    )

    task.algorithm_plan = restored
    task.algorithm_plan_status = "draft"
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanResponse(
            task_id=task.id,
            status=task.algorithm_plan_status,
            algorithm_plan=restored,
            pipeline_config=preview_pipeline,
        ).model_dump(),
    }


@router.post("/plan/{task_id}/negotiate")
def negotiate_algorithm_plan(
    task_id: str,
    payload: AlgorithmPlanNegotiateRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=404, detail="Algorithm plan not found")

    task.negotiation_summary = payload.negotiation_summary
    if payload.offline_evaluation is not None:
        task.offline_evaluation = payload.offline_evaluation
    db.commit()

    return {
        "code": 0,
        "msg": "ok",
        "data": AlgorithmPlanNegotiateResponse(
            task_id=task.id,
            negotiation_summary=task.negotiation_summary,
            offline_evaluation=task.offline_evaluation,
        ).model_dump(),
    }


@router.post("/package/{task_id}")
def export_algorithm_package(task_id: str, current_user: dict = Depends(require_auth), db: Session = Depends(get_db)):
    task = get_task_for_user(db, task_id, current_user)
    if not task.pipeline_config:
        raise HTTPException(status_code=400, detail="Pipeline config not ready")

    package_result = export_task_algorithm_package(
        task_id=task.id,
        pipeline_config=task.pipeline_config,
        artifacts=task.artifact_paths or {},
    )
    merged_artifacts = dict(task.artifact_paths or {})
    merged_artifacts.update(
        {
            "pipeline.json": package_result["pipeline_path"],
            "manifest.json": package_result["manifest_path"],
            "README.md": package_result["readme_path"],
            "run_pipeline.py": package_result["entrypoint_path"],
            "sample_input.json": package_result.get("sample_input_path"),
            "sample_output.json": package_result.get("sample_output_path"),
            "runtime_support.py": package_result.get("runtime_support_path"),
        }
    )
    task.artifact_paths = merged_artifacts
    db.commit()

    return {"code": 0, "msg": "ok", "data": package_result}


@router.post("/preview/{task_id}")
def preview_algorithm(
    task_id: str,
    payload: AlgorithmPreviewRequest,
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=404, detail="Algorithm plan not found")

    preview_result = preview_algorithm_events(
        algorithm_plan=task.algorithm_plan,
        sample_boxes=payload.sample_boxes,
        observation_frames=payload.observation_frames,
        region_overrides=payload.region_overrides,
    )
    return {"code": 0, "msg": "ok", "data": preview_result}


# ---------------------------------------------------------------------------
# Model Registry 端点
# ---------------------------------------------------------------------------

@router.get("/models")
def list_available_models(
    device_tier: Optional[str] = None,
    task_type: Optional[str] = None,
    family: Optional[str] = None,
    current_user: dict = Depends(require_auth),
):
    registry = get_model_registry()
    models = registry.list_models(
        family=family,
        task_type=task_type,
        max_device_tier=device_tier,
    )
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "models": [{
                "model_id": m.model_id,
                "family": m.family,
                "variant": m.variant,
                "display_name": m.display_name,
                "display_name_zh": m.display_name_zh,
                "task_types": m.task_types,
                "params_m": m.params_m,
                "map50_coco": m.map50_coco,
                "fps_gpu": m.fps_gpu,
                "fps_cpu": m.fps_cpu,
                "min_device_tier": m.min_device_tier,
                "recommended_device_tiers": m.recommended_device_tiers,
                "description_zh": m.description_zh,
                "strengths": m.strengths,
                "weaknesses": m.weaknesses,
                "use_cases": m.use_cases,
                "export_formats": m.export_formats,
            } for m in models],
            "total": len(models),
        },
    }


@router.get("/models/cached")
def list_cached_models(current_user: dict = Depends(require_auth)):
    registry = get_model_registry()
    cached = registry.list_cached_models()
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "cached_models": [{
                "cache_id": c.cache_id,
                "source_model_id": c.source_model_id,
                "classes": c.classes,
                "scenario_type": c.scenario_type,
                "map50": c.map50,
                "trained_at": c.trained_at,
                "reuse_count": c.reuse_count,
            } for c in cached],
            "total": len(cached),
        },
    }


@router.get("/device-tier")
def detect_device_tier(
    gpu_type: Optional[str] = None,
    platform: Optional[str] = None,
    current_user: dict = Depends(require_auth),
):
    tier = infer_device_tier(gpu_type, platform)
    return {"code": 0, "msg": "ok", "data": {"device_tier": tier}}


# ---------------------------------------------------------------------------
# Video Validation 端点
# ---------------------------------------------------------------------------

@router.post("/validate-video/{task_id}")
async def validate_video(
    task_id: str,
    video: UploadFile = File(...),
    current_user: dict = Depends(require_auth),
    db: Session = Depends(get_db),
):
    task = get_task_for_user(db, task_id, current_user)
    if not task.algorithm_plan:
        raise HTTPException(status_code=400, detail="请先生成算法方案")

    import tempfile, os
    from services.video_processor import validate_video_with_vlm
    from services.vlm_adapter import VLMAdapter
    from services.settings_manager import decrypt_value

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename or ".mp4")[1]) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = validate_video_with_vlm(
            video_path=tmp_path,
            algorithm_plan=task.algorithm_plan,
            vlm_adapter=vlm_adapter,
        )
        task.offline_evaluation = result
        db.commit()
        return {"code": 0, "msg": "ok", "data": result}
    finally:
        os.unlink(tmp_path)
