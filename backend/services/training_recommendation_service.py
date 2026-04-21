from copy import deepcopy
from typing import Any, Dict, Iterable, Optional

from services.model_registry import get_model_registry, infer_device_tier


_DEFAULT_EXPORT_FORMATS = ["onnx"]
_DEVICE_EXPORT_FORMATS = {
    "mps": ["onnx", "coreml"],
    "cuda": ["onnx", "engine", "openvino"],
    "cpu": ["onnx", "openvino"],
}


def _normalized_gpu_type(settings: Optional[Any]) -> str:
    gpu_type = getattr(settings, "default_gpu_type", None) or "RTX 4090"
    return str(gpu_type)


def _detect_preferred_device(settings: Optional[Any], runtime_capability: Optional[Dict[str, Any]]) -> str:
    runtime_device = (runtime_capability or {}).get("preferred_device")
    if runtime_device:
        return str(runtime_device).lower()

    gpu_type = _normalized_gpu_type(settings).lower()
    if "apple" in gpu_type or "mps" in gpu_type:
        return "mps"
    if any(token in gpu_type for token in ("rtx", "nvidia", "cuda", "a100", "h100", "l40")):
        return "cuda"
    return "cpu"


def _derive_complexity(algorithm_plan: Dict[str, Any], pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    targets = algorithm_plan.get("targets", [])
    rules = pipeline_config.get("rules", [])
    regions = pipeline_config.get("regions", [])
    training_requirements = algorithm_plan.get("training_requirements", {})

    has_temporal_rule = any(int(rule.get("duration_seconds", 0)) > 0 for rule in rules)
    has_region_rule = any(rule.get("region_id") for rule in rules)
    tracking_required = bool(training_requirements.get("tracking_required")) or bool(pipeline_config.get("trackers"))

    temporal_or_region_intensity = 0
    if has_region_rule:
        temporal_or_region_intensity += 1
    if has_temporal_rule:
        temporal_or_region_intensity += 2

    score = 0
    score += len(targets)
    score += len(rules)
    score += 2 if tracking_required else 0
    score += temporal_or_region_intensity

    if len(targets) >= 4 or score >= 10:
        complexity_level = "high"
    elif len(targets) >= 2 or score >= 5:
        complexity_level = "medium"
    else:
        complexity_level = "low"

    return {
        "target_count": len(targets),
        "rule_count": len(rules),
        "region_count": len(regions),
        "tracking_required": tracking_required,
        "has_temporal_rule": has_temporal_rule,
        "has_region_rule": has_region_rule,
        "temporal_or_region_intensity": temporal_or_region_intensity,
        "score": score,
        "level": complexity_level,
    }


def _recommended_model(
    complexity: Dict[str, Any],
    settings: Optional[Any],
    algorithm_plan: Optional[Dict[str, Any]] = None,
    runtime_capability: Optional[Dict[str, Any]] = None,
) -> tuple[str, str]:
    # 1) 优先使用 VLM 算法规划中推荐的模型
    if algorithm_plan and algorithm_plan.get("model_pipeline"):
        for step in algorithm_plan["model_pipeline"]:
            if step.get("role") in ("primary_detector", "secondary_detector"):
                model_id = step.get("recommended_model_id")
                if model_id:
                    return model_id, "vlm_plan"

    # 2) 使用 model_registry 根据设备等级选择
    gpu_type = _normalized_gpu_type(settings)
    platform = (runtime_capability or {}).get("platform")
    device_tier = infer_device_tier(gpu_type, platform)
    registry = get_model_registry()
    candidates = registry.list_models(task_type="detection", max_device_tier=device_tier)
    if not candidates:
        candidates = registry.list_models(task_type="detection")

    if candidates:
        candidates.sort(key=lambda m: (m.map50_coco or 0), reverse=True)
        if complexity["level"] == "high":
            pick = next((m for m in candidates if m.variant in ("medium", "large")), candidates[0])
        elif complexity["level"] == "medium":
            pick = next((m for m in candidates if m.variant in ("small", "medium")), candidates[0])
        else:
            pick = next((m for m in candidates if m.variant in ("nano", "small")), candidates[0])
        return pick.model_id, "model_registry"

    # 3) 兜底
    if complexity["level"] == "high":
        return "yolo11m.pt", "algorithm"
    default_model = getattr(settings, "default_model", None)
    if default_model and complexity["level"] == "low":
        return default_model, "user_default"
    return "yolo11s.pt", "algorithm"


def _local_training_available(runtime_capability: Optional[Dict[str, Any]], settings: Optional[Any]) -> bool:
    if runtime_capability and "local_training_available" in runtime_capability:
        return bool(runtime_capability["local_training_available"])
    return getattr(settings, "default_train_mode", "local") == "local"


def _supports_cloud_training(runtime_capability: Optional[Dict[str, Any]]) -> bool:
    if runtime_capability and "supports_cloud_training" in runtime_capability:
        return bool(runtime_capability["supports_cloud_training"])
    return True


def _recommended_train_mode(
    complexity: Dict[str, Any],
    settings: Optional[Any],
    runtime_capability: Optional[Dict[str, Any]],
) -> tuple[str, str]:
    if _local_training_available(runtime_capability, settings):
        return "local", "runtime"
    if _supports_cloud_training(runtime_capability):
        return "cloud", "runtime"
    return getattr(settings, "default_train_mode", "local"), "user_default"


def _available_export_formats(
    settings: Optional[Any],
    runtime_capability: Optional[Dict[str, Any]],
) -> list[str]:
    declared = (runtime_capability or {}).get("available_export_formats")
    if isinstance(declared, Iterable) and not isinstance(declared, (str, bytes)):
        values = [str(item) for item in declared if item]
        if values:
            return values

    preferred_device = _detect_preferred_device(settings, runtime_capability)
    return list(_DEVICE_EXPORT_FORMATS.get(preferred_device, _DEFAULT_EXPORT_FORMATS))


def _recommended_imgsz(complexity: Dict[str, Any]) -> tuple[int, str]:
    if complexity["level"] == "high":
        return 1280, "algorithm"
    return 640, "algorithm"


def _recommended_epochs(complexity: Dict[str, Any]) -> tuple[int, str]:
    if complexity["level"] == "high":
        return 140, "algorithm"
    if complexity["level"] == "medium":
        return 120, "algorithm"
    return 100, "algorithm"


def _recommended_patience(complexity: Dict[str, Any]) -> tuple[int, str]:
    if complexity["level"] == "high":
        return 30, "algorithm"
    return 20, "algorithm"


def _build_reason_summary(
    algorithm_plan: Dict[str, Any],
    complexity: Dict[str, Any],
    recommended_config: Dict[str, Any],
) -> str:
    scenario_label = {
        "occupancy_monitoring": "占位监测",
        "parking_violation": "违规停车",
        "intrusion_monitoring": "区域闯入",
        "dwell_time_monitoring": "滞留监测",
        "object_tracking": "目标跟踪",
        "object_counting": "目标计数",
        "safety_compliance": "安全合规",
        "quality_inspection": "质量检测",
        "feature_matching": "特征匹配",
        "classification": "分类识别",
        "custom_event_monitoring": "自定义事件",
    }.get(algorithm_plan.get("scenario_type"), "当前任务")

    reasons = [f"当前任务属于{scenario_label}场景"]
    reasons.append(f"包含 {complexity['target_count']} 个目标类别")

    if complexity["tracking_required"]:
        reasons.append("需要跟踪")

    if complexity["has_temporal_rule"] and complexity["has_region_rule"]:
        reasons.append("同时包含区域与时序规则")
    elif complexity["has_temporal_rule"]:
        reasons.append("包含时序规则")
    elif complexity["has_region_rule"]:
        reasons.append("包含区域规则")

    if complexity["level"] == "high":
        tail = (
            f"因此推荐使用 {recommended_config['model']}、{recommended_config['imgsz']} 图像尺寸、"
            f"{recommended_config['epochs']} 轮训练和更长的早停耐心值。"
        )
    elif complexity["level"] == "medium":
        tail = (
            f"因此推荐使用 {recommended_config['model']}、{recommended_config['imgsz']} 图像尺寸，"
            f"并适度提高训练轮次。"
        )
    else:
        tail = (
            f"因此推荐使用 {recommended_config['model']} 和标准训练配置。"
        )

    return "，".join(reasons) + "，" + tail


def build_training_recommendation(
    algorithm_plan: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    settings: Optional[Any] = None,
    runtime_capability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    complexity = _derive_complexity(algorithm_plan, pipeline_config)
    preferred_device = _detect_preferred_device(settings, runtime_capability)
    model, model_source = _recommended_model(complexity, settings, algorithm_plan, runtime_capability)
    train_mode, train_mode_source = _recommended_train_mode(complexity, settings, runtime_capability)
    export_formats = _available_export_formats(settings, runtime_capability)
    imgsz, imgsz_source = _recommended_imgsz(complexity)
    epochs, epochs_source = _recommended_epochs(complexity)
    patience, patience_source = _recommended_patience(complexity)

    recommended_config = {
        "model": model,
        "train_mode": train_mode,
        "export_formats": export_formats,
        "imgsz": imgsz,
        "epochs": epochs,
        "lr0": 0.01,
        "patience": patience,
        "conf": 0.25,
        "iou": 0.7,
        "gpu_type": _normalized_gpu_type(settings),
    }

    source_map = {
        "model": model_source,
        "train_mode": train_mode_source,
        "export_formats": "runtime" if runtime_capability and runtime_capability.get("available_export_formats") else "runtime",
        "imgsz": imgsz_source,
        "epochs": epochs_source,
        "lr0": "system_default",
        "patience": patience_source,
        "conf": "system_default",
        "iou": "system_default",
        "gpu_type": "user_default",
    }

    legacy = {
        "recommended_model": recommended_config["model"],
        "train_mode": recommended_config["train_mode"],
        "export_formats": recommended_config["export_formats"],
        "requires_detector_training": bool(
            algorithm_plan.get("training_requirements", {}).get("detector_training_required", False)
        ),
    }

    return {
        "recommended_config": recommended_config,
        "reason_summary": _build_reason_summary(algorithm_plan, complexity, recommended_config),
        "source_map": source_map,
        "legacy": legacy,
    }


def enrich_pipeline_with_training_recommendation(
    algorithm_plan: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    settings: Optional[Any] = None,
    runtime_capability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    enriched = deepcopy(pipeline_config)
    recommendation = build_training_recommendation(
        algorithm_plan=algorithm_plan,
        pipeline_config=enriched,
        settings=settings,
        runtime_capability=runtime_capability,
    )
    enriched["training_recommendation"] = {
        **recommendation["legacy"],
        "recommended_config": recommendation["recommended_config"],
        "reason_summary": recommendation["reason_summary"],
        "source_map": recommendation["source_map"],
    }
    return enriched
