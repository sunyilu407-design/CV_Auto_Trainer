from __future__ import annotations

from typing import Any, Dict, List


def _recommended_model(targets: List[Dict[str, Any]], plan: Dict[str, Any] | None = None) -> str:
    # 优先从 VLM model_pipeline 获取
    plan = plan or {}
    for step in plan.get("model_pipeline", []):
        if step.get("role") in ("primary_detector", "secondary_detector"):
            model_id = step.get("recommended_model_id")
            if model_id:
                return model_id
    if len(targets) >= 4:
        return "yolo11m.pt"
    return "yolo11s.pt"


def _build_detectors(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    target_classes = [target["class_name"] for target in targets if target.get("class_name")]
    return [
        {
            "detector_id": "primary_detector",
            "detector_type": "yolo",
            "target_classes": target_classes,
            "prompt_bindings": {
                target["class_name"]: target.get("prompt", target["class_name"])
                for target in targets
                if target.get("class_name")
            },
        }
    ]


def _build_trackers(training_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not training_requirements.get("tracking_required", False):
        return []
    return [
        {
            "tracker_id": "primary_tracker",
            "tracker_type": "bytetrack",
            "source_detector_id": "primary_detector",
        }
    ]


def _duration_fields(temporal_constraint: Dict[str, Any]) -> Dict[str, Any]:
    duration_seconds = int(temporal_constraint.get("duration_seconds", 0) or 0)
    return {
        "duration_seconds": duration_seconds,
        "duration_ms": duration_seconds * 1000,
    }


def _infer_rule_type(event: Dict[str, Any], trigger: Dict[str, Any]) -> str:
    if event.get("event_type"):
        return str(event["event_type"])
    if trigger.get("from_region_id") and trigger.get("to_region_id"):
        return "cross_region_transition"
    if trigger.get("temporal_constraint_id"):
        return "region_presence_duration"
    return "region_presence_duration"


def _build_rules(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    temporal_constraints = {
        item["constraint_id"]: item
        for item in plan.get("temporal_constraints", [])
        if item.get("constraint_id")
    }

    for index, event in enumerate(plan.get("events", []), start=1):
        trigger = event.get("trigger", {})
        temporal_constraint = temporal_constraints.get(trigger.get("temporal_constraint_id", ""), {})
        rule_type = _infer_rule_type(event, trigger)
        rule = {
            "rule_id": f"rule_{index}",
            "rule_type": rule_type,
            "event_code": event.get("event_code", f"event_{index}"),
            "target_class": trigger.get("target_class"),
        }

        if rule_type in {"region_enter", "region_exit", "region_presence_duration"}:
            rule["region_id"] = trigger.get("region_id")
        if rule_type == "cross_region_transition":
            rule["from_region_id"] = trigger.get("from_region_id")
            rule["to_region_id"] = trigger.get("to_region_id")
        if rule_type == "region_presence_duration":
            rule.update(_duration_fields(temporal_constraint))

        rules.append(rule)
    return rules


def compile_algorithm_pipeline(plan: Dict[str, Any]) -> Dict[str, Any]:
    targets = plan.get("targets", [])
    training_requirements = plan.get("training_requirements", {})
    model_pipeline = plan.get("model_pipeline", [])
    training_strategy = plan.get("training_strategy", {})

    # 确定训练模式
    train_mode = training_strategy.get("train_mode_recommendation", "local")
    if train_mode in ("cloud_ssh", "cloud_autodl"):
        train_mode = "cloud"

    # 确定导出格式
    export_formats = ["onnx"]
    for step in model_pipeline:
        if step.get("role") in ("primary_detector", "secondary_detector"):
            from services.model_registry import get_model_registry
            registry = get_model_registry()
            model = registry.get_model(step.get("recommended_model_id", ""))
            if model:
                export_formats = model.export_formats[:3]
            break

    result = {
        "version": "v1",
        "metadata": {
            "summary": plan.get("summary_zh") or plan.get("summary", ""),
            "scenario_type": plan.get("scenario_type", "custom_event_monitoring"),
            "confidence": plan.get("confidence"),
            "difficulty_level": plan.get("difficulty_level"),
        },
        "inputs": {
            "runtime_modes": plan.get("runtime_modes", ["offline", "stream"]),
        },
        "detectors": _build_detectors(targets),
        "trackers": _build_trackers(training_requirements),
        "regions": plan.get("regions", []),
        "temporal_windows": plan.get("temporal_constraints", []),
        "rules": _build_rules(plan),
        "outputs": [
            {
                "output_id": "primary_event_output",
                "type": "event_stream",
                "event_codes": [event.get("event_code") for event in plan.get("events", []) if event.get("event_code")],
            }
        ],
        "packaging": {
            "format": "project_bundle",
            "entrypoint": "run_pipeline.py",
            "config_path": "pipeline.json",
        },
        "training_recommendation": {
            "recommended_model": _recommended_model(targets, plan),
            "train_mode": train_mode,
            "export_formats": export_formats,
            "requires_detector_training": training_requirements.get("detector_training_required", False),
        },
    }

    # 附加 model_pipeline 信息到编译结果
    if model_pipeline:
        result["model_pipeline"] = model_pipeline

    return result
