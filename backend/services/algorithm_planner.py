import re
from typing import Any, Dict, List, Optional


def _unique_non_empty(items: List[str]) -> List[str]:
    unique_items: List[str] = []
    for item in items:
        value = item.strip()
        if value and value not in unique_items:
            unique_items.append(value)
    return unique_items


def _detect_scenario_type(user_description: str) -> str:
    desc = user_description.lower()
    if any(token in user_description for token in ("仓位", "占位", "占用", "空位")):
        return "occupancy_monitoring"
    if any(token in user_description for token in ("停车", "违停")):
        return "parking_violation"
    if any(token in user_description for token in ("闯入", "进入区域", "越线")):
        return "intrusion_monitoring"
    if "滞留" in user_description:
        return "dwell_time_monitoring"
    if "tracking" in desc or "track" in desc:
        return "object_tracking"
    return "custom_event_monitoring"


def _extract_duration_seconds(user_description: str) -> int:
    match = re.search(r"(\d+)\s*秒", user_description)
    if match:
        return max(1, int(match.group(1)))

    minute_match = re.search(r"(\d+)\s*分", user_description)
    if minute_match:
        return max(1, int(minute_match.group(1)) * 60)

    return 3


def _build_event_code(scenario_type: str) -> str:
    mapping = {
        "occupancy_monitoring": "occupancy_detected",
        "parking_violation": "parking_violation_detected",
        "intrusion_monitoring": "intrusion_detected",
        "dwell_time_monitoring": "dwell_timeout_detected",
        "object_tracking": "tracking_event_detected",
    }
    return mapping.get(scenario_type, "custom_event_detected")


def _build_targets(vlm_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classes = []
    if vlm_result and isinstance(vlm_result.get("classes"), list):
        classes = vlm_result["classes"]

    targets: List[Dict[str, Any]] = []
    for item in classes:
        class_name = item.get("class_name", "target")
        targets.append(
            {
                "class_name": class_name,
                "prompt": item.get("prompt", class_name),
                "role": "primary_subject" if not targets else "context_subject",
                "requires_training": True,
            }
        )
    return targets


def _extract_region_labels(user_description: str) -> List[str]:
    labels = re.findall(r"([A-Za-z0-9一二三四五六七八九十甲乙丙丁]+区)", user_description)
    unique_labels: List[str] = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)
    return unique_labels


def _region_id_from_label(label: str, index: int) -> str:
    latin_match = re.fullmatch(r"([A-Za-z0-9]+)区", label)
    if latin_match:
        return f"zone_{latin_match.group(1).lower()}"
    chinese_map = {
        "一区": "zone_1",
        "二区": "zone_2",
        "三区": "zone_3",
        "四区": "zone_4",
        "五区": "zone_5",
    }
    if label in chinese_map:
        return chinese_map[label]
    return "primary_region" if index == 0 else f"region_{index + 1}"


def _build_regions(user_description: str) -> List[Dict[str, Any]]:
    labels = _extract_region_labels(user_description)
    if not labels:
        return [
            {
                "region_id": "primary_region",
                "name": "主监测区域",
                "source": "user_defined",
                "required": True,
            }
        ]

    return [
        {
            "region_id": _region_id_from_label(label, index),
            "name": label,
            "source": "user_defined",
            "required": True,
        }
        for index, label in enumerate(labels)
        ]


def _build_negotiation_regions(user_description: str, regions: List[Dict[str, Any]]) -> List[str]:
    mentions = re.findall(r"(?:进入|离开|在|从|到)([A-Za-z0-9\u4e00-\u9fff]{1,8}(?:区域|区))", user_description)
    if mentions:
        return _unique_non_empty(mentions)

    named_regions = [
        region.get("name", "")
        for region in regions
        if region.get("name") and region.get("name") != "主监测区域"
    ]
    return _unique_non_empty(named_regions)


def _build_negotiation_objects(
    user_description: str,
    targets: List[Dict[str, Any]],
) -> List[str]:
    lowered = user_description.lower()
    objects: List[str] = []
    keyword_map = [
        (("人员", "员工", "行人", "person"), "人员"),
        (("工帽", "安全帽", "helmet"), "工帽"),
        (("货箱", "货物箱", "cargo_box", "box"), "货箱"),
        (("仓位", "货位", "slot", "rack_slot"), "仓位"),
        (("车辆", "汽车", "卡车", "truck", "car", "vehicle"), "车辆"),
    ]

    for tokens, label in keyword_map:
        if any(token in user_description or token in lowered for token in tokens):
            objects.append(label)

    for target in targets:
        class_name = str(target.get("class_name", "")).strip()
        prompt = str(target.get("prompt", "")).strip()
        if class_name and class_name != "target":
            objects.append(class_name)
        if prompt:
            objects.append(prompt)

    return _unique_non_empty(objects) or ["目标"]


def _build_negotiation_events(
    *,
    user_description: str,
    duration_seconds: int,
    regions: List[str],
    events: List[Dict[str, Any]],
) -> List[str]:
    summaries: List[str] = []
    primary_region = regions[0] if regions else "监测区域"
    secondary_region = regions[1] if len(regions) > 1 else None

    if any(token in user_description for token in ("进入", "闯入", "越线")):
        summaries.append(f"进入{primary_region}")
    if any(token in user_description for token in ("离开", "退出")):
        summaries.append(f"离开{primary_region}")
    if secondary_region and (("从" in user_description and "进入" in user_description) or "跨区" in user_description):
        summaries.append(f"从{primary_region}进入{secondary_region}")
    if any(token in user_description for token in ("持续", "滞留", "停留", "超时", "超过", "占位", "占用")):
        summaries.append(f"在{primary_region}停留{duration_seconds}秒告警")

    if summaries:
        return _unique_non_empty(summaries)

    return _unique_non_empty([str(item.get("name", "")).strip() for item in events]) or ["主事件"]


def _slugify_capability_token(token: str) -> str:
    token_map = {
        "人员": "person",
        "工帽": "helmet",
        "货箱": "cargo_box",
        "仓位": "slot",
        "车辆": "vehicle",
        "危险区域": "danger_zone",
    }
    if token in token_map:
        return token_map[token]

    slug = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return slug or "custom"


def _build_negotiation_summary(
    *,
    user_description: str,
    targets: List[Dict[str, Any]],
    regions: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    duration_seconds: int,
) -> Dict[str, Any]:
    negotiation_regions = _build_negotiation_regions(user_description, regions)
    negotiation_objects = _build_negotiation_objects(user_description, targets)
    negotiation_events = _build_negotiation_events(
        user_description=user_description,
        duration_seconds=duration_seconds,
        regions=negotiation_regions,
        events=events,
    )

    return {
        "scenario_label": user_description,
        "objects": negotiation_objects,
        "regions": negotiation_regions,
        "duration_seconds": duration_seconds,
        "events": negotiation_events,
    }


def _build_capabilities(
    *,
    user_description: str,
    targets: List[Dict[str, Any]],
    negotiation_summary: Dict[str, Any],
) -> List[Dict[str, Any]]:
    capabilities: List[Dict[str, Any]] = []
    objects = negotiation_summary.get("objects", [])
    regions = negotiation_summary.get("regions", [])
    primary_object = next((item for item in objects if item not in {"工帽"}), None)
    primary_region = regions[0] if regions else "监测区域"

    if primary_object:
        capabilities.append(
            {
                "capability_id": f"detect_{_slugify_capability_token(primary_object)}",
                "label": f"识别{primary_object}",
                "trainable": True,
                "kind": "detection",
            }
        )
    elif targets:
        primary_target = targets[0].get("class_name", "target")
        capabilities.append(
            {
                "capability_id": f"detect_{_slugify_capability_token(str(primary_target))}",
                "label": f"识别{primary_target}",
                "trainable": True,
                "kind": "detection",
            }
        )

    if any(token in user_description for token in ("工帽", "安全帽")):
        capabilities.append(
            {
                "capability_id": "classify_helmet",
                "label": "判断是否佩戴工帽",
                "trainable": True,
                "kind": "classification",
            }
        )

    rule_label = f"{primary_region}停留规则"
    if any(token in user_description for token in ("进入", "闯入", "越线")) and not any(
        token in user_description for token in ("持续", "滞留", "停留", "超时", "超过", "占位", "占用")
    ):
        rule_label = f"{primary_region}进入告警规则"

    capabilities.append(
        {
            "capability_id": f"rule_{_slugify_capability_token(primary_region)}",
            "label": rule_label,
            "trainable": False,
            "kind": "rule",
        }
    )
    return capabilities


def _presence_event_code(scenario_type: str, region_id: str) -> str:
    if region_id == "primary_region":
        return _build_event_code(scenario_type)
    return f"dwell_{region_id}"


def _should_emit_presence_event(user_description: str, scenario_type: str) -> bool:
    return any(token in user_description for token in ("持续", "滞留", "停留", "超时", "超过", "占位", "占用")) or (
        scenario_type in {"occupancy_monitoring", "dwell_time_monitoring"}
    )


def _build_events(
    *,
    user_description: str,
    scenario_type: str,
    primary_target: str,
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    first_region_id = regions[0]["region_id"] if regions else "primary_region"
    second_region_id = regions[1]["region_id"] if len(regions) > 1 else None
    events: List[Dict[str, Any]] = []

    if any(token in user_description for token in ("进入", "闯入", "越线")):
        events.append(
            {
                "event_type": "region_enter",
                "event_code": f"entered_{first_region_id}",
                "name": "进入区域事件",
                "trigger": {
                    "target_class": primary_target,
                    "region_id": first_region_id,
                },
            }
        )

    if _should_emit_presence_event(user_description, scenario_type):
        events.append(
            {
                "event_type": "region_presence_duration",
                "event_code": _presence_event_code(scenario_type, first_region_id),
                "name": "持续停留事件",
                "trigger": {
                    "target_class": primary_target,
                    "region_id": first_region_id,
                    "temporal_constraint_id": "primary_duration",
                },
            }
        )

    if any(token in user_description for token in ("离开", "退出")):
        events.append(
            {
                "event_type": "region_exit",
                "event_code": f"left_{first_region_id}",
                "name": "离开区域事件",
                "trigger": {
                    "target_class": primary_target,
                    "region_id": first_region_id,
                },
            }
        )

    if second_region_id and (("从" in user_description and "进入" in user_description) or "跨区" in user_description):
        events.append(
            {
                "event_type": "cross_region_transition",
                "event_code": f"crossed_{first_region_id.split('zone_', 1)[-1]}_to_{second_region_id.split('zone_', 1)[-1]}",
                "name": "跨区域事件",
                "trigger": {
                    "target_class": primary_target,
                    "from_region_id": first_region_id,
                    "to_region_id": second_region_id,
                },
            }
        )

    if not events:
        events.append(
            {
                "event_type": "region_presence_duration",
                "event_code": _build_event_code(scenario_type),
                "name": "主事件",
                "trigger": {
                    "target_class": primary_target,
                    "region_id": first_region_id,
                    "temporal_constraint_id": "primary_duration",
                },
            }
        )

    return events


def build_algorithm_plan(user_description: str, vlm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    scenario_type = _detect_scenario_type(user_description)
    duration_seconds = _extract_duration_seconds(user_description)
    targets = _build_targets(vlm_result)
    primary_target = targets[0]["class_name"] if targets else "target"
    regions = _build_regions(user_description)
    events = _build_events(
        user_description=user_description,
        scenario_type=scenario_type,
        primary_target=primary_target,
        regions=regions,
    )
    negotiation_summary = _build_negotiation_summary(
        user_description=user_description,
        targets=targets,
        regions=regions,
        events=events,
        duration_seconds=duration_seconds,
    )
    capabilities = _build_capabilities(
        user_description=user_description,
        targets=targets,
        negotiation_summary=negotiation_summary,
    )

    return {
        "summary": f"基于 {primary_target} 的 {scenario_type} 算法草案",
        "scenario_type": scenario_type,
        "negotiation_summary": negotiation_summary,
        "capabilities": capabilities,
        "runtime_modes": ["offline", "stream"],
        "targets": targets,
        "regions": regions,
        "temporal_constraints": [
            {
                "constraint_id": "primary_duration",
                "type": "sustain",
                "duration_seconds": duration_seconds,
            }
        ],
        "events": events,
        "training_requirements": {
            "detector_training_required": bool(targets),
            "tracking_required": True,
            "rule_engine_required": True,
        },
        "confidence": 0.72 if targets else 0.45,
    }
