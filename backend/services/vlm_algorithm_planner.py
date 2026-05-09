"""
VLM 驱动的智能算法规划器。

替代原有纯规则引擎 (algorithm_planner.py)，核心逻辑：
1. 将用户需求、图片/视频帧、设备信息 和 模型目录 一起传给 VLM
2. VLM 输出完整的算法方案（含模型选择、多模型优先级、pipeline 设计）
3. 系统自动判断所有技术决策，用户无需任何计算机知识
4. 自动查找可复用的已训练模型，避免重复训练
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services.model_registry import (
    get_model_registry,
    infer_device_tier,
    TrainedModelCache,
)
from services.reasoning_adapter import (
    build_reasoning_adapter_from_settings,
    get_reasoning_adapter,
    normalize_categories,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLM 算法方案 JSON Schema
# ---------------------------------------------------------------------------

ALGORITHM_PLAN_SCHEMA = {
    "type": "object",
    "required": [
        "summary",
        "summary_zh",
        "scenario_type",
        "difficulty_level",
        "targets",
        "regions",
        "events",
        "model_pipeline",
        "training_strategy",
    ],
    "properties": {
        "summary": {"type": "string", "description": "英文一句话摘要"},
        "summary_zh": {"type": "string", "description": "中文一句话摘要，面向非技术用户"},
        "scenario_type": {
            "type": "string",
            "enum": [
                "occupancy_monitoring", "parking_violation", "intrusion_monitoring",
                "dwell_time_monitoring", "object_tracking", "object_counting",
                "safety_compliance", "quality_inspection", "feature_matching",
                "classification", "custom_event_monitoring",
            ],
        },
        "difficulty_level": {
            "type": "string",
            "enum": ["simple", "moderate", "complex", "very_complex"],
            "description": "任务复杂度评估",
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "targets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["class_name", "display_name_zh", "purpose"],
                "properties": {
                    "class_name": {"type": "string"},
                    "display_name_zh": {"type": "string"},
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "purpose": {"type": "string", "description": "该目标在整个算法中的作用"},
                },
            },
        },
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "label": {"type": "string"},
                    "purpose_zh": {"type": "string"},
                },
            },
        },
        "temporal_constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "constraint_id": {"type": "string"},
                    "type": {"type": "string"},
                    "duration_seconds": {"type": "number"},
                },
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["event_code", "name", "name_zh", "trigger"],
                "properties": {
                    "event_code": {"type": "string"},
                    "name": {"type": "string"},
                    "name_zh": {"type": "string"},
                    "trigger": {
                        "type": "object",
                        "properties": {
                            "target_class": {"type": "string"},
                            "region_id": {"type": "string"},
                            "temporal_constraint_id": {"type": "string"},
                            "from_region_id": {"type": "string"},
                            "to_region_id": {"type": "string"},
                        },
                    },
                },
            },
        },
        "model_pipeline": {
            "type": "array",
            "description": "多模型 pipeline，按执行优先级排序",
            "items": {
                "type": "object",
                "required": ["step_id", "role", "recommended_model_id", "reason_zh"],
                "properties": {
                    "step_id": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["primary_detector", "secondary_detector", "classifier", "feature_matcher", "tracker", "rule_engine", "ocr"],
                    },
                    "recommended_model_id": {"type": "string", "description": "从可选模型列表中选择的 model_id"},
                    "alternative_model_ids": {
                        "type": "array", "items": {"type": "string"},
                        "description": "备选模型列表",
                    },
                    "reason_zh": {"type": "string", "description": "选择该模型的中文理由"},
                    "requires_training": {"type": "boolean"},
                    "training_priority": {"type": "integer", "description": "训练优先级，1 最高"},
                    "estimated_training_hours": {"type": "number"},
                    "input_size": {"type": "integer"},
                    "epochs": {"type": "integer"},
                },
            },
        },
        "training_strategy": {
            "type": "object",
            "required": ["total_models_to_train", "estimated_total_hours", "train_mode_recommendation"],
            "properties": {
                "total_models_to_train": {"type": "integer"},
                "estimated_total_hours": {"type": "number"},
                "train_mode_recommendation": {
                    "type": "string",
                    "enum": ["local", "cloud_ssh", "cloud_autodl"],
                },
                "train_mode_reason_zh": {"type": "string"},
                "special_requirements": {"type": "array", "items": {"type": "string"}},
            },
        },
        "capabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "capability_id": {"type": "string"},
                    "label": {"type": "string"},
                    "kind": {"type": "string"},
                    "trainable": {"type": "boolean"},
                },
            },
        },
        "negotiation_summary": {
            "type": "object",
            "properties": {
                "objects": {"type": "array", "items": {"type": "string"}},
                "regions": {"type": "array", "items": {"type": "string"}},
                "events": {"type": "array", "items": {"type": "string"}},
                "duration_seconds": {"type": "number"},
                "user_facing_description_zh": {
                    "type": "string",
                    "description": "面向非技术用户的一段通俗描述，用于协商确认",
                },
            },
        },
    },
}


def _build_system_prompt(
    model_catalog: str,
    cached_models_info: str,
    device_tier: str,
    device_description: str,
) -> str:
    return f"""你是一位专业的计算机视觉算法架构师。你的任务是根据用户提供的图片/视频和需求描述，设计一个完整的视觉算法方案。

## 你的职责
1. 分析用户的图片/视频内容和业务需求
2. 从可选模型目录中选择最适合的模型组合
3. 设计完整的算法 pipeline（可能包含多个模型）
4. 所有技术决策由你自动完成，用户没有任何计算机知识

## 用户部署设备信息
- 设备等级: {device_tier}
- 设备描述: {device_description}
- **请根据设备算力选择合适大小的模型，边缘设备必须选轻量模型**

## 可选预训练模型目录
{model_catalog}

## 已训练可复用的模型
{cached_models_info}
如果列表中有模型完全覆盖当前需求的类别且精度达标，请直接引用该缓存模型，将其 requires_training 设为 false，避免重复训练。

## 设计原则
1. **模型选择**：优先选择与设备匹配的模型，精度和速度需要平衡
2. **多模型组合**：复杂场景可以设计多步骤 pipeline（如先检测再分类、先检测再匹配）
3. **训练优先级**：如需多个模型训练，按重要性排序，priority=1 最先训练
4. **训练模式**：根据数据量和模型大小建议 local/cloud_ssh/cloud_autodl
5. **复用优先**：如果已有训练好的模型可用，直接引用，不要重复训练
6. **用户描述**：negotiation_summary.user_facing_description_zh 要通俗易懂，不能出现技术术语
7. **文字识别（OCR）**：如果场景需要从图像中识别文字（如油品名称、车牌、编号等），必须设计一个 role="ocr" 的 pipeline 步骤，recommended_model_id 固定为 "easyocr"，requires_training 必须为 false。OCR 步骤通常接在检测步骤之后，对检测框裁剪区域进行文字识别。

## 输出格式
请严格按照 JSON 格式输出，不要包含任何额外文本。"""


def _build_user_prompt(
    user_description: str,
    vlm_result: Optional[Dict[str, Any]],
    image_count: int,
) -> str:
    parts = [f"## 用户业务需求\n{user_description}\n"]

    if vlm_result:
        classes = vlm_result.get("classes", [])
        if classes:
            parts.append("## 图片视觉分析结果")
            for cls in classes:
                name = cls.get("display_name_zh") or cls.get("class_name", "")
                prompt = cls.get("display_prompt_zh") or cls.get("prompt", "")
                parts.append(f"- {name}: {prompt}")
            parts.append("")

    parts.append(f"## 数据信息\n- 用户上传的样本图片约 {image_count} 张")
    parts.append("\n请设计完整的算法方案（JSON 格式）。")
    return "\n".join(parts)


def _build_cached_models_info(registry) -> str:
    cached = registry.list_cached_models()
    if not cached:
        return "暂无已训练的可复用模型。"
    lines = ["已有以下训练好的模型可直接使用："]
    for c in cached:
        lines.append(
            f"- cache_id={c.cache_id}, 基于 {c.source_model_id}, "
            f"类别={c.classes}, mAP50={c.map50}, 场景={c.scenario_type}, "
            f"权重路径={c.weight_path}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 核心函数
# ---------------------------------------------------------------------------

def build_vlm_algorithm_plan(
    user_description: str,
    vlm_result: Optional[Dict[str, Any]],
    vlm_adapter,
    images_base64: Optional[List[str]] = None,
    gpu_type: str | None = None,
    platform: str | None = None,
    device_description: str = "",
    image_count: int = 0,
    reasoning_settings=None,
    algorithm_hints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    用 VLM 生成完整算法方案。

    参数:
        user_description: 用户的业务需求文本
        vlm_result: 之前 VLM 意图解析的结果
        vlm_adapter: VLMAdapter 实例
        images_base64: 用户上传的样板图 (base64)
        gpu_type: 用户 GPU 型号
        platform: 用户操作系统平台
        device_description: 用户设备描述
        image_count: 上传图片数量
    """
    registry = get_model_registry()
    device_tier = infer_device_tier(gpu_type, platform)

    model_catalog = registry.get_models_summary_for_vlm(
        device_tier=device_tier,
        task_type="detection",
    )
    cached_info = _build_cached_models_info(registry)

    if not device_description:
        device_description = gpu_type or platform or "未知设备"

    system_prompt = _build_system_prompt(
        model_catalog=model_catalog,
        cached_models_info=cached_info,
        device_tier=device_tier,
        device_description=device_description,
    )
    user_prompt = _build_user_prompt(user_description, vlm_result, image_count)

    # 注入 algorithm_hints（来自需求确认对话）
    if algorithm_hints:
        import json as _json
        hints_text = _json.dumps(algorithm_hints, ensure_ascii=False, indent=2)
        user_prompt += (
            "\n\n## 需求确认阶段产出的 algorithm_hints（请优先参考）\n"
            f"```json\n{hints_text}\n```"
        )

    # 调用 VLM
    try:
        raw = vlm_adapter.call_with_system_prompt(
            system_prompt=system_prompt,
            user_text=user_prompt,
            images_base64=images_base64 or [],
            response_format="json",
        )
        plan = _parse_plan_response(raw)
    except Exception as e:
        logger.warning("VLM algorithm planning failed, falling back to rule-based: %s", e)
        plan = _fallback_rule_based_plan(user_description, vlm_result, device_tier, registry)

    # P1-A 决策层：用推理模型对类别词做归一化（CLIP 友好度提升）
    plan = _apply_reasoning_normalization(plan, vlm_adapter, reasoning_settings)

    # 查找可复用模型
    plan = _apply_cached_models(plan, registry)

    # 补充 negotiation_summary (确保存在)
    if "negotiation_summary" not in plan or not plan["negotiation_summary"]:
        plan["negotiation_summary"] = _build_fallback_negotiation(plan, user_description)

    # 确保兼容旧字段
    plan.setdefault("runtime_modes", ["offline", "stream"])
    plan.setdefault("training_requirements", {
        "detector_training_required": any(
            step.get("requires_training", True)
            for step in plan.get("model_pipeline", [])
            if step.get("role") in ("primary_detector", "secondary_detector")
        ),
        "tracking_required": any(
            step.get("role") == "tracker"
            for step in plan.get("model_pipeline", [])
        ),
        "rule_engine_required": any(
            step.get("role") == "rule_engine"
            for step in plan.get("model_pipeline", [])
        ),
    })

    return plan


def revise_vlm_algorithm_plan(
    existing_plan: Dict[str, Any],
    user_feedback: str,
    vlm_adapter,
    reasoning_settings=None,
    revision_history: Optional[List[Dict[str, str]]] = None,
    gpu_type: str | None = None,
    platform: str | None = None,
    device_description: str = "",
    images_base64: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    根据用户反馈修订算法方案。

    参数:
        existing_plan: 当前方案（VLM 生成的完整 plan）
        user_feedback: 用户提出的修改意见，如 "太复杂，模型换小一点" / "增加一个颜色分类"
        revision_history: 历次修订对话记录 [{"role": "user"|"assistant", "content": "..."}]
        gpu_type/platform/device_description: 设备上下文
    """
    registry = get_model_registry()
    device_tier = infer_device_tier(gpu_type, platform)

    model_catalog = registry.get_models_summary_for_vlm(
        device_tier=device_tier,
        task_type="detection",
    )
    cached_info = _build_cached_models_info(registry)
    if not device_description:
        device_description = gpu_type or platform or "未知设备"

    system_prompt = _build_system_prompt(
        model_catalog=model_catalog,
        cached_models_info=cached_info,
        device_tier=device_tier,
        device_description=device_description,
    )

    # 构建修订 prompt：包含当前方案摘要 + 历史对话 + 本轮反馈
    existing_summary = json.dumps(
        {
            "summary": existing_plan.get("summary_zh") or existing_plan.get("summary", ""),
            "scenario_type": existing_plan.get("scenario_type"),
            "difficulty_level": existing_plan.get("difficulty_level"),
            "model_pipeline": [
                {
                    "role": step.get("role"),
                    "recommended_model_id": step.get("recommended_model_id"),
                    "requires_training": step.get("requires_training"),
                    "reason_zh": step.get("reason_zh"),
                }
                for step in existing_plan.get("model_pipeline", [])
            ],
            "training_strategy": existing_plan.get("training_strategy", {}),
            "targets": existing_plan.get("targets", []),
        },
        ensure_ascii=False,
        indent=2,
    )

    history_text = ""
    if revision_history:
        history_lines = []
        for msg in revision_history:
            prefix = "用户" if msg.get("role") == "user" else "系统"
            history_lines.append(f"{prefix}：{msg.get('content', '')}")
        history_text = "\n历史对话：\n" + "\n".join(history_lines)

    revise_prompt = f"""你之前生成的算法方案如下：

```json
{existing_summary}
```
{history_text}

用户对这个方案提出以下修改意见：
"{user_feedback}"

请根据用户反馈重新生成完整的算法方案 JSON。要求：
1. 保持原方案中用户未质疑的合理部分
2. 针对用户反馈做出明确调整（如换模型、增减步骤、调整训练优先级等）
3. 在新方案的 `summary_zh` 中简要说明"相比上一版做了哪些调整"
4. 输出格式必须与首次规划一致（包含 summary/scenario_type/targets/model_pipeline/training_strategy 等字段）
"""

    try:
        raw = vlm_adapter.call_with_system_prompt(
            system_prompt=system_prompt,
            user_text=revise_prompt,
            images_base64=images_base64 or [],
            response_format="json",
        )
        plan = _parse_plan_response(raw)
    except Exception as e:
        logger.exception("VLM plan revision failed")
        # 降级：保留原方案，但在 summary_zh 中说明失败
        plan = dict(existing_plan)
        plan["summary_zh"] = (plan.get("summary_zh") or plan.get("summary", "")) + f"\n[修订失败：{e}]"
        return plan

    plan = _apply_reasoning_normalization(plan, vlm_adapter, reasoning_settings)
    plan = _apply_cached_models(plan, registry)
    if "negotiation_summary" not in plan or not plan["negotiation_summary"]:
        plan["negotiation_summary"] = _build_fallback_negotiation(plan, user_feedback)
    plan.setdefault("runtime_modes", existing_plan.get("runtime_modes", ["offline", "stream"]))
    plan.setdefault("training_requirements", existing_plan.get("training_requirements", {}))
    return plan


def _parse_plan_response(raw: str) -> Dict[str, Any]:
    """从 VLM 原始响应中提取 JSON"""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    plan = json.loads(text)

    required_keys = ["summary", "scenario_type", "targets", "model_pipeline"]
    for key in required_keys:
        if key not in plan:
            raise ValueError(f"VLM response missing required key: {key}")

    return plan


def _apply_reasoning_normalization(
    plan: Dict[str, Any],
    vlm_adapter,
    reasoning_settings=None,
) -> Dict[str, Any]:
    """
    P1-A 决策层：用推理模型把 targets/classes 的类别词归一化为 CLIP/YOLO-World 友好的英文。
    - 不可用（未配置 API Key 且无 fallback）→ 静默跳过，保留原 plan
    - 输出附加到每个 target 的 reasoning_normalization 字段，并且如果有更优 prompt/aliases，写回 prompt + prompt_aliases
    - 同时把 plan["reasoning_layer"] 置为后端名称，便于前端展示「已经过推理模型校验」徽章

    reasoning_settings: 可选，含 reasoning_enabled/provider/base_url/api_key/model 的对象（来自用户 DB 设置）。
    若为 None 则降级到 env-var 自动选择 + vlm_adapter。
    """
    targets = plan.get("targets", [])
    if not targets:
        return plan

    if reasoning_settings is not None:
        adapter = build_reasoning_adapter_from_settings(reasoning_settings, vlm_adapter)
    else:
        adapter = get_reasoning_adapter(vlm_adapter)
    if adapter is None:
        return plan

    # 只对真正会进入 YOLO-World 的检测目标做归一化（targets 已经是这一层）
    result = normalize_categories(targets, adapter=adapter)
    if not result or "items" not in result:
        return plan

    # 合并归一化结果回 targets
    items = {item.get("input_class_name"): item for item in result.get("items", []) if isinstance(item, dict)}
    for t in targets:
        item = items.get(t.get("class_name"))
        if not item:
            continue
        normalized_name = (item.get("class_name_en") or "").strip()
        aliases = [a.strip() for a in (item.get("aliases") or []) if isinstance(a, str) and a.strip()]
        t["reasoning_normalization"] = {
            "class_name_en": normalized_name,
            "aliases": aliases,
            "abstract": bool(item.get("abstract", False)),
            "reasoning": item.get("reasoning", ""),
            "provider": adapter.name,
        }
        # 仅在确信归一化结果合理时覆盖原 prompt（非空且非抽象词）
        if normalized_name and not item.get("abstract", False):
            t["prompt"] = normalized_name
            existing_aliases = t.get("prompt_aliases") or []
            merged: List[str] = []
            seen = set()
            for v in [normalized_name, *aliases, *existing_aliases]:
                k = v.lower().strip()
                if k and k not in seen:
                    seen.add(k)
                    merged.append(v)
            t["prompt_aliases"] = merged

    plan["reasoning_layer"] = adapter.name
    return plan


def _apply_cached_models(plan: Dict[str, Any], registry) -> Dict[str, Any]:
    """对 pipeline 中的每个需要训练的步骤，检查缓存是否可复用"""
    targets = plan.get("targets", [])
    scenario_type = plan.get("scenario_type", "")

    for step in plan.get("model_pipeline", []):
        if not step.get("requires_training", True):
            continue
        if step.get("role") not in ("primary_detector", "secondary_detector", "classifier"):
            continue

        required_classes = [t.get("class_name", "") for t in targets]
        cached = registry.find_reusable_model(
            required_classes=required_classes,
            scenario_type=scenario_type,
        )
        if cached:
            step["requires_training"] = False
            step["reuse_cache_id"] = cached.cache_id
            step["reuse_weight_path"] = cached.weight_path
            step["reuse_info_zh"] = (
                f"复用已训练模型 (mAP50={cached.map50:.2f}), "
                f"来自任务 {cached.task_id}"
            )
            registry.increment_reuse(cached.cache_id)

    # 重算训练数量
    training_strategy = plan.get("training_strategy", {})
    models_needing_training = [
        s for s in plan.get("model_pipeline", [])
        if s.get("requires_training", True) and s.get("role") not in ("tracker", "rule_engine")
    ]
    training_strategy["total_models_to_train"] = len(models_needing_training)
    plan["training_strategy"] = training_strategy

    return plan


def _build_fallback_negotiation(plan: Dict[str, Any], user_description: str) -> Dict[str, Any]:
    targets = plan.get("targets", [])
    events = plan.get("events", [])
    regions = plan.get("regions", [])
    temporal = plan.get("temporal_constraints", [])

    duration = 0
    for tc in temporal:
        d = tc.get("duration_seconds", 0)
        if d and d > duration:
            duration = d

    return {
        "objects": [t.get("display_name_zh") or t.get("class_name", "") for t in targets],
        "regions": [r.get("label", r.get("region_id", "")) for r in regions],
        "events": [e.get("name_zh") or e.get("name", "") for e in events],
        "duration_seconds": duration,
        "user_facing_description_zh": plan.get("summary_zh", f"基于您的需求进行智能视觉分析"),
    }


# ---------------------------------------------------------------------------
# 规则兜底（VLM 不可用时）
# ---------------------------------------------------------------------------

def _fallback_rule_based_plan(
    user_description: str,
    vlm_result: Optional[Dict[str, Any]],
    device_tier: str,
    registry,
) -> Dict[str, Any]:
    """当 VLM 不可用时，退回到规则引擎"""
    from services.algorithm_planner import build_algorithm_plan as legacy_build

    legacy_plan = legacy_build(user_description, vlm_result)

    # 选择合适的模型
    models = registry.list_models(
        task_type="detection",
        max_device_tier=device_tier,
    )
    if not models:
        models = registry.list_models(task_type="detection")

    # 按精度排序，取中间偏小的
    models.sort(key=lambda m: m.params_m)
    selected = models[len(models) // 3] if models else None
    model_id = selected.model_id if selected else "yolo11s.pt"

    targets = legacy_plan.get("targets", [])
    for t in targets:
        t.setdefault("display_name_zh", t.get("class_name", ""))
        t.setdefault("purpose", "检测目标")

    events = legacy_plan.get("events", [])
    for e in events:
        e.setdefault("name_zh", e.get("name", ""))

    legacy_plan["summary_zh"] = legacy_plan.get("summary", "")
    legacy_plan["difficulty_level"] = "moderate"
    legacy_plan["model_pipeline"] = [
        {
            "step_id": "step_1",
            "role": "primary_detector",
            "recommended_model_id": model_id,
            "alternative_model_ids": [m.model_id for m in models[:3]] if models else [],
            "reason_zh": f"根据设备等级 {device_tier} 自动选择",
            "requires_training": bool(targets),
            "training_priority": 1,
            "estimated_training_hours": 2.0,
            "input_size": 640,
            "epochs": 100,
        },
        {
            "step_id": "step_tracker",
            "role": "tracker",
            "recommended_model_id": "bytetrack",
            "alternative_model_ids": [],
            "reason_zh": "ByteTrack 无需训练，用于目标跟踪",
            "requires_training": False,
            "training_priority": 0,
        },
        {
            "step_id": "step_rules",
            "role": "rule_engine",
            "recommended_model_id": "built_in_rules",
            "alternative_model_ids": [],
            "reason_zh": "内置规则引擎处理业务逻辑",
            "requires_training": False,
            "training_priority": 0,
        },
    ]
    legacy_plan["training_strategy"] = {
        "total_models_to_train": 1 if targets else 0,
        "estimated_total_hours": 2.0,
        "train_mode_recommendation": "local",
        "train_mode_reason_zh": "数据量适中，建议本地训练",
        "special_requirements": [],
    }

    return legacy_plan
