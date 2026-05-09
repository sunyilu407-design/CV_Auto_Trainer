"""
Agent B: 配置生成智能体 (推理模型)

职责:
- 接收 Agent A 收集的语义需求摘要
- 生成 CLIP 友好的 class_config（VLMClass 格式）
- 生成检测规则 detection_rules（置信度阈值、后处理过滤）
- 生成开放词汇表 vocab（prompt_aliases 扩展）
- 生成 algorithm_hints（场景类型、事件、多模型需求）

使用推理模型（DeepSeek-R1/QwQ/o3-mini）驱动，需要强逻辑推理能力。
如果用户未配置推理模型，降级使用 VLM。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from services.reasoning_adapter import (
    ReasoningAdapter,
    VLMFallbackReasoner,
    build_reasoning_adapter_from_settings,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output Schema
# ---------------------------------------------------------------------------

CONFIG_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["classes", "detection_rules", "vocab", "algorithm_hints"],
    "properties": {
        "classes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["class_name", "prompt", "display_name_zh"],
                "properties": {
                    "class_name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "prompt_aliases": {"type": "array", "items": {"type": "string"}},
                    "negative_prompt": {"type": "string"},
                    "color_hint": {"type": "string"},
                    "display_name_zh": {"type": "string"},
                    "display_prompt_zh": {"type": "string"},
                    "display_negative_prompt_zh": {"type": "string"},
                    "display_color_hint_zh": {"type": "string"},
                },
            },
        },
        "detection_rules": {
            "type": "object",
            "properties": {
                "conf_threshold": {"type": "number"},
                "iou_threshold": {"type": "number"},
                "post_filters": {"type": "array", "items": {"type": "object"}},
            },
        },
        "vocab": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "primary": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "context_anchors": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "algorithm_hints": {
            "type": "object",
            "properties": {
                "scenario_type": {"type": "string"},
                "needs_tracking": {"type": "boolean"},
                "needs_ocr": {"type": "boolean"},
                "events": {"type": "array", "items": {"type": "object"}},
                "regions": {"type": "array", "items": {"type": "object"}},
                "performance_hint": {"type": "string"},
                "multi_model_needed": {"type": "boolean"},
                "suggested_pipeline_roles": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CONFIG_GENERATOR_SYSTEM_PROMPT = """\
你是结构化配置生成专家。根据已确认的业务需求，生成检测系统的完整配置。

## 生成规则

### classes（类别配置，给 YOLO-World 自动标注用）

1. `class_name`：必须是 CLIP 友好的 1-3 词英文小写名词，下划线连接
   - 优先从 CLIP 训练分布常见词中选择
   - 禁止抽象词（danger/safety/zone/violation）
   - 禁止状态/动作词（running/working）
2. `prompt`：简短英文名词短语（1-4 个词），这是交给 YOLO-World set_classes() 的文本
3. `prompt_aliases`：3-5 个同义词/等价描述（短名词），用于扩大召回
   - 最多 5 个（超过反而会降低检测置信度）
   - 覆盖不同方面（名称、形状、材质、功能），不要冗余颜色变体
4. `negative_prompt`：容易混淆但不应检测的排除项
5. `color_hint`：主要颜色特征
6. `display_name_zh`：中文展示名（简洁）
7. `display_prompt_zh`：中文详细描述
8. `display_negative_prompt_zh`：中文排除项说明
9. `display_color_hint_zh`：中文颜色提示

### detection_rules（检测规则）

1. `conf_threshold`：置信度阈值（0.1-0.5，小目标建议 0.15-0.25）
2. `iou_threshold`：NMS IoU 阈值（通常 0.45）
3. `post_filters`：后处理过滤规则数组
   - `{"type": "min_area", "value": 0.005, "unit": "relative"}`（最小相对面积）
   - `{"type": "max_area", "value": 0.3, "unit": "relative"}`（最大相对面积）
   - 只包含用户明确确认过的约束，不要编造未确认的精确数值

### vocab（开放词汇表）

每个 class_name 对应一个词汇条目：
- `primary`：主检测词（= prompt）
- `aliases`：别名列表（= prompt_aliases）
- `context_anchors`：上下文锚点词（关联对象，帮助理解场景）

### algorithm_hints（算法方案提示，给后续 AlgorithmPlan 用）

- `scenario_type`：场景类型（occupancy_monitoring/parking_violation/intrusion_monitoring/
  dwell_time_monitoring/object_tracking/object_counting/safety_compliance/
  quality_inspection/feature_matching/classification/custom_event_monitoring）
- `needs_tracking`：是否需要目标跟踪
- `needs_ocr`：是否需要文字识别
- `events`：事件定义列表 [{"name_zh": "...", "trigger": "..."}]
- `regions`：区域定义 [{"label": "...", "purpose": "..."}]
- `performance_hint`：性能要求（real_time/near_real_time/offline）
- `multi_model_needed`：是否需要多模型协作
- `suggested_pipeline_roles`：建议的 pipeline 角色列表
  （primary_detector/secondary_detector/classifier/tracker/ocr/rule_engine）

## CLIP 友好参考词表

- People: person, worker, pedestrian, cyclist, driver
- Vehicles: car, truck, bus, motorcycle, bicycle, forklift, excavator
- Safety: hard_hat, safety_vest, face_mask, glove
- Industrial: bottle, box, pallet, pipe, valve, screw, bearing
- Animals: dog, cat, bird, cow, horse
- Medical: syringe, pill, bandage

## 输出格式

严格输出以下 JSON 结构，不允许额外文本：

{
  "classes": [...],
  "detection_rules": {...},
  "vocab": {...},
  "algorithm_hints": {...}
}
"""


# ---------------------------------------------------------------------------
# Agent B Core
# ---------------------------------------------------------------------------

class ConfigGeneratorAgent:
    """推理模型驱动的结构化配置生成智能体"""

    def __init__(self, reasoning_adapter: ReasoningAdapter):
        self.reasoner = reasoning_adapter

    @classmethod
    def create(cls, settings_obj=None, vlm_adapter=None) -> "ConfigGeneratorAgent":
        """
        工厂方法：推理模型优先，VLM 兜底。
        """
        reasoner = build_reasoning_adapter_from_settings(settings_obj, vlm_adapter)
        if reasoner is None and vlm_adapter is not None:
            reasoner = VLMFallbackReasoner(vlm_adapter)
        if reasoner is None:
            raise RuntimeError("无可用的推理后端：请至少配置 VLM 模型")
        return cls(reasoner)

    def generate_config(
        self,
        intent_summary: Dict[str, Any],
        conversation_context: str = "",
    ) -> Dict[str, Any]:
        """
        根据 Agent A 收集的需求摘要，生成完整配置。

        Args:
            intent_summary: Agent A 输出的 intent_update 结构
            conversation_context: 对话上下文摘要文本

        Returns:
            完整配置 dict (classes + detection_rules + vocab + algorithm_hints)
        """
        user_text = self._build_user_prompt(intent_summary, conversation_context)

        try:
            result = self.reasoner.reason_json(
                system_prompt=CONFIG_GENERATOR_SYSTEM_PROMPT,
                user_text=user_text,
                max_tokens=4096,
            )
            # 基本校验
            self._validate_config(result)
            # 限制 aliases 数量（防止 CLIP 召回退化）
            self._cap_aliases(result)
            return result
        except Exception as exc:
            logger.error("ConfigGeneratorAgent failed: %s", exc)
            raise RuntimeError(f"配置生成失败: {exc}") from exc

    def _build_user_prompt(
        self,
        intent_summary: Dict[str, Any],
        conversation_context: str,
    ) -> str:
        parts = ["请根据以下已确认需求，生成检测系统的完整配置。\n"]

        # 检测目标
        targets = intent_summary.get("targets", [])
        if targets:
            parts.append("## 检测目标")
            for t in targets:
                line = f"- {t.get('name', '未知')}"
                if t.get("description"):
                    line += f"：{t['description']}"
                if t.get("color"):
                    line += f"（颜色: {t['color']}）"
                if t.get("size"):
                    line += f"（大小: {t['size']}）"
                parts.append(line)
            parts.append("")

        # 事件
        events = intent_summary.get("events", [])
        if events:
            parts.append("## 事件/告警")
            for e in events:
                parts.append(f"- {e.get('name', '')}: {e.get('trigger_description', '')}")
            parts.append("")

        # 区域
        regions = intent_summary.get("regions", [])
        if regions:
            parts.append("## 区域约束")
            for r in regions:
                parts.append(f"- {r.get('label', '')}: {r.get('purpose', '')}")
            parts.append("")

        # 排除条件
        exclusions = intent_summary.get("exclusions", [])
        if exclusions:
            parts.append("## 排除条件")
            for ex in exclusions:
                parts.append(f"- {ex}")
            parts.append("")

        # 约束
        constraints = intent_summary.get("constraints", [])
        if constraints:
            parts.append("## 特殊约束")
            for c in constraints:
                parts.append(f"- {c}")
            parts.append("")

        # 额外能力
        extra = intent_summary.get("extra_capabilities", [])
        if extra:
            parts.append(f"## 额外能力需求: {', '.join(extra)}\n")

        # 对话上下文
        if conversation_context:
            parts.append(f"## 对话上下文摘要\n{conversation_context}\n")

        parts.append("严格按照系统提示中的 JSON 结构返回。")
        return "\n".join(parts)

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """基本结构校验"""
        if "classes" not in config or not config["classes"]:
            raise ValueError("配置缺少 classes 或为空")
        for cls in config["classes"]:
            if not cls.get("class_name"):
                raise ValueError(f"类别缺少 class_name: {cls}")
            if not cls.get("prompt"):
                raise ValueError(f"类别缺少 prompt: {cls}")

    @staticmethod
    def _cap_aliases(config: Dict[str, Any], max_aliases: int = 5) -> None:
        """限制 aliases 数量，防止 CLIP 召回退化"""
        for cls in config.get("classes", []):
            aliases = cls.get("prompt_aliases", [])
            if len(aliases) > max_aliases:
                cls["prompt_aliases"] = aliases[:max_aliases]

        for key, vocab_entry in config.get("vocab", {}).items():
            aliases = vocab_entry.get("aliases", [])
            if len(aliases) > max_aliases:
                vocab_entry["aliases"] = aliases[:max_aliases]
