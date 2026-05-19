"""
Agent A: 对话引导智能体 (VLM 多模态)

职责:
- 与用户自然语言对话，理解完整算法需求
- 看图辅助理解（样张 + 预览检测结果）
- 引导追问（颜色/大小/事件/区域/排除条件/多模型需求）
- 判断需求是否已收敛（converged）
- 输出语义需求摘要给 Agent B

使用 VLM（多模态模型）驱动，因为需要看图。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

CONVERSATION_SYSTEM_PROMPT = """\
你是 CV Auto Trainer 的需求确认助手，帮助用户明确他们想实现的完整视觉算法方案。

## 你的沟通范围

不限于"检测什么对象"，还包括：
- 业务场景是什么？（占位监测？安全合规？闯入告警？质检？计数？）
- 需要检测哪些对象？每个对象的视觉特征？（颜色、大小、形状）
- 什么事件需要触发告警？（对象进入区域？离开？停留超时？缺失？）
- 是否需要多个模型协作？（检测+分类？检测+OCR？检测+跟踪？）
- 有什么特殊约束？（实时性？边缘设备？精度要求？）
- 什么情况不应该检测？（排除条件）

## 沟通策略

1. **初始**：基于系统的初始理解，先展示你理解了什么，然后一次追问 2-3 个最关键的模糊点
2. **中期**：每轮聚焦 1-2 个维度深挖，不要一次问太多
3. **触发预览**：当类别定义有更新时，设置 should_regenerate=true 让系统做一次预览
4. **收敛判断**：当以下条件都满足时，设置 converged=true
   - 所有检测类别都有明确定义（至少知道是什么、大概什么样）
   - 事件/告警逻辑已明确（或用户明确表示不需要）
   - 排除条件已明确
   - 至少做过 1 次预览且用户未提异议

## 重要规则

- 用中文回复，语气简洁友好
- 不要使用技术术语（CLIP、prompt、class_name、YOLO 等），用户是普通业务人员
- 不要一次问超过 3 个问题
- 如果用户说"可以了"/"没问题"/"就这样"等确认词，且之前条件满足，设置 converged=true
- 如果用户提出新的修改意见，设置 converged=false 并继续对话

## 输出格式

你必须严格输出以下 JSON 格式，不要有任何额外文字：

```json
{
  "reply": "你的中文回复文本",
  "intent_update": {
    "targets": [{"name": "目标名", "description": "视觉描述", "color": "颜色", "size": "大小"}],
    "events": [{"name": "事件名", "trigger_description": "触发条件描述"}],
    "regions": [{"label": "区域名", "purpose": "用途"}],
    "exclusions": ["排除项1", "排除项2"],
    "constraints": ["约束1"],
    "extra_capabilities": ["ocr", "tracking"]
  },
  "should_regenerate": false,
  "should_preview": false,
  "convergence": {
    "converged": false,
    "missing": ["还需确认的内容"]
  }
}
```

注意：
- intent_update 是累积的，每次输出当前已确认的全部需求（不只是本轮新增的）
- should_regenerate=true 表示需求有实质性变化，需要重新生成配置
- should_preview=true 表示建议做一次检测预览让用户验证
- 首轮对话必须设置 should_regenerate=true（需要根据初始理解生成配置）
"""


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

class ConversationResponse:
    """Agent A 的输出结构"""

    def __init__(
        self,
        reply: str,
        intent_update: Optional[Dict[str, Any]] = None,
        should_regenerate: bool = False,
        should_preview: bool = False,
        converged: bool = False,
        missing: Optional[List[str]] = None,
    ):
        self.reply = reply
        self.intent_update = intent_update
        self.should_regenerate = should_regenerate
        self.should_preview = should_preview
        self.converged = converged
        self.missing = missing or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply,
            "intent_update": self.intent_update,
            "should_regenerate": self.should_regenerate,
            "should_preview": self.should_preview,
            "convergence": {
                "converged": self.converged,
                "missing": self.missing,
            },
        }


# ---------------------------------------------------------------------------
# Agent A Core
# ---------------------------------------------------------------------------

class ConversationAgent:
    """VLM 驱动的对话引导智能体"""

    def __init__(self, vlm_adapter):
        self.vlm = vlm_adapter

    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        initial_understanding: Optional[Dict[str, Any]] = None,
        current_config_summary: Optional[str] = None,
        preview_stats: Optional[Dict[str, Any]] = None,
        sample_images_base64: Optional[List[str]] = None,
        is_first_message: bool = False,
    ) -> ConversationResponse:
        """
        处理一轮对话（一次性返回）。
        """
        # 构建 user prompt
        user_prompt = self._build_user_prompt(
            user_message=user_message,
            initial_understanding=initial_understanding,
            current_config_summary=current_config_summary,
            preview_stats=preview_stats,
            is_first_message=is_first_message,
        )

        # 构建多轮消息
        messages = self._build_messages(conversation_history, user_prompt)

        # 调用 VLM
        try:
            raw = self.vlm.backend.call_api(
                messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            )
            return self._parse_response(raw)
        except Exception as exc:
            logger.error("ConversationAgent VLM call failed: %s", exc)
            return ConversationResponse(
                reply="抱歉，AI 助手暂时无法响应，请稍后重试。",
                should_regenerate=False,
                should_preview=False,
                converged=False,
                missing=["系统错误，需重试"],
            )

    def stream_chat(
        self,
        user_message: str,
        conversation_history: List[Dict[str, str]],
        initial_understanding: Optional[Dict[str, Any]] = None,
        current_config_summary: Optional[str] = None,
        preview_stats: Optional[Dict[str, Any]] = None,
        sample_images_base64: Optional[List[str]] = None,
        is_first_message: bool = False,
    ):
        """
        流式对话：yield 文本片段，最后 yield 结构化 metadata。

        Yields:
            str: 文本片段（逐 token/逐句）
            dict: 最后一条元数据 {"type": "done", "response": ConversationResponse}
        """
        user_prompt = self._build_user_prompt(
            user_message=user_message,
            initial_understanding=initial_understanding,
            current_config_summary=current_config_summary,
            preview_stats=preview_stats,
            is_first_message=is_first_message,
        )
        messages = self._build_messages(conversation_history, user_prompt)

        try:
            # 流式调用，拼接完整响应
            full_text = ""
            for chunk in self.vlm.stream_call_api(
                messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            ):
                full_text += chunk
                yield chunk

            response = self._parse_response(full_text)
            yield {"type": "done", "response": response}
        except Exception as exc:
            logger.error("ConversationAgent stream_chat failed: %s", exc)
            yield "抱歉，AI 助手暂时无法响应，请稍后重试。"
            yield {"type": "done", "response": ConversationResponse(
                reply="抱歉，AI 助手暂时无法响应，请稍后重试。",
                should_regenerate=False,
                should_preview=False,
                converged=False,
                missing=["系统错误，需重试"],
            )}

    def generate_opening(
        self,
        initial_understanding: Dict[str, Any],
        user_description: str,
        sample_images_base64: Optional[List[str]] = None,
    ) -> ConversationResponse:
        """
        生成首轮开场白（基于 VLM parse 结果）。
        """
        return self.chat(
            user_message=user_description,
            conversation_history=[],
            initial_understanding=initial_understanding,
            sample_images_base64=sample_images_base64,
            is_first_message=True,
        )

    def _build_user_prompt(
        self,
        user_message: str,
        initial_understanding: Optional[Dict[str, Any]],
        current_config_summary: Optional[str],
        preview_stats: Optional[Dict[str, Any]],
        is_first_message: bool,
    ) -> str:
        parts = []

        # 每轮都注入 VLM parse 的检测目标，确保 Agent A 始终知道用户想检测什么
        if initial_understanding:
            parts.append("## 系统初始理解（VLM 图片分析结果，用户的检测目标）")
            classes = initial_understanding.get("classes", [])
            if classes:
                for cls in classes:
                    name = cls.get("display_name_zh") or cls.get("class_name", "")
                    desc = cls.get("display_prompt_zh") or cls.get("prompt", "")
                    parts.append(f"- {name}: {desc}")
            confidence = initial_understanding.get("confidence")
            if confidence:
                parts.append(f"- 置信度: {confidence}")
            parts.append("重要：intent_update.targets 必须忠实于上述检测目标，而不是图片中你看到的其他物体。")
            parts.append("")

        if current_config_summary:
            parts.append(f"## 当前已确认的配置摘要\n{current_config_summary}\n")

        if preview_stats:
            parts.append("## 最近一次检测预览结果")
            parts.append(f"- 命中: {preview_stats.get('hits', 0)} 个")
            parts.append(f"- 误检: {preview_stats.get('false_positives', 0)} 个")
            parts.append(f"- 漏检: {preview_stats.get('misses', 0)} 个")
            parts.append("")

        if is_first_message:
            parts.append(f"## 用户原始需求描述（最高优先级，以此为准）\n{user_message}")
            parts.append("\n请基于以上信息生成开场白：展示你理解了什么，然后追问 2-3 个关键模糊点。")
        else:
            parts.append(f"## 用户本轮输入\n{user_message}")

        return "\n".join(parts)

    def _build_messages(
        self,
        conversation_history: List[Dict[str, str]],
        current_user_prompt: str,
    ) -> List[Dict[str, Any]]:
        messages = [{"role": "system", "content": CONVERSATION_SYSTEM_PROMPT}]

        # 历史对话（最多保留最近 20 轮，避免 token 超限）
        recent_history = conversation_history[-40:] if len(conversation_history) > 40 else conversation_history
        for msg in recent_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })

        # 当前用户输入
        messages.append({"role": "user", "content": current_user_prompt})
        return messages

    def _parse_response(self, raw: str) -> ConversationResponse:
        """解析 VLM 返回的 JSON 响应"""
        try:
            data = self._extract_json(raw)
        except Exception:
            # VLM 可能返回非 JSON 格式，做容错
            logger.warning("Agent A response not valid JSON, treating as plain reply")
            return ConversationResponse(
                reply=raw.strip(),
                should_regenerate=False,
                should_preview=False,
                converged=False,
                missing=["VLM 输出格式异常"],
            )

        convergence = data.get("convergence", {})
        return ConversationResponse(
            reply=data.get("reply", ""),
            intent_update=data.get("intent_update"),
            should_regenerate=data.get("should_regenerate", False),
            should_preview=data.get("should_preview", False),
            converged=convergence.get("converged", False),
            missing=convergence.get("missing", []),
        )

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """从响应中提取 JSON 对象"""
        if not text:
            raise ValueError("Empty response")
        # 尝试 ```json ``` 包裹
        fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
        if fence_match:
            return json.loads(fence_match.group(1))
        # 尝试直接 JSON
        text_stripped = text.strip()
        if text_stripped.startswith("{"):
            return json.loads(text_stripped)
        # 尝试定位 JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise ValueError(f"No JSON found in: {text[:200]}")
