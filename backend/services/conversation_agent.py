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

## 你的职责

通过自然对话，帮助用户补充算法方案中缺失的关键信息，而不是重复用户已经说过的话。

## 核心原则

1. 不重复用户说过的话：如果用户已经说了"禁止闯入"，不要再问"需要检测闯入吗"
2. 追问缺失的维度：用户说"检测人"，但没说场景 → 问场景；用户说了"告警"，但没说方式 → 问告警方式
3. 基于上下文智能生成：根据用户已经提供的信息，推理出还需要补充什么
4. 回复简洁自然：像人与人对话，不要机械列表式提问

## 收敛条件（最重要！）

当用户的需求已经足够清晰时，**必须**设置 converged=true：

**必须收敛的情况（满足任一即可）：**
- 用户明确确认：说"可以了"、"没问题"、"就这样"、"直接生成"、"直接生成配置"、"跳过追问"、"确认"、"好的，生成配置"、"开始训练"等
- 核心要素已明确：
  - 检测目标已明确（用户说了"任何人"或具体类别）**且**
  - 事件/告警逻辑已明确（用户说了"出现就告警"、"禁止闯入"、"进入就报警"等）**且**
  - 排除条件已明确或用户表示不需要

**常见收敛示例：**
- 用户说"走廊检测人员，出现就告警" → 收敛 ✓
- 用户说"检测人，有人就报警" → 收敛 ✓
- 用户说"检测任何物体出现" → 收敛 ✓
- 用户说"可以了，生成配置吧" → 收敛 ✓
- 用户说"直接生成配置，不需要更多追问了" → 收敛 ✓（最高优先级）
- 用户说"跳过追问" → 收敛 ✓（最高优先级）

**必须继续追问的情况：**
- 用户说的很模糊（"检测人"但没说场景、没说告警）
- 用户明确表达了不确定性（"我不太确定..."、"帮我建议..."）

## 常见算法维度（按需追问，不要全部问）

场景类型：用户说了"闯入告警"等则跳过，用户只说"检测人"时要问
检测目标：用户说了"任何人"则跳过，用户抽象说要问具体
告警方式：用户说了"声音报警"等则跳过，用户没说事件逻辑时要问
排除条件：用户说了"不要检测XX"则跳过，用户没说时要问
精度/实时性：用户说了"200ms内"等则跳过，用户没说时可问
多模型协作：用户说了"检测+跟踪"等则跳过，用户抽象说要问

## 沟通策略

1. 基于上下文智能生成追问，不要固定模板
2. 每轮聚焦 1-2 个维度深挖，不要一次问太多
3. 当类别定义有更新时，设置 should_regenerate=true 让系统做一次预览
4. **当满足收敛条件时，直接收敛，不要继续追问！**

## 重要规则

- 用中文回复，语气简洁友好
- 不要使用技术术语（CLIP、prompt、class_name、YOLO 等），用户是普通业务人员
- 不要一次问超过 3 个问题
- 如果用户说"可以了"/"没问题"/"就这样"等确认词，设置 converged=true
- 如果用户提出新的修改意见，设置 converged=false 并继续对话

## 输出格式

你必须严格输出以下 JSON 格式，不要有任何额外文字：

```json
{
  "reply": "你的中文回复文本（简洁自然，像人与人对话）",
  "intent_update": {
    "targets": [{"name": "目标名", "description": "视觉描述"}],
    "events": [{"name": "事件名", "trigger_description": "触发条件描述"}],
    "regions": [{"label": "区域名", "purpose": "用途"}],
    "exclusions": ["排除项1", "排除项2"],
    "constraints": ["约束1"],
    "extra_capabilities": ["ocr", "tracking"]
  },
  "should_regenerate": true,
  "should_preview": false,
  "convergence": {
    "converged": false,
    "missing": ["还需确认的内容（如果有的话）"]
  }
}
```

注意：
- reply 必须是自然的对话，不要机械列表
- should_regenerate=true 表示需要根据 intent_update 生成配置
- 首轮对话必须设置 should_regenerate=true
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
        流式对话：先收集完整响应，解析后模拟打字效果流式发送 reply。

        Yields:
            str: reply 文本片段（逐句）
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
            # 收集完整响应
            full_text = ""
            for chunk in self.vlm.stream_call_api(
                messages,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            ):
                full_text += chunk

            # 解析 JSON 获取 reply
            response = self._parse_response(full_text)
            reply_text = response.reply

            # 模拟打字效果：按句子/段落分批发送
            # 先处理换行分段
            sentences = reply_text.split('\n')
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    yield sentence
                if i < len(sentences) - 1:
                    yield '\n'

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
                    if not cls or not isinstance(cls, dict):
                        continue
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
            # 跳过空消息或无效消息
            if not msg or not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                continue
            messages.append({
                "role": role,
                "content": content,
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
        reply = data.get("reply") or ""
        if not reply:
            logger.warning("Agent A response has empty reply, using fallback")
            reply = "抱歉，暂时无法生成回复，请重试。"
        return ConversationResponse(
            reply=reply,
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
