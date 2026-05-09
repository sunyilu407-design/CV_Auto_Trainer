"""
需求确认编排器 (Negotiation Orchestrator)

协调 Agent A (对话引导) + Agent B (配置生成) 的调用顺序：
1. 从 DB 加载对话历史
2. 调用 Agent A → 获取 reply + intent_update + should_regenerate
3. if should_regenerate → 调用 Agent B → 获取 updated_config
4. 追加本轮消息到历史
5. 返回前端: reply + config + convergence
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.db import NegotiationConversation
from services.conversation_agent import ConversationAgent, ConversationResponse
from services.config_generator_agent import ConfigGeneratorAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response DTO
# ---------------------------------------------------------------------------

class OrchestratorResponse:
    """编排器最终返回给前端的结构"""

    def __init__(
        self,
        conversation_id: str,
        reply: str,
        updated_config: Optional[Dict[str, Any]] = None,
        should_preview: bool = False,
        converged: bool = False,
        missing: Optional[List[str]] = None,
    ):
        self.conversation_id = conversation_id
        self.reply = reply
        self.updated_config = updated_config
        self.should_preview = should_preview
        self.converged = converged
        self.missing = missing or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "reply": self.reply,
            "updated_config": self.updated_config,
            "should_preview": self.should_preview,
            "convergence": {
                "converged": self.converged,
                "missing": self.missing,
            },
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class NegotiationOrchestrator:
    """
    编排 Agent A + Agent B 的协作。
    """

    def __init__(
        self,
        vlm_adapter,
        settings_obj=None,
        db: Optional[Session] = None,
    ):
        self.vlm_adapter = vlm_adapter
        self.settings_obj = settings_obj
        self.db = db
        self.agent_a = ConversationAgent(vlm_adapter)
        self.agent_b = ConfigGeneratorAgent.create(settings_obj, vlm_adapter)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_chat(
        self,
        task_id: str,
        message: str,
        conversation_id: Optional[str] = None,
        preview_stats: Optional[Dict[str, Any]] = None,
        sample_images_base64: Optional[List[str]] = None,
        initial_understanding: Optional[Dict[str, Any]] = None,
        user_description: Optional[str] = None,
        is_init_signal: bool = False,
    ) -> OrchestratorResponse:
        """
        处理一轮对话。

        Args:
            task_id: 任务 ID
            message: 用户消息
            conversation_id: 对话 ID（首次为 None，后端创建）
            preview_stats: 预览统计
            sample_images_base64: 样张
            initial_understanding: VLM parse 初始结果（首轮时提供）
        """
        # 1. 加载或创建对话
        conv = self._get_or_create_conversation(task_id, conversation_id)

        # 判断是否首轮
        existing_messages = conv.messages or []
        is_first = len(existing_messages) == 0
        logger.info(
            "handle_chat: task=%s, conv_id=%s, is_first=%s, is_init=%s, msg_count=%d, "
            "has_initial=%s, user_desc=%s",
            task_id, conv.id, is_first, is_init_signal, len(existing_messages),
            bool(initial_understanding),
            (user_description or "")[:60],
        )

        # __INIT__ + 已有对话 + 有用户真实消息：恢复上次状态
        has_user_messages = any(m.get("role") == "user" for m in existing_messages)
        if is_init_signal and has_user_messages:
            last_assistant = next(
                (m for m in reversed(existing_messages) if m.get("role") == "assistant"),
                None,
            )
            reply = last_assistant["content"] if last_assistant else "对话已恢复，请继续。"
            logger.info("__INIT__ on existing conv (%d msgs, has user msgs): restoring", len(existing_messages))
            return OrchestratorResponse(
                conversation_id=conv.id,
                reply=reply,
                updated_config=conv.current_config,
                should_preview=False,
                converged=conv.confirmed or False,
                missing=[],
            )

        # __INIT__ + 旧对话只有 AI 回复（无用户消息）：视为需要重新开始
        if is_init_signal and len(existing_messages) > 0 and not has_user_messages:
            logger.info("__INIT__ on stale conv (%d msgs, no user msgs): resetting to first turn", len(existing_messages))
            conv.messages = []
            existing_messages = []
            is_first = True

        # 当前配置摘要
        config_summary = self._summarize_config(conv.current_config) if conv.current_config else None

        import time as _time

        # 2. 首轮优化：跳过 Agent A + Agent B，直接用结构化开场白
        #    响应 <1s，配置留到用户回复后再生成
        if is_first and initial_understanding:
            classes = initial_understanding.get("classes", [])
            target_names = [
                c.get("display_name_zh") or c.get("class_name", "")
                for c in classes if c.get("display_name_zh") or c.get("class_name")
            ]
            targets_str = "、".join(target_names) if target_names else "目标"
            structured_reply = (
                f"你好！根据你上传的图片和需求描述，"
                f"我理解你需要检测以下目标：**{targets_str}**。\n\n"
                f"在为你生成检测配置之前，我需要确认几个问题：\n\n"
                f"1. 除了{targets_str}，还有其他需要检测的对象吗？\n"
                f"2. 检测到这些目标后，需要触发什么动作或告警吗？\n"
                f"3. 有什么情况下出现的{targets_str}不需要检测？（排除条件）\n\n"
                f"请直接回复你的补充说明，或者输入「没有其他要求」让我直接生成配置。"
            )
            agent_a_response = ConversationResponse(
                reply=structured_reply,
                should_regenerate=False,
                should_preview=False,
                converged=False,
                missing=["待确认事件/告警", "待确认排除条件"],
            )
            # 首轮就用 VLM parse 生成基础配置，前端立即可展示
            updated_config = self._fallback_config_from_vlm(initial_understanding)
            conv.current_config = updated_config
            logger.info("First turn: structured opening + fallback config, no LLM call (<1s)")

        else:
            # 非首轮：调用 Agent A（始终传入 initial_understanding + user_description）
            _t0 = _time.monotonic()
            agent_a_response = self.agent_a.chat(
                user_message=message,
                conversation_history=conv.messages or [],
                initial_understanding=initial_understanding,
                current_config_summary=config_summary,
                preview_stats=preview_stats,
                sample_images_base64=sample_images_base64,
                is_first_message=False,
            )
            logger.info(
                "Agent A done in %.1fs: reply_len=%d, should_regen=%s, converged=%s",
                _time.monotonic() - _t0,
                len(agent_a_response.reply),
                agent_a_response.should_regenerate,
                agent_a_response.converged,
            )

            # 生成/更新配置
            updated_config = None
            if agent_a_response.should_regenerate and agent_a_response.intent_update:
                # 后续轮次：正常使用 Agent A 的 intent_update
                try:
                    context = self._build_conversation_context(conv.messages or [], message)
                    if user_description:
                        context = f"用户原始需求描述：{user_description}\n\n{context}"
                    updated_config = self.agent_b.generate_config(
                        intent_summary=agent_a_response.intent_update,
                        conversation_context=context,
                    )
                    conv.current_config = updated_config
                    conv.algorithm_hints = updated_config.get("algorithm_hints")
                except Exception as exc:
                    logger.warning("Agent B config generation failed: %s", exc)
                    # Agent B 失败时用 VLM parse 兜底，确保前端有配置可展示
                    if initial_understanding and not conv.current_config:
                        updated_config = self._fallback_config_from_vlm(initial_understanding)
                        conv.current_config = updated_config
                        logger.info("Using fallback config from VLM parse after Agent B failure")
                    agent_a_response.reply += "\n\n（配置生成暂时失败，已使用基础配置，不影响继续沟通）"

        # 4. 更新对话历史（__INIT__ 信号不记录为用户消息）
        messages = list(conv.messages or [])
        if message and not is_init_signal:
            messages.append({
                "role": "user",
                "content": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        messages.append({
            "role": "assistant",
            "content": agent_a_response.reply,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "should_regenerate": agent_a_response.should_regenerate,
                "should_preview": agent_a_response.should_preview,
                "converged": agent_a_response.converged,
                "config_updated": updated_config is not None,
            },
        })
        conv.messages = messages
        conv.confirmed = False  # 有新消息，未确认
        conv.updated_at = datetime.now(timezone.utc)

        # 5. 持久化
        if self.db:
            self.db.add(conv)
            self.db.commit()
            self.db.refresh(conv)

        # 6. 返回
        return OrchestratorResponse(
            conversation_id=conv.id,
            reply=agent_a_response.reply,
            updated_config=updated_config,
            should_preview=agent_a_response.should_preview,
            converged=agent_a_response.converged,
            missing=agent_a_response.missing,
        )

    def confirm(self, task_id: str, conversation_id: str) -> Dict[str, Any]:
        """
        确认需求，标记对话完成，返回最终配置。
        """
        conv = self._get_conversation(task_id, conversation_id)
        if not conv:
            raise ValueError(f"对话不存在: {conversation_id}")

        conv.confirmed = True
        conv.updated_at = datetime.now(timezone.utc)

        if self.db:
            self.db.add(conv)
            self.db.commit()

        return {
            "finalized_config": conv.current_config,
            "algorithm_hints": conv.algorithm_hints,
            "conversation_id": conv.id,
        }

    def get_conversation(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        获取任务的对话记录（用于恢复）。
        """
        conv = self._get_latest_conversation(task_id)
        if not conv:
            return None
        return {
            "conversation_id": conv.id,
            "messages": conv.messages or [],
            "current_config": conv.current_config,
            "algorithm_hints": conv.algorithm_hints,
            "confirmed": conv.confirmed,
            "preview_count": conv.preview_count,
        }

    def increment_preview_count(self, task_id: str, conversation_id: str) -> None:
        """预览计数 +1"""
        conv = self._get_conversation(task_id, conversation_id)
        if conv:
            conv.preview_count = (conv.preview_count or 0) + 1
            conv.updated_at = datetime.now(timezone.utc)
            if self.db:
                self.db.add(conv)
                self.db.commit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_or_create_conversation(
        self, task_id: str, conversation_id: Optional[str]
    ) -> NegotiationConversation:
        """获取已有对话或创建新对话"""
        if conversation_id and self.db:
            conv = (
                self.db.query(NegotiationConversation)
                .filter_by(id=conversation_id, task_id=task_id)
                .first()
            )
            if conv:
                return conv

        # 尝试找到该任务的最新未确认对话
        existing = self._get_latest_conversation(task_id)
        if existing and not existing.confirmed:
            return existing

        # 创建新对话（立即 commit 释放写锁，避免后续 VLM 调用期间阻塞其他请求）
        conv = NegotiationConversation(task_id=task_id)
        if self.db:
            self.db.add(conv)
            self.db.commit()
        return conv

    def _get_conversation(
        self, task_id: str, conversation_id: str
    ) -> Optional[NegotiationConversation]:
        if not self.db:
            return None
        return (
            self.db.query(NegotiationConversation)
            .filter_by(id=conversation_id, task_id=task_id)
            .first()
        )

    def _get_latest_conversation(self, task_id: str) -> Optional[NegotiationConversation]:
        if not self.db:
            return None
        return (
            self.db.query(NegotiationConversation)
            .filter_by(task_id=task_id)
            .order_by(NegotiationConversation.updated_at.desc())
            .first()
        )

    @staticmethod
    def _fallback_config_from_vlm(initial_understanding: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agent B 失败时的兜底：直接把 VLM parse 的 classes 包装为初始配置。
        不如 Agent B 生成的完整（缺少 vocab/algorithm_hints），但至少能显示正确的类别。
        """
        classes = initial_understanding.get("classes", [])
        return {
            "classes": classes,
            "detection_rules": {
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "post_filters": [{"type": "min_area", "value": 0.001, "unit": "relative"}],
            },
            "vocab": {
                cls.get("class_name", f"class_{i}"): {
                    "primary": cls.get("prompt", ""),
                    "aliases": cls.get("prompt_aliases", []),
                    "context_anchors": [],
                }
                for i, cls in enumerate(classes)
            },
            "algorithm_hints": None,
        }

    @staticmethod
    def _build_intent_from_vlm_result(
        initial_understanding: Dict[str, Any],
        agent_a_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        从 VLM parse 的可靠结果构建 intent_summary 给 Agent B。
        首轮使用：确保 Agent B 配置基于真实 VLM 结果，不走 Agent A 可能偏差的输出。
        """
        classes = initial_understanding.get("classes", [])
        targets = []
        for cls in classes:
            target = {
                "name": cls.get("display_name_zh") or cls.get("class_name", ""),
                "description": cls.get("display_prompt_zh") or cls.get("prompt", ""),
            }
            color = cls.get("display_color_hint_zh") or cls.get("color_hint", "")
            if color:
                target["color"] = color
            neg = cls.get("display_negative_prompt_zh") or cls.get("negative_prompt", "")
            if neg:
                target["exclusion_note"] = neg
            targets.append(target)

        # 合并 Agent A 可能补充的事件、区域等（如果有）
        intent: Dict[str, Any] = {"targets": targets}
        if agent_a_intent:
            for key in ("events", "regions", "exclusions", "constraints", "extra_capabilities"):
                val = agent_a_intent.get(key)
                if val:
                    intent[key] = val

        return intent

    @staticmethod
    def _summarize_config(config: Dict[str, Any]) -> str:
        """简要摘要当前配置，供 Agent A 参考"""
        parts = []
        classes = config.get("classes", [])
        if classes:
            names = [c.get("display_name_zh") or c.get("class_name", "") for c in classes]
            parts.append(f"检测目标: {', '.join(names)}")

        hints = config.get("algorithm_hints", {})
        if hints.get("scenario_type"):
            parts.append(f"场景: {hints['scenario_type']}")
        events = hints.get("events", [])
        if events:
            event_names = [e.get("name_zh", "") for e in events]
            parts.append(f"事件: {', '.join(event_names)}")

        rules = config.get("detection_rules", {})
        if rules.get("conf_threshold"):
            parts.append(f"置信度阈值: {rules['conf_threshold']}")

        return " | ".join(parts) if parts else "无"

    @staticmethod
    def _build_conversation_context(
        history: List[Dict[str, Any]], current_message: str
    ) -> str:
        """构建精简对话上下文给 Agent B"""
        # 最近 5 轮用户消息作为上下文
        user_msgs = [
            m["content"] for m in history if m.get("role") == "user"
        ][-5:]
        user_msgs.append(current_message)
        return "用户对话摘要：\n" + "\n".join(f"- {m}" for m in user_msgs)
