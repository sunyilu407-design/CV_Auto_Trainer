"""
推理模型适配器（v9.0 P1-A 决策层）。

为 vlm_algorithm_planner 等场景提供「结构化推理」能力，
对应优化文档中「o3-mini / DeepSeek-R1 用作类别词归一化、质检边界判断」的需求。

设计要点：
- 统一为 OpenAI 兼容格式（DeepSeek / OpenAI / Kimi / Qwen / 智谱 / 自定义）
- 单独保留 VLMFallback 以便用户未配置任何推理模型时仍可走 vlm_adapter
- 既支持环境变量配置（部署时方便），也支持从用户 DB 设置注入

调用顺序：
- 显式传入 settings_obj → 优先
- 环境变量 REASONING_PROVIDER + 对应 API Key
- vlm_adapter fallback
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Protocol

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider 默认配置
# ---------------------------------------------------------------------------


REASONING_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
        "supports_json_format": False,
        "use_max_completion_tokens": False,
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "o3-mini",
        "supports_json_format": True,
        "use_max_completion_tokens": True,
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-thinking-preview",
        "supports_json_format": True,
        "use_max_completion_tokens": False,
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwq-plus",
        "supports_json_format": True,
        "use_max_completion_tokens": False,
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-zero-preview",
        "supports_json_format": True,
        "use_max_completion_tokens": False,
    },
    "custom": {
        "base_url": "",
        "model": "",
        "supports_json_format": False,
        "use_max_completion_tokens": False,
    },
}


# ---------------------------------------------------------------------------
# Reasoning Adapter Protocol
# ---------------------------------------------------------------------------


class ReasoningAdapter(Protocol):
    """所有推理后端必须实现的接口。"""

    name: str
    provider: str

    def reason_json(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        ...

    def test_connection(self) -> Dict[str, Any]:
        ...


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Dict[str, Any]:
    """从模型响应中提取首个完整 JSON 对象。容错性较强，能跳过 ```json 包裹。"""
    if not text:
        raise RuntimeError("Empty response")
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fence_match:
        return json.loads(fence_match.group(1))
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            return json.loads(text_stripped)
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise RuntimeError(f"No JSON object found in response: {text[:200]}")


# ---------------------------------------------------------------------------
# OpenAI 兼容推理后端（覆盖 DeepSeek / OpenAI / Kimi / Qwen / 智谱 / Custom）
# ---------------------------------------------------------------------------


class OpenAICompatibleReasoner:
    """单一类覆盖所有 OpenAI 兼容的推理 API。"""

    name = "openai_compatible"

    def __init__(
        self,
        provider: str,
        api_key: str,
        base_url: str = "",
        model: str = "",
    ):
        cfg = REASONING_PROVIDER_CONFIG.get(provider, REASONING_PROVIDER_CONFIG["custom"])
        self.provider = provider
        self.api_key = api_key
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.model = model or cfg["model"]
        self.supports_json_format = cfg["supports_json_format"]
        self.use_max_completion_tokens = cfg["use_max_completion_tokens"]
        self.name = provider

    def _build_payload(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int,
        force_json: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        }
        if self.use_max_completion_tokens:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if force_json and self.supports_json_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def reason_json(
        self,
        system_prompt: str,
        user_text: str,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        payload = self._build_payload(
            system_prompt=system_prompt + "\n\n严格只输出符合要求的 JSON，不要任何额外文字。",
            user_text=user_text,
            max_tokens=max_tokens,
            force_json=True,
        )
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=120,  # 推理模型通常耗时
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _extract_json(content)

    def test_connection(self) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "message": "未配置 API Key"}
        if not self.base_url:
            return {"success": False, "message": "未配置 Base URL"}
        if not self.model:
            return {"success": False, "message": "未配置模型名称"}
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
            }
            if self.use_max_completion_tokens:
                payload["max_completion_tokens"] = 16
            else:
                payload["max_tokens"] = 16
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                return {"success": True, "message": f"连接成功 ({self.provider} / {self.model})"}
            return {
                "success": False,
                "message": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
        except httpx.TimeoutException:
            return {"success": False, "message": "请求超时"}
        except Exception as exc:  # noqa: BLE001 - friendly message
            return {"success": False, "message": str(exc)[:200]}


# ---------------------------------------------------------------------------
# VLM Fallback 后端
# ---------------------------------------------------------------------------


class VLMFallbackReasoner:
    """复用现有 vlm_adapter 做轻量推理；推理稳定性低于专用模型，仅作 fallback。"""

    name = "vlm_fallback"
    provider = "vlm_fallback"

    def __init__(self, vlm_adapter):
        self.vlm = vlm_adapter

    def reason_json(self, system_prompt: str, user_text: str, max_tokens: int = 2048) -> Dict[str, Any]:
        raw = self.vlm.call_with_system_prompt(
            system_prompt=system_prompt + "\n\n严格只输出 JSON 对象，不要解释。",
            user_text=user_text,
            images_base64=[],
            response_format="json",
        )
        return _extract_json(raw)

    def test_connection(self) -> Dict[str, Any]:
        try:
            backend = getattr(self.vlm, "backend", None)
            if backend and hasattr(backend, "test_connection"):
                return backend.test_connection()
            return {"success": False, "message": "VLM Adapter 未提供测试接口"}
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "message": str(exc)[:200]}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _build_from_explicit(
    provider: str,
    api_key: str,
    base_url: str = "",
    model: str = "",
) -> Optional[OpenAICompatibleReasoner]:
    if not api_key:
        return None
    if provider not in REASONING_PROVIDER_CONFIG:
        provider = "custom"
    return OpenAICompatibleReasoner(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def build_reasoning_adapter_from_settings(settings_obj, vlm_adapter=None) -> Optional[ReasoningAdapter]:
    """
    从 UserSettings 行（或同名属性的对象）构建推理适配器。
    settings_obj 期望含字段：
      reasoning_enabled, reasoning_provider, reasoning_base_url,
      reasoning_api_key (已解密), reasoning_model
    """
    if settings_obj is None:
        return get_reasoning_adapter(vlm_adapter)

    enabled = getattr(settings_obj, "reasoning_enabled", True)
    if enabled is False:  # 显式 False（None/未设置则视作启用）
        return None

    provider = (getattr(settings_obj, "reasoning_provider", "") or "").strip() or "deepseek"
    api_key = getattr(settings_obj, "reasoning_api_key", "") or ""
    base_url = getattr(settings_obj, "reasoning_base_url", "") or ""
    model = getattr(settings_obj, "reasoning_model", "") or ""

    explicit = _build_from_explicit(provider, api_key, base_url, model)
    if explicit is not None:
        return explicit

    return get_reasoning_adapter(vlm_adapter)


def get_reasoning_adapter(vlm_adapter=None) -> Optional[ReasoningAdapter]:
    """
    从环境变量构建（兼容老路径）：
    1. REASONING_PROVIDER 环境变量强制指定
    2. DEEPSEEK_API_KEY → DeepSeek
    3. OPENAI_API_KEY → OpenAI
    4. vlm_adapter 不为 None → VLMFallback
    5. 否则 None
    """
    forced = os.getenv("REASONING_PROVIDER", "").strip().lower()

    if forced == "deepseek" or (not forced and os.getenv("DEEPSEEK_API_KEY")):
        key = os.getenv("DEEPSEEK_API_KEY")
        if key:
            return OpenAICompatibleReasoner(
                provider="deepseek",
                api_key=key,
                base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
                model=os.getenv("DEEPSEEK_MODEL", ""),
            )

    if forced == "openai" or (not forced and os.getenv("OPENAI_API_KEY")):
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return OpenAICompatibleReasoner(
                provider="openai",
                api_key=key,
                base_url=os.getenv("OPENAI_BASE_URL", ""),
                model=os.getenv("OPENAI_REASONING_MODEL", ""),
            )

    if (forced == "vlm_fallback" or not forced) and vlm_adapter is not None:
        return VLMFallbackReasoner(vlm_adapter)

    return None


# ---------------------------------------------------------------------------
# 高层用法 #1：类别词归一化
# ---------------------------------------------------------------------------


CATEGORY_NORMALIZATION_SYSTEM_PROMPT = """\
你是计算机视觉系统的「类别词专家」，负责把用户提供的中英混合类别名归一化成 YOLO-World / CLIP 友好的英文检测词。

规则：
1. class_name_en 必须是单一英文名词或短名词短语（≤3 个单词），描述具体可见物体
2. aliases 是 3-8 个同义词或视觉等价描述（颜色 + 物体、形状 + 物体），用于扩大召回
3. 抽象词（danger/safety/zone/violation/事件名）必须替换为具体物体；如果实在没有，标记 "abstract": true
4. 状态/动作（running/working/工作中）必须改为对应的静态实体
5. 提供 1-3 句 reasoning 解释为什么这样归一化

返回 JSON 结构：
{
  "items": [
    {
      "input_class_name": "<原 class_name>",
      "class_name_en": "<归一化英文名>",
      "aliases": ["alias1", "alias2", ...],
      "abstract": false,
      "reasoning": "..."
    }
  ]
}
"""


def normalize_categories(
    classes: List[Dict[str, Any]],
    adapter: Optional[ReasoningAdapter] = None,
) -> Optional[Dict[str, Any]]:
    """
    给 vlm_algorithm_planner 输出的 targets/classes 列表做后置归一化。
    返回 None 表示推理后端不可用，调用方应保留原始数据。
    """
    if adapter is None:
        return None

    payload = []
    for c in classes:
        payload.append({
            "input_class_name": c.get("class_name", ""),
            "display_name_zh": c.get("display_name_zh", ""),
            "current_prompt": c.get("prompt", ""),
            "current_aliases": c.get("prompt_aliases", []) or [],
            "color_hint": c.get("color_hint", ""),
        })

    user_text = (
        "请把以下类别归一化为 CLIP/YOLO-World 友好的英文检测词。\n"
        f"输入：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "严格按照系统提示中的 JSON 结构返回。"
    )

    try:
        return adapter.reason_json(
            system_prompt=CATEGORY_NORMALIZATION_SYSTEM_PROMPT,
            user_text=user_text,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning("Category normalization failed via %s: %s", adapter.name, exc)
        return None
