import json
import re
import httpx
from abc import ABC, abstractmethod
from jsonschema import validate, ValidationError
from typing import Optional

TASK_SCHEMA = {
    "type": "object",
    "required": ["classes"],
    "properties": {
        "confidence": {"type": "number"},
        "classes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["class_name", "prompt"],
                "properties": {
                    "class_name": {"type": "string"},
                    "prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "color_hint": {"type": ["string", "null"]},
                    "display_name_zh": {"type": "string"},
                    "display_prompt_zh": {"type": "string"},
                    "display_negative_prompt_zh": {"type": "string"},
                    "display_color_hint_zh": {"type": ["string", "null"]},
                },
            },
        }
    },
}

SYSTEM_PROMPT_TEMPLATE = """
你是一个计算机视觉标注任务专家。用户会提供若干参考图片（已画框）和口语化描述。
{box_context}
你必须同时参考用户的文字描述和参考图片，不允许只看图片忽略文字描述。
其中，用户的文字描述是高优先级约束；图片和参考框用于补充外观、颜色、视角、遮挡、尺度等视觉细节。
你需要输出一个标准 JSON，描述每个需要检测的类别，并给出整体解析置信度。

要求：
1. `class_name` 仍然使用英文小写下划线，作为内部标识。
2. `prompt` 必须是更详细的英文视觉描述，长度优先充分，不要过短，需覆盖形状、材质、典型外观、场景、姿态、颜色、排除项等关键信息。
3. `negative_prompt` 用英文写清楚容易混淆但不应算作该类的情况。
4. `color_hint` 可用简短中文或英文。
5. `display_name_zh` 必须是给前端页面展示用的简洁中文类别名。
6. `display_prompt_zh` 必须是给前端展示的中文详细描述，要和英文 `prompt` 语义一致。
7. `display_negative_prompt_zh` 必须是给前端展示的中文排除项描述。
8. `display_color_hint_zh` 必须是给前端展示的中文颜色提示，可为 null。
9. `confidence` 为 0 到 1 之间的小数，表示你对整体类别解析结果的把握程度。

输出格式（仅输出 JSON，不要有任何额外文字）：
{{
  "confidence": 0.82,
  "classes": [
    {{
      "class_name": "英文类别名（小写下划线）",
      "prompt": "详细的英文视觉描述，用于引导目标检测模型定位目标",
      "negative_prompt": "该类别不包含的特征描述",
      "color_hint": "主要颜色特征（可为null）",
      "display_name_zh": "给页面展示的中文类别名",
      "display_prompt_zh": "给页面展示的中文详细描述",
      "display_negative_prompt_zh": "给页面展示的中文排除项描述",
      "display_color_hint_zh": "给页面展示的中文颜色提示（可为null）"
    }}
  ]
}}
"""


# Provider 默认配置（使用各厂家最新多模态模型，2026年4月）
PROVIDER_CONFIG = {
    "openai": {
        "api_format": "openai",
        "model": "gpt-4.1",
        "base_url": "https://api.openai.com/v1",
    },
    "kimi": {
        "api_format": "openai",
        "model": "kimi-k2.5",
        "base_url": "https://api.moonshot.cn/v1",
    },
    "minimax": {
        "api_format": "openai",
        "model": "MiniMax-M2.7",
        "base_url": "https://api.minimax.chat/v1",
    },
    "zhipu": {
        "api_format": "openai",
        "model": "glm-4v-plus",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
    },
    "gemini": {
        "api_format": "gemini",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
    },
    "claude": {
        "api_format": "anthropic",
        "model": "claude-sonnet-4-6",
        "base_url": "https://api.anthropic.com/v1",
    },
    "custom": {
        "api_format": "openai",  # 默认，可切换
        "model": "",
        "base_url": "",
    },
}


class VLMBackend(ABC):
    """VLM 后端抽象基类"""

    @abstractmethod
    def call_api(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """调用 API，返回原始响应内容"""
        pass

    @abstractmethod
    def test_connection(self) -> dict:
        """测试连接，返回 {"success": bool, "message": str}"""
        pass

    def build_content(self, images_base64: list[str], user_text: str) -> list:
        """构建消息内容（图片+文字）"""
        content = [{"type": "text", "text": user_text}]
        for img_b64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        return content


class OpenAICompatibleBackend(VLMBackend):
    """OpenAI 兼容格式后端（OpenAI/Kimi/MiniMax/智谱/自定义中转）"""

    def __init__(self, base_url: str, api_key: str, model: str = "gpt-4o"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def call_api(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = stop
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def test_connection(self) -> dict:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            }
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                return {"success": True, "message": "连接成功"}
            else:
                return {"success": False, "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "message": "请求超时"}
        except Exception as e:
            return {"success": False, "message": str(e)[:200]}


class AnthropicBackend(VLMBackend):
    """Anthropic Claude 格式后端"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def call_api(
        self,
        messages: list,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        # Anthropic 格式：分离 system 和 user messages
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        payload: dict = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": max_tokens,
        }
        if system_content:
            payload["system"] = system_content
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop_sequences"] = stop

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = httpx.post(
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def test_connection(self) -> dict:
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            resp = httpx.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                return {"success": True, "message": "连接成功"}
            else:
                return {"success": False, "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "message": "请求超时"}
        except Exception as e:
            return {"success": False, "message": str(e)[:200]}

    def build_content(self, images_base64: list[str], user_text: str) -> list:
        """Anthropic 格式的图片内容"""
        content = [{"type": "text", "text": user_text}]
        for img_b64 in images_base64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_b64,
                },
            })
        return content


class GeminiBackend(VLMBackend):
    """Google Gemini 格式后端"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def call_api(self, messages: list, max_tokens: int = 1024, **kwargs) -> str:
        # Gemini: extract system instruction and user content (with images)
        system_text = ""
        user_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_text = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
            elif msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    user_parts = content
                elif isinstance(content, str):
                    user_parts = [{"text": content}]

        payload: dict = {
            "contents": [{"parts": user_parts}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}

        resp = httpx.post(
            f"{self.base_url}/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def test_connection(self) -> dict:
        try:
            payload = {
                "contents": [{"parts": [{"text": "Hi"}]}],
                "generationConfig": {"maxOutputTokens": 10},
            }
            resp = httpx.post(
                f"{self.base_url}/models/{self.model}:generateContent",
                params={"key": self.api_key},
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                return {"success": True, "message": "连接成功"}
            else:
                return {"success": False, "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except httpx.TimeoutException:
            return {"success": False, "message": "请求超时"}
        except Exception as e:
            return {"success": False, "message": str(e)[:200]}

    def build_content(self, images_base64: list[str], user_text: str) -> list:
        """Gemini 格式"""
        parts = [{"text": user_text}]
        for img_b64 in images_base64:
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": img_b64,
                }
            })
        return parts


class VLMAdapter:
    def __init__(
        self,
        provider: str,
        base_url: str,
        api_key: str,
        api_format: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # 获取 provider 配置
        config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["custom"])

        # API 格式：优先使用传入值，否则用 provider 默认
        self.api_format = api_format or config.get("api_format", "openai")

        # 模型名：优先使用传入值，否则用 provider 默认
        self.model = model or config.get("model", "gpt-4o")

        # 生成参数
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop

        # 如果 base_url 为空，使用默认值
        if not self.base_url:
            self.base_url = config.get("base_url", "https://api.openai.com/v1")

        # 创建后端实例
        self.backend = self._create_backend()

    def _create_backend(self) -> VLMBackend:
        if self.api_format == "anthropic":
            return AnthropicBackend(self.base_url, self.api_key, self.model)
        elif self.api_format == "gemini":
            return GeminiBackend(self.base_url, self.api_key, self.model)
        else:
            return OpenAICompatibleBackend(self.base_url, self.api_key, self.model)

    def parse_intent(
        self,
        images_base64: list[str],
        user_text: str,
        sample_boxes: Optional[list] = None,
        max_retry: int = 3,
    ) -> dict:
        last_err = None
        last_raw = ""
        for _attempt in range(max_retry):
            try:
                raw = self._call_api(images_base64, user_text, sample_boxes)
                last_raw = raw
                result = self._parse_and_validate(raw)
                result["status"] = "success"
                result["message"] = ""
                result["retryable"] = False
                result["raw_vlm_response"] = raw
                return result
            except (json.JSONDecodeError, ValidationError, ValueError, httpx.HTTPError) as e:
                last_err = e
                continue
        return self._build_failed_result(last_err, last_raw)

    def _build_failed_result(self, error: Optional[Exception], raw: str) -> dict:
        message = "VLM 服务暂时无法完成视觉理解，当前将先根据文字需求生成草案"
        retryable = True

        if isinstance(error, json.JSONDecodeError):
            if not raw.strip():
                message = "VLM 返回了空内容，当前将先根据文字需求生成草案"
            else:
                message = "VLM 已响应，但返回格式不符合系统要求，当前将先根据文字需求生成草案"
        elif isinstance(error, ValidationError):
            message = "VLM 已响应，但返回字段不符合系统要求，当前将先根据文字需求生成草案"
        elif isinstance(error, ValueError):
            message = "VLM 返回内容无法解析，当前将先根据文字需求生成草案"
        elif isinstance(error, httpx.HTTPError):
            message = "VLM 服务暂时不可用，当前将先根据文字需求生成草案"

        return {
            "status": "failed",
            "message": message,
            "retryable": retryable,
            "raw_vlm_response": raw,
            "classes": [],
            "confidence": None,
        }

    def _build_system_prompt(self, sample_boxes: Optional[list] = None) -> str:
        if sample_boxes:
            box_context = "用户已在参考图上标注了目标区域，请参考这些框的位置和大小来理解用户意图：\n"
            for i, box in enumerate(sample_boxes):
                box_context += f"图{i+1}: 框坐标 x={box.get('x')}, y={box.get('y')}, w={box.get('width')}, h={box.get('height')}\n"
        else:
            box_context = ""
        return SYSTEM_PROMPT_TEMPLATE.format(box_context=box_context)

    def _call_api(self, images_base64: list[str], user_text: str, sample_boxes: Optional[list] = None) -> str:
        system_prompt = self._build_system_prompt(sample_boxes)
        user_content = self.backend.build_content(images_base64, user_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.backend.call_api(
            messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=self.stop,
        )

    def _parse_and_validate(self, raw: str) -> dict:
        match = re.search(r"```json\s*([\s\S]*?)```", raw)
        json_str = match.group(1).strip() if match else raw.strip()
        data = json.loads(json_str)
        validate(instance=data, schema=TASK_SCHEMA)
        confidence = data.get("confidence")
        if not isinstance(confidence, (int, float)):
            data["confidence"] = None
        else:
            data["confidence"] = max(0.0, min(1.0, float(confidence)))

        for item in data.get("classes", []):
            item["display_name_zh"] = item.get("display_name_zh") or item.get("class_name") or ""
            item["display_prompt_zh"] = item.get("display_prompt_zh") or item.get("prompt") or ""
            item["display_negative_prompt_zh"] = item.get("display_negative_prompt_zh") or item.get("negative_prompt") or ""
            item["display_color_hint_zh"] = item.get("display_color_hint_zh") or item.get("color_hint")
        return data

    def call_with_system_prompt(
        self,
        system_prompt: str,
        user_text: str,
        images_base64: Optional[list[str]] = None,
        response_format: Optional[str] = None,
        max_tokens: int = 4096,
        max_retry: int = 3,
    ) -> str:
        """
        通用方法：使用自定义 system prompt 调用 VLM。
        用于算法规划、视频验证等需要自定义 prompt 的场景。
        """
        user_content = self.backend.build_content(images_base64 or [], user_text)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        last_err = None
        for _attempt in range(max_retry):
            try:
                raw = self.backend.call_api(
                    messages,
                    max_tokens=max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stop=self.stop,
                )
                return raw
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"VLM call failed after {max_retry} retries: {last_err}")

    def test_connection(self) -> dict:
        """测试 API 连接"""
        return self.backend.test_connection()
