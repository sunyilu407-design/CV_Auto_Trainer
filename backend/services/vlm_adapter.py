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
        "scenario_hint": {"type": "string", "description": "从视觉推断的场景类型"},
        "difficulty_hint": {"type": "string", "description": "从视觉推断的难度级别"},
        "visual_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "从图片中观察到的关键视觉特征（供后续增强和训练参考）",
        },
        "special_considerations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "需要特别注意的事项，如小目标、遮挡、旋转等",
        },
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
                    # 新增字段
                    "estimated_size_hint": {
                        "type": "string",
                        "description": "目标相对画面的尺寸估算：small(<5%), medium(5-30%), large(>30%)",
                    },
                    "typical_perspective": {
                        "type": "string",
                        "description": "典型视角：front/back/side/aerial/arbitrary",
                    },
                    "rotation_invariant": {
                        "type": "boolean",
                        "description": "是否需要旋转不变性",
                    },
                    "occlusion_tolerance": {
                        "type": "string",
                        "description": "遮挡容忍度：low(<20%)/medium(20-60%)/high(>60%)",
                    },
                    "color_consistency": {
                        "type": "string",
                        "description": "颜色一致性：consistent/intermediate/variable",
                    },
                    "data_augmentation_priority": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "建议优先应用的增强策略",
                    },
                },
            },
        }
    },
}

SYSTEM_PROMPT_TEMPLATE = """
你是一位顶级的计算机视觉算法专家，专注于目标检测领域。你需要像一个人类视觉专家一样，从用户提供的参考图片和文字描述中，深度理解检测需求。

## ⚠️ 类别词的硬纪律（v9.0 优化版核心约束 / 必须遵守）

下游打标使用 YOLO-World，依赖 **CLIP 文本编码器**理解类别词。CLIP 对词汇极度敏感，违反以下规则会直接导致召回率<30%。

**`class_name` 字段必须满足**：
1. 必须是单一、物理可见的英文名词（一个实体），形如 `worker` / `forklift` / `hard_hat`
2. 使用 CLIP 训练分布中常见的标准命名，**不要造词**
3. 多词时用下划线连接（`hard_hat`、`safety_vest`），最多 3 个词
4. 优先选**具体而非泛化**的词：`sedan` 优于 `vehicle`、`hard_hat` 优于 `equipment`

**禁止出现以下类型**（发现请立刻替换为最近的具体物体名，否则用户业务会失败）：
- 抽象概念：`danger`、`safety`、`area`、`zone`、`violation`、事件名（`fire_event` 等）
- 状态/动作：`running`、`broken`、`working`、`工作中`
- 单独的属性：`red`、`large`、`hot`
- 空间关系：`near`、`inside`、`above`
- 中文/拼音/品牌专名：除非品牌已是 CLIP 训练分布中的常见词

**类别数量上限**：单任务**最多 10 个**类别。多余的合并到最相似的，或归入 `other` 后舍弃。

**CLIP-friendly 参考词表**（优先从中挑选最近的英文等价物）：
- People:    `person`, `worker`, `pedestrian`, `cyclist`, `driver`, `patient`
- Vehicles:  `car`, `sedan`, `truck`, `bus`, `motorcycle`, `bicycle`, `forklift`, `excavator`
- Safety:    `hard_hat`, `safety_vest`, `face_mask`, `glove`, `safety_shoe`
- Industrial:`bottle`, `box`, `pallet`, `conveyor_belt`, `pipe`, `valve`, `screw`, `bearing`
- Animals:   `dog`, `cat`, `bird`, `cow`, `horse`
- Medical:   `syringe`, `pill`, `tablet`, `bandage`, `stethoscope`

中文展示名（`display_name_zh`）不受此约束，可保留行业说法。**只有 `class_name` 进入打标管线**。

## 输入分析任务

你将同时收到：
1. **参考图片**（用户手动画了目标框）：图片会告诉你目标的外观、大小、遮挡情况、拍摄角度
2. **文字描述**：用户用自然语言描述他们想检测什么

**核心原则**：
- 文字描述是**硬约束**（用户明确想要的，必须包含）
- 参考图片是**视觉锚点**（帮助你理解目标在真实场景中的样子）
- 如果图片和文字有歧义，优先遵循文字描述，但在 prompt 中体现不确定性

## 你必须输出的内容

### 第一层：全局理解
- `scenario_hint`：从图片推断场景类型（indoor/outdoor/industrial/medical/agricultural/surveillance/retail/other）
- `difficulty_hint`：从图片和描述评估检测难度（easy/moderate/hard/very_hard）
- `visual_insights`：3-5 条从图片观察到的关键视觉特征（如"目标经常被大面积遮挡""背景与目标颜色接近""存在大量小目标"）
- `special_considerations`：需要特别注意的事项

### 第二层：每个类别
对于每个要检测的类别，你需要从**视觉角度**分析并输出：

| 字段 | 要求 | 示例 |
|------|------|------|
| `class_name` | 英文小写下划线，内部标识 | `vehicle` |
| `prompt` | 详细的英文视觉描述，用于引导检测模型定位目标。必须覆盖：形状/轮廓、材质/纹理、典型颜色/外观、常见视角/姿态、上下文背景。不要写负面排除（那是 negative_prompt 的工作）| `A passenger vehicle with four wheels, typically with a rectangular body shape, windows on the sides, commonly seen from side or rear view in road scenes. Roof racks, rear-view mirrors, and wheel wells are distinctive structural features. Color varies widely but sedan shapes have characteristic proportions (length ~3-4m, width ~1.5-1.8m).` |
| `negative_prompt` | 明确写出**容易混淆但不应被识别为该类**的情况 | `Do NOT detect bicycles, motorcycles, buses, trucks, or pedestrians. Do NOT detect vehicles viewed only from directly above (aerial view). Do NOT detect vehicle parts without the full vehicle body in frame.` |
| `color_hint` | 主要颜色特征，可为 null | `White, silver, black are most common; red and blue are also frequent` |
| `display_name_zh` | 简洁中文展示名 | `小汽车` |
| `display_prompt_zh` | 中文详细描述，与英文 prompt 语义一致 | `一种典型的乘用车，车身呈长方形，有车窗、车灯、后视镜等结构特征，常见于道路场景，颜色多样` |
| `display_negative_prompt_zh` | 中文排除项描述 | `不要检测自行车、摩托车、公交车、卡车或行人。不要从正上方俯视检测。不要检测单独的车轮或车窗等局部部件` |
| `display_color_hint_zh` | 中文颜色提示，可为 null | `白色、银色、黑色最为常见` |
| `estimated_size_hint` | 目标占画面比例估算 | `medium (5-30%)` |
| `typical_perspective` | 典型视角 | `side or rear` |
| `rotation_invariant` | 是否需要旋转不变性 | `false` |
| `occlusion_tolerance` | 遮挡容忍度 | `medium (20-60%)` |
| `color_consistency` | 颜色一致性 | `variable` |
| `data_augmentation_priority` | 建议优先增强策略 | `["rotation", "blur", "brightness"]` |

## 输出格式

你必须输出 JSON，不要有任何额外文字：

```json
{{
  "confidence": 0.87,
  "scenario_hint": "surveillance",
  "difficulty_hint": "moderate",
  "visual_insights": [
    "目标在画面中占比中等，约为5-15%",
    "存在一定的遮挡情况，目标可能仅部分可见",
    "背景为室内环境，光照条件稳定",
    "同类目标间存在颜色差异，颜色提示仅作参考"
  ],
  "special_considerations": [
    "小目标检测：部分目标在画面中占比小于5%，需要增强小目标召回",
    "遮挡处理：建议使用 soft-NMS 而非 hard-NMS"
  ],
  "classes": [
    {{
      "class_name": "...",
      "prompt": "...",
      "negative_prompt": "...",
      "color_hint": "...",
      "display_name_zh": "...",
      "display_prompt_zh": "...",
      "display_negative_prompt_zh": "...",
      "display_color_hint_zh": "...",
      "estimated_size_hint": "...",
      "typical_perspective": "...",
      "rotation_invariant": ...,
      "occlusion_tolerance": "...",
      "color_consistency": "...",
      "data_augmentation_priority": [...]
    }}
  ]
}}
```

## 注意事项

1. `prompt` 中的尺寸信息尽量用**相对比例**而非绝对像素值（如"占画面5-15%"而非"50-150像素"）
2. 如果参考图片中目标存在明显遮挡，在 `occlusion_tolerance` 中体现，并建议 `data_augmentation_priority` 包含 `cutout` 或 `mosaic`
3. 如果用户描述中存在否定词（如"不要检测XXX"），必须在 `negative_prompt` 中体现
4. 如果某个字段在图片中无法确定，输出你认为最可能的值，但降低 `confidence`
5. `confidence` 反映你对整体类别解析的把握程度，不确定性高时应低于 0.7
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

    @abstractmethod
    def stream_call_api(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ):
        """流式调用 API，yield 每块文本内容。子类应覆盖以支持真正的 SSE 流。"""
        # 默认实现：一次性调用，按字符yield，模拟打字效果
        full = self.call_api(messages, max_tokens, temperature, top_p, stop)
        for ch in full:
            yield ch

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

    def stream_call_api(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ):
        """同步生成器，yield 每块文本。"""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop"] = stop
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content") or delta.get("text") or ""
                if content:
                    yield content

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
            resp = httpx.post(
                f"{self.base_url}/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
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

    def stream_call_api(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ):
        """Anthropic SSE 流式调用。"""
        system_content = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                user_messages.append(msg)

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict = {
            "model": self.model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system_content:
            payload["system"] = system_content
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop:
            payload["stop_sequences"] = stop

        with httpx.stream(
            "POST",
            f"{self.base_url}/messages",
            headers=headers,
            json=payload,
            timeout=60,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                chunk_type = chunk.get("type", "")
                if chunk_type == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text

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
        video_info: Optional[dict] = None,
        max_retry: int = 3,
    ) -> dict:
        last_err = None
        last_raw = ""
        for _attempt in range(max_retry):
            try:
                raw = self._call_api(images_base64, user_text, sample_boxes, video_info)
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

    def _build_system_prompt(self, sample_boxes: Optional[list] = None, video_info: Optional[dict] = None) -> str:
        parts = [SYSTEM_PROMPT_TEMPLATE.format(box_context="")]

        # 注入视频上下文（如果有）
        if video_info:
            video_context = f"""
## 视频上下文信息
- 视频帧率: {video_info.get('fps', 'N/A')} fps
- 视频时长: {video_info.get('duration_seconds', 'N/A')} 秒
- 视频分辨率: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}
- 采样帧数: {video_info.get('frame_count', len([]))} 帧
请结合视频的时间跨度理解画面变化，避免将某一帧的瞬时状态误判为常态。
"""
            parts.append(video_context)

        # 注入参考框信息
        if sample_boxes:
            box_context_parts = [
                "## 用户标注的参考框信息",
                "用户已在参考图上标注了目标区域。以下是各参考框的位置和尺寸信息（归一化坐标）：",
            ]
            for i, box in enumerate(sample_boxes):
                x = box.get('x', 0)
                y = box.get('y', 0)
                w = box.get('width', 0)
                h = box.get('height', 0)
                cx = (x + w / 2) / 100
                cy = (y + h / 2) / 100
                area_pct = round(w * h / 10000, 2)
                aspect = round(w / max(h, 1), 2) if h else 0
                box_context_parts.append(
                    f"图{i+1}: 中心坐标(归一化) cx={cx:.2f}, cy={cy:.2f}; "
                    f"宽高比 w/h≈{aspect}; 粗估占画面面积≈{area_pct}% "
                    f"({'**小目标**' if area_pct < 5 else '**中等目标**' if area_pct < 30 else '**大目标**'})"
                )
            box_context_parts.append(
                "\n请综合这些框的分布、尺寸和位置特征，结合用户文字描述，准确推断检测目标的视觉特征。"
            )
            parts.append("\n".join(box_context_parts))
        else:
            parts.append(
                "## 重要说明\n"
                "用户未提供手绘参考框。请仅依靠文字描述推断目标外观。\n"
                "这种情况下，你对目标大小、遮挡情况的判断会存在较大不确定性，"
                "请适当降低 confidence，并在 prompt 中使用更保守的描述。"
            )
        return "\n".join(parts)

    def _call_api(
        self,
        images_base64: list[str],
        user_text: str,
        sample_boxes: Optional[list] = None,
        video_info: Optional[dict] = None,
    ) -> str:
        system_prompt = self._build_system_prompt(sample_boxes, video_info)
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
            if not item or not isinstance(item, dict):
                continue
            item["display_name_zh"] = item.get("display_name_zh") or item.get("class_name") or ""
            item["display_prompt_zh"] = item.get("display_prompt_zh") or item.get("prompt") or ""
            item["display_negative_prompt_zh"] = item.get("display_negative_prompt_zh") or item.get("negative_prompt") or ""
            item["display_color_hint_zh"] = item.get("display_color_hint_zh") or item.get("color_hint")

            # 新增字段默认值
            item.setdefault("estimated_size_hint", "medium")
            item.setdefault("typical_perspective", "arbitrary")
            item.setdefault("rotation_invariant", False)
            item.setdefault("occlusion_tolerance", "medium")
            item.setdefault("color_consistency", "intermediate")
            item.setdefault("data_augmentation_priority", ["flip", "brightness", "contrast"])

        # 全局字段默认值
        data.setdefault("scenario_hint", "")
        data.setdefault("difficulty_hint", "moderate")
        data.setdefault("visual_insights", [])
        data.setdefault("special_considerations", [])

        # ── CLIP-friendly 后验证 ────────────────────────────────────────────────
        # 检查 VLM 输出的 class_name 是否符合 CLIP 要求，不符合则自动修正
        self._fix_clip_unfriendly_classes(data)

        return data

    # ── CLIP-friendly class_name 验证 ───────────────────────────────────────────────
    # 规则来源：SYSTEM_PROMPT_TEMPLATE 的"类别词的硬纪律"
    _CLIP_BAD_PATTERNS = re.compile(
        r"(^(danger|safety|area|zone|violation|fire_event|accident|"
        r"running|broken|working|active|inactive|normal|abnormal|"
        r"normal_state|abnormal_state|"
        r"red|blue|green|yellow|large|small|hot|cold|fast|slow|"
        r"near|inside|outside|above|below|"
        r"[\u4e00-\u9fff]+))"  # 任何中文字符
    )
    _CLIP_GOOD_FALLBACKS = {
        "vehicle": "car",
        "equipment": "forklift",
        "safety": "hard_hat",
        "danger": "worker",
        "area": "car",
        "zone": "car",
        "violation": "worker",
        "running": "person",
        "broken": "forklift",
        "working": "forklift",
        "active": "worker",
        "inactive": "worker",
        "normal": "person",
        "abnormal": "worker",
    }

    def _fix_clip_unfriendly_classes(self, data: dict) -> None:
        """检查并修正 class_name 中不符合 CLIP 友好规则的词，自动修复或警告。"""
        for item in data.get("classes", []):
            if not item or not isinstance(item, dict):
                continue
            original_name = item.get("class_name", "") or ""
            warnings: list[str] = []

            # 规则1：包含中文/拼音
            if re.search(r"[\u4e00-\u9fff]", original_name):
                warnings.append(f"class_name 包含中文字符（已自动替换）")
                item["class_name"] = self._CLIP_GOOD_FALLBACKS.get(original_name.lower().replace(" ", "_"), "target")
                item["prompt"] = original_name

            # 规则2：包含禁用词（抽象/状态/动作）
            elif self._CLIP_BAD_PATTERNS.match(original_name.lower()):
                warnings.append(f"class_name 「{original_name}」是抽象词或状态词，CLIP 召回率低（已自动替换）")
                item["class_name"] = self._CLIP_GOOD_FALLBACKS.get(
                    original_name.lower().replace(" ", "_"),
                    original_name.lower().replace(" ", "_")
                )
                # 用 prompt 中的具体描述替换
                item["class_name"] = self._CLIP_GOOD_FALLBACKS.get(
                    original_name.lower().replace(" ", "_").strip("_"),
                    "target"
                )

            # 规则3：超过3个词
            elif len(original_name.split("_")) > 3:
                warnings.append(f"class_name 超过3个词（{len(original_name.split('_'))}个），CLIP 泛化能力下降")

            # 规则4：包含空格但不是下划线连接
            elif " " in original_name and "_" not in original_name:
                fixed = original_name.replace(" ", "_").lower()
                warnings.append(f"class_name 包含空格（已自动转为下划线：{fixed}）")
                item["class_name"] = fixed

            if warnings:
                item["clip_validation_warnings"] = warnings

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

    def stream_call_api(
        self,
        messages: list,
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ):
        """流式调用 VLM，返回文本片段生成器"""
        return self.backend.stream_call_api(
            messages,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            stop=stop if stop is not None else self.stop,
        )
