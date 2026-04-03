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
                },
            },
        }
    },
}

SYSTEM_PROMPT_TEMPLATE = """
你是一个计算机视觉标注任务专家。用户会提供若干参考图片（已画框）和口语化描述。
{box_context}
你需要输出一个标准 JSON，描述每个需要检测的类别。

输出格式（仅输出 JSON，不要有任何额外文字）：
{{
  "classes": [
    {{
      "class_name": "英文类别名（小写下划线）",
      "prompt": "详细的英文视觉描述，用于引导目标检测模型定位目标",
      "negative_prompt": "该类别不包含的特征描述",
      "color_hint": "主要颜色特征（可为null）"
    }}
  ]
}}
"""


class VLMAdapter:
    def __init__(self, provider: str, base_url: str, api_key: str):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def parse_intent(
        self,
        images_base64: list[str],
        user_text: str,
        sample_boxes: Optional[list] = None,
        max_retry: int = 3,
    ) -> dict:
        last_err = None
        for attempt in range(max_retry):
            try:
                raw = self._call_api(images_base64, user_text, sample_boxes)
                result = self._parse_and_validate(raw)
                return result
            except (json.JSONDecodeError, ValidationError, ValueError, httpx.HTTPError) as e:
                last_err = e
                continue
        raise RuntimeError(f"VLM 解析失败，已重试 {max_retry} 次: {last_err}")

    def _build_content(self, images_base64: list[str], user_text: str, sample_boxes: Optional[list] = None):
        content = [{"type": "text", "text": user_text}]
        for img_b64 in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        return content

    def _build_system_prompt(self, sample_boxes: Optional[list] = None) -> str:
        if sample_boxes:
            box_context = "用户已在参考图上标注了目标区域，请参考这些框的位置和大小来理解用户意图：\n"
            for i, box in enumerate(sample_boxes):
                box_context += f"图{i+1}: 框坐标 x={box.get('x')}, y={box.get('y')}, w={box.get('width')}, h={box.get('height')}\n"
        else:
            box_context = ""
        return SYSTEM_PROMPT_TEMPLATE.format(box_context=box_context)

    def _call_api(self, images_base64: list[str], user_text: str, sample_boxes: Optional[list] = None) -> str:
        content = self._build_content(images_base64, user_text, sample_boxes)
        payload = {
            "model": self._get_model_name(),
            "messages": [
                {"role": "system", "content": self._build_system_prompt(sample_boxes)},
                {"role": "user", "content": content},
            ],
            "max_tokens": 1024,
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_and_validate(self, raw: str) -> dict:
        match = re.search(r"```json\s*([\s\S]*?)```", raw)
        json_str = match.group(1).strip() if match else raw.strip()
        data = json.loads(json_str)
        validate(instance=data, schema=TASK_SCHEMA)
        return data

    def _get_model_name(self) -> str:
        mapping = {
            "openai": "gpt-4o",
            "kimi": "moonshot-v1-8k",
            "gemini": "gemini-1.5-pro",
        }
        return mapping.get(self.provider, "gpt-4o")
