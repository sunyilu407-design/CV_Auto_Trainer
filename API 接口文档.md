# API 接口文档

## 1. 对话（OpenAI 兼容）

此端点用于创建聊天对话，模型将根据对话历史生成回复。该接口兼容 OpenAI 的 Chat Completion 格式。

### 请求地址

`POST http://tokenapi.boundlessai.tech/v1/chat/completions`

### 认证

使用 Bearer Token 进行认证。在 HTTP 头中添加：

```http
Authorization: Bearer YOUR_API_KEY
```

### 请求 Body

| 参数 | 类型 | 必选 | 描述 |
| --- | --- | --- | --- |
| `model` | `string` | 是 | 使用的模型名称。例如：`"Pro/zai-org/GLM-4.7"`。可用模型列表请参考平台文档。 |
| `messages` | `array` | 是 | 组成对话的消息列表。数组长度：1-10。 |
| `max_tokens` | `integer` | 否 | 生成的最大 token 数。确保输入 token + `max_tokens` 不超过模型的上下文窗口。示例：`4096`。 |
| `temperature` | `number` | 否 | 控制响应随机性的程度。取值范围 `[0, 2]`。示例：`0.7`。 |
| `top_p` | `number` | 否 | 核采样参数，动态调整每个预测 token 的选择范围。示例：`0.7`。 |
| `stream` | `boolean` | 否 | 若设置为 `true`，令牌将以 Server-Sent Events 的形式流式传输。示例：`false`。 |
| `stop` | `array` | 否 | 最多 4 个序列，API 在生成这些序列后将停止。示例：`null`。 |
| `n` | `integer` | 否 | 为每条输入消息生成几个回复。示例：`1`。 |
| `reasoning_effort` | `string` | 否 | 在思考模式和非思考模式间切换。部分模型支持。示例：`false`。 |
| `max_reasoning_tokens` | `integer` | 否 | 思维链输出的最大 token 数。适用于所有推理模型。取值范围 `[128, 32768]`。示例：`4096`。 |
| `min_p` | `number` | 否 | 基于 token 概率动态调整的过滤阈值。仅适用于 Qwen3 模型。取值范围 `[0, 1]`。示例：`0.05`。 |
| `top_k` | `integer` | 否 | 示例：`50`。 |
| `repetition_penalty` | `number` | 否 | 示例：`0.5`。 |
| `response_format` | `object` | 否 | 指定模型必须输出的格式。 |
| `tools` | `array` | 否 | 模型可能调用的工具列表，目前仅支持函数。最多支持 128 个函数。 |

### 响应示例

```json
{
  "id": "019bdaa55225ef854b320e9b838f77ce",
  "object": "chat.completion",
  "created": 1768899826,
  "model": "Pro/zai-org/GLM-4.7",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！...",
        "reasoning_content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 1540,
    "total_tokens": 1555,
    "completion_tokens_details": {
      "reasoning_tokens": 1190
    },
    "prompt_tokens_details": {
      "cached_tokens": 0
    },
    "prompt_cache_hit_tokens": 0,
    "prompt_cache_miss_tokens": 15
  },
  "system_fingerprint": ""
}
```

## 2. 对话（Anthropic 兼容）

此端点用于创建聊天对话，模型将根据对话历史生成回复。该接口兼容 Anthropic 的 Messages API 格式。

### 请求地址

`POST http://tokenapi.boundlessai.tech/v1/messages`

### 认证

使用 Bearer Token 进行认证。在 HTTP 头中添加：

```http
Authorization: Bearer YOUR_API_KEY
```

### 请求 Body

| 参数 | 类型 | 必选 | 描述 |
| --- | --- | --- | --- |
| `model` | `string` | 是 | 使用的模型名称。例如：`"Pro/zai-org/GLM-4.7"`。可用模型列表请参考平台文档。 |
| `messages` | `array` | 是 | 组成对话的消息列表。数组长度：1-10。 |
| `system` | `string` | 否 | 系统提示词，用于为大模型提供上下文和指令，如指定特定目标或角色。 |
| `max_tokens` | `integer` | 否 | 停止前生成的最大 token 数。不同模型的最大值不同。示例：`8192`。 |
| `temperature` | `number` | 否 | 控制响应随机性的程度。取值范围 `[0, 2]`。示例：`0.7`。 |
| `top_p` | `number` | 否 | 核采样参数。取值范围 `[0.1, 1]`。示例：`0.7`。 |
| `stream` | `boolean` | 否 | 若设置为 `true`，令牌将以 Server-Sent Events 的形式流式传输。示例：`true`。 |
| `stop_sequences` | `array` | 否 | 自定义文本序列，模型遇到时会停止生成。 |
| `top_k` | `integer` | 否 | 取值范围 `[0, 50]`。示例：`50`。 |
| `tools` | `array` | 否 | 工具定义列表。每个工具包含 `name`、`description`（推荐）和 `input_schema`（JSON Schema）。 |
| `tool_choice` | `object` | 否 | 模型应如何使用提供的工具。可选值：`auto`、`tool`、`none`。 |

### 响应示例

```json
{
  "id": "msg_T15jjp718fACotrwiLp3KwVu",
  "type": "message",
  "role": "assistant",
  "model": "Pro/zai-org/GLM-4.7",
  "content": [
    {
      "type": "thinking",
      "thinking": "...",
      "signature": "tvshsltrjs"
    },
    {
      "type": "text",
      "text": "Hello! I'm GLM, trained by Z.ai. How can I assist you today?"
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 6,
    "output_tokens": 215
  }
}
```

## 3. 创建嵌入（Embeddings）

此端点用于创建代表输入文本的嵌入向量。

### 请求地址

`POST http://tokenapi.boundlessai.tech/v1/embeddings`

### 认证

使用 Bearer Token 进行认证。在 HTTP 头中添加：

```http
Authorization: Bearer YOUR_API_KEY
```

### 请求 Body

| 参数 | 类型 | 必选 | 描述 |
| --- | --- | --- | --- |
| `model` | `string` | 是 | 使用的模型名称。例如：`"BAAI/bge-large-zh-v1.5"`。可用模型列表请参考平台文档。 |
| `input` | `string 或 array` | 是 | 要嵌入的输入文本。可以是字符串、token 数组、字符串数组或 token 数组的数组（用于批量处理）。输入长度不能超过模型的 max token 限制。各模型限制详见文档。 |
| `encoding_format` | `string` | 否 | 返回嵌入的格式。可选值：`float`、`base64`。示例：`"float"`。 |
| `dimensions` | `integer` | 否 | 输出嵌入向量的维度。仅在 Qwen/Qwen3 系列模型中支持。示例：`1024`。 |

### 响应示例

```json
{
  "object": "list",
  "model": "BAAI/bge-large-zh-v1.5",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, "..."],
      "index": 0
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

## 4. 创建重排序（Rerank）

此端点用于对文档列表根据与查询的相关性进行重排序。

### 请求地址

`POST http://tokenapi.boundlessai.tech/v1/rerank`

### 认证

使用 Bearer Token 进行认证。在 HTTP 头中添加：

```http
Authorization: Bearer YOUR_API_KEY
```

### 请求 Body

| 参数 | 类型 | 必选 | 描述 |
| --- | --- | --- | --- |
| `model` | `string` | 是 | 使用的模型名称。例如：`"BAAI/bge-reranker-v2-m3"`。可用模型列表请参考平台文档。 |
| `query` | `string` | 是 | 搜索查询。 |
| `documents` | `array` | 是 | 需要重排序的文档列表。目前仅支持字符串列表。最小数组长度为 1。 |
| `instruction` | `string` | 否 | 重排器的指令。仅支持 Qwen/Qwen3-Reranker 系列模型。 |
| `top_n` | `integer` | 否 | 返回的最相关文档或索引的数量。 |
| `return_documents` | `boolean` | 否 | 若为 `false`，响应中不包含文档文本；若为 `true`，则包含。 |
| `max_chunks_per_doc` | `integer` | 否 | 单个文档内生成的最大 chunk 数量。仅部分模型支持。 |
| `overlap_tokens` | `integer` | 否 | 文档分块时相邻块之间重叠的 token 数。仅部分模型支持。取值范围 `x <= 80`。 |

### 响应示例

```json
{
  "id": "a2b3c4d5e6f7g8h9i0j1",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "apple"
      },
      "relevance_score": 0.95
    },
    {
      "index": 2,
      "document": {
        "text": "fruit"
      },
      "relevance_score": 0.62
    }
  ],
  "meta": [
    {
      "tokens": {
        "input_tokens": 10,
        "output_tokens": 5
      }
    }
  ]
}
```

## 通用说明

- `Trace ID`：所有 API 的响应头中都包含 `x-siliconcloud-trace-id` 字段，作为请求的唯一标识符，方便日志查询和问题排查。
- `认证`：所有请求均需在 HTTP 头中包含 `Authorization: Bearer YOUR_API_KEY` 进行认证。
- `模型列表`：可用的模型列表会动态更新，请定期查阅 SiliconFlow 官方文档或控制台以获取最新信息。

## 模型列表

| 序号 | 模型 ID | 说明 |
| --- | --- | --- |
| 1 | `Pro/MiniMaxAI/MiniMax-M2.5` | MiniMax 最新模型 |
| 2 | `Pro/zai-org/GLM-5` | GLM 系列最新 |
| 3 | `Pro/moonshotai/Kimi-K2.5` | Kimi 最新 |
| 4 | `Pro/zai-org/GLM-4.7` | GLM 4.7 |
| 5 | `deepseek-ai/DeepSeek-V3.2` | DeepSeek V3.2 |
| 6 | `Pro/deepseek-ai/DeepSeek-V3.2` | DeepSeek V3.2 Pro |
| 7 | `deepseek-ai/DeepSeek-V3.1-Terminus` | DeepSeek V3.1 Terminus |
| 8 | `Pro/deepseek-ai/DeepSeek-V3.1-Terminus` | DeepSeek Pro 版 |
| 9 | `Qwen/Qwen3.5-397B-A17B` | Qwen 3.5 397B |
| 10 | `Qwen/Qwen3.5-122B-A10B` | Qwen 3.5 122B |
| 11 | `Qwen/Qwen3.5-35B-A3B` | Qwen 3.5 35B |
| 12 | `Qwen/Qwen3.5-27B` | Qwen 3.5 标准版 |
| 13 | `Qwen/Qwen3.5-9B` | Qwen 3.5 轻量版 |
| 14 | `Qwen/Qwen3.5-4B` | Qwen 3.5 微量型 |
| 16 | `deepseek-ai/DeepSeek-R1` | DeepSeek R1 |
| 17 | `Pro/deepseek-ai/DeepSeek-R1` | DeepSeek R1 Pro 版 |
| 18 | `deepseek-ai/DeepSeek-V3` | DeepSeek V3 |
| 19 | `Pro/deepseek-ai/DeepSeek-V3` | DeepSeek V3 Pro 版 |
| 20 | `Pro/MiniMaxAI/MiniMax-M2.1` | MiniMax M2.1 |
| 21 | `stepfun-ai/Step-3.5-Flash` | Step 系列 |
| 23 | `moonshotai/Kimi-K2-Thinking` | Kimi K2 推理版 |
| 24 | `Pro/moonshotai/Kimi-K2-Thinking` | Kimi K2 Pro 版 |
| 25 | `zai-org/GLM-4.6` | GLM 4.6 |
| 26 | `Kwaipilot/KAT-Dev` | KAT 开发版 |
| 40 | `moonshotai/Kimi-K2-Instruct-0905` | Kimi K2 Instruct |
| 41 | `Pro/moonshotai/Kimi-K2-Instruct-0905` | Pro 版 |
| 42 | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Qwen Next Instruct |
| 43 | `Qwen/Qwen3-Next-80B-A3B-Thinking` | Qwen Next 推理版 |
| 44 | `inclusionAI/Ring-flash-2.0` | Ring 系列 |
| 45 | `inclusionAI/Ling-flash-2.0` | Ling 系列 |
| 46 | `inclusionAI/Ling-mini-2.0` | Ling 迷你版 |
| 50 | `tencent/Hunyuan-MT-7B` | 腾讯混元 MT |
| 51 | `ByteDance-Seed/Seed-OSS-36B-Instruct` | 字节 |
| 54 | `zai-org/GLM-4.5V` | GLM 多模态 |
| 55 | `zai-org/GLM-4.5-Air` | GLM 轻量版 |
| 57 | `Qwen/Qwen3-Coder-30B-A3B-Instruct` | Qwen Coder |
| 58 | `Qwen/Qwen3-Coder-480B-A35B-Instruct` | Qwen Coder |
| 59 | `Qwen/Qwen3-30B-A3B-Thinking-2507` | Qwen 推理版 |
| 60 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Qwen Instruct |
| 61 | `Qwen/Qwen3-235B-A22B-Thinking-2507` | Qwen 推理 |
| 62 | `Qwen/Qwen3-235B-A22B-Instruct-2507` | Qwen Instruct |
| 64 | `baidu/ERNIE-4.5-300B-A47B` | 百度文心 |
| 65 | `tencent/Hunyuan-A13B-Instruct` | 腾讯混元 A13B |
| 66 | `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` | DeepSeek R1 蒸馏版 |
| 67 | `Qwen/Qwen3-32B` | Qwen 3 标准版 |
| 68 | `Qwen/Qwen3-14B` | Qwen 3 轻量版 |
| 69 | `Qwen/Qwen3-8B` | Qwen 3 微型板 |
| 77 | `THUDM/GLM-Z1-32B-0414` | GLM Z1 |
| 78 | `THUDM/GLM-4-32B-0414` | GLM 4 |
| 79 | `THUDM/GLM-Z1-9B-0414` | GLM Z1 轻量 |
| 80 | `THUDM/GLM-4-9B-0414` | GLM 4 轻量 |
| 82 | `Qwen/QwQ-32B` | QwQ 推理模型 |
| 85 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | R1 蒸馏 Qwen |
| 86 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | R1 蒸馏 Qwen |
| 87 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | R1 蒸馏 Qwen |
| 88 | `deepseek-ai/DeepSeek-V2.5` | DeepSeek V2.5 |
| 97 | `Qwen/Qwen2.5-Coder-32B-Instruct` | Qwen2.5 Coder |
| 100 | `Qwen/Qwen2.5-72B-Instruct-128K` | Qwen2.5 超长上下文 |
| 101 | `Qwen/Qwen2.5-72B-Instruct` | Qwen2.5 超大杯 |
| 103 | `Qwen/Qwen2.5-32B-Instruct` | Qwen2.5 中杯 |
| 104 | `Qwen/Qwen2.5-14B-Instruct` | Qwen2.5 小杯 |
| 105 | `Qwen/Qwen2.5-7B-Instruct` | Qwen2.5 轻量 |
| 106 | `Qwen/Qwen2.5-Coder-7B-Instruct` | Qwen2.5 Coder 轻量 |
| 107 | `internlm/internlm2_5-7b-chat` | internlm |
| 108 | `Qwen/Qwen2-7B-Instruct` | Qwen2 轻量 |
| 109 | `THUDM/glm-4-9b-chat` | GLM4 对话版 |
| 112 | `LoRA/Qwen/Qwen2.5-32B-Instruct` | LoRA 微调版 |
| 113 | `LoRA/Qwen/Qwen2.5-14B-Instruct` | LoRA 微调版 |
| 114 | `Pro/Qwen/Qwen2.5-Coder-7B-Instruct` | Pro 版 |
| 116 | `Pro/Qwen/Qwen2.5-7B-Instruct` | Pro 版 |
| 118 | `LoRA/Qwen/Qwen2.5-72B-Instruct` | LoRA 微调版 |
| 119 | `Pro/Qwen/Qwen2-7B-Instruct` | Pro 版 |
| 120 | `LoRA/Qwen/Qwen2.5-7B-Instruct` | LoRA 微调版 |
| 121 | `Pro/THUDM/glm-4-9b-chat` | Pro 版 |
