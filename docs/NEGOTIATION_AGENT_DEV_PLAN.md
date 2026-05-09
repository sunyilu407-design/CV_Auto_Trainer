# 多智能体需求确认助手 — 开发计划

> 基于 NEGOTIATION_AGENT_DESIGN.md 架构设计，分 4 个 Sprint 实施

---

## 总览

| Sprint | 名称 | 预估工作量 | 依赖 |
|--------|------|-----------|------|
| **A** | 后端 Agent 核心层 | 2-3 天 | 无 |
| **B** | 前端对话 UI | 2-3 天 | Sprint A |
| **C** | YOLO-World 预览对接 | 1-2 天 | Sprint A+B |
| **D** | AlgorithmPlan 增强 + 回路 | 1 天 | Sprint A+B+C |

**总计**: 6-9 天

---

## Sprint A: 后端 Agent 核心层

### A.1 数据库模型 — 对话持久化

**文件**: `backend/models/database.py`

```python
# negotiation_conversations 表
class NegotiationConversation:
    id: str (UUID)
    task_id: str (FK → tasks.id)
    messages: JSON           # [{ role, content, timestamp, metadata }]
    current_config: JSON     # Agent B 最新输出的完整配置
    algorithm_hints: JSON    # 给 AlgorithmPlan 的提示
    confirmed: bool          # 用户是否已确认
    preview_count: int       # 预览次数
    created_at: datetime
    updated_at: datetime
```

**验收**: 表创建成功，migration 通过

---

### A.2 Agent A — 对话引导 (VLM)

**文件**: `backend/services/conversation_agent.py`

**功能**:
- `negotiate_chat(message, history, images, preview_stats, current_config) → AgentAResponse`
- 使用 `vlm_adapter.call_with_system_prompt()` + 多轮 history
- System Prompt 引导追问（颜色/大小/事件/区域/排除条件）
- 输出: reply + intent_update + should_regenerate + convergence

**关键逻辑**:
```python
def negotiate_chat(...):
    # 1. 构建多轮消息 (system + history + current user msg)
    # 2. 如有预览图 → 加入 image_base64 让 VLM 看
    # 3. 调用 VLM
    # 4. 解析结构化返回 (reply + intent_update + flags)
    # 5. 判断 convergence (所有维度已确认 + 预览通过)
```

**验收**: 单元测试 — 模拟对话 3 轮后正确输出 intent_update

---

### A.3 Agent B — 配置生成 (推理模型)

**文件**: `backend/services/config_generator_agent.py`

**功能**:
- `generate_config(intent_summary, conversation_context) → FullConfig`
- 使用 `reasoning_adapter.reason_json()` 生成完整配置
- 输出: classes[] + detection_rules{} + vocab{} + algorithm_hints{}
- 含 CLIP 友好性校验 + schema 验证

**关键逻辑**:
```python
def generate_config(...):
    # 1. 构建 system prompt (CLIP 规则 + schema 约束)
    # 2. 构建 user prompt (已确认需求 + 当前上下文)
    # 3. 调用推理模型 → JSON
    # 4. schema 验证 (jsonschema.validate)
    # 5. CLIP 友好性检查 (复用现有 normalize_categories 逻辑)
    # 6. 返回 FullConfig
```

**验收**: 单元测试 — 给定需求摘要，输出合法的三件套 JSON

---

### A.4 Orchestrator — 编排器

**文件**: `backend/services/negotiation_orchestrator.py`

**功能**:
- `handle_chat(task_id, conversation_id, message, preview_stats) → OrchestratorResponse`
- 协调 Agent A + Agent B 的调用顺序
- 管理对话历史读写 (DB)
- 决定是否触发 Agent B 重新生成配置

**流程**:
```
1. 从 DB 加载对话历史
2. 调用 Agent A → 获取 reply + intent_update + should_regenerate
3. if should_regenerate:
      调用 Agent B → 获取 updated_config
      写入 DB
4. 追加本轮消息到历史
5. 返回前端: reply + config + convergence
```

**验收**: 集成测试 — 完整对话 → 生成配置 → 确认流程

---

### A.5 API 路由

**文件**: `backend/routers/negotiate.py`

```python
router = APIRouter(prefix="/api/negotiate", tags=["negotiate"])

# 对话
POST /api/negotiate/chat
  → negotiation_orchestrator.handle_chat()

# 确认进入下一阶段
POST /api/negotiate/confirm
  → 标记 confirmed=True, 返回最终配置

# 获取对话历史（恢复用）
GET /api/negotiate/conversation/{task_id}
  → 返回完整对话 + 当前配置 + confirmed 状态

# 快速预览（触发 YOLO-World 检测 2-3 张图）
POST /api/negotiate/preview
  → 调用 stage2_labeler 做小批量检测，返回结果
```

**验收**: Postman/curl 测试全部 API 正常响应

---

## Sprint B: 前端对话 UI

### B.1 状态管理

**文件**: `frontend/src/store/taskStore.ts`

新增字段:
```typescript
interface TaskState {
  // ... 现有字段 ...
  
  // 需求协商
  conversationId: string | null
  negotiationMessages: NegotiationMessage[]
  negotiatedConfig: NegotiatedConfig | null
  negotiationConverged: boolean
  
  // Actions
  addNegotiationMessage: (msg: NegotiationMessage) => void
  setNegotiatedConfig: (config: NegotiatedConfig) => void
  setNegotiationConverged: (v: boolean) => void
  resetNegotiation: () => void
}

interface NegotiationMessage {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
  metadata?: {
    should_preview?: boolean
    config_updated?: boolean
  }
}

interface NegotiatedConfig {
  classes: VLMClass[]
  detection_rules: DetectionRules
  vocab: Record<string, VocabEntry>
  algorithm_hints: AlgorithmHints
}
```

**验收**: store 定义完成，类型正确

---

### B.2 API 层

**文件**: `frontend/src/api/backend.ts`

```typescript
export const negotiateApi = {
  chat: (taskId: string, message: string, previewStats?: PreviewStats) =>
    request<NegotiateChatResponse>('/negotiate/chat', { ... }),
  
  confirm: (taskId: string, conversationId: string) =>
    request<NegotiateConfirmResponse>('/negotiate/confirm', { ... }),
  
  getConversation: (taskId: string) =>
    request<ConversationData>(`/negotiate/conversation/${taskId}`),
  
  preview: (taskId: string, config: NegotiatedConfig) =>
    request<PreviewResult>('/negotiate/preview', { ... }),
}
```

**验收**: API 类型定义完成

---

### B.3 对话面板组件

**文件**: `frontend/src/components/NegotiationChat.tsx`

**功能**:
- 消息列表渲染（AI/用户气泡）
- 输入框 + 发送
- 加载状态（AI 思考中...）
- 自动滚动到底部
- "正在更新配置..." 状态提示

**验收**: 组件渲染正常，可发送消息

---

### B.4 配置预览面板

**文件**: `frontend/src/components/ConfigPreview.tsx`

**功能**:
- Tab 切换: [检测配置] [预览结果] [完整需求]
- 检测配置 Tab: 展示 classes + rules（可折叠卡片）
- 预览结果 Tab: 展示检测图 + 统计
- 完整需求 Tab: 展示 algorithm_hints 摘要

**验收**: 三个 Tab 正常展示数据

---

### B.5 确认面板

**文件**: `frontend/src/components/ConfirmPanel.tsx`

**功能**:
- 当 `negotiationConverged=true` 时激活
- 展示确认摘要（目标、事件、排除条件）
- [继续修改] 按钮 → 继续对话
- [确认，进入方案规划 →] 按钮 → 调用 confirm API → setStage('algorithm_plan')
- converged=false 时按钮置灰 + 提示"还需确认..."

**验收**: 按钮状态正确，确认后正常跳转

---

### B.6 IntentConfirm 页面重构

**文件**: `frontend/src/pages/IntentConfirm.tsx`

**改造**:
- 布局改为左右分栏（对话 40% + 配置预览 60%）
- 首次进入：调用 `GET /negotiate/conversation/{taskId}` 检查是否有历史
  - 有 → 恢复对话
  - 无 → Agent A 生成开场白（基于 VLM parse 初始结果）
- 对话发送 → 调用 `POST /negotiate/chat`
- 配置更新 → 自动刷新右侧面板
- should_preview → 触发预览
- converged → 显示确认面板

**验收**: 完整对话流程可跑通

---

## Sprint C: YOLO-World 预览对接

### C.1 预览 API 实现

**文件**: `backend/routers/negotiate.py` (preview 端点)

**功能**:
- 接收当前 vocab 配置
- 从上传图片中随机选 2-3 张
- 调用 `stage2_labeler.run_detection()` 快速检测
- 应用 `detection_rules.post_filters` 过滤
- 返回检测结果 + base64 标注图

**关键**: 复用现有 `run_detection`，只是用 vocab 的 aliases 做 `set_classes`

---

### C.2 stage2_labeler 扩展

**文件**: `worker/pipeline/stage2_labeler.py`

**改动**:
- `run_detection` 新增可选参数 `vocab_aliases: dict[str, list[str]]`
- 如果传入 vocab_aliases → 用 primary + aliases 的全集做 `set_classes`
- 新增可选参数 `post_filters: list[dict]`
- 检测后按 post_filters 过滤（面积范围、置信度等）

**验收**: 传入 vocab 后检测结果正确使用别名

---

### C.3 前端预览展示

**文件**: `frontend/src/components/ConfigPreview.tsx` (预览 Tab)

**功能**:
- 调用预览 API 后展示标注图
- 显示统计: 命中 / 误检 / 总数
- 支持点击框标记"误检"（可选，作为反馈给 Agent A）

**验收**: 预览图正常展示，统计数据准确

---

## Sprint D: AlgorithmPlan 增强 + 回路

### D.1 AlgorithmPlan 接收 algorithm_hints

**文件**: `backend/services/vlm_algorithm_planner.py`

**改动**:
- `build_vlm_algorithm_plan` 新增参数 `algorithm_hints: dict`
- 将 hints 中的场景类型、事件定义、多模型需求注入 user_prompt
- 减少 VLM 重复推理（hints 已包含确认过的信息）

**验收**: 有 hints 时方案生成更精准、更快

---

### D.2 从后续阶段返回对话

**文件**: `frontend/src/pages/LabelingProgress.tsx`, `ReviewSamples.tsx`

**改动**:
- 添加 "回到需求调整" 按钮
- 点击 → `setStage('intent_confirm')`
- IntentConfirm 检测到已有对话历史 → 恢复对话，不重新开始
- 自动补充上下文："上次试打标效果：命中X个，误检Y个"

**验收**: 从后续阶段返回后，对话历史完整保留

---

### D.3 AlgorithmPlan 的"回到需求调整"

**文件**: `frontend/src/pages/AlgorithmPlan.tsx`

**改动**:
- 添加 "修改需求" 按钮 → 回到 IntentConfirm 对话
- 保留已生成的方案草稿，下次 confirm 后自动刷新

**验收**: 方案阶段也能返回对话

---

## 文件清单（新增）

| 文件 | Sprint | 说明 |
|------|--------|------|
| `backend/services/conversation_agent.py` | A.2 | Agent A 核心逻辑 |
| `backend/services/config_generator_agent.py` | A.3 | Agent B 核心逻辑 |
| `backend/services/negotiation_orchestrator.py` | A.4 | 编排器 |
| `backend/routers/negotiate.py` | A.5 | API 路由 |
| `frontend/src/components/NegotiationChat.tsx` | B.3 | 对话面板 |
| `frontend/src/components/ConfigPreview.tsx` | B.4 | 配置预览 |
| `frontend/src/components/ConfirmPanel.tsx` | B.5 | 确认面板 |

## 文件清单（修改）

| 文件 | Sprint | 改动 |
|------|--------|------|
| `backend/models/database.py` | A.1 | 新增 negotiation_conversations 表 |
| `backend/main.py` | A.5 | 注册 negotiate router |
| `frontend/src/store/taskStore.ts` | B.1 | 新增协商相关状态 |
| `frontend/src/api/backend.ts` | B.2 | 新增 negotiateApi |
| `frontend/src/pages/IntentConfirm.tsx` | B.6 | 重构为对话式 |
| `frontend/src/App.tsx` | B.6 | stage 标签可能微调 |
| `worker/pipeline/stage2_labeler.py` | C.2 | 支持 vocab_aliases + post_filters |
| `backend/services/vlm_algorithm_planner.py` | D.1 | 接收 algorithm_hints |
| `frontend/src/pages/LabelingProgress.tsx` | D.2 | 回路按钮 |
| `frontend/src/pages/AlgorithmPlan.tsx` | D.3 | 回路按钮 |

---

## 开发顺序建议

```
Day 1-2: Sprint A (A.1 → A.2 → A.3 → A.4 → A.5)
          后端全部 Agent 逻辑 + API，可用 curl 测试

Day 3-4: Sprint B (B.1 → B.2 → B.3 → B.4 → B.5 → B.6)
          前端对话 UI，对接后端 API

Day 5-6: Sprint C (C.1 → C.2 → C.3)
          YOLO-World 预览接入对话循环

Day 7:   Sprint D (D.1 → D.2 → D.3)
          AlgorithmPlan 增强 + 回路

Day 8-9: 集成测试 + Bug 修复 + 边界条件处理
```

---

## 模型降级策略

### 核心原则：VLM 为唯一强依赖，推理模型可选

```
用户必须配置:
  ✅ VLM 模型（必填）— Agent A 对话 + Agent B 降级后端

用户可选配置:
  ⬜ 推理模型（选填）— Agent B 首选后端，提升结构化输出质量
```

### 降级逻辑

```
Agent A (对话引导):
  → 始终使用 VLM（需要看图，无替代）

Agent B (配置生成):
  → 优先使用推理模型（reasoning_adapter）
  → 如果用户未配置推理模型 → 降级使用 VLM（vlm_adapter.call_with_system_prompt）
```

### 实现方式

```python
# negotiation_orchestrator.py
def _get_config_generator(self):
    """获取 Agent B 的模型后端，推理模型优先，VLM 兜底"""
    # 1. 尝试获取推理模型
    reasoner = build_reasoning_adapter_from_settings(
        self.settings, vlm_adapter=None
    )
    if reasoner is not None:
        return reasoner  # 用户配置了推理模型，使用它
    
    # 2. 降级：用 VLM 做结构化输出（VLMFallbackReasoner）
    return VLMFallbackReasoner(self.vlm_adapter)
```

### 前端提示

```
设置页面:
┌────────────────────────────────────────────┐
│  VLM 模型配置（必填）                        │
│  ├─ Provider: [Kimi ▼]                     │
│  ├─ API Key: [••••••••]                    │
│  └─ Model: [kimi-k2.5]                    │
│                                            │
│  推理模型配置（选填，提升配置生成质量）        │
│  ├─ Provider: [DeepSeek ▼]                 │
│  ├─ API Key: [••••••••]                    │
│  └─ Model: [deepseek-reasoner]             │
│                                            │
│  ℹ️ 未配置推理模型时，系统将使用 VLM 模型     │
│     进行结构化配置生成（效果略低但可用）       │
└────────────────────────────────────────────┘
```

### 进入对话的前置检查

```typescript
// IntentConfirm.tsx 进入时检查
if (!settings.vlm_provider || !settings.vlm_api_key) {
  // 弹窗提示: "请先在设置中配置 VLM 模型才能开始需求确认"
  // 引导去设置页
  return
}
// VLM 已配置 → 可以开始对话（不管有没有推理模型）
```

---

## 测试验收标准

| 场景 | 验收条件 |
|------|---------|
| 新任务首次对话 | 系统基于 VLM parse 结果生成开场白 + 追问 |
| 多轮对话细化 | 3-5 轮后 Agent A 判断 converged=true |
| 配置生成 | Agent B 输出合法 classes + rules + vocab JSON |
| 预览触发 | 对话中配置更新后自动触发 YOLO-World 预览 |
| 用户确认 | 点击确认按钮后正确跳转 AlgorithmPlan |
| 对话恢复 | 从后续阶段返回后对话完整恢复 |
| 仅配置 VLM | Agent B 使用 VLM fallback 生成配置，功能正常 |
| VLM + 推理模型都配置 | Agent B 优先用推理模型，质量更高 |
| VLM 未配置 | 阻止进入对话，提示配置 |
| VLM 超时 | 友好错误提示，不丢失对话历史 |
