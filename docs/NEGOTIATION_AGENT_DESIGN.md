# 多智能体需求确认助手 — 详细设计文档

> 基于 LLM_RULE_AUTODISTILL_ANALYSIS.md 的分析结论，给出完整实施方案

---

## 一、当前系统阶段流程（改造前）

```
Upload → IntentConfirm(静态表单) → AlgorithmPlan(完整方案) → Environment → Labeling → ...
              ↑                          ↑
              单轮VLM解析                 VLM+推理模型生成
              只管检测类别               完整pipeline(多模型/事件/区域)
```

**问题**：
- IntentConfirm 只管检测类别编辑，和用户零交互
- AlgorithmPlan 生成完整方案，但输入（类别定义）可能不准
- 试打标后发现不对，没有回路修正

---

## 二、改造后的阶段流程

```
Upload
  ↓ VLM parse (现有，生成初始理解)
  ↓
IntentConfirm (改造为「需求确认对话」)
  ┌─────────────────────────────────────────┐
  │  Agent A (VLM): 对话引导                  │
  │  - 和用户聊完整的算法需求                  │
  │  - 不仅是"检测什么"，还包括:              │
  │    · 要识别哪些对象？                     │
  │    · 什么场景/事件需要触发告警？           │
  │    · 区域划分？时间约束？                  │
  │    · 可能需要几个模型协作？                │
  │  - 看图辅助理解                           │
  │                                          │
  │  Agent B (推理模型): 结构化配置生成         │
  │  - classes 类别配置 → 给 YOLO-World 打标用 │
  │  - detection_rules → 后处理过滤规则        │
  │  - vocab → 开放词汇表(prompt_aliases)      │
  │  - algorithm_hints → 给后续方案规划的提示   │
  └─────────────────────────────────────────┘
  ↓ 用户确认("可以了") → 显示确认按钮 → 点击
  ↓
AlgorithmPlan (现有，但输入质量大幅提升)
  - 接收 Agent B 的精确 classes + algorithm_hints
  - 生成完整 pipeline（多模型、事件、区域）
  ↓
Environment → Labeling (试打标)
  ↓
  ┌─ 效果不好？ ─→ 回到 IntentConfirm 对话（带上下文） ─┐
  │                                                      │
  └──────────────── 效果好 → 继续后续流程 ───────────────┘
```

---

## 三、对话范围：不只是训练一个模型

你说得对。用户和 VLM 聊的是**完整算法需求**，不只是"检测什么"。

### 对话要覆盖的维度

| 维度 | 示例对话 | 输出产物 |
|------|---------|---------|
| **检测目标** | "检测三角木和货车轮胎" | `classes[]` |
| **场景/事件** | "轮胎旁没有三角木时告警" | `algorithm_hints.events[]` |
| **区域约束** | "只关注车位区域" | `algorithm_hints.regions[]` |
| **时间约束** | "停留超过30秒才算" | `algorithm_hints.temporal[]` |
| **多模型需求** | "还需要识别车牌号" | `algorithm_hints.extra_capabilities[]` (如 OCR) |
| **排除条件** | "不要阴影误检" | `classes[].negative_prompt` |
| **精度/速度** | "边缘设备要实时" | `algorithm_hints.performance_constraints` |

**关键**：Agent A 收集的不只是类别信息，而是完整的算法意图。Agent B 把这些意图转成两部分：
1. **给 YOLO-World 打标用的**：`classes` + `detection_rules` + `vocab`
2. **给 AlgorithmPlan 用的**：`algorithm_hints`（场景、事件、多模型需求）

---

## 四、两个 Agent 的精确职责分工

### Agent A: 对话引导 (VLM 多模态)

```
为什么用 VLM：需要看图。用户说"这个东西"，AI 要看图理解。
               预览结果也需要 VLM 看懂才能解释给用户。

输入：
  - 用户消息（文字）
  - 对话历史（完整上下文，不丢失）
  - 用户样张（图片 base64）
  - 当前预览检测结果截图（可选）
  - 当前已确认的需求摘要

输出：
  {
    "reply": "中文自然语言回复",
    "intent_update": {           // 本轮对话收集到的信息增量
      "confirmed_targets": [...],
      "confirmed_events": [...],
      "confirmed_regions": [...],
      "confirmed_exclusions": [...],
      "confirmed_constraints": [...]
    } | null,
    "should_regenerate": true,   // 是否触发 Agent B 重新生成配置
    "should_preview": true,      // 是否触发 YOLO-World 预览
    "convergence": {
      "converged": false,        // 需求是否已明确
      "reason": "用户还未确认排除条件"
    }
  }
```

**Agent A 的 System Prompt 要点**：

```
你是 CV Auto Trainer 的需求确认助手，帮助用户明确他们想实现的完整视觉算法方案。

你的沟通范围不限于"检测什么对象"，还包括：
- 业务场景是什么？（占位监测？安全合规？闯入告警？）
- 需要检测哪些对象？每个对象的视觉特征？
- 什么事件需要触发告警？（对象进入区域？离开？停留超时？）
- 是否需要多个模型协作？（检测+分类？检测+OCR？检测+跟踪？）
- 有什么特殊约束？（实时性？边缘设备？精度要求？）

沟通策略：
1. 初始：基于 VLM 的初始理解，先展示你理解了什么，然后一次追问 2-3 个最关键的模糊点
2. 中期：每轮聚焦 1 个维度深挖，不要一次问太多
3. 触发预览：当类别定义有更新时，建议做一次预览让用户看效果
4. 收敛判断：当以下条件都满足时，告知用户可以确认
   - 所有检测类别都有明确定义
   - 事件/告警逻辑已明确
   - 排除条件已明确
   - 至少做过 1 次预览且用户未提异议

你要用中文回复，语气简洁友好。不要使用技术术语（CLIP、prompt、class_name 等），
用户是普通业务人员。
```

### Agent B: 配置生成 (推理模型)

```
为什么用推理模型：不需要看图，但需要严格的逻辑推理。
                 把自然语言需求转成精确的 JSON 格式，
                 必须遵守 schema 约束、CLIP 友好性规则。

输入：
  - Agent A 输出的 intent_update（已确认的需求摘要）
  - 当前对话的完整上下文摘要
  - 约束条件（CLIP 词表、schema 格式等）

输出（三件套 + 方案提示）：
  {
    // 1. 给 YOLO-World 自动打标用
    "classes": [
      {
        "class_name": "wheel_chock",
        "prompt": "wheel chock",
        "prompt_aliases": ["chock block", "wheel wedge", "yellow triangular block"],
        "negative_prompt": "stone, brick, shadow, debris",
        "color_hint": "yellow or red",
        "display_name_zh": "三角木",
        "display_prompt_zh": "货车轮胎旁用于防滑的三角形木块",
        "display_negative_prompt_zh": "石块、砖头、阴影",
        "display_color_hint_zh": "黄色或红色"
      }
    ],
    
    // 2. 检测后处理规则
    "detection_rules": {
      "conf_threshold": 0.3,
      "iou_threshold": 0.45,
      "post_filters": [
        { "type": "min_area", "value": 0.005, "unit": "relative" },
        { "type": "max_area", "value": 0.08, "unit": "relative" }
      ]
    },

    // 3. 开放词汇表（给 YOLO-World set_classes 用）
    "vocab": {
      "wheel_chock": {
        "primary": "wheel chock",
        "aliases": ["chock block", "wheel wedge"],
        "context_anchors": ["truck tire", "wheel"]
      }
    },

    // 4. 算法方案提示（给后续 AlgorithmPlan 用）
    "algorithm_hints": {
      "scenario_type": "safety_compliance",
      "needs_tracking": false,
      "needs_ocr": false,
      "events": [
        {
          "name_zh": "缺少三角木告警",
          "trigger": "truck_tire 存在但 wheel_chock 不在其附近"
        }
      ],
      "regions": [
        { "label": "车位区域", "purpose": "限定检测范围" }
      ],
      "performance_hint": "real_time",
      "multi_model_needed": false,
      "suggested_pipeline_roles": ["primary_detector"]
    }
  }
```

**Agent B 的 System Prompt 要点**：

```
你是结构化配置生成专家。把已确认的业务需求转换成检测系统的精确配置。

转换规则：
1. class_name 必须是 CLIP 友好的 1-3 词英文名词（参考 CLIP 词表）
2. prompt_aliases 至少 3 个同义词，用于提高召回
3. negative_prompt 列出视觉上容易混淆的排除项
4. detection_rules.post_filters 只包含用户明确确认过的约束
   - 不要编造未确认的精确数值（如 RGB 范围）
   - 可以用相对面积（relative area）等粗粒度过滤
5. vocab.primary 是交给 YOLO-World set_classes() 的文本
6. algorithm_hints 提供给后续 AlgorithmPlan 阶段，帮助生成完整 pipeline

输出格式必须严格符合 schema，不允许额外文本。
```

---

## 五、关键交互细节

### 5.1 试打标后回到对话（上下文不丢失）

```
IntentConfirm (对话完成，用户确认)
  ↓
AlgorithmPlan → Environment → Labeling (试打标)
  ↓
  用户看了打标结果，发现问题
  ↓ 点击 "回到需求调整"
  ↓
IntentConfirm (对话恢复)
  - 对话历史完整保留（从 conversationStore 读取）
  - 当前配置保留
  - 新增上下文："上次试打标结果：命中12个，误检5个，漏检3个"
  - Agent A 自动开场："上次打标有些问题，请告诉我哪里不满意？"
```

**实现方式**：对话历史持久化到后端数据库（conversation 表），不随 stage 切换丢失。

```python
# 后端数据模型
class NegotiationConversation(Base):
    __tablename__ = 'negotiation_conversations'
    id = Column(String, primary_key=True)
    task_id = Column(String, ForeignKey('tasks.id'))
    messages = Column(JSON)         # [{ role, content, timestamp }]
    current_config = Column(JSON)   # Agent B 最新输出
    confirmed = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
```

### 5.2 用户确认流程（不自动跳转）

```
对话中...
  ↓
用户: "可以了" / "没问题" / "就这样吧"
  ↓
Agent A 检测到确认意图，返回 converged=true
  ↓
前端展示确认面板：
  ┌────────────────────────────────────────────────┐
  │  ✅ 需求已确认                                   │
  │                                                  │
  │  检测目标: 三角木(wheel_chock), 货车轮胎(truck_tire) │
  │  告警事件: 轮胎旁无三角木                          │
  │  排除条件: 石块、阴影                              │
  │  预览检测: 命中15个，误检0个                        │
  │                                                  │
  │  [继续修改]         [确认，进入方案规划 →]          │
  └────────────────────────────────────────────────┘
```

**关键**：
- `converged=true` 只是让确认按钮亮起来，**不会自动跳转**
- 用户必须主动点击「确认，进入方案规划」才会 `setStage('algorithm_plan')`
- 用户点「继续修改」可以继续对话

### 5.3 从 Labeling 返回时的入口

在 LabelingProgress 页面和后续页面添加返回入口：

```typescript
// LabelingProgress.tsx / ReviewSamples.tsx
<button onClick={() => setStage('intent_confirm')}>
  回到需求调整
</button>
```

前端检测到 stage 是 `intent_confirm` 且已有对话历史时，自动恢复对话，不重新开始。

---

## 六、数据流全景

```
                  ┌──────────────────┐
                  │  用户对话输入      │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │   Agent A (VLM)   │──→ reply + intent_update
                  └────────┬─────────┘
                           │ should_regenerate=true?
                           ▼
                  ┌────────────────────┐
                  │  Agent B (推理模型)  │
                  └──┬─────┬─────┬──┬─┘
                     │     │     │  │
            ┌────────┘     │     │  └──────────┐
            ▼              ▼     ▼              ▼
    ┌──────────────┐ ┌────────┐ ┌─────┐ ┌──────────────┐
    │ classes[]     │ │ rules  │ │vocab│ │algorithm_hints│
    │ (VLMClass[])  │ │ (JSON) │ │(JSON│ │  (JSON)       │
    └──────┬───────┘ └───┬────┘ └──┬──┘ └──────┬───────┘
           │             │         │            │
           │     ┌───────┘         │            │
           ▼     ▼                 ▼            │
    ┌──────────────────────┐                    │
    │  YOLO-World 预览/打标  │                    │
    │  set_classes(vocab)   │                    │
    │  post_filter(rules)   │                    │
    └──────────────────────┘                    │
                                                │
           ┌────────────────────────────────────┘
           ▼
    ┌──────────────────────────────────────────┐
    │  AlgorithmPlan (vlm_algorithm_planner)    │
    │  输入: classes + algorithm_hints           │
    │  输出: 完整 pipeline (多模型/事件/区域)     │
    │  - primary_detector (YOLO)                │
    │  - tracker (ByteTrack, 如需要)             │
    │  - ocr (EasyOCR, 如需要)                   │
    │  - rule_engine (事件逻辑)                   │
    └──────────────────────────────────────────┘
```

**清晰分工**：
- **Agent B 的 classes + rules + vocab → 给 YOLO-World 打标用**
- **Agent B 的 algorithm_hints → 给 AlgorithmPlan 用，生成完整多模型方案**
- **Agent A 只负责对话，不直接输出技术配置**

---

## 七、与现有模块的对接

| 现有模块 | 改动 | 说明 |
|---------|------|------|
| `vlm_adapter.py` | 新增 `chat_negotiate()` | 支持多轮对话（传入 history），Agent A 用 |
| `reasoning_adapter.py` | 新增 `generate_full_config()` | 扩展 `reason_json` 输出三件套，Agent B 用 |
| `vlm_algorithm_planner.py` | 接收 `algorithm_hints` | `build_vlm_algorithm_plan` 额外接收 hints 参数 |
| `stage2_labeler.py` | 接收 `vocab` + `detection_rules` | `run_detection` 使用 vocab 的 aliases 做 set_classes，用 rules 做后处理过滤 |
| `IntentConfirm.tsx` | 重构为对话式 | 左对话 + 右配置预览 + 确认面板 |
| `taskStore.ts` | 新增字段 | `conversationId`, `negotiatedConfig`, `algorithmHints` |
| `LabelingProgress.tsx` | 新增返回入口 | "回到需求调整" 按钮 |
| 后端 DB | 新增 `negotiation_conversations` 表 | 持久化对话历史 + 配置 |

---

## 八、后端 API 设计

### 8.1 对话 API

```
POST /api/negotiate/chat
{
  "task_id": "xxx",
  "conversation_id": "conv_xxx",     // 首次为空，后端创建
  "message": "漏检了一个小的三角木",
  "include_image": true,              // 是否传当前预览图给 VLM 看
  "preview_stats": {                  // 最近预览统计
    "hits": 12, "false_positives": 5, "misses": 3
  }
}

Response:
{
  "conversation_id": "conv_xxx",
  "reply": "明白，我调整了...",
  "updated_config": {                // Agent B 重新生成的配置（如有变化）
    "classes": [...],
    "detection_rules": {...},
    "vocab": {...},
    "algorithm_hints": {...}
  } | null,
  "should_preview": true,
  "convergence": {
    "converged": false,
    "summary": "还需确认排除条件"
  }
}
```

### 8.2 确认并进入下一阶段

```
POST /api/negotiate/confirm
{
  "task_id": "xxx",
  "conversation_id": "conv_xxx"
}

Response:
{
  "finalized_config": {
    "classes": [...],
    "detection_rules": {...},
    "vocab": {...},
    "algorithm_hints": {...}
  }
}

→ 前端收到后写入 taskStore，然后 setStage('algorithm_plan')
```

### 8.3 获取对话历史（恢复用）

```
GET /api/negotiate/conversation/{task_id}

Response:
{
  "conversation_id": "conv_xxx",
  "messages": [...],
  "current_config": {...},
  "confirmed": false
}
```

---

## 九、前端 UI 结构

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage: 需求确认                                                  │
├──────────────────────────┬──────────────────────────────────────┤
│  对话面板 (左侧 40%)      │  配置 & 预览面板 (右侧 60%)           │
│                          │                                      │
│  ┌──────────────────┐    │  Tab: [检测配置] [预览结果] [完整需求]  │
│  │ [AI] 您好！根据您  │    │                                      │
│  │ 上传的图片，我理解 │    │  [检测配置 Tab]                        │
│  │ 您需要检测三角木。 │    │  ┌────────────────────────────┐      │
│  │ 请问：             │    │  │ wheel_chock (三角木)        │      │
│  │ 1. 颜色是？        │    │  │ prompt: wheel chock         │      │
│  │ 2. 只要轮胎旁的？  │    │  │ aliases: chock block, ...   │      │
│  │                    │    │  │ exclude: stone, shadow      │      │
│  │ [User] 黄色的，    │    │  └────────────────────────────┘      │
│  │ 轮胎旁边的         │    │                                      │
│  │                    │    │  [预览结果 Tab]                       │
│  │ [AI] 明白。已更新  │    │  ┌────────────────────────────┐      │
│  │ 配置并检测了2张图， │    │  │ [图片+检测框]               │      │
│  │ 请看右侧预览。     │    │  │ 命中: 5  误检: 1  漏检: 0   │      │
│  │                    │    │  └────────────────────────────┘      │
│  │ [User] 可以了       │    │                                      │
│  │                    │    │  [完整需求 Tab]                       │
│  │ [AI] 需求已明确 ✅ │    │  ┌────────────────────────────┐      │
│  │ 请点击右侧确认按钮 │    │  │ 场景: 安全合规               │      │
│  │ 进入下一步。        │    │  │ 目标: 三角木, 货车轮胎       │      │
│  └──────────────────┘    │  │ 事件: 缺少三角木告警          │      │
│                          │  │ 约束: 仅车位区域内             │      │
│  [输入框]       [发送]   │  └────────────────────────────┘      │
│                          │                                      │
│                          │ ┌──────────────────────────────────┐ │
│                          │ │ [继续修改]  [✅ 确认，进入方案规划] │ │
│                          │ └──────────────────────────────────┘ │
└──────────────────────────┴──────────────────────────────────────┘

注: 确认按钮仅在 converged=true 时高亮可点击
```

---

## 十、实施顺序

### Sprint A: 后端 Agent 层 (P0)

```
1. negotiation_conversations DB 模型
2. negotiation_orchestrator.py — 编排 Agent A + Agent B
3. conversation_agent.py — VLM 多轮对话 (Agent A)
4. config_generator_agent.py — 推理模型配置生成 (Agent B)
5. /api/negotiate/chat、/confirm、/conversation API
```

### Sprint B: 前端对话 UI (P0)

```
1. NegotiationChat 组件（对话面板 + 消息渲染）
2. ConfigPreview 组件（右侧配置展示 + 预览图）
3. ConfirmPanel 组件（确认面板，converged 时亮起）
4. IntentConfirm.tsx 重构为三栏布局
5. 对话历史恢复逻辑
```

### Sprint C: YOLO-World 对接 (P1)

```
1. stage2_labeler.py 接收 vocab aliases 做 set_classes
2. stage2_labeler.py 接收 detection_rules 做后处理过滤
3. 预览 API 接入对话循环
4. LabelingProgress 添加"回到需求调整"入口
```

### Sprint D: AlgorithmPlan 增强 (P1)

```
1. vlm_algorithm_planner.py 接收 algorithm_hints
2. 利用 hints 减少 VLM 重复推理
3. 方案生成质量因精确输入而提升
```
