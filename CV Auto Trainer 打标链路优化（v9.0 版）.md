# CV Auto Trainer 打标链路优化（v9.0 版）

# CV Auto Trainer 打标链路优化方案 & 推理模型集成指南

**版本：v9.0 优化版**  

面向 AI Agent 快速理解与执行

---

## 文档摘要（AI 快速读取入口）

- **本文档目的**：定位 CV Auto Trainer 现有打标链路的核心缺陷，并提供完整的优化方案。

- **核心结论**：当前系统最大问题在于 VLM 生成的类别词与 YOLO-World 词汇空间不对齐，导致打标召回率低、框画错。修复方案分四个层次，可独立实施。

- **关于推理模型**：VLM 不需要完全替换，而是引入推理模型（如 o3/QwQ/DeepSeek-R1）作为「决策层」，VLM 继续负责「感知层」，两者串联。

---

## 问题优先级总览

|优先级|问题描述|影响|解决方案|
|---|---|---|---|
|P0 🔴|VLM 词汇不对齐 CLIP 空间|打标完全失效，召回率<30%|修改 System Prompt + 验证层|
|P0 🔴|质检缺少 category_match 维度|错误框混入训练集，模型学歪|扩展 Moondream2 评分维度|
|P1 🟡|YOLO-World 不适合精细场景|工业/自定义类别基本检不到|引入 Grounding DINO 双引擎|
|P1 🟡|无人工确认节点导致批量错误|1000张图白跑|在意图确认页加词汇审查|
|P2 🟢|VLM 推理能力弱，结构化输出差|类别词生成不稳定|引入推理模型作决策层|
|P2 🟢|极端自定义类别无法检测|新类别训练数据为零|SAM2 点提示打标方案|
---

## 一、根本问题分析

### 1.1 当前打标链路的三个断层

#### 断层 1：VLM 词汇 ≠ CLIP 词汇空间（最严重）

YOLO-World 使用 CLIP 文本编码器来理解类别词。CLIP 对词汇极度敏感，同义词在向量空间中的距离可能差异巨大。VLM 自由生成的词很可能落在 YOLO-World 的弱覆盖区域。

- ❌ **VLM 自由生成（当前）**输出: "工人"、"危险区域"、"设备"、"人员""危险区域" → CLIP 无法理解的抽象概念"工人" → 不如 "worker" 或 "person" 精确→ YOLO-World 大量漏检

- ✅ **约束生成（修复后）**输出: person, hard_hat, safety_vest, forklift全部为 CLIP 可理解的具体视觉实体单词而非短语，符合 CLIP 训练分布→ YOLO-World 正常检测

#### 断层 2：YOLO-World 不适合精细/自定义场景

YOLO-World 的 open-vocabulary 能力基于 COCO+Objects365 预训练。对标准类别（人、车、动物）表现好，对工业零件、医疗器械、特定产品等细粒度类别基本检不到。这不是 prompt 问题，是模型能力边界问题。

- 适合 YOLO-World 的场景：COCO 80 类及其近义词，人、车、动物、日常物品

- 不适合 YOLO-World 的场景：工业零件、医疗设备、自定义产品、品牌特有物品、细粒度子类别

#### 断层 3：VLM 推理能力不足，结构化输出不稳定

当前使用的 VLM（如 GPT-4V / Kimi Vision / Gemini Vision）在「图片描述」任务上表现优秀，但在「结构化任务规划、约束推理、格式保证」上能力较弱。具体表现：

- 生成的 JSON 格式不稳定，偶尔夹杂解释文字导致解析失败

- 无法自主判断哪些词对 CLIP 友好，哪些不友好

- 对复杂约束（最多N个类别、不能包含抽象概念）的遵循率低

- 质检评分时对边界情况判断不一致，评分飘忽

---

## 二、P0 修复：VLM System Prompt 重写

### 2.1 新版 System Prompt（直接替换）

```Plain Text

SYSTEM PROMPT v2 — CV Intent Parser
─────────────────────────────────────────────────────────
You are a computer vision annotation expert.
Convert user descriptions into YOLO-World detection categories.

STRICT RULES:
1. Output ONLY a JSON array of strings — no explanation, no markdown
2. Each string must be a SINGLE, PHYSICALLY VISIBLE noun (one entity)
3. Use the MOST COMMON English name that a CLIP model would recognize
4. Prefer SPECIFIC over GENERIC:
   "hard_hat" not "equipment"
   "sedan" not "vehicle"
   "german_shepherd" not "dog" (if breed matters)
5. Maximum 10 categories per task
6. NEVER include:
   - Abstract concepts: "danger", "safety", "area", "zone"
   - States/actions: "running", "broken", "working"
   - Attributes alone: "red", "large"
   - Spatial relationships: "near", "inside"
7. For industrial/custom objects: use closest common-noun equivalent

CLIP-FRIENDLY VOCABULARY REFERENCE:
  People:     person, worker, pedestrian, cyclist, driver, patient
  Vehicles:   car, truck, bus, motorcycle, bicycle, forklift, excavator
  Safety:     hard_hat, safety_vest, face_mask, glove, safety_shoe
  Industrial: bottle, box, pallet, conveyor_belt, pipe, valve, screw, bearing
  Animals:    dog, cat, bird, cow, horse
  Medical:    syringe, pill, tablet, bandage, stethoscope

User description: {user_input}
Image analysis: {vlm_image_description}

Output: ["category1", "category2", ...]
```

### 2.2 追加验证层（生成后立即执行）

```Python

VALIDATION_PROMPT = """
Given these detection categories: {categories}

For each category, rate 0-10:
  clip_score:    Can a CLIP vision model detect this visually? (10=easy)
  specificity:   Is it a concrete, single, visible object? (10=very specific)

Rules:
- Remove any category with clip_score < 6
- Remove any category with specificity < 6
- If a category is too generic, suggest a more specific replacement

Return JSON only:
{
  "valid": ["cat1", "cat2"],
  "removed": [{"word": "x", "reason": "abstract concept"}],
  "suggestions": [{"original": "vehicle", "replace_with": "car"}]
}
"""
```

### 2.3 UI 改动：意图确认页增加词汇审查组件

- 展示内容：

    - 每个类别词 + CLIP 置信度分数（颜色标注：绿色≥7、黄色4-6、红色<4）

    - 「删除」按钮：移除不需要的词

    - 「修改」输入框：直接编辑词汇

    - 「添加」按钮：补充遗漏的类别

    - 预计打标数量估算（根据类别数量和图片数量）

- 交互要求：

    - 红色词汇必须用户主动确认或删除才能继续

    - 确认后记录到任务元数据，用于后续训练报告

---

## 三、P1 优化：打标引擎三路由策略

### 3.1 引擎选择逻辑

```Python

def select_labeling_engine(categories: list[str], task_type: str) -> str:
    """
    三引擎自动路由策略
    Returns: "yolo_world" | "grounding_dino" | "sam2_point" | "hybrid"
    """
    # YOLO-World 擅长的核心词汇集
    YOLO_WORLD_STRONG = {
        "person","car","truck","bus","bicycle","motorcycle","dog","cat","bird",
        "bottle","chair","table","cup","fork","knife","laptop","phone","book",
        "worker","forklift","hard_hat","safety_vest",  # 安全场景常见词
    }

    overlap = len(set(categories) & YOLO_WORLD_STRONG)
    coverage = overlap / max(len(categories), 1)

    if task_type == "industrial" or coverage < 0.3:
        return "grounding_dino"   # 工业/自定义：用 Grounding DINO
    elif coverage >= 0.8:
        return "yolo_world"       # 大部分是通用类别：保持 YOLO-World
    else:
        return "hybrid"           # 混合：通用词 YOLO-W，自定义词 GDINO
```

### 3.2 Grounding DINO 接入方案（核心升级）

```Python

from groundingdino.util.inference import load_model, predict
import torch

class GroundingDINOLabeler:
    def __init__(self, model_path: str, config_path: str):
        self.model = load_model(config_path, model_path)
        self.model.eval()

    def label(self, image, categories: list[str],
              box_threshold=0.35, text_threshold=0.25):
        text_prompt = " . ".join(categories)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        return self._to_yolo_format(boxes, logits, phrases)

    def release(self):
        del self.model
        torch.cuda.empty_cache()
        import gc; gc.collect()
```

### 3.3 显存分配时间表（4GB 低显存机器）

|时间段|模型|显存用量|适用场景|结束后操作|
|---|---|---|---|---|
|T0|YOLO-World FP16|~2.5 GB|通用类别|del + empty_cache|
|T1|Grounding DINO tiny|~1.5 GB|自定义/工业|del + empty_cache|
|T2|SAM2-small (可选)|~3.0 GB|极细粒度|del + empty_cache|
|T3|Moondream2 质检|~2.2 GB|所有框质检|del + empty_cache|
---

## 四、推理模型集成方案

### 4.1 为什么需要推理模型

- VLM 擅长：图片理解、场景描述、视觉问答、目标识别

- VLM 不擅长：复杂约束推理、结构化格式保证、多步骤逻辑判断、边界情况处理

- 推理模型擅长：严格遵循约束规则、稳定输出结构化 JSON、多维度权衡判断、逻辑一致性

- **结论**：不是「换掉 VLM」，而是「VLM 感知 → 推理模型决策」两层串联架构

### 4.2 两层架构设计

```Plain Text

┌─────────────────────────────────────────────────────────────┐
│  感知层（VLM）          决策层（推理模型）                    │
│                                                             │
│  输入：用户描述 + 样板图                                    │
│     ↓                                                       │
│  VLM 任务：                推理模型任务：                   │
│  · 描述图片中的物体        · 将 VLM 描述转化为标准类别词    │
│  · 识别用户意图            · 验证词汇 CLIP 友好度           │
│  · 估算目标尺寸分布        · 质检评分的边界情况判断         │
│  · 判断场景类型            · 训练参数推荐                   │
│     ↓                         ↓                            │
│  输出：图片描述文本        输出：结构化 JSON                │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 推荐模型选型

|模型|推理能力|成本|速度|推荐场景|
|---|---|---|---|---|
|o3-mini|⭐⭐⭐⭐⭐|中|中|最推荐，推理强+成本合理|
|DeepSeek-R1|⭐⭐⭐⭐⭐|低|慢|成本最低，可本地部署|
|QwQ-32B|⭐⭐⭐⭐|低|中|中文友好，开源可本地|
|Claude 3.5 Sonnet|⭐⭐⭐⭐|中|快|稳定性最好，格式遵循强|
|Gemini 2.0 Flash|⭐⭐⭐|低|极快|预算有限时备选|
### 4.4 推理模型调用代码

```Python

class ReasoningModelAdapter:
    CATEGORY_VALIDATION_PROMPT = """
    You are a computer vision expert. Validate these detection categories.
    Categories to validate: {categories}
    Task description: {task_description}
    Return ONLY valid JSON, no explanation
    """

    def validate_categories(self, categories: list[str], task_desc: str) -> dict:
        response = self.client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": self.CATEGORY_VALIDATION_PROMPT.format(
                categories=categories, task_description=task_desc)}],
            reasoning_effort="medium"
        )
        import json
        return json.loads(response.choices[0].message.content)
```

---

## 五、P0 修复：质检维度强化

### 5.1 当前质检的关键缺失

- 缺失维度 1 — **category_match**（最重要）：框内容是否真的是这个类别？

- 缺失维度 2 — **tightness**（框松紧）：框太松/太紧都会影响模型学习

### 5.2 扩展质检代码

```Python

ENHANCED_QC_PROMPT = """
Evaluate this bounding box annotation carefully.
Category label: {category}
Score each dimension from 0.0 to 1.0:
1. category_match
2. completeness
3. tightness
4. clarity
5. no_occlusion_error
Return ONLY JSON
"""

def should_keep(scores: dict) -> tuple[bool, str]:
    if scores["category_match"] < 0.5:
        return False, "category_mismatch"
    if scores["tightness"] < 0.35:
        return False, "box_too_loose"
    if scores["completeness"] < 0.4:
        return False, "object_cut_off"
    if scores["clarity"] < 0.35:
        return False, "too_blurry"
    avg = sum(scores.values()) / len(scores)
    if avg < 0.5:
        return False, "low_overall"
    return True, "pass"
```

### 5.3 训练前类别平衡预警

```Python

def check_class_balance(dataset_dir: str) -> dict:
    counts = count_labels_per_class(dataset_dir)
    max_count = max(counts.values())
    warnings, errors = [], []
    for cls, cnt in counts.items():
        ratio = cnt / max_count
        if cnt < 50:
            errors.append(f"🔴 [{cls}] 样本严重不足: {cnt}张（建议≥100张）")
        elif ratio < 0.1:
            warnings.append(f"⚠️ [{cls}] 严重不平衡")
        elif ratio < 0.3:
            warnings.append(f"⚡ [{cls}] 偏少")
    if errors:
        raise DatasetImbalanceError(errors)
    return {"counts": counts, "warnings": warnings}
```

---

## 六、P1 优化：人机协作两节点

### 6.1 新增流程总览

1. 用户上传样板图 + 描述

2. [感知层] VLM 分析图片 → 输出场景描述文本

3. [决策层] 推理模型 → 生成并验证类别词

4. ★ 【人工节点 1】意图确认页：词汇审查

5. [打标引擎] 自动路由 → YOLO-World / Grounding DINO / 混合

6. [质检] Moondream2 五维度评分 → 过滤低质量框

7. ★ 【人工节点 2】样本抽检（抽取 5%，每类≥5张）

8. [数据增强] Albumentations

9. [类别平衡检查] → 预警或阻断

10. [数据集分割] 8:1:1

11. [训练] 本地 / 云端

12. 交付 [best.pt](best.pt) + ONNX + 训练报告

### 6.2 样本抽检界面功能要求

- 框拖拽调整

- 类别标签修改

- 删除错误框

- 手动补框

- 标记此类全错

- 跳过/确认（快捷键）

---

## 七、进阶方案：SAM2 点提示打标

### 7.1 适用场景

- Grounding DINO 置信度 < 0.3（检测失败）

- 用户在抽检时标记「此类完全错误」

- 用户在意图确认时手动选择「精细打标模式」

### 7.2 SAM2 点提示流程

```Python

# Step 1: VLM 输出目标中心点坐标
# Step 2: SAM2 点提示生成 Mask
# Step 3: Mask → BBox（外接矩形）

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

def sam2_label(image, center_points: list[tuple[float,float]]):
    h, w = image.shape[:2]
    input_points = [[x*w, y*h] for x, y in center_points]
    input_labels = [1] * len(input_points)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )
    # 转 BBox
    bboxes = []
    for mask in masks:
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]
        bboxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
    return bboxes, masks, scores
```

---

## 八、实施计划与优先级

### 8.1 分阶段实施路线

|阶段|时间|实施内容|预期效果|风险|
|---|---|---|---|---|
|Phase 1|1-2天|替换 VLM System Prompt + 验证层|打标召回率提升 40-60%|低|
|Phase 2|1天|IntentConfirm 页加词汇审查 UI|消除批量错误|低|
|Phase 3|3-5天|接入 Grounding DINO 双引擎|工业场景成功率 >70%|中|
|Phase 4|2-3天|扩展质检到五维度|训练集质量提升|低|
|Phase 5|5-7天|推理模型作决策层|稳定性与准确率全面提升|中|
|Phase 6|7-10天|SAM2 点提示打标|覆盖极端自定义场景|高|
### 8.2 对现有架构的改动范围

- backend/services/[vlm_adapter.py](vlm_adapter.py) — 修改 System Prompt + 验证层

- backend/services/[reasoning_adapter.py](reasoning_adapter.py) — 新增推理模型适配器

- worker/pipeline/[stage2_labeler.py](stage2_labeler.py) — 新增 Grounding DINO + 路由

- worker/pipeline/[stage2_qc.py](stage2_qc.py) — 扩展质检维度

- frontend/src/pages/IntentConfirm.tsx — 词汇审查 UI

- frontend/src/pages/ReviewSamples.tsx — 抽检界面扩展

- worker/pipeline/[stage2_sam2.py](stage2_sam2.py) — 新增 SAM2 引擎

### 8.3 AI Agent 执行指令

- **立即执行**：

    - 修改 [vlm_adapter.py](vlm_adapter.py) 的 System Prompt

    - 追加词汇验证层逻辑

    - 扩展质检到 5 个维度

- **需确认后执行**：

    - Grounding DINO 接入

    - 推理模型 API Key 配置

    - UI 页面改动

- **暂缓**：

    - SAM2 点提示方案

    - 推理模型完整替换决策层

---

**文档版本：v9.0 优化版 | 生成日期：2026-05**

---

要不要我帮你把这份 MD 再**精简成一页执行版**，方便直接贴到项目仓库做 README？
> （注：文档部分内容可能由 AI 生成）