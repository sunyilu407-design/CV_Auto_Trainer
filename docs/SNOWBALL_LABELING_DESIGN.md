# 增量打标（Snowball Labeling）详细设计文档

> 当 YOLO-World 零样本检测效果极差时（如小目标三角木 ≤6% 置信度），
> 通过「人工种子标注 → 种子模型训练 → 自动打标 → 合并」的滚雪球循环，
> 让用户只标 50-100 张就能利用上万张数据集。

---

## 1. 整体流程

```
Upload → IntentConfirm → AlgorithmPlan → Environment
  │
  ├─ [YOLO-World 预览效果好] → 正常自动打标 (labeling) → augment → ...
  │
  └─ [YOLO-World 预览效果差] → 进入增量打标模式 ──┐
                                                    │
    ┌───────────────────────────────────────────────┘
    ▼
    Stage: manual_annotation
      用户在数据集图片上手动画框标注
      最少 20 张, 建议 50-100 张
      标注保存为 YOLO .txt 格式
    │
    ▼
    Stage: seed_training
      用种子标注快速训练 YOLOv8n
      epochs: 50, imgsz: 640, patience: 15
      约 3-10 分钟 (GPU) / 10-30 分钟 (CPU)
    │
    ▼
    Stage: labeling (种子模型模式)
      用 seed_model.pt 对剩余未标注图片做推理
      置信度分层:
        >= 0.5 → 自动采纳 (auto_accepted)
        0.2~0.5 → 低置信待审 (needs_review)
        < 0.2 → 丢弃
    │
    ▼
    合并所有标注:
      seed_labels/ (手动) + auto_labels/ (种子模型)
      → labeled_images/ + labels/
      进入正常后续流程: augment → review → train_config → training
```

### 1.1 可选的多轮滚雪球

如果种子模型自动打标效果仍不理想（自动采纳比例 < 40%），用户可以：
1. 审核/修正低置信框
2. 把修正后的数据加入训练集
3. 重新训练更好的模型
4. 用新模型再次自动打标

每轮都会提升模型质量，一般 2-3 轮即可收敛。

---

## 2. 新增 Stage 定义

```typescript
// frontend/src/store/taskStore.ts
export type Stage =
  | 'upload'
  | 'intent_confirm'
  | 'algorithm_plan'
  | 'environment'
  | 'manual_annotation'   // 新增: 手动种子标注
  | 'seed_training'       // 新增: 种子模型训练
  | 'labeling'            // 现有: YOLO-World 自动打标 / 种子模型自动打标
  | 'augment'
  | 'review'
  | ...
```

---

## 3. 模块设计

### 3.1 手动标注页面 (ManualAnnotation.tsx)

**入口条件:**
- AlgorithmPlan 预览结果差 (`accepted === 0 && candidates <= 2`) 时显示入口按钮
- 或用户主动选择「手动标注模式」

**UI 布局:**

```
+----------------------------------------------------------+
|  手动标注种子框                                           |
|  已标注 23/200 张 (最少 20 张)        [完成标注, 开始训练] |
+-------------+--------------------------------------------+
|  图片列表    |  MultiClassAnnotationCanvas                |
|  (缩略图)    |                                            |
|  [1] done   |  +----------------------------+            |
|  [2] done   |  |   当前图片 + 标注框         |            |
|  [3]        |  |   [画框] [撤销] [清除]      |            |
|  [4]        |  +----------------------------+            |
|  ...        |                                            |
|             |  当前类别: wheel_chock (三角木)              |
|  筛选:      |  [上一张] [跳过] [下一张]                   |
|  [全部]     |  快捷键: A=上一张 D=下一张 S=跳过           |
|  [已标注]   |                                            |
|  [未标注]   |                                            |
+-------------+--------------------------------------------+
```

**关键 State:**

```typescript
interface ManualAnnotationState {
  imageList: string[]                              // 数据集图片文件名
  currentIndex: number                             // 当前图片索引
  annotations: Record<string, AnnotationBox[]>     // 每张图的标注
  annotatedCount: number
  totalCount: number
  classes: Array<{ class_name: string; display_name_zh: string }>
  activeClassIndex: number                         // 多类别时当前类别
}

interface AnnotationBox {
  classIndex: number
  cx: number   // YOLO 归一化 center_x (0~1)
  cy: number   // YOLO 归一化 center_y (0~1)
  w: number    // YOLO 归一化 width (0~1)
  h: number    // YOLO 归一化 height (0~1)
}
```

**AnnotationCanvas 改造:**

新建 `MultiClassAnnotationCanvas.tsx`，保留原组件不变。新组件增加：
- 多类别支持：每个框携带 classIndex，不同类别不同颜色
- 类别标签显示：框上方显示类别名
- 单框删除：右键点击可删除单个框
- YOLO 归一化坐标输出

```typescript
interface MultiClassAnnotationCanvasProps {
  imageUrl: string
  boxes: MultiClassBox[]
  onBoxesChange: (boxes: MultiClassBox[]) => void
  classes: Array<{ name: string; color: string }>
  activeClassIndex: number
  readOnly?: boolean
}
```

**数据持久化:**

标注实时保存到后端，防页面刷新丢失。

```
POST /api/files/{task_id}/seed-annotations
Body: { image_name: "1.PNG", boxes: [{ class_index: 0, cx: 0.45, cy: 0.72, w: 0.08, h: 0.06 }] }

GET /api/files/{task_id}/seed-annotations
Response: { annotations: { "1.PNG": [...], "2.PNG": [...] }, annotated_count: 23, total_count: 200 }
```

后端存储: `backend/uploads/{task_id}/seed_labels/{stem}.txt` (标准 YOLO 格式)

---

### 3.2 种子模型训练 (seed_training)

**训练参数:**

| 参数 | 值 | 理由 |
|------|-----|------|
| model | yolov8n.pt | 最轻量，少数据也能快速收敛 |
| epochs | 50 | 数据少不需要太多 epoch |
| imgsz | 640 | 标准尺寸，平衡速度和精度 |
| patience | 15 | 小数据集容易过拟合，早停 |
| batch | 8 | 少数据用小 batch |

**数据准备:**

```
seed_dataset/
  data.yaml
  train/
    images/ (80% 的手动标注图片)
    labels/
  val/
    images/ (20% 的手动标注图片)
    labels/
```

**Worker 命令:**

```json
{
  "type": "start_seed_training",
  "payload": {
    "task_id": "xxx",
    "seed_label_dir": "../backend/uploads/{task_id}/seed_labels",
    "image_dir": "../backend/uploads/{task_id}/images",
    "classes": ["wheel_chock"],
    "train_config": {
      "model": "yolov8n.pt", "epochs": 50, "imgsz": 640,
      "patience": 15, "lr0": 0.01, "batch": 8, "device": 0
    }
  }
}
```

**新文件 `worker/pipeline/seed_trainer.py`:**

两个核心函数:
- `prepare_seed_dataset()` — 组织 YOLO 训练目录，80/20 分割，生成 data.yaml
- `run_seed_training()` — 复用 `LocalTrainer` 但用轻量参数

**WebSocket 消息:**

```
← { type: "seed_training_progress", currentEpoch, totalEpochs, currentMap }
← { type: "seed_training_complete", seed_model_path, best_map, training_time_seconds }
```

---

### 3.3 种子模型自动打标 (seed_auto_labeler)

**与 YOLO-World 打标的区别:**

| | YOLO-World | 种子模型 |
|---|---|---|
| 模型 | yolov8s-world.pt (零样本) | seed_best.pt (微调后) |
| 类别设置 | model.set_classes(prompts) | 不需要，类别固定在权重中 |
| 置信度 | 通常 0.05-0.15 | 通常 0.3-0.8 |
| 适用场景 | 常见物体 | 任意物体（只要有种子标注） |

**Worker 命令:**

```json
{
  "type": "start_seed_auto_label",
  "payload": {
    "seed_model_path": "/path/to/seed_best.pt",
    "image_dir": "../backend/uploads/{task_id}/images",
    "already_labeled": ["1.PNG", "2.PNG"],
    "conf_threshold": 0.2,
    "auto_accept_threshold": 0.5,
    "imgsz": 640
  }
}
```

**新文件 `worker/pipeline/seed_auto_labeler.py`:**

核心函数 `run_seed_auto_label()`:
1. 加载种子模型
2. 过滤已手动标注的图片
3. 对每张未标注图片推理
4. 按置信度分层: >= 0.5 自动采纳, 0.2~0.5 待审, < 0.2 丢弃
5. 返回统计和分层结果

**置信度分层:**

```
>= 0.5  → 自动采纳: 直接写入 labels/
0.2~0.5 → 低置信待审: 标记 needs_review, 可选人工审核
< 0.2   → 丢弃: 模型不确定, 不纳入
```

---

### 3.4 数据合并

自动打标完成后合并到正式打标目录:
1. 手动标注优先（同一张图如果有手动 + 自动, 用手动的）
2. 自动采纳的标注直接写入
3. 更新 labeledImageCount
4. 进入正常后续流程 (augment)

---

### 3.5 前端状态管理

```typescript
// 新增到 TaskState
snowballMode: boolean               // 是否进入增量打标模式
snowballRound: number               // 当前轮次
seedAnnotatedCount: number          // 已手动标注图片数
seedModelPath: string | null        // 种子模型路径
seedModelMap: number | null         // 种子模型 mAP
seedAutoLabelStats: {               // 种子自动打标统计
  autoAccepted: number
  needsReview: number
  noDetection: number
  avgConfidence: number
} | null
```

---

## 4. API 接口汇总

### 4.1 后端 REST API

```
POST   /api/files/{task_id}/seed-annotations          保存单张图标注
GET    /api/files/{task_id}/seed-annotations          获取所有标注
DELETE /api/files/{task_id}/seed-annotations/{image}  删除单张标注
GET    /api/files/{task_id}/dataset-images?page=1&size=50  数据集图片分页列表
```

### 4.2 Worker WebSocket 命令

```
start_seed_training     → seed_training_progress → seed_training_complete
start_seed_auto_label   → progress (seed_auto_label) → seed_auto_label_complete
merge_seed_labels       → merge_complete
```

---

## 5. 目录结构

```
backend/uploads/{task_id}/
  images/                  原始数据集图片 (不变)
  seed_labels/             手动种子标注 (.txt YOLO 格式)
  seed_dataset/            种子训练目录 (data.yaml + train/val)
  seed_training_output/    种子模型产物 (weights/best.pt)
  auto_labels/
    accepted/              高置信自动采纳
    review/                低置信待审
  labeled_images/          最终合并结果 (现有)
  labels/                  最终合并结果 (现有)
  dataset/                 数据分割后训练集 (现有)
```

---

## 6. 新增/改动文件清单

### 前端新增

| 文件 | 说明 |
|------|------|
| `frontend/src/pages/ManualAnnotation.tsx` | 手动标注页面 |
| `frontend/src/pages/SeedTraining.tsx` | 种子训练进度页 |
| `frontend/src/components/MultiClassAnnotationCanvas.tsx` | 多类别标注画布 |

### 前端改动

| 文件 | 改动 |
|------|------|
| `store/taskStore.ts` | 新增 Stage + snowball 状态字段 |
| `App.tsx` | Stage 路由新增两个页面 |
| `pages/AlgorithmPlan.tsx` | 预览差时显示「进入手动标注」入口 |
| `pages/LabelingProgress.tsx` | 支持种子模型打标模式 |

### Worker 新增

| 文件 | 说明 |
|------|------|
| `worker/pipeline/seed_trainer.py` | 种子数据准备 + 训练 |
| `worker/pipeline/seed_auto_labeler.py` | 种子模型自动打标 |

### Worker 改动

| 文件 | 改动 |
|------|------|
| `worker/main.py` | 新增 3 个命令处理 |

### 后端改动

| 文件 | 改动 |
|------|------|
| `backend/routers/files.py` | 种子标注 CRUD + 图片分页 |

---

## 7. 实现优先级

**P0 — 核心闭环（最小可用）**
- MultiClassAnnotationCanvas 组件
- ManualAnnotation 页面（画框 + 保存）
- seed_trainer.py（准备数据集 + 训练）
- SeedTraining 页面（训练进度展示）
- seed_auto_labeler.py（种子模型自动打标）
- 合并标注到 labeled_images/ + labels/
- 进入正常后续流程

**P1 — 体验优化**
- 标注数据后端持久化（防页面刷新丢失）
- 低置信框人工审核页面
- 种子训练预估时间
- 滚雪球多轮迭代

**P2 — 进阶功能**
- 主动学习: 智能选择最有价值的图片让用户标注
- 半监督学习: 用未标注图片做一致性正则
- 标注导入/导出

---

## 8. 边界条件与错误处理

| 场景 | 处理方式 |
|------|----------|
| 用户只标了 5 张就点训练 | 前端阻止, 提示"最少标注 20 张" |
| 种子训练 mAP < 0.3 | 警告"模型质量偏低", 建议增加标注数量或检查标注质量 |
| 种子模型自动采纳率 < 20% | 建议用户进入下一轮: 审核低置信框 + 重训练 |
| 数据集 > 10000 张 | 种子打标分批推理, 避免内存溢出 |
| 种子训练中途断开 | 训练在 Worker 子进程中, 不受 WS 断连影响; 重连后可查进度 |
| 多类别场景 | 标注页支持类别切换, 每个框都有 classIndex |
| 用户中途放弃回到上一步 | 已保存的种子标注不丢失, 下次可继续 |
| GPU 显存不足 | 种子训练用 yolov8n + batch=8, 显存需求 < 4GB |
| CPU-only 设备 | 种子训练仍可运行, 约 10-30 分钟, 前端预估并提示 |

---

## 9. 增量训练（模型迭代优化）

> 用户训练好模型 v1 并部署后，发现某些场景识别效果不好。
> 此时用户**不应从零开始**，而是追加失败案例数据，基于 v1 微调得到 v2。

### 9.1 增量训练 vs 现有功能对比

| 概念 | 含义 | 现有实现 |
|------|------|----------|
| **resume** (断点续训) | 训练中途断了，从 last.pt 接着跑 | 已有 `resume_last: true` |
| **增量训练** (fine-tune) | 训练完了但效果不好，加新数据基于 best.pt 再训 | **未实现 ←** |
| **滚雪球打标** (本文档) | 零样本打不了，人工标少量再自动标多的 | 本文档设计 |

### 9.2 增量训练流程

```
用户完成训练 → 部署 → 发现 badcase
    │
    ▼
Stage: incremental_upload (增量数据上传)
  上传新的失败案例图片 (10-500 张)
  可选: 在新图上手动标注 / 用已有模型辅助预标注
    │
    ▼
Stage: incremental_merge (数据合并)
  新数据 + 旧训练集合并
  自动去重 (按文件 hash)
  更新 data.yaml
    │
    ▼
Stage: training (增量微调)
  model = YOLO("previous_best.pt")  ← 从上次训练结果开始
  model.train(data="merged_data.yaml", epochs=30)
  比从零训练 (100 epoch) 快 3-5 倍
    │
    ▼
  训练完成 → v2 模型 → 重新部署
```

### 9.3 与从零训练的关键区别

```python
# 从零训练 (首次)
model = YOLO("yolov8s.pt")       # COCO 预训练权重
model.train(data="data.yaml", epochs=100)

# 增量训练 (fine-tune)
model = YOLO("runs/exp/best.pt") # 用户上次训练的 best.pt
model.train(data="merged_data.yaml", epochs=30, lr0=0.005)
                                  # epochs 更少, lr 更小
```

### 9.4 技术实现

**前端入口:**

在 Delivery / 训练完成页面 添加「追加数据 & 增量训练」按钮：
- 用户点击后进入增量上传页
- 可上传新图片 + 可选手动标注
- 确认后合并数据，跳到训练配置页（model 字段锁定为上次 best.pt）

**后端 API:**

```
POST /api/training/{task_id}/incremental
Body: {
  "new_image_dir": "...",
  "new_label_dir": "...",         // 可选: 用户已标注的
  "base_model_path": "best.pt",  // 上次训练产物
  "auto_label_new": true          // 是否用旧模型辅助预标注新图
}
Response: {
  "merged_dataset_dir": "...",
  "total_images": 1500,           // 旧 1200 + 新 300
  "new_images": 300,
  "recommended_epochs": 30,
  "recommended_lr0": 0.005
}
```

**训练参数自动调整:**

| 参数 | 首次训练 | 增量训练 | 理由 |
|------|---------|---------|------|
| model | yolov8s.pt | previous/best.pt | 从已学到的特征继续 |
| epochs | 100 | 30 | 已有基础，不需要太多 |
| lr0 | 0.01 | 0.005 | 学习率减半，避免遗忘 |
| patience | 20 | 10 | 微调收敛快 |

**数据合并策略:**

1. 旧数据 100% 保留
2. 新数据追加
3. 按文件 hash 去重
4. 重新 80/20 分割（新数据均匀分配到 train/val）
5. 如果新数据有手动标注，直接用；没有则用旧模型预标注 + 人工审核

**版本管理:**

```
backend/uploads/{task_id}/
  training_history/
    v1/
      best.pt
      data.yaml
      stats.json       # { images: 1200, classes: 3, map50: 0.82 }
    v2/
      best.pt          # 增量训练产物
      data.yaml
      stats.json       # { images: 1500, classes: 3, map50: 0.89, base: "v1" }
    v3/
      ...
```

每次训练完成自动归档到版本目录，支持：
- 查看训练历史
- 回滚到旧版本
- 对比不同版本的 mAP

### 9.5 增量训练的 UX

**入口 1: Delivery 页面**

训练完成后展示：
```
模型 v1 已就绪 ✓  mAP50: 82%

[下载部署包]  [追加数据，增量训练 →]
```

**入口 2: 独立的「模型迭代」页面**

```
训练历史:
┌──────┬──────────┬─────────┬──────────┬─────────┐
│ 版本 │ 训练图片  │ mAP50   │ 新增数据  │ 状态    │
├──────┼──────────┼─────────┼──────────┼─────────┤
│ v1   │ 1200 张  │ 82%     │ -        │ 已完成  │
│ v2   │ 1500 张  │ 89%     │ +300 张  │ 已完成  │
│ v3   │ 1650 张  │ 91%     │ +150 张  │ 训练中  │
└──────┴──────────┴─────────┴──────────┴─────────┘

[上传新数据，开始 v4 训练]
```

### 9.6 辅助预标注

用户上传新图片后，可选择用旧模型先跑一遍：
- 高置信框直接采纳
- 低置信框让用户修正
- 比纯手动标注效率高 5-10 倍

这里复用 `seed_auto_labeler.py` 的逻辑，只是模型换成上次训练的 best.pt。

---

## 10. 关键设计决策

1. **种子模型用 yolov8n 而非 yolov8s** — 数据少 (50-100 张) 时小模型更不容易过拟合, 训练也快很多
2. **置信度分层而非全部采纳** — 避免低质量标注污染训练集
3. **手动标注优先级 > 自动标注** — 人工标注是 ground truth
4. **复用 LocalTrainer 而非新写训练器** — 减少代码量, 已验证的训练逻辑
5. **种子标注存后端而非纯前端** — 防刷新丢失, 支持多设备访问
6. **不强制多轮** — 大部分场景一轮 (手动 + 种子自动) 就够用, 多轮是可选
7. **增量训练用 best.pt 而非 last.pt** — best.pt 是验证集最优, last.pt 可能过拟合
8. **增量训练自动降 lr** — 微调场景学习率过大会导致灾难性遗忘
9. **版本归档** — 每次训练自动归档, 支持回滚和对比

---

## 11. 完整开发计划与优先级

### Sprint 1: 手动标注核心 (P0, 预计 2 天)

| # | 任务 | 涉及文件 | 预计耗时 |
|---|------|---------|---------|
| 1.1 | 新建 `MultiClassAnnotationCanvas.tsx` 组件 | 前端 components/ | 3h |
| 1.2 | 新建 `ManualAnnotation.tsx` 页面 (图片列表 + 画框 + 键盘快捷键) | 前端 pages/ | 4h |
| 1.3 | 后端种子标注 CRUD API (`/seed-annotations`, `/dataset-images`) | backend/routers/files.py | 2h |
| 1.4 | taskStore 新增 Stage + snowball 状态字段 | store/taskStore.ts | 1h |
| 1.5 | App.tsx 路由 + AlgorithmPlan 入口按钮 | App.tsx, AlgorithmPlan.tsx | 1h |

### Sprint 2: 种子训练 + 自动打标 (P0, 预计 1.5 天)

| # | 任务 | 涉及文件 | 预计耗时 |
|---|------|---------|---------|
| 2.1 | `worker/pipeline/seed_trainer.py` (数据准备 + 训练) | Worker 新文件 | 3h |
| 2.2 | `worker/pipeline/seed_auto_labeler.py` (种子模型推理 + 分层) | Worker 新文件 | 2h |
| 2.3 | `worker/main.py` 新增 3 个 WebSocket 命令 | Worker main | 2h |
| 2.4 | `SeedTraining.tsx` 训练进度页 | 前端 pages/ | 2h |
| 2.5 | 合并逻辑 + 对接正常后续流程 (augment) | Worker + 前端 | 2h |

### Sprint 3: 增量训练 (P0, 预计 1.5 天)

| # | 任务 | 涉及文件 | 预计耗时 |
|---|------|---------|---------|
| 3.1 | 后端增量训练 API (`POST /training/{task_id}/incremental`) | backend/routers/training.py | 2h |
| 3.2 | 数据合并 + 去重 + 重新分割 | Worker 新工具函数 | 2h |
| 3.3 | 训练配置页增量模式 (锁定 model=best.pt, 自动调参) | TrainConfig.tsx | 2h |
| 3.4 | Delivery 页「追加数据 & 增量训练」入口 | Delivery.tsx | 1h |
| 3.5 | 训练版本归档 + 历史展示 | 后端 + 前端 | 3h |

### Sprint 4: 体验优化 (P1, 预计 1 天)

| # | 任务 | 涉及文件 | 预计耗时 |
|---|------|---------|---------|
| 4.1 | 低置信框人工审核页 | 新页面 | 3h |
| 4.2 | 增量训练辅助预标注 (旧模型自动标新图) | 复用 seed_auto_labeler | 1h |
| 4.3 | 训练时间预估 (根据设备档位 + 数据量) | 前端 utils | 1h |
| 4.4 | 版本对比 (v1 vs v2 mAP 差异可视化) | 前端组件 | 2h |

### 总计: ~6 天工作量

```
Sprint 1 (2天)  ████████████████████░░░░░░░░░░  手动标注
Sprint 2 (1.5天) ░░░░░░░░░░░░░░████████████████  种子训练+打标
Sprint 3 (1.5天) ░░░░░░░░░░░░░░░░░░░░░░████████  增量训练
Sprint 4 (1天)   ░░░░░░░░░░░░░░░░░░░░░░░░░░████  体验优化
```

### 开发顺序建议

```
1.1 → 1.2 → 1.3 → 1.4 → 1.5 (手动标注页可独立运行)
    → 2.1 → 2.3 → 2.4 (种子训练能跑)
    → 2.2 → 2.5 (自动打标 + 合并, 打通全流程)
    → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 (增量训练)
    → 4.x (优化)
```
