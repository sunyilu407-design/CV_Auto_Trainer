# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**CV 自动化训练中台** — 面向「一人公司」的零代码 CV 模型训练平台。用户上传手动画框的样板图 + 口语描述，系统自动完成从意图理解 → 海量打标 → 数据增强 → 云端训练 → 模型交付的全流程。

### 核心四阶段

```
阶段一（云端 VLM）：上传样板图 + 描述 → VLM 解析为结构化任务书
阶段二（本地 GPU，两段式）：YOLO-World 画框 → Moondream VQA 质检 → YOLO 标注
阶段二点五（本地）：Albumentations 数据增强（零 API 成本）
阶段三（Web UI）：数据集打包 + 训练配置
阶段四（AutoDL 云端）：云端训练 → 自动关机 → 模型交付
```

---

## 关键约束（必须严格遵守）

### 显存安全（阶段二绝对核心）

**GTX 1650 最低仅 4GB 显存。两段之间必须彻底释放显存，绝不能同时驻留两个模型。**

```
第一段：YOLO-World 推理 → raw_boxes.json → del model → empty_cache → gc.collect()
第二段：Moondream VQA 质检 → YOLO .txt 标注 → del model → empty_cache → gc.collect()
```

### 兜底关机（阶段四最高优先级）

**AutoDL 状态机的 `finally` 块必须覆盖所有退出路径，无论成功、异常、还是被中断。**

### 两段式打标是核心

阶段二的两段之间必须严格执行：`del model` + `torch.cuda.empty_cache()` + `gc.collect()`，绝不能同时加载两个模型。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React + TypeScript + Vite + Zustand |
| 实时通信 | WebSocket（前端 ↔ 本地 Worker） |
| 后端 API | FastAPI（Python） |
| 本地 Worker | Python 独立进程 |
| 目标检测（打标） | Ultralytics YOLO-World（FP16） |
| VQA 质检 | Moondream2 |
| 数据增强 | Albumentations >= 1.4.0 |
| 云端训练 | Ultralytics CLI（YOLO11 / RT-DETR） |
| 云端调度 | AutoDL OpenAPI + SSH |
| 数据库 | SQLite（开发）/ PostgreSQL（生产），SQLAlchemy ORM |

---

## 目录结构

```
project-root/
├── frontend/                 # React 前端
│   ├── src/pages/            # 页面：Upload / IntentConfirm / LabelingProgress /
│   │                         #   AugmentConfig / ReviewSamples / TrainConfig /
│   │                         #   TrainingMonitor / Delivery
│   ├── src/components/       # AnnotationCanvas / AugPreview / MetricsChart / GpuMonitor
│   ├── src/store/            # taskStore.ts（Zustand 全局状态）
│   └── src/api/              # backend.ts / worker.ts
│
├── backend/                   # FastAPI 后端
│   ├── routers/              # vlm.py / tasks.py / autodl.py / files.py / settings.py
│   ├── services/             # vlm_adapter.py / autodl_scheduler.py /
│   │                         # dataset_packer.py / model_exporter.py
│   └── models/               # db.py（SQLAlchemy 模型）
│
├── worker/                    # 本地 Worker（独立进程，监听 localhost:7860）
│   ├── pipeline/             # stage2_labeler.py（两段式打标）/
│   │                         # stage25_augmentor.py / gpu_manager.py
│   └── utils/                # yolo_io.py / dataset_splitter.py
│
└── cloud_scripts/            # 上传至 AutoDL 实例执行的脚本
    ├── train.py              # 统一训练入口（支持断点续训）
    ├── export.py             # 模型导出
    └── health_check.py
```

---

## 启动命令

```bash
# 终端 1：后端 API
cd backend
uvicorn main:app --reload --port 8000

# 终端 2：本地 Worker（独立进程，会加载 GPU 模型）
cd worker
python main.py

# 终端 3：前端开发服务器
cd frontend
pnpm dev
```

Worker 监听 `http://127.0.0.1:7860`，前端通过本地 HTTP/WS 与其通信。

---

## 架构决策记录

### 两段式打标设计

- **第一段**：YOLO-World 负责画框，输出归一化 xywh + class_idx + conf，保存 raw_boxes.json
- **第二段**：Moondream2 VQA 三维度质检（清晰度/完整性/目标一致性），任一维度 < 0.4 丢弃，< 0.5 中性
- 两段之间显存必须清空，GTX 1650 4GB 最低适配

### VLM Adapter

支持 OpenAI-compatible API：openai（GPT-4o）/ kimi（moonshot-v1-8k）/ gemini（gemini-1.5-pro）
用户在前端设置面板配置 provider / base_url / api_key

### AutoDL 状态机

`IDLE → CREATING → UPLOADING → TRAINING → PULLING → SHUTTING_DOWN → DONE/ERROR`
`finally` 块强制关机，最多重试 3 次

---

## 依赖版本约束

| 依赖 | 版本要求 |
|------|----------|
| ultralytics | >= 8.3.0 |
| albumentations | >= 1.4.0（BboxParams clip 参数） |
| transformers | >= 4.36.0（Moondream2） |
| torch | >= 2.0.0（FP16 推理） |
| Python | >= 3.10（类型注解语法） |
