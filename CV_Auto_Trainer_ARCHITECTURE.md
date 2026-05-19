# CV 自动化训练中台 — 项目架构说明书

> **文档定位**：完整的开发交付物，开发者/AI 可直接依据本文档进行实现，无需额外询问细节。
> **版本**：v8.1 | **状态**：可直接开始开发 | **更新**：新增快速预览模式、视频推理演示、训练回调机制、数据集质量报告，数据流与代码完全同步

---

## 目录

1. [项目概述与技术选型](#1-项目概述与技术选型)
2. [整体系统架构](#2-整体系统架构)
3. [目录结构](#3-目录结构)
4. [阶段一：意图解析（云端 VLM）](#4-阶段一意图解析云端-vlm)
5. [阶段二：两段式打标与清洗（本地 GPU）](#5-阶段二两段式打标与清洗本地-gpu)
6. [阶段二点五：离线数据增强（本地 CPU/GPU）](#6-阶段二点五离线数据增强本地-cpugpu)
7. [阶段三：数据集打包与训练配置（Web UI）](#7-阶段三数据集打包与训练配置web-ui)
8. [阶段四：模型训练（本地 GPU / AutoDL 云端）](#8-阶段四模型训练本地-gpu--autodl-云端)
9. [前端页面与状态机](#9-前端页面与状态机)
10. [后端 API 接口规范](#10-后端-api-接口规范)
11. [本地 Worker 进程规范](#11-本地-worker-进程规范)
12. [数据库模型](#12-数据库模型)
13. [错误处理与异常边界](#13-错误处理与异常边界)
14. [配置与可选项](#14-配置与可选项)
15. [技术栈汇总](#15-技术栈汇总)

---

## 1. 项目概述与技术选型

### 1.1 产品定位

面向「一人公司」的**零代码 CV 模型训练平台**。用户上传几张手动画好框的「样板图」+ 口语描述，平台自动完成从「意图理解」→「海量打标」→「数据增强」→「本地/GPU 服务器训练」→「模型交付」的全流程。支持**本地训练**（用户自有 GPU）和**云端训练**（AutoDL）两种模式，用户可在训练配置页面自由切换。

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| 算力极致分离 | 云端大脑（VLM 意图解析）+ 本地小脑（GPU 打标）+ 本地/云端超算（模型训练） |
| 显存安全第一 | 本地 GPU 最低适配 GTX 1650 4GB，所有本地推理严格两段式串行，绝不并发加载模型 |
| 零 API 成本增强 | 数据增强完全在本地用 Albumentations 实现，不依赖任何云端 API |
| 幂等与可恢复 | 所有阶段支持中断续跑，云端训练支持从 checkpoint 恢复 |
| 训练灵活选择 | 支持本地训练（免排队、免上传、适合中大规模数据集）和云端训练（4090/A100 顶级算力），用户按需切换 |
| 兜底关机 | 无论训练成功还是异常，AutoDL 实例必须被关闭，防止扣费；本地训练则确保 GPU 资源正确释放 |
| 无账号体系 | 单人单机器使用，本地存储所有数据，无需登录 |
| 图片清理可控 | 用户可配置任务完成后是否自动删除原图 |

### 1.3 关键约束（必须严格遵守）

> **两段式打标是绝对核心**。阶段二的两段之间必须彻底释放显存（`del model` + `torch.cuda.empty_cache()` + `gc.collect()`），绝不能同时驻留两个模型。

> **兜底关机是最高优先级**。AutoDL 状态机的 `finally` 块必须覆盖所有退出路径，无论成功、异常、还是被中断。

> **本地训练 GPU 隔离**。本地训练时，打标阶段必须完全退出并释放显存（`gpu_stage` 上下文管理器），训练子进程独立管理显存，不与打标共享。

### 1.4 技术选型

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端 | React + TypeScript + Vite | Canvas API 标注框可视化 |
| 前端状态 | Zustand | 轻量，适合单人项目 |
| 实时通信 | WebSocket（前端 ↔ 本地 Worker） | 训练进度推送 |
| 后端 API | FastAPI（Python） | VLM 调用、AutoDL 调度、文件管理 |
| 本地 Worker | Python 独立进程 | 打标、增强，通过 HTTP/WS 与前端通信 |
| 目标检测（打标） | Ultralytics YOLO-World | 本地画框，FP16 半精度 |
| VQA 质检 | Moondream2 | 本地质检，显存占用低 |
| 数据增强 | Albumentations >= 1.4.0 | 纯本地，支持 bbox 同步变换 |
| 云端训练 | Ultralytics CLI（YOLO11 / RT-DETR 统一 API） | 统一 API，降低运维复杂度 |
| 云端调度 | AutoDL OpenAPI + SSH | API + SSH 双通道 |
| 模型导出 | Ultralytics export | ONNX / TensorRT / CoreML / OpenVINO |
| 数据库 | SQLite（开发/单人）/ PostgreSQL（生产） | SQLAlchemy ORM |
| 文件存储 | 本地文件系统 | 训练数据集和模型文件持久化在本地 |

---

## 2. 整体系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                     用户浏览器（前端 React SPA）                  │
│   Upload → IntentConfirm → LabelingProgress → AugmentConfig   │
│        → ReviewSamples → TrainConfig → TrainingMonitor       │
│        → Delivery                                            │
└──────────┬───────────────────────────────┬───────────────────┘
           │ REST API                       │ HTTP / WebSocket
           ▼                               ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│     后端 API 服务      │    │      本地 Worker 进程（用户电脑）    │
│     FastAPI           │    │   Python 独立进程，监听 localhost   │
│                       │    │                                  │
│  · VLM Adapter       │    │  阶段二：YOLO-World → 显存释放 →    │
│  · AutoDL 调度       │    │          Moondream VQA 质检        │
│  · 本地训练调度       │    │  阶段二点五：Albumentations 增强    │
│  · 任务状态管理       │    │  阶段四-本地：独立子进程 Ultralytics  │
│  · 文件管理           │    │  · GPU 显存安全管理（gpu_stage）   │
│  · 设置管理           │    │  · 进度 WebSocket 推送             │
└──────────┬───────────┘    └─────────────────────────────────┘
           │ SSH + AutoDL API
           ▼
┌──────────────────────────────────────────────────────────────┐
│                   阶段四-A：AutoDL 云端 GPU 实例                │
│     4090 / A100 · Ultralytics 训练 · 训练结束自动关机            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                   阶段四-B：本地 GPU 训练（Worker 子进程）         │
│    打标完成后释放 GPU → fork/spawn 训练子进程 → 显存隔离训练       │
└──────────────────────────────────────────────────────────────┘
```

### 2.1 数据流总览

```
用户上传样板图 + 描述
        │
        ▼ 阶段一（云端 VLM）
  JSON 检测任务书 [{class_name, prompt, negative_prompt, color_hint}]
        │
        ▼ 阶段二（本地 GPU，两段式）
  第一段：YOLO-World 画框 → 保存 raw_boxes.json → 释放显存
  第二段：Moondream VQA 质检 → 过滤低质量框 → 保存 YOLO .txt 标注
  输出：images/ + labels/（YOLO .txt 格式）
        │
        ▼ 阶段二点五（本地 CPU/GPU）
  用户设定目标数量 → Albumentations 增强
  输出：images_aug/ + labels_aug/
        │
        ▼ 阶段三（Web UI 配置）
  train/val/test 分层分割（8:1:1）
  生成 data.yaml
  用户配置：模型选型 + 超参数 + 图片清理策略
        │
        ▼ 阶段四（用户选择本地或云端）
  【模式 A：本地训练】
    打标/增强阶段 GPU 显存完全释放后
    → fork 训练子进程（独立显存空间）
    → Ultralytics 本地训练
    → 直接输出 best.pt + 报告
  【模式 B：AutoDL 云端】
    打包 dataset.zip → 上传 → 启动实例 → SSH 训练
    → 拉取 best.pt + 报告 → 关机
        │
        ▼
  交付物：best.pt + ONNX/TensorRT（可选）+ 训练报告
```

---

## 3. 目录结构

```
project-root/
├── frontend/                          # React 前端
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Upload.tsx             # 阶段一：上传样板图 + 描述
│   │   │   ├── IntentConfirm.tsx       # 阶段一：确认 VLM 解析结果
│   │   │   ├── LabelingProgress.tsx    # 阶段二：打标进度监控
│   │   │   ├── AugmentConfig.tsx       # 阶段二点五：增强配置
│   │   │   ├── ReviewSamples.tsx       # 阶段三：抽样查看
│   │   │   ├── TrainConfig.tsx         # 阶段三：模型选型 + 超参数
│   │   │   ├── TrainingMonitor.tsx     # 阶段四：训练进度实时监控
│   │   │   └── Delivery.tsx            # 阶段四：交付物下载
│   │   ├── components/
│   │   │   ├── AnnotationCanvas.tsx    # Canvas 标注框可视化
│   │   │   ├── AugPreview.tsx          # 增强效果实时预览
│   │   │   ├── MetricsChart.tsx        # mAP/loss 实时曲线
│   │   │   ├── GpuMonitor.tsx          # 本地 GPU 显存监控
│   │   │   ├── ModelCacheTab.tsx       # 已训练模型缓存管理
│   │   │   ├── PreAnnotatedToggle.tsx  # 预标注数据跳过打标开关
│   │   │   └── SettingsPanel.tsx        # 全局设置面板
│   │   ├── store/
│   │   │   ├── taskStore.ts            # 任务全局状态（Zustand）
│   │   │   └── settingsStore.ts        # 用户设置状态
│   │   └── api/
│   │       ├── backend.ts              # 后端 API 调用封装
│   │       └── worker.ts               # 本地 Worker API 调用封装
│   └── package.json
│
├── backend/                           # FastAPI 后端
│   ├── main.py                        # 应用入口
│   ├── routers/
│   │   ├── vlm.py                     # VLM 意图解析接口
│   │   ├── tasks.py                   # 任务管理 CRUD
│   │   ├── algorithm.py              # 算法规划/协商/修订接口
│   │   ├── files.py                  # 文件上传/下载/预标注检测接口
│   │   ├── training.py               # 训练启动/状态/取消接口
│   │   ├── models.py                 # 已训练模型缓存管理接口
│   │   └── settings.py               # 用户设置接口
│   ├── services/
│   │   ├── vlm_adapter.py             # VLM 多厂商适配器
│   │   ├── vlm_algorithm_planner.py  # VLM 驱动智能算法规划器
│   │   ├── algorithm_planner.py       # 规则引擎兜底算法规划器
│   │   ├── algorithm_preview_service.py # 事件预览服务
│   │   ├── algorithm_package_service.py # 算法工程包导出服务
│   │   ├── pipeline_compiler.py      # 算法 Pipeline 编译
│   │   ├── model_registry.py          # 预训练模型目录 + 已训练缓存
│   │   ├── video_processor.py         # 视频拆帧/抽帧/离线验证
│   │   ├── cloud_trainer.py          # 云端训练抽象基类
│   │   ├── autodl_trainer.py         # AutoDL API 云端训练器
│   │   ├── generic_ssh_trainer.py    # 通用 SSH 云端训练器
│   │   ├── train_dispatcher.py        # 训练分发器（本地/云端路由）
│   │   ├── multi_model_orchestrator.py # 多模型优先级编排训练器
│   │   ├── training_recommendation_service.py # 训练推荐服务
│   │   ├── task_access.py            # 任务访问权限控制
│   │   ├── settings_manager.py        # 加密设置管理器
│   │   └── alert_manager.py          # 告警管理器
│   └── models/
│       ├── db.py                    # SQLAlchemy ORM 数据库模型
│       └── database.py              # 数据库连接管理
│
├── worker/                            # 本地 Worker（用户电脑运行）
│   ├── main.py                        # FastAPI + WebSocket 服务入口
│   ├── pipeline/
│   │   ├── stage2_labeler.py        # 阶段二：两段式打标（含预标注跳过）
│   │   ├── stage25_augmentor.py     # 阶段二点五：数据增强
│   │   ├── gpu_manager.py             # 显存安全管理（gpu_stage）
│   │   ├── package_exporter.py      # 算法工程包导出
│   │   ├── tracking_runtime.py      # 目标跟踪运行时
│   │   ├── event_engine.py          # 事件引擎
│   │   ├── frame_adapter.py         # 帧适配器
│   │   ├── runtime_session.py        # 运行时会话管理
│   │   └── local_trainer.py         # 本地训练子进程管理
│   └── utils/
│       ├── yolo_io.py               # YOLO .txt 格式读写
│       ├── dataset_splitter.py       # train/val/test 分层分割
│       └── image_files.py           # 图片文件工具

└── cloud_scripts/                     # 上传至云端实例执行的脚本
    ├── train.py                       # 统一训练入口（支持断点续训）
    ├── export.py                      # 模型导出
    └── health_check.py                # 训练状态心跳
```

---

## 4. 阶段一：意图解析（云端 VLM）

### 4.1 用户操作

1. 用户上传 1~3 张「样板图」（已手动画框的参考图，支持 JPG/PNG，单张 <= 10MB）
2. 输入口语化需求文本（如："把戴红帽子的人框出来，戴的叫 helmet，没戴的叫 no_helmet"）
3. 系统调用用户配置的 VLM API 返回结构化任务书
4. 前端展示解析结果，用户可手动微调每个 class 的 prompt 后确认

**重要**：仅用样板图做意图解析，不需要对大量图片调用 VLM API。

### 4.2 VLM 输出数据结构

```json
{
  "task_id": "uuid-xxxx",
  "classes": [
    {
      "class_name": "helmet",
      "prompt": "construction worker wearing a red or yellow hard hat",
      "negative_prompt": "person without hat, bare head",
      "color_hint": "red or yellow"
    },
    {
      "class_name": "no_helmet",
      "prompt": "construction worker without any hard hat, bare head visible",
      "negative_prompt": "person wearing any hat or helmet",
      "color_hint": null
    }
  ],
  "raw_vlm_response": "...",
  "confidence": 0.92
}
```

### 4.3 VLM Adapter 实现

```python
# backend/services/vlm_adapter.py

import json
import re
import httpx
from abc import ABC, abstractmethod
from jsonschema import validate, ValidationError

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
                    "color_hint": {"type": ["string", "null"]}
                }
            }
        }
    }
}

SYSTEM_PROMPT = """
你是一个计算机视觉标注任务专家。用户会提供若干参考图片（已画框）和口语化描述。
你需要输出一个标准 JSON，描述每个需要检测的类别。

输出格式（仅输出 JSON，不要有任何额外文字）：
{
  "classes": [
    {
      "class_name": "英文类别名（小写下划线）",
      "prompt": "详细的英文视觉描述，用于引导目标检测模型定位目标",
      "negative_prompt": "该类别不包含的特征描述",
      "color_hint": "主要颜色特征（可为null）"
    }
  ]
}
"""

class VLMAdapter:
    def __init__(self, provider: str, base_url: str, api_key: str):
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def parse_intent(
        self, images_base64: list[str], user_text: str, max_retry: int = 3
    ) -> dict:
        """调用 VLM 解析用户意图，失败自动重试"""
        last_err = None
        for attempt in range(max_retry):
            try:
                raw = self._call_api(images_base64, user_text)
                result = self._parse_and_validate(raw)
                return result
            except (json.JSONDecodeError, ValidationError, ValueError, httpx.HTTPError) as e:
                last_err = e
                continue
        raise RuntimeError(f"VLM 解析失败，已重试 {max_retry} 次: {last_err}")

    def _call_api(self, images_base64: list[str], user_text: str) -> str:
        """统一调用接口，根据 provider 构造不同请求体"""
        content = [{"type": "text", "text": user_text}]
        for img in images_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })
        payload = {
            "model": self._get_model_name(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ],
            "max_tokens": 1024
        }
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _parse_and_validate(self, raw: str) -> dict:
        """提取 JSON 并做 schema 校验"""
        # 支持 ```json ... ``` 包裹的情况
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
```

### 4.4 支持的 VLM 厂商

| provider | model | API 格式 |
|----------|-------|----------|
| openai | GPT-4o | OpenAI-compatible |
| kimi | moonshot-v1-8k | OpenAI-compatible |
| gemini | gemini-1.5-pro | OpenAI-compatible（via proxy）或 Google 原生 |

用户在前端「设置面板」中配置：provider、base_url、api_key。

---

## 5. 阶段二：两段式打标与清洗（本地 GPU）

### 5.1 核心约束（必须严格遵守）

> **GTX 1650 仅有 4GB 显存。两段之间必须彻底释放显存，绝不能同时驻留两个模型。**

```
第一段：加载 YOLO-World → 推理全量图片 → 保存 raw_boxes.json → del model → empty_cache → gc.collect
第二段：加载 Moondream   → VQA 质检      → 过滤低质量框 → del model → empty_cache → gc.collect
```

### 5.1.5 预标注数据跳过打标

若用户已通过 LabelImg / roLabelImg 等工具预先标注好数据（YOLO .txt 格式，放在图片同目录下），
系统支持跳过两段式打标，直接使用已有标注。

**触发流程：**

1. 用户在「需求协商」页面打开「我已用 LabelImg 标注好数据」开关
2. 后端 `GET /api/files/{task_id}/check-annotations` 自动检测目录下是否包含 `.txt` 标注文件
3. 返回检测结果：已标注图片数、总框数、类别索引列表
4. 打标阶段收到 `use_existing_labels=True` 后，跳过 YOLO-World 和 Moondream，直接从 `.txt` 读取检测框
5. 打标阶段跳过 VQA 质检（质检分数统一设为 1.0），直接保存 YOLO 标注

**API 接口：**

```bash
GET /api/files/{task_id}/check-annotations?subdir=images
# 返回：
{
  "has_annotations": true,
  "total_images": 100,
  "annotated_images": 87,
  "total_boxes": 312,
  "detected_classes": [0, 1, 2],
  "message": "发现 87/100 张图片已标注，共 312 个框，类别索引 [0, 1, 2]"
}
```

**前端状态：**

```typescript
// frontend/src/store/taskStore.ts
skipLabeling: boolean  // 跳过打标标志
setSkipLabeling: (skip: boolean) => void
```

**Worker 端逻辑：**

```python
# worker/pipeline/stage2_labeler.py

def run_detection(
    image_dir: str,
    classes: list[dict],
    output_raw_dir: str,
    # ...
    use_existing_labels: bool = False,  # 新增参数
) -> dict:
    if use_existing_labels:
        # 直接读取 YOLO .txt 标注文件，跳过 YOLO-World 推理
        image_paths = list_image_files(image_dir)
        results_map = {}
        for img_path in image_paths:
            label_path = Path(image_dir) / f"{img_path.stem}.txt"
            if label_path.exists():
                bboxes, labels = load_yolo_labels(str(label_path))
                results_map[str(img_path)] = [
                    {
                        "class_idx": cls_idx,
                        "class_name": classes[cls_idx]["class_name"],
                        "prompt": classes[cls_idx]["prompt"],
                        "bbox_xywhn": bbox,
                        "conf": 1.0,
                        "_source": "existing_label",
                    }
                    for cls_idx, bbox in zip(labels, bboxes)
                ]
            else:
                results_map[str(img_path)] = []
        return results_map

    # 原有 YOLO-World 推理逻辑...
```

**WebSocket 消息：**

Worker 跳过质检阶段时发送 `stage: "quality_check_skipped"`，前端渲染为"标注导入"一步完成。



### 5.2 显存安全管理器

```python
# worker/pipeline/gpu_manager.py

import torch
import gc
import threading
from contextlib import contextmanager

_current_stage = {"name": None, "cancelled": False, "lock": threading.Lock()}

@contextmanager
def gpu_stage(stage_name: str, required_gb: float = 2.0):
    """
    显存安全上下文管理器。
    进入时检查显存是否充足，退出时强制释放。
    支持取消：外部设置 _current_stage['cancelled'] = True 时，
    推理循环检测到标志后主动跳出并释放资源。
    """
    if torch.cuda.is_available():
        free_gb = (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1e9
        if free_gb < required_gb:
            raise MemoryError(
                f"阶段 [{stage_name}] 需要 {required_gb:.1f}GB 显存，"
                f"当前仅剩 {free_gb:.1f}GB"
            )
    with _current_stage["lock"]:
        _current_stage["name"] = stage_name
        _current_stage["cancelled"] = False
    try:
        yield
    finally:
        with _current_stage["lock"]:
            _current_stage["name"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

def is_cancelled() -> bool:
    with _current_stage["lock"]:
        return _current_stage["cancelled"]

def cancel_current_stage():
    """收到 cancel 命令时调用，设置取消标志"""
    with _current_stage["lock"]:
        _current_stage["cancelled"] = True

def check_cancel_and_yield():
    """推理循环中周期性调用：检测到取消标志则抛出 CancelError"""
    if is_cancelled():
        raise CancelError("Stage cancelled by user")
```

### 5.3 第一段：YOLO-World 画框

```python
# worker/pipeline/stage2_labeler.py

import torch
import gc
import json
import os
from pathlib import Path
from ultralytics import YOLOWorld

def run_detection(
    image_dir: str,
    classes: list[dict],       # [{class_name, prompt, ...}]
    output_raw_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 4,
    progress_callback=None
) -> dict:
    """
    第一段：使用 YOLO-World 对全量图片进行目标检测，输出原始框 JSON。
    返回：{image_path: [box_dict]}，其中 box_dict 包含 class_idx, class_name,
         bbox_xywhn (归一化 xywh), conf
    """
    model = None
    try:
        model = YOLOWorld("yolov8s-world.pt")
        model.half()  # FP16 半精度，大幅降低显存占用
        model.set_classes([c["prompt"] for c in classes])

        image_paths = (
            list(Path(image_dir).glob("*.jpg")) +
            list(Path(image_dir).glob("*.png"))
        )
        results_map: dict = {}

        for i in range(0, len(image_paths), batch_size):
            batch = [str(p) for p in image_paths[i:i + batch_size]]
            results = model.predict(
                batch,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            for img_path, result in zip(batch, results):
                boxes = []
                for box in result.boxes:
                    cls_idx = int(box.cls[0])
                    boxes.append({
                        "class_idx": cls_idx,
                        "class_name": classes[cls_idx]["class_name"],
                        "prompt": classes[cls_idx]["prompt"],
                        "bbox_xywhn": box.xywhn[0].tolist(),  # 归一化 xywh
                        "conf": float(box.conf[0])
                    })
                results_map[str(img_path)] = boxes

            if progress_callback:
                progress_callback(
                    min(i + batch_size, len(image_paths)),
                    len(image_paths),
                    "detection"
                )

        # 保存原始框到磁盘（供第二段读取）
        os.makedirs(output_raw_dir, exist_ok=True)
        with open(f"{output_raw_dir}/raw_boxes.json", "w", encoding="utf-8") as f:
            json.dump(results_map, f, ensure_ascii=False)

        return results_map

    finally:
        # 无论是否异常，必须释放显存
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()
```

### 5.4 第二段：Moondream VQA 质检（三维度）

采用**三维度 VQA** 而非单一问题，提升质检准确率：

```python
def run_quality_check(
    raw_boxes_path: str,
    min_confidence: float = 0.5,
    progress_callback=None
) -> dict:
    """
    第二段：使用 Moondream2 对每个裁剪框进行三维度 VQA 质检。
    三维度：清晰度 + 完整性 + 目标一致性
    任一维度 < 0.4 则丢弃该框。
    返回：{image_path: [通过质检的 box]}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import cv2

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True
        )

        with open(raw_boxes_path, encoding="utf-8") as f:
            raw_boxes: dict = json.load(f)

        passed_boxes: dict = {}
        total = sum(len(v) for v in raw_boxes.values())
        processed = 0

        for img_path, boxes in raw_boxes.items():
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            passed = []

            for box in boxes:
                cx, cy, bw, bh = box["bbox_xywhn"]
                x1 = max(0, int((cx - bw / 2) * w))
                y1 = max(0, int((cy - bh / 2) * h))
                x2 = min(w, int((cx + bw / 2) * w))
                y2 = min(h, int((cy + bh / 2) * h))

                # 过小的框直接丢弃
                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    processed += 1
                    continue

                crop = img[y1:y2, x1:x2]
                prompt_text = box["prompt"]

                # 三维度打分
                scores = _multi_dim_vqa(model, tokenizer, crop, prompt_text)
                avg_score = sum(scores) / len(scores)

                if avg_score >= min_confidence and all(s >= 0.4 for s in scores):
                    box["qa_score"] = avg_score
                    box["qa_dimensions"] = {
                        "clarity": scores[0],
                        "completeness": scores[1],
                        "match": scores[2]
                    }
                    passed.append(box)

                processed += 1
                if progress_callback:
                    progress_callback(processed, total, "quality_check")

            passed_boxes[img_path] = passed

        return passed_boxes

    finally:
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()


def _multi_dim_vqa(model, tokenizer, crop, prompt_text: str) -> list[float]:
    """
    三维度 VQA：清晰度、完整性、目标一致性。
    每个维度返回 0.0~1.0 的分数。
    """
    import re

    questions = [
        (
            "Is this image region clear and in focus, not blurry or severely distorted? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is perfectly clear.",
            "clarity"
        ),
        (
            "Does this cropped image show a complete or mostly complete object "
            "(not severely cropped or truncated)? "
            "Answer with a number from 0.0 to 1.0, where 1.0 is complete.",
            "completeness"
        ),
        (
            f"Does this image clearly show: {prompt_text}? "
            f"Answer with a number from 0.0 to 1.0, where 1.0 is a clear match.",
            "match"
        ),
    ]

    scores = []
    enc_img = model.encode_image(crop)

    for question, _ in questions:
        answer = model.answer_question(enc_img, question, tokenizer)
        score = _parse_confidence(answer)
        scores.append(score)

    return scores


def _parse_confidence(answer: str) -> float:
    """
    从 Moondream 回答中提取置信度分数，支持多种格式。
    支持：0.85 / 0.85 Yes / 0.85, yes / Yes 0.85 / 85%
    无法解析时返回 0.5（中性分数，不丢弃也不保留边缘样本）。
    """
    import re

    answer_clean = answer.strip()

    # 匹配 "0.85" 或 "1.0" 格式（最常见）
    match = re.search(r"(0(?:\.\d+|\.0)|1(?:\.0)?)", answer_clean)
    if match:
        return float(match.group(1))

    # 匹配 "85%" 格式
    match = re.search(r"(\d{1,3})%", answer_clean)
    if match:
        return float(match.group(1)) / 100.0

    # 降级：Yes / No
    lower = answer_clean.lower()
    if lower.startswith("yes"):
        num_match = re.search(r"[\d.]+", answer_clean[len("yes"):])
        if num_match:
            return float(num_match.group())
        return 0.8
    if lower.startswith("no"):
        num_match = re.search(r"[\d.]+", answer_clean[len("no"):])
        if num_match:
            return float(num_match.group())
        return 0.1

    return 0.5  # 无法解析时返回中性分数
```

### 5.5 清洗结果转 YOLO 标注格式

```python
# worker/utils/yolo_io.py

import os
import shutil
from pathlib import Path


def save_yolo_labels(
    passed_boxes: dict,
    output_label_dir: str,
    output_image_dir: str
):
    """
    将质检通过的框保存为 YOLO .txt 格式。
    每行格式：class_idx cx cy w h
    无有效框的图片不写入数据集。
    """
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    for img_path, boxes in passed_boxes.items():
        if not boxes:
            continue
        stem = Path(img_path).stem
        label_path = f"{output_label_dir}/{stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            for box in boxes:
                cx, cy, w, h = box["bbox_xywhn"]
                f.write(
                    f"{box['class_idx']} {cx:.6f} {cy:.6f} "
                    f"{w:.6f} {h:.6f}\n"
                )
        shutil.copy(img_path, f"{output_image_dir}/{Path(img_path).name}")
```

---

## 6. 阶段二点五：离线数据增强（本地 CPU/GPU）

### 6.1 模块定位

在阶段二清洗完成后、打包上传前执行。用户指定「目标图片总数」，系统从清洗后的干净数据自动扩充。**完全基于 Albumentations，零 Token 消耗，无需任何 API**。

### 6.2 UI 配置项

| 控件 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 目标总图片数 | 数字输入 | 当前数量 × 5 | 系统将清洗后数据扩充至此数量 |
| 增强强度 | 单选（轻/中/重） | 中 | 轻=几何+色彩，中+噪声模糊，重+天气遮挡 |
| 几何变换 | 多选勾选 | 全选 | 翻转/旋转/缩放/透视/平移 |
| 色彩扰动 | 多选勾选 | 全选 | 亮度/饱和度/Gamma/CLAHE |
| 噪声与模糊 | 多选勾选 | 全选 | 高斯噪声/运动模糊/JPEG压缩 |
| 天气模拟 | 多选勾选 | 不选 | 雨/雾/阳光光斑 |
| 遮挡模拟 | 多选勾选 | 不选 | Cutout/Mosaic |
| 预览按钮 | 按钮 | — | 随机抽取 3 张原图，实时展示增强效果 |
| 完成后删除原图 | 开关 | 关闭 | 任务完成后自动删除原始上传图片 |

### 6.3 增强 Pipeline 实现

```python
# worker/pipeline/stage25_augmentor.py

import albumentations as A
import cv2
import math
import os
import shutil
import random
from pathlib import Path
from typing import Optional


def build_pipeline(
    strength: str = "medium",
    enabled: Optional[dict] = None
) -> A.Compose:
    """
    构建增强 pipeline。
    enabled 示例：{
        "geometric": True, "color": True, "noise": True,
        "weather": False, "occlusion": False
    }
    """
    if enabled is None:
        enabled = {k: True for k in ["geometric", "color", "noise", "weather", "occlusion"]}

    transforms = []

    # A. 几何变换（自动同步 bbox 坐标）
    if enabled.get("geometric", True):
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.3,
                rotate_limit=15 if strength != "light" else 5,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ]

    # B. 色彩与亮度扰动
    if enabled.get("color", True):
        transforms += [
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30,
                val_shift_limit=20, p=0.4
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ]
        if strength in ("medium", "heavy"):
            transforms.append(A.CLAHE(clip_limit=4.0, p=0.2))

    # C. 噪声与模糊
    if enabled.get("noise", True) and strength != "light":
        transforms += [
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),
        ]

    # D. 天气模拟（户外场景）
    if enabled.get("weather", False) and strength == "heavy":
        transforms += [
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        ]

    # E. 遮挡模拟
    if enabled.get("occlusion", False) and strength == "heavy":
        transforms.append(
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                fill_value=0, p=0.3
            )
        )

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["labels"],
            min_visibility=0.3,   # 被裁剪超过 70% 的框自动丢弃
            clip=True             # 超出边界的框自动裁剪至边界
        )
    )


def augment_dataset(
    src_image_dir: str,
    src_label_dir: str,
    output_image_dir: str,
    output_label_dir: str,
    target_count: int,
    strength: str = "medium",
    enabled: Optional[dict] = None,
    progress_callback=None
) -> dict:
    """
    将数据集从 N 张扩充至 target_count 张。
    同时保留原始数据（原图直接复制到 output 目录）。
    返回：{existing: int, generated: int, total: int}
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    src_images = (
        list(Path(src_image_dir).glob("*.jpg")) +
        list(Path(src_image_dir).glob("*.png"))
    )
    if not src_images:
        raise ValueError(f"源目录无图片: {src_image_dir}")

    pipeline = build_pipeline(strength, enabled)

    # 1. 先复制原始数据
    for img_path in src_images:
        label_path = Path(src_label_dir) / f"{img_path.stem}.txt"
        shutil.copy(img_path, output_image_dir)
        if label_path.exists():
            shutil.copy(label_path, output_label_dir)

    # 2. 计算每张图需要增强的次数
    existing = len(src_images)
    needed = max(0, target_count - existing)
    per_image = math.ceil(needed / existing) if needed > 0 else 0
    generated = 0

    for img_path in src_images:
        if generated >= needed:
            break

        label_path = Path(src_label_dir) / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        bboxes, labels = _load_yolo_label(label_path)

        if img is None:
            continue

        for aug_idx in range(per_image):
            if generated >= needed:
                break
            try:
                result = pipeline(image=img, bboxes=bboxes, labels=labels)
                if not result["bboxes"]:
                    continue  # 增强后无有效框，跳过

                out_stem = f"{img_path.stem}_aug{generated:05d}"
                cv2.imwrite(
                    f"{output_image_dir}/{out_stem}.jpg",
                    result["image"]
                )
                _save_yolo_label(
                    f"{output_label_dir}/{out_stem}.txt",
                    result["bboxes"],
                    result["labels"]
                )
                generated += 1

                if progress_callback:
                    progress_callback(generated, needed, "augmentation")

            except Exception:
                # 单张增强失败不中断整体流程
                continue

    return {"existing": existing, "generated": generated, "total": existing + generated}


def _load_yolo_label(label_path: Path) -> tuple[list, list]:
    bboxes, labels = [], []
    if not label_path.exists():
        return bboxes, labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:]])
    return bboxes, labels


def _save_yolo_label(path: str, bboxes: list, labels: list):
    with open(path, "w", encoding="utf-8") as f:
        for label, bbox in zip(labels, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
```

---

## 7. 阶段三：数据集打包与训练配置（Web UI）

### 7.1 数据集分层分割

```python
# worker/utils/dataset_splitter.py

import random
import shutil
import os
from pathlib import Path
from collections import defaultdict, Counter


def split_dataset(
    image_dir: str,
    label_dir: str,
    output_root: str,
    ratios: tuple = (0.8, 0.1, 0.1),   # train / val / test
    seed: int = 42
) -> dict:
    """
    分层抽样分割：确保每个类别在 train/val/test 中的比例一致。
    输出目录结构：
        output_root/
            images/train/  images/val/  images/test/
            labels/train/  labels/val/  labels/test/
    返回：{split: count} 统计
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "比例之和必须为 1"
    random.seed(seed)

    class_groups: dict = defaultdict(list)
    for lbl_path in Path(label_dir).glob("*.txt"):
        img_path = Path(image_dir) / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            img_path = Path(image_dir) / f"{lbl_path.stem}.png"
        if not img_path.exists():
            continue
        dominant_class = _get_dominant_class(lbl_path)
        class_groups[dominant_class].append((img_path, lbl_path))

    splits: dict = {"train": [], "val": [], "test": []}

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    for split, items in splits.items():
        img_out = Path(output_root) / "images" / split
        lbl_out = Path(output_root) / "labels" / split
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for img_path, lbl_path in items:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(lbl_path, lbl_out / lbl_path.name)

    return {k: len(v) for k, v in splits.items()}


def _get_dominant_class(label_path: Path) -> int:
    classes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(int(parts[0]))
    if not classes:
        return -1
    return Counter(classes).most_common(1)[0][0]
```

### 7.2 data.yaml 自动生成

```python
# backend/services/dataset_packer.py

import yaml
import json
import zipfile
import os
from pathlib import Path


def generate_data_yaml(
    dataset_root: str,
    class_names: list[str],
    output_path: str
) -> str:
    """生成 YOLO 标准 data.yaml"""
    config = {
        "path": dataset_root,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    return output_path


def pack_dataset_zip(
    dataset_root: str,
    output_zip_path: str,
    train_config: dict
) -> str:
    """
    打包数据集 + 训练配置为 zip 文件。
    train_config 包含：model_name, epochs, imgsz, lr0, patience, conf, iou,
                       export_formats
    """
    config_path = f"{dataset_root}/train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in Path(dataset_root).rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(dataset_root))
    return output_zip_path
```

### 7.3 模型选型与超参数配置

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| model | yolo11s.pt | 见下表 | 基础模型 |
| epochs | 100 | 50~300 | 训练轮次 |
| imgsz | 640 | 416/512/640/1280 | 输入图像尺寸 |
| lr0 | 0.01 | 0.001~0.1 | 初始学习率 |
| batch | -1（自动） | 4~64 | -1 = 自动根据显存选择 |
| patience | 20 | 10~50 | Early stopping 轮次 |
| conf | 0.25 | 0.1~0.9 | 推理置信度阈值 |
| iou | 0.7 | 0.3~0.9 | NMS IoU 阈值 |

**模型选型对照表：**

| 模型文件 | 适用场景 | 参数量 | 推理速度（RTX 4090） |
|----------|----------|--------|---------------------|
| yolo11n.pt | 手机/嵌入式极限压缩 | 2.6M | ~2ms |
| yolo11s.pt | 边缘盒子/树莓派（默认推荐） | 9.4M | ~3ms |
| yolo11m.pt | 工控机/普通服务器 | 20.1M | ~5ms |
| yolo11l.pt | 高端 GPU 服务器，精度优先 | 25.3M | ~7ms |
| rtdetr-l.pt | 密集/遮挡严重场景 | 32.9M | ~9ms |

### 7.4 数据质量评估（自动生成）

在阶段三页面展示数据集统计，帮助用户判断数据是否足够：

```typescript
// 前端展示的数据质量报告
interface DataQualityReport {
  totalImages: number
  classDistribution: {
    className: string
    boxCount: number
    avgBoxesPerImage: number
  }[]
  avgBoxesPerImage: number
  warnings: string[]  // 警告信息（如某类别 < 50 bbox）
}
```

**警告规则**：
- 某类别 bbox 总数 < 50 → 建议补充更多样板图
- 某类别 bbox 总数 < 200 → 建议增加增强强度
- 某类别 bbox 总数 > 5000 → 建议减少该类别数据（类别不平衡）

---

## 7.5 增量训练流程（Snowball / 迭代优化）

> 适用场景：模型已上线部署一段时间后，发现某些场景（badcase）识别效果差，需要补充新数据继续微调而非从头训练。

### 7.5.1 触发路径

```
Delivery 页面
  └─ "追加数据 & 增量训练" 按钮
       └─ 设置 incrementalMode = true → 跳转 Upload 页面
            └─ 上传 badcase 图片 → 调用 POST /training/{task_id}/incremental
                 └─ 后端合并新旧数据 + 自动标注 → 返回合并统计
                      └─ 前端显示确认对话框（用户确认）
                           └─ 写入 trainConfig → 跳转 TrainConfig 页面
                                └─ 开始增量训练（使用已有 best.pt 作为预训练权重）
```

### 7.5.2 关键 API

**POST /training/{task_id}/incremental**

请求体：
```json
{
  "auto_label_new": true,          // 用已有模型对新图片自动预标注
  "class_names": ["worker", "forklift"]
}
```

返回体：
```json
{
  "merged_dataset_dir": "uploads/{task_id}/incremental_dataset",
  "data_yaml": "uploads/{task_id}/incremental_dataset/data.yaml",
  "total_images": 120,
  "old_images": 80,              // 来自首次训练
  "new_images": 40,             // 新增 badcase
  "duplicates_removed": 5,       // 与旧数据重复的图片
  "auto_labeled": 40,            // 新图片已自动标注
  "train_count": 96,
  "val_count": 24,
  "base_model_path": "uploads/{task_id}/training_history/v1/best.pt",
  "current_version": 1,
  "next_version": 2,
  "recommended_config": {
    "model": "uploads/{task_id}/training_history/v1/best.pt",
    "epochs": 30,               // 自动降低以保护已有精度
    "lr0": 0.005,              // 自动降低学习率
    "patience": 10,
    "imgsz": 640
  }
}
```

### 7.5.3 数据合并逻辑（worker/utils/incremental_merger.py）

1. **读取旧数据**：从 `training_history/v{latest}/` 获取已有数据集
2. **去重**：用感知哈希（pHash）对新旧图片去重，避免相同图片重复训练
3. **自动标注**：若 `auto_label_new=true`，用已有 `best.pt` 对新图片做推理，生成 YOLO 格式标签文件
4. **合并数据集**：新旧图片 + 标签 → `incremental_dataset/train/` 和 `val/`
5. **生成 data.yaml**：指向合并后的 train/val 路径，写入 `incremental_dataset/data.yaml`
6. **归档**：将本次 `best.pt` + `data.yaml` + `stats.json` 存入 `training_history/v{N}/`

### 7.5.4 训练时的路径解析

```
TrainConfig 页面（incrementalMode=true）
  ├─ model = base_model_path（已有 best.pt，用于微调）
  ├─ incrementalDatasetDir = merged_dataset_dir（数据集目录）
  ├─ incrementalDataYaml = data_yaml（配置文件路径）
  │
  └─ Worker（local_trainer.py）
       ├─ data_yaml = incrementalDataYaml（跳过自动生成，使用合并后的）
       └─ model.train(..., data=yaml_path, pretrained=model)
```

### 7.5.5 与首次训练的对比

| 维度 | 首次训练 | 增量训练 |
|------|---------|---------|
| VLM 意图解析 | ✓ 需要 | ✗ 跳过 |
| 数据来源 | 用户上传 | 旧数据 + 新 badcase |
| 模型权重 | 预训练模型（如 yolo11s.pt） | 已有 best.pt |
| 学习率 | 默认 0.01 | 自动降至 0.005 |
| Epochs | 用户选择（默认 100） | 自动降至 30 |
| 标注方式 | YOLO-World 半自动 + 用户修正 | 自动预标注 + 用户可选修正 |

---

## 7.6 质量门控（Quality Gate）与模型优化路径

### 7.6.1 质量门控机制

在 Delivery 页面，系统根据两个维度综合判定模型是否达标：

| 维度 | 数据来源 | 达标阈值 | 说明 |
|------|----------|---------|------|
| 视频场景匹配置信度 | `algorithmPlan.offline_evaluation.validation_passed` + `confidence` | ≥ 75% | 离线视频预识别结果 |
| 训练评分 | `TrainingReport.score` | ≥ 65 分 | AI 解读训练报告的评分 |

**达标判定**：`validation_passed === true` AND `score ≥ 65` → 显示绿色"模型质量达标"

**未达标**时，触发优化行动面板（Delivery 页面 A/B/C/D 四选项）。

### 7.6.2 优化路径矩阵

| 选项 | 触发条件 | 路径 | 数据影响 |
|------|---------|------|---------|
| **A · 追加 badcase（推荐）** | 场景匹配低 / 训练评分低 | Delivery → Upload → startIncremental → TrainConfig → Training | 新增 badcase 自动预标注，合并旧数据 |
| **B · 调整增强策略** | 小目标 / 遮挡场景召回差 | Delivery → AlgorithmPlan → 调整 augment_config → TrainConfig → Training | 保持当前数据集，改增强参数 |
| **C · 扩充检测类别** | 漏检是因为类别定义不全 | Delivery → IntentConfirm（扩充模式）→ AlgorithmPlan → TrainConfig → Training | 复用已有数据集，新增类别 |
| **D · 接受并交付** | 用户决定接受当前效果 | 留在 Delivery，导出模型 | 不改动 |

### 7.6.3 扩充类别模式（IntentConfirm 扩充模式）

从 Delivery 选项 C 进入 IntentConfirm 时，系统检测到 `userDescription === ''` 且 `vlmResult` 已存在，自动进入扩充模式：

- 左侧对话面板预填充已有类别
- 右侧配置预览保留已有类别
- 底部"补充遗漏的类别"按钮可追加新类别
- 确认后回到 AlgorithmPlan，不重新走 Upload 阶段

### 7.6.4 触发增量训练的其他原因

除 Badcase 积累外，触发增量训练的原因还包括：

| 原因 | 表现 | 处理方式 |
|------|------|---------|
| 场景漂移（Domain Shift） | 换季 / 换摄像头 / 环境变化 | A · 追加新场景数据 |
| 数据增强不够 | 小目标 / 遮挡场景召回差 | B · 调增强策略 |
| 误检率高 | CLIP 词汇太泛化 | C · 细化类别定义 |
| 新增检测类别 | 业务扩展 | C · 扩充类别 |

---

## 8. 阶段四：模型训练（本地 GPU / 云端服务器）

> **训练模式选择**：用户在 TrainConfig 页面选择「本地训练」或「云端训练」。云端训练支持任意可通过 SSH 连接的 GPU 服务器（AutoDL / 阿里云 / 腾讯云 / AWS / 学员自有服务器等）。两套路径共享同一套训练配置（model / epochs / imgsz / lr0 / patience 等），最终交付物格式完全一致。

### 8.1 模式选择与分发

```python
# backend/services/train_dispatcher.py

from enum import Enum

class TrainMode(Enum):
    LOCAL = "local"    # 本地 GPU 训练
    CLOUD = "cloud"    # 云端 SSH 服务器训练

def dispatch_training(
    mode: TrainMode,
    dataset_dir: str,
    train_config: dict,
    cloud_config: dict,          # 云端连接配置（见 8.3）
    progress_callback=None
) -> dict:
    """
    统一训练分发器。根据 mode 选择本地或云端路径。
    返回：{best_pt_path, last_pt_path, metrics, export_paths, mode}
    """
    if mode == TrainMode.LOCAL:
        from .local_trainer import LocalTrainer
        trainer = LocalTrainer()
        return trainer.train(dataset_dir, train_config, progress_callback)
    else:
        from .cloud_trainer import CloudTrainer
        provider_type = cloud_config.get("provider", "generic")  # "autodl" | "generic"
        trainer = CloudTrainer.for_provider(provider_type, cloud_config)
        return trainer.train(dataset_dir, train_config, progress_callback)
```

### 8.2 本地训练（Worker 子进程）

#### 8.2.1 设计原则

- 打标/增强阶段完成后，Worker 必须彻底释放 GPU 显存（`gpu_stage` 保证）
- 本地训练通过 `subprocess.Popen` 启动**独立子进程**，不与 Worker 主进程共享 GPU 上下文
- 子进程训练期间，Worker 主进程可继续响应前端 WebSocket 心跳（但不执行 GPU 操作）
- 训练子进程的进度通过**临时文件轮询**（results.csv）→ Worker 主进程 → WebSocket 推送给前端
- 取消时：Worker 主进程向子进程发 `SIGTERM`（Linux）/ `CTRL_BREAK_EVENT`（Windows），子进程负责优雅停止

#### 8.2.2 本地训练器实现

```python
# worker/pipeline/local_trainer.py

import os
import time
import signal
import subprocess
import threading
import shutil
from pathlib import Path
from typing import Optional, Callable


class LocalTrainer:
    """
    本地 GPU 训练器。
    通过子进程运行 Ultralytics 训练，显存与 Worker 主进程完全隔离。
    """

    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._stop_flag = False
        self._output_dir = None

    def train(
        self,
        dataset_dir: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        启动本地训练子进程，轮询 results.csv 推送进度。
        返回：{best_pt_path, last_pt_path, metrics}
        """
        # 生成输出目录
        self._output_dir = (
            Path(dataset_dir).parent / "local_training_output" / "exp"
        )
        os.makedirs(self._output_dir, exist_ok=True)

        # 构造 data.yaml 路径
        data_yaml = Path(dataset_dir) / "data.yaml"

        # 构建训练命令
        cmd = self._build_command(train_config, data_yaml)

        # 启动子进程（不继承 Worker 主进程的 GPU 状态）
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False  # Windows 兼容
        )

        # 启动进度轮询线程
        poller = threading.Thread(
            target=self._poll_progress,
            args=(progress_callback, train_config),
            daemon=True
        )
        poller.start()

        # 等待子进程结束
        returncode = self._process.wait()

        if returncode != 0 and not self._stop_flag:
            stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"本地训练子进程异常退出（code {returncode}）: {stderr}")

        return self._collect_artifacts()

    def _build_command(self, cfg: dict, data_yaml: Path) -> list[str]:
        model = cfg.get("model", "yolov8s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = str(Path(self._output_dir).parent)
        resume_str = (
            [f"--resume={self._output_dir / 'weights' / 'last.pt'}"]
            if cfg.get("resume_last", False) else []
        )

        cmd = [
            "python", "-c",
            "from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='{data_yaml}', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device=0)"
        ]
        return cmd

    def _poll_progress(
        self,
        progress_callback: Optional[Callable],
        cfg: dict
    ):
        """每 10 秒轮询 results.csv，推送训练进度"""
        results_csv = self._output_dir / "results.csv"
        while True:
            time.sleep(10)
            if self._stop_flag:
                break
            if not results_csv.exists():
                continue
            try:
                with open(results_csv, "r") as f:
                    lines = f.readlines()
                if not lines:
                    continue
                last_line = lines[-1].strip()
                parts = last_line.split(",")
                if len(parts) > 3:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
                    if progress_callback:
                        progress_callback({
                            "current_epoch": current_epoch,
                            "total_epochs": cfg.get("epochs", 100),
                            "current_map": current_map,
                            "done": current_epoch >= cfg.get("epochs", 100)
                        })
            except (ValueError, IndexError, FileNotFoundError):
                continue

    def _collect_artifacts(self) -> dict:
        """收集训练产物"""
        weights_dir = self._output_dir / "weights"
        artifacts = {}
        for fname in ["best.pt", "last.pt"]:
            fpath = weights_dir / fname
            if fpath.exists():
                artifacts[fname] = str(fpath)
        results_csv = self._output_dir / "results.csv"
        if results_csv.exists():
            artifacts["results.csv"] = str(results_csv)
        return artifacts

    def cancel(self):
        """
        取消训练：发 SIGTERM/CTRL_BREAK_EVENT 优雅终止，
        等待子进程退出（最多 30s），超时强制 kill。
        """
        self._stop_flag = True
        if self._process is None:
            return
        try:
            if os.name == "nt":  # Windows
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:  # Linux/macOS
                self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


### 8.3 云端训练

#### 8.3.1 抽象接口

```python
# backend/services/cloud_trainer.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Callable


class CloudTrainState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    UPLOADING = "uploading"
    TRAINING = "training"
    PULLING = "pulling"
    SHUTTING_DOWN = "shutting_down"
    DONE = "done"
    ERROR = "error"


class CloudTrainer(ABC):
    """
    云端训练抽象基类。
    支持任意可通过 SSH 连接 GPU 服务器（AutoDL / 阿里云 / 腾讯云 / AWS / 自有服务器）。
    """

    @staticmethod
    def for_provider(provider: str, config: dict) -> "CloudTrainer":
        """工厂方法：根据 provider 类型返回对应实现"""
        if provider == "autodl":
            from .autodl_trainer import AutoDLCloudTrainer
            return AutoDLCloudTrainer(config)
        else:  # generic
            from .generic_ssh_trainer import GenericSSHCloudTrainer
            return GenericSSHCloudTrainer(config)

    @abstractmethod
    def connect(self) -> None:
        """建立 SSH 连接"""
        pass

    @abstractmethod
    def upload_dataset(self, zip_path: str) -> None:
        """上传数据集 zip 到远程服务器"""
        pass

    @abstractmethod
    def run_training(
        self,
        train_config: dict,
        progress_callback: Optional[Callable] = None
    ) -> None:
        """在远程服务器上执行训练命令"""
        pass

    @abstractmethod
    def pull_artifacts(self, train_config: dict) -> dict:
        """从远程服务器拉取训练产物"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """关闭远程服务器（关机/退订实例）"""
        pass

    def train(
        self,
        dataset_dir: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        完整云端训练流水线。finally 块保证无论如何都执行关机。
        返回：{best_pt_path, last_pt_path, metrics, export_paths}
        """
        try:
            self.connect()

            # 打包数据集
            zip_path = self._pack_dataset(dataset_dir)
            self.upload_dataset(zip_path)

            self.run_training(train_config, progress_callback)
            artifacts = self.pull_artifacts(train_config)

            return artifacts

        except Exception as e:
            raise

        finally:
            self.shutdown()

    def _pack_dataset(self, dataset_dir: str) -> str:
        """打包数据集为 zip"""
        import zipfile
        from pathlib import Path
        zip_path = Path(dataset_dir).parent / "dataset.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for fp in Path(dataset_dir).rglob("*"):
                if fp.is_file():
                    zf.write(fp, fp.relative_to(dataset_dir))
        return str(zip_path)
```

#### 8.3.2 通用 SSH 云端训练（适用于任意云服务器）

```python
# backend/services/generic_ssh_trainer.py

import time
import os
import paramiko
from pathlib import Path
from typing import Optional, Callable
from .cloud_trainer import CloudTrainer, CloudTrainState


class GenericSSHCloudTrainer(CloudTrainer):
    """
    通用 SSH 云端训练器。
    适用于：阿里云、腾讯云、AWS、Google Cloud、AutoDL（SSH 直连）、
            学员自有 GPU 服务器、实验室服务器等任意提供 SSH 访问的机器。

    用户需提供：host / port / username / password（或 private_key_path）
    """

    def __init__(self, config: dict):
        self.host = config["ssh_host"]
        self.port = config.get("ssh_port", 22)
        self.username = config["ssh_username"]
        self.password = config.get("ssh_password")
        self.private_key_path = config.get("ssh_private_key_path")
        self.remote_work_dir = config.get("remote_work_dir", "/root/workspace")
        self.gpu_device = config.get("gpu_device", "0")
        self.state = CloudTrainState.IDLE
        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._alert_manager = None  # 注入告警管理器

    def connect(self):
        self.state = CloudTrainState.CONNECTING
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if self.private_key_path:
            pkey = paramiko.RSAKey.from_private_key_file(self.private_key_path)
            self._ssh.connect(
                self.host, port=self.port,
                username=self.username, pkey=pkey, timeout=30
            )
        else:
            self._ssh.connect(
                self.host, port=self.port,
                username=self.username, password=self.password, timeout=30
            )
        self._sftp = self._ssh.open_sftp()

    def upload_dataset(self, zip_path: str):
        self.state = CloudTrainState.UPLOADING
        remote_zip = f"{self.remote_work_dir}/dataset.zip"
        self._sftp.put(zip_path, remote_zip)
        self._ssh.exec_command(
            f"cd {self.remote_work_dir} && unzip -q dataset.zip -d dataset"
        )
        time.sleep(5)

    def run_training(
        self,
        train_config: dict,
        progress_callback: Optional[Callable] = None
    ):
        self.state = CloudTrainState.TRAINING
        train_cmd = self._build_train_command(train_config)
        # 后台运行训练，不阻塞
        self._ssh.exec_command(
            f"cd {self.remote_work_dir} && screen -dmS train bash -c '{train_cmd}'"
        )

        while True:
            time.sleep(30)
            status = self._check_training_status(train_config)
            if progress_callback:
                progress_callback(status)
            if status["done"]:
                break
            if status["error"]:
                raise RuntimeError(f"训练失败: {status['error_msg']}")

    def _build_train_command(self, cfg: dict) -> str:
        model = cfg.get("model", "yolo11s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = f"{self.remote_work_dir}/training_output"
        resume_str = (
            f", resume='{project}/exp/weights/last.pt'"
            if cfg.get("resume_last", False) else ""
        )
        return (
            f"python -c \"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f\"model.train(data='{self.remote_work_dir}/dataset/data.yaml', \"
            f\"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, \"
            f\"patience={patience}, project='{project}', \"
            f\"name='exp', exist_ok=True, device={self.gpu_device}{resume_str})\"
        )

    def _check_training_status(self, cfg: dict) -> dict:
        _, stdout, _ = self._ssh.exec_command(
            f"tail -1 {self.remote_work_dir}/training_output/exp/results.csv 2>/dev/null"
        )
        line = stdout.read().decode().strip()

        _, stdout2, _ = self._ssh.exec_command(
            f"[ -f {self.remote_work_dir}/training_output/exp/weights/best.pt ] && echo done"
        )
        is_done = "done" in stdout2.read().decode()

        _, stdout3, _ = self._ssh.exec_command(
            f"tail -5 {self.remote_work_dir}/training_output/exp/train.log 2>/dev/null | grep -i error"
        )
        error_line = stdout3.read().decode().strip()

        current_epoch, current_map = 0, 0.0
        if line:
            parts = line.split(",")
            if len(parts) > 3:
                try:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
                except (ValueError, IndexError):
                    pass

        return {
            "done": is_done,
            "error": bool(error_line),
            "error_msg": error_line,
            "current_epoch": current_epoch,
            "total_epochs": cfg.get("epochs", 100),
            "current_map": current_map,
        }

    def pull_artifacts(self, cfg: dict) -> dict:
        self.state = CloudTrainState.PULLING
        local_dir = Path("/tmp/cloud_artifacts") / f"{self.host}_{int(time.time())}"
        local_dir.mkdir(parents=True, exist_ok=True)

        artifacts = {}
        files_to_pull = [
            f"{self.remote_work_dir}/training_output/exp/weights/best.pt",
            f"{self.remote_work_dir}/training_output/exp/weights/last.pt",
            f"{self.remote_work_dir}/training_output/exp/results.csv",
            f"{self.remote_work_dir}/training_output/exp/confusion_matrix.png",
            f"{self.remote_work_dir}/training_output/exp/PR_curve.png",
            f"{self.remote_work_dir}/training_output/exp/F1_curve.png",
            f"{self.remote_work_dir}/training_output/exp/results.png",
        ]

        for remote_path in files_to_pull:
            fname = Path(remote_path).name
            local_path = local_dir / fname
            try:
                self._sftp.get(remote_path, str(local_path))
                artifacts[fname] = str(local_path)
            except FileNotFoundError:
                pass

        for fmt in cfg.get("export_formats", []):
            export_local = self._export_model(fmt, local_dir, cfg)
            if export_local:
                artifacts[f"model.{fmt}"] = export_local

        return artifacts

    def _export_model(self, fmt: str, local_dir: Path, cfg: dict) -> Optional[str]:
        export_cmd = (
            f"python -c \"from ultralytics import YOLO; \"
            f\"YOLO('{self.remote_work_dir}/training_output/exp/weights/best.pt')\"
            f\".export(format='{fmt}')\""
        )
        self._ssh.exec_command(
            f"cd {self.remote_work_dir} && {export_cmd}"
        )
        time.sleep(60)
        remote = f"{self.remote_work_dir}/training_output/exp/weights/best.{fmt}"
        local_path = local_dir / f"best.{fmt}"
        try:
            self._sftp.get(remote, str(local_path))
            return str(local_path)
        except FileNotFoundError:
            return None

    def shutdown(self):
        """执行关机命令（通用：shutdown now），失败后触发告警"""
        self.state = CloudTrainState.SHUTTING_DOWN
        try:
            # 通用 Linux 关机命令
            self._ssh.exec_command("shutdown now")
        except Exception as e:
            self._alert("关机命令发送失败，请手动关闭服务器", str(e))
        finally:
            if self._ssh:
                self._ssh.close()

    def _alert(self, title: str, detail: str):
        """关机失败时发送告警通知"""
        if self._alert_manager:
            self._alert_manager.send(title, detail)

    def cancel(self):
        """取消训练：SSH 发 kill 信号到 screen 训练会话"""
        if self._ssh:
            self._ssh.exec_command(
                "screen -S train -X quit 2>/dev/null; "
                "killall -SIGTERM python 2>/dev/null; true"
            )


from pathlib import Path
from typing import Optional
```

#### 8.3.3 AutoDL 云端训练（专用实现）

> AutoDL 提供实例自动创建/销毁 API，适合「不想自己管理服务器」的用户。
> 对于已有云服务器的用户，推荐使用 GenericSSHCloudTrainer（8.3.2）。

```python
# backend/services/autodl_trainer.py

import time
import os
import requests
import paramiko
from enum import Enum
from typing import Optional, Callable
from .cloud_trainer import CloudTrainer, CloudTrainState


class AutoDLCloudTrainer(CloudTrainer):
    """
    AutoDL 专用云端训练器。
    与 GenericSSHCloudTrainer 的区别：AutoDL 提供实例创建/销毁 API，
    用户只需提供 token，系统自动创建和销毁 GPU 实例。
    """

    def __init__(self, config: dict):
        self.token = config["autodl_token"]
        self.api_base = "https://www.autodl.com/api/v1"
        self.gpu_type = config.get("gpu_type", "RTX 4090")
        self.remote_work_dir = "/root"
        self.gpu_device = "0"
        self.state = CloudTrainState.IDLE
        self._instance_id: Optional[str] = None
        self._ssh: Optional[paramiko.SSHClient] = None
        self._sftp: Optional[paramiko.SFTPClient] = None
        self._alert_manager = None

    def connect(self):
        self.state = CloudTrainState.CONNECTING
        self._instance_id = self._create_instance()
        self._wait_for_running()
        self._ssh = self._get_ssh()
        self._sftp = self._ssh.open_sftp()

    def _create_instance(self) -> str:
        resp = requests.post(
            f"{self.api_base}/instance/create",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"gpu_type": self.gpu_type, "image": "pytorch:2.1.0-cuda11.8"},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["data"]["instance_id"]

    def _wait_for_running(self, timeout: int = 300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{self.api_base}/instance/status/{self._instance_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if resp.json()["data"]["status"] == "running":
                return
            time.sleep(5)
        raise TimeoutError(f"AutoDL 实例 {self._instance_id} 启动超时（{timeout}s）")

    def _get_ssh(self) -> paramiko.SSHClient:
        info = self._get_instance_info()
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            info["host"], port=info["port"],
            username="root", password=info["password"], timeout=30
        )
        return ssh

    def _get_instance_info(self) -> dict:
        resp = requests.get(
            f"{self.api_base}/instance/info/{self._instance_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def upload_dataset(self, zip_path: str):
        self.state = CloudTrainState.UPLOADING
        self._sftp.put(zip_path, "/root/dataset.zip")
        self._ssh.exec_command("cd /root && unzip -q dataset.zip -d dataset")
        time.sleep(5)

    def run_training(self, train_config: dict, progress_callback=None):
        self.state = CloudTrainState.TRAINING
        train_cmd = self._build_train_command(train_config)
        self._ssh.exec_command(
            f"cd /root && screen -dmS train bash -c '{train_cmd}'"
        )
        while True:
            time.sleep(30)
            status = self._check_training_status(train_config)
            if progress_callback:
                progress_callback(status)
            if status["done"]:
                break
            if status["error"]:
                raise RuntimeError(f"训练失败: {status['error_msg']}")

    def _build_train_command(self, cfg: dict) -> str:
        model = cfg.get("model", "yolo11s.pt")
        epochs = cfg.get("epochs", 100)
        imgsz = cfg.get("imgsz", 640)
        lr0 = cfg.get("lr0", 0.01)
        patience = cfg.get("patience", 20)
        project = "/root/training_output"
        resume_str = (
            f", resume='/root/training_output/exp/weights/last.pt'"
            if cfg.get("resume_last", False) else ""
        )
        return (
            f"python -c \"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='/root/dataset/data.yaml', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device={self.gpu_device}{resume_str})\""
        )

    def _check_training_status(self, cfg: dict) -> dict:
        _, stdout, _ = self._ssh.exec_command(
            "tail -1 /root/training_output/exp/results.csv 2>/dev/null"
        )
        line = stdout.read().decode().strip()
        _, stdout2, _ = self._ssh.exec_command(
            "[ -f /root/training_output/exp/weights/best.pt ] && echo done"
        )
        is_done = "done" in stdout2.read().decode()
        _, stdout3, _ = self._ssh.exec_command(
            "tail -5 /root/training_output/exp/train.log 2>/dev/null | grep -i error"
        )
        error_line = stdout3.read().decode().strip()
        current_epoch, current_map = 0, 0.0
        if line:
            parts = line.split(",")
            if len(parts) > 3:
                try:
                    current_epoch = int(parts[0].strip())
                    current_map = float(parts[3].strip())
                except (ValueError, IndexError):
                    pass
        return {
            "done": is_done,
            "error": bool(error_line),
            "error_msg": error_line,
            "current_epoch": current_epoch,
            "total_epochs": cfg.get("epochs", 100),
            "current_map": current_map,
        }

    def pull_artifacts(self, cfg: dict) -> dict:
        self.state = CloudTrainState.PULLING
        local_dir = f"/tmp/artifacts/{self._instance_id}"
        os.makedirs(local_dir, exist_ok=True)
        artifacts = {}
        files = [
            "/root/training_output/exp/weights/best.pt",
            "/root/training_output/exp/weights/last.pt",
            "/root/training_output/exp/results.csv",
            "/root/training_output/exp/confusion_matrix.png",
            "/root/training_output/exp/PR_curve.png",
            "/root/training_output/exp/F1_curve.png",
            "/root/training_output/exp/results.png",
        ]
        for remote_path in files:
            fname = os.path.basename(remote_path)
            local_path = f"{local_dir}/{fname}"
            try:
                self._sftp.get(remote_path, local_path)
                artifacts[fname] = local_path
            except FileNotFoundError:
                pass
        for fmt in cfg.get("export_formats", []):
            export_local = self._export_model(fmt, local_dir)
            if export_local:
                artifacts[f"model.{fmt}"] = export_local
        return artifacts

    def _export_model(self, fmt: str, local_dir: str) -> Optional[str]:
        self._ssh.exec_command(
            f"python -c \"from ultralytics import YOLO; \"
            f\"YOLO('/root/training_output/exp/weights/best.pt').export(format='{fmt}')\""
        )
        time.sleep(60)
        remote = f"/root/training_output/exp/weights/best.{fmt}"
        local = f"{local_dir}/best.{fmt}"
        try:
            self._sftp.get(remote, local)
            return local
        except FileNotFoundError:
            return None

    def shutdown(self):
        """AutoDL API 关机（最多重试 3 次），失败触发告警"""
        self.state = CloudTrainState.SHUTTING_DOWN
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.api_base}/instance/shutdown",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"instance_id": self._instance_id},
                    timeout=30
                )
                resp.raise_for_status()
                return
            except Exception:
                time.sleep(5 * (attempt + 1))
        self._alert(
            f"[CRITICAL] AutoDL 实例 {self._instance_id} 关机失败",
            "请立即登录 AutoDL 控制台手动关闭实例以防继续扣费"
        )

    def _alert(self, title: str, detail: str):
        if self._alert_manager:
            self._alert_manager.send(title, detail)

    def cancel(self):
        if self._ssh:
            self._ssh.exec_command(
                "screen -S train -X quit; killall -SIGTERM python; true"
            )


from typing import Optional
import os
```

---

## 9. 前端页面与状态机

### 9.1 页面流转

```
Upload (阶段一)
    ↓ [上传样板图 + 描述，点击「解析意图」]
IntentConfirm (阶段一)
    ↓ [确认/微调 class list，点击「开始打标」]
LabelingProgress (阶段二，两段式打标，实时 WebSocket 进度)
    ↓ [打标完成，点击「前往增强」]
AugmentConfig (阶段二点五，Albumentations 配置)
    ↓ [点击「开始增强」]
ReviewSamples (阶段三，数据质量评估 + 抽样查看)
    ↓ [点击「确认并前往训练」]
TrainConfig (阶段三，模型选型 + 超参数 + 训练模式选择)
    ├─ [选择「本地训练」，点击「开始本地训练」]
    │       ↓
    │   TrainingMonitor (阶段四-A，本地子进程训练，WebSocket 进度)
    │       ↓ [训练完成]
    │   Delivery (下载 best.pt / ONNX / 报告)
    │
    └─ [选择「云端训练（AutoDL）」，点击「开始云端训练」]
            ↓
        TrainingMonitor (阶段四-B，AutoDL 云端训练，WebSocket 进度)
            ↓ [训练完成]
        Delivery (下载 best.pt / ONNX / 报告)
```

### 9.2 任务全局状态（Zustand）

```typescript
// frontend/src/store/taskStore.ts

type Stage =
  | "upload"
  | "intent_confirm"
  | "labeling"
  | "augment"
  | "review"
  | "train_config"
  | "training"
  | "delivery"

interface TaskState {
  // 基本信息
  taskId: string | null
  taskName: string
  stage: Stage

  // 阶段一
  sampleImages: File[]
  userDescription: string
  vlmResult: VLMResult | null

  // 阶段二
  labelingProgress: {
    current: number
    total: number
    phase: "detection" | "quality_check"
  }
  labeledImageCount: number

  // 阶段二点五
  augConfig: AugmentConfig
  totalImageCount: number

  // 阶段三
  splitStats: { train: number; val: number; test: number }
  qualityReport: DataQualityReport | null

  // 阶段四
  trainConfig: TrainConfig
  trainingProgress: TrainingProgress | null
  artifacts: ArtifactMap
}

interface AugmentConfig {
  targetCount: number
  strength: "light" | "medium" | "heavy"
  enabled: {
    geometric: boolean
    color: boolean
    noise: boolean
    weather: boolean
    occlusion: boolean
  }
  // 图片清理策略
  deleteOriginalImages: boolean  // 任务完成后删除原始上传图片
}

interface TrainConfig {
  model: string           // "yolo11s.pt" 等；增量训练时为已有 best.pt 路径
  epochs: number
  imgsz: number
  lr0: number
  patience: number
  conf: number
  iou: number
  exportFormats: ("onnx" | "engine" | "coreml" | "openvino")[]
  gpuType: string        // "RTX 4090" / "A100" 等（云端训练时使用）
  trainMode: "local" | "cloud"  // 训练模式选择
  // ── 增量训练模式 ─────────────────────────────
  incrementalMode: boolean           // true = 增量训练，跳过 VLM 意图解析
  baseModelPath: string | null       // 已有 best.pt 的路径
  /** startIncremental API 返回的合并数据集目录（由 worker 解析为绝对路径） */
  incrementalDatasetDir: string | null
  /** startIncremental API 返回的合并后 data.yaml 路径 */
  incrementalDataYaml: string | null
}

interface TrainingProgress {
  state: string
  currentEpoch: number
  totalEpochs: number
  currentMap: number
  startedAt: string
}
```

### 9.3 WebSocket 协议（Worker → 前端）

连接地址：`ws://localhost:7860/ws`

```typescript
// 进度推送
interface ProgressMessage {
  type: "progress"
  stage: "detection" | "quality_check" | "augmentation"
  current: number
  total: number
}

// 阶段完成
interface StageCompleteMessage {
  type: "stage_complete"
  stage: string
  result: Record<string, unknown>
}

// GPU 信息（定期推送，前端显示显存使用情况）
interface GpuInfoMessage {
  type: "gpu_info"
  name: string
  totalMemoryGB: number
  freeMemoryGB: number
  usedMemoryGB: number
}

// 训练进度推送（本地训练 + 云端训练共用）
interface TrainingProgressMessage {
  type: "training_progress"
  currentEpoch: number
  totalEpochs: number
  currentMap: number
  done: boolean
}

// 训练完成
interface TrainingCompleteMessage {
  type: "training_complete"
  mode: "local" | "cloud"
  artifacts: Record<string, string>  // filename -> local_path
  metrics: {
    bestMap: number
    lastMap: number
  }
}

// 训练错误
interface TrainingErrorMessage {
  type: "training_error"
  message: string
  recoverable: boolean  // true = 可从 checkpoint 恢复
}

// 错误
interface ErrorMessage {
  type: "error"
  stage: string
  message: string
  recoverable: boolean
}

// 前端 → Worker 指令
interface CommandMessage {
  type: "start" | "pause" | "cancel"
  payload?: Record<string, unknown>
}

// 心跳
interface HeartbeatMessage {
  type: "pong"
}
```

---

## 10. 后端 API 接口规范

所有接口返回格式：`{ "code": 0, "msg": "ok", "data": {} }`
错误时：`{ "code": 非0, "msg": "错误描述", "data": null }`

### 10.1 接口列表

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/tasks` | 创建新任务 |
| GET | `/api/tasks` | 获取任务列表 |
| GET | `/api/tasks/{task_id}` | 获取任务详情 |
| DELETE | `/api/tasks/{task_id}` | 删除任务 |
| POST | `/api/vlm/parse` | 调用 VLM 解析意图 |
| PUT | `/api/vlm/result/{task_id}` | 用户确认/修改 VLM 结果 |
| GET | `/api/settings` | 获取用户设置 |
| PUT | `/api/settings` | 保存用户设置 |
| POST | `/api/training/start` | 提交训练任务（根据 train_mode 选择本地或 AutoDL） |
| GET | `/api/training/{task_id}/status` | 查询训练状态 |
| POST | `/api/training/{task_id}/cancel` | 取消训练（本地发 cancel 命令，云端发 shutdown） |
| POST | `/api/files/upload` | 上传数据集 zip |
| GET | `/api/files/{task_id}/artifacts` | 获取交付物列表 |
| GET | `/api/files/{task_id}/artifacts/{filename}` | 下载指定交付物 |

### 10.2 关键接口详情

**POST /api/vlm/parse**

```json
// Request
{
  "images_base64": ["base64str1", "base64str2"],
  "user_text": "把戴红帽子的人框出来，戴的叫 helmet"
}

// Response
{
  "code": 0,
  "data": {
    "classes": [
      {
        "class_name": "helmet",
        "prompt": "construction worker wearing a red or yellow hard hat",
        "negative_prompt": "person without hat",
        "color_hint": "red or yellow"
      }
    ],
    "confidence": 0.94
  }
}
```

**PUT /api/settings**

```json
// Request（用户设置）
{
  "vlm_provider": "openai",
  "vlm_base_url": "https://api.openai.com/v1",
  "vlm_api_key": "sk-xxx",
  "autodl_token": "xxx",
  "default_model": "yolo11s.pt",
  "default_augment_strength": "medium",
  "default_delete_original": false
}
```

**POST /api/training/start**

```json
// Request
{
  "task_id": "uuid",
  "model": "yolo11s.pt",
  "epochs": 100,
  "imgsz": 640,
  "lr0": 0.01,
  "patience": 20,
  "conf": 0.25,
  "iou": 0.7,
  "export_formats": ["onnx"],
  "train_mode": "local",       // "local" | "cloud"
  "gpu_type": "RTX 4090",      // 云端训练时使用，本地训练时忽略
  "resume_last": false
}

// Response
{
  "code": 0,
  "data": {
    "instance_id": "autodl-instance-xxx"
  }
}
```

---

## 11. 本地 Worker 进程规范

### 11.1 启动与端口

Worker 监听 `http://127.0.0.1:7860`，前端通过本地 HTTP/WS 与其通信。

### 11.2 Worker 入口

```python
# worker/main.py

import uvicorn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CV Auto Trainer Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            await _handle_command(ws, data)
    except WebSocketDisconnect:
        pass


async def _handle_command(ws: WebSocket, data: dict):
    cmd = data.get("type")
    payload = data.get("payload", {})

    if cmd == "start_detection":
        from .pipeline.stage2_labeler import run_detection, run_quality_check
        from .utils.yolo_io import save_yolo_labels

        def make_progress(current, total, phase):
            ws.send_json(_build_gpu_info_msg())
            ws.send_json({
                "type": "progress",
                "stage": phase,
                "current": current,
                "total": total
            })

        # 第一段
        raw_boxes = run_detection(
            image_dir=payload["image_dir"],
            classes=payload["classes"],
            output_raw_dir=payload["output_raw_dir"],
            conf_threshold=payload.get("conf_threshold", 0.25),
            iou_threshold=payload.get("iou_threshold", 0.45),
            batch_size=payload.get("batch_size", 4),
            progress_callback=make_progress
        )

        # 第二段
        passed = run_quality_check(
            raw_boxes_path=f"{payload['output_raw_dir']}/raw_boxes.json",
            min_confidence=payload.get("qa_threshold", 0.5),
            progress_callback=make_progress
        )

        save_yolo_labels(
            passed,
            output_label_dir=payload["output_label_dir"],
            output_image_dir=payload["output_image_dir"]
        )

        ws.send_json({
            "type": "stage_complete",
            "stage": "labeling",
            "result": {"labeled_count": sum(len(v) > 0 for v in passed.values())}
        })

    elif cmd == "start_augmentation":
        from .pipeline.stage25_augmentor import augment_dataset

        def aug_progress(current, total, _phase):
            ws.send_json({
                "type": "progress",
                "stage": "augmentation",
                "current": current,
                "total": total
            })

        result = augment_dataset(
            src_image_dir=payload["src_image_dir"],
            src_label_dir=payload["src_label_dir"],
            output_image_dir=payload["output_image_dir"],
            output_label_dir=payload["output_label_dir"],
            target_count=payload["target_count"],
            strength=payload.get("strength", "medium"),
            enabled=payload.get("enabled"),
            progress_callback=aug_progress
        )

        ws.send_json({
            "type": "stage_complete",
            "stage": "augmentation",
            "result": result
        })

    elif cmd == "start_local_training":
        # 打标/增强完成后，确保 GPU 显存已释放
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        from .pipeline.local_trainer import LocalTrainer

        trainer = LocalTrainer()

        def training_progress_cb(status: dict):
            ws.send_json({
                "type": "training_progress",
                "currentEpoch": status.get("current_epoch", 0),
                "totalEpochs": status.get("total_epochs", 100),
                "currentMap": status.get("current_map", 0.0),
                "done": status.get("done", False)
            })

        try:
            artifacts = trainer.train(
                dataset_dir=payload["dataset_dir"],
                train_config=payload["train_config"],
                progress_callback=training_progress_cb
            )
            ws.send_json({
                "type": "training_complete",
                "mode": "local",
                "artifacts": artifacts,
                "metrics": {
                    "bestMap": artifacts.get("best_map", 0.0),
                    "lastMap": artifacts.get("last_map", 0.0)
                }
            })
        except Exception as e:
            ws.send_json({
                "type": "training_error",
                "message": str(e),
                "recoverable": payload.get("train_config", {}).get("resume_last", False)
            })

    elif cmd == "cancel":
        # 取消当前运行中的阶段（打标/增强/训练）
        from .pipeline.gpu_manager import cancel_current_stage
        cancel_current_stage()
        ws.send_json({"type": "cancel_ack", "cancelled": True})

    elif cmd == "ping":
        ws.send_json({"type": "pong"})


def _build_gpu_info_msg() -> dict:
    if not torch.cuda.is_available():
        return {"type": "gpu_info", "available": False}
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    used = torch.cuda.memory_allocated(0) / 1e9
    return {
        "type": "gpu_info",
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "totalMemoryGB": round(total, 1),
        "usedMemoryGB": round(used, 1),
        "freeMemoryGB": round(total - used, 1)
    }


@app.get("/gpu-info")
def get_gpu_info():
    return _build_gpu_info_msg()


@app.post("/check-health")
def check_health():
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
```

---

## 12. 数据库模型

```python
# backend/models/db.py

from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import uuid as uuid_lib
from datetime import datetime

Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String, primary_key=True, default=lambda: str(uuid_lib.uuid4()))
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 状态
    status = Column(String, default="created")

    # 阶段一
    vlm_result = Column(JSON)

    # 阶段二统计
    raw_image_count = Column(Integer, default=0)
    labeled_image_count = Column(Integer, default=0)

    # 阶段二点五
    augment_config = Column(JSON)

    training_state = Column(String)         # 训练状态：training_local | training_cloud | done | error | cancelled（由 Worker 回调写入）
    training_progress = Column(JSON)        # 训练进度详情（供前端轮询）
    training_started_at = Column(DateTime)  # 训练开始时间
    training_finished_at = Column(DateTime) # 训练结束时间

    # 阶段三统计（数据集打包阶段）
    total_image_count = Column(Integer, default=0)
    train_split_count = Column(Integer, default=0)
    val_split_count = Column(Integer, default=0)
    test_split_count = Column(Integer, default=0)

    # 数据集打包结果（prepare_dataset 接口写入）
    split_stats = Column(JSON)        # {"train": N, "val": N, "test": N}
    quality_report = Column(JSON)     # {"total_images": N, "class_distribution": [...], "avg_boxes_per_image": N, "warnings": [...]}

    # 训练配置
    train_config = Column(JSON)

    # 图片清理策略
    delete_original_images = Column(Boolean, default=False)

    # 阶段四结果
    autodl_instance_id = Column(String)
    best_map50 = Column(Float)
    best_map50_95 = Column(Float)
    artifact_paths = Column(JSON)
    error_message = Column(Text)

    # 文件路径
    image_dir = Column(String)
    label_dir = Column(String)
    dataset_dir = Column(String)


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, default=1)

    vlm_provider = Column(String, default="openai")
    vlm_base_url = Column(String, default="https://api.openai.com/v1")
    vlm_api_key_encrypted = Column(String)

    # 云端训练通用配置
    cloud_provider = Column(String, default="generic")  # "generic" | "autodl"
    # 通用 SSH 配置
    ssh_host = Column(String)
    ssh_port = Column(Integer, default=22)
    ssh_username = Column(String, default="root")
    ssh_password_encrypted = Column(String)
    ssh_private_key_path = Column(String)  # 私钥路径（密码和私钥二选一）
    remote_work_dir = Column(String, default="/root/workspace")
    # AutoDL 专用配置
    autodl_token_encrypted = Column(String)
    # 通用配置
    default_model = Column(String, default="yolo11s.pt")
    default_augment_strength = Column(String, default="medium")
    default_delete_original = Column(Boolean, default=False)
    default_gpu_type = Column(String, default="RTX 4090")
    default_train_mode = Column(String, default="local")  # "local" | "cloud"
```

---

## 13. 错误处理与异常边界

### 13.1 错误分级

| 级别 | 触发条件 | 处理方式 |
|------|----------|----------|
| FATAL | 云端服务器未关机（通用 SSH / AutoDL 均适用） | 钉钉/飞书/邮件告警，用户需手动处理 |
| ERROR | VLM 解析失败 | 自动重试 3 次，仍失败则提示用户检查 API Key |
| ERROR | 训练脚本崩溃 | 触发兜底关机，保存 last.pt，提示用户可续训 |
| ERROR | 云端服务器连接失败（SSH timeout / Auth failed） | 提示检查网络、用户名密码、密钥配置 |
| ERROR | AutoDL 实例启动超时（> 5 分钟） | 重试一次，失败后提示用户选择其他 GPU 类型 |
| WARN | 显存不足 | 自动将 batch_size 减半重试，最多 3 次 |
| WARN | 某张图片增强失败 | 跳过该图片，继续处理，统计失败数量 |
| WARN | 某个框质检失败 | 丢弃该框，继续处理 |

### 13.2 各阶段异常处理要点

**阶段二（本地打标）：**
- GPU 显存不足 → 自动将 `batch_size` 减半重试，最多 3 次，仍不足则抛出 `MemoryError`
- 单张图片读取失败 → 跳过，记录到 `failed_images.log`
- 任务中断 → 下次启动时读取已存在的 `raw_boxes.json`，跳过已完成图片

**阶段二点五（数据增强）：**
- 增强后所有 bbox 被 clip 消失 → 丢弃该增强样本，重新抽取原图增强
- 目标数量设置过高（> 原始数据 × 50）→ 弹出警告，但仍允许继续

**阶段四-A（本地训练）：**
- 子进程崩溃 → 捕获 returncode != 0，发送 `training_error` 消息，`recoverable` = `resume_last`
- 用户取消 → 发 SIGTERM/CTRL_BREAK_EVENT，超时 30s 强制 kill，清理 GPU 状态
- 显存不足 → 本地训练子进程独立显存空间，不影响 Worker 主进程，但子进程内部可能 OOM
- 断点续训 → 检测 `last.pt` 存在，train_config 中 `resume_last: true`，子进程加 `--resume` 参数

**阶段四-B（AutoDL 云端训练）：**
- SSH 连接断开 → 重连最多 3 次，每次等待 30s
- 训练过程中实例被抢占 → 检测到 `last.pt` 存在则下次可从断点续训
- **任何异常（包括 KeyboardInterrupt）** → `finally` 块必须触发关机
- 关机失败 → 自动发送钉钉/飞书/邮件告警

**阶段四（云端训练）：**
- SSH 连接断开 → 重连最多 3 次，每次等待 30s
- 训练过程中实例被抢占 → 检测到 `last.pt` 存在则下次可从断点续训
- **任何异常（包括 KeyboardInterrupt）** → `finally` 块必须触发关机

---

## 14. 配置与可选项

### 14.1 用户设置（SettingsPanel）

用户可在任意页面右上角打开设置面板：

**VLM 配置：**

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| VLM Provider | 下拉选择 | openai | openai / kimi / gemini |
| Base URL | 文本输入 | https://api.openai.com/v1 | API 端点 |
| API Key | 密码输入 | — | 加密存储在本地 SQLite |

**云端训练配置（通用 SSH）：**

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| 云端提供商 | 单选 | generic | generic（通用 SSH）/ autodl（AutoDL 专用 API） |
| SSH Host | 文本输入 | — | 云服务器 IP 或域名（如 123.45.67.89） |
| SSH Port | 数字输入 | 22 | SSH 端口 |
| SSH 用户名 | 文本输入 | root | 连接用户名 |
| SSH 密码 | 密码输入 | — | 密码/私钥二选一，加密存储 |
| SSH 私钥路径 | 文件选择 | — | 私钥文件路径（如 ~/.ssh/id_rsa） |
| 远程工作目录 | 文本输入 | /root/workspace | 服务器上的工作根目录 |

**AutoDL 专用配置（仅 provider=autodl 时显示）：**

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| AutoDL Token | 密码输入 | — | AutoDL API Token，加密存储 |
| GPU 类型 | 下拉选择 | RTX 4090 | 实例规格（系统自动创建/销毁实例） |

**训练全局配置：**

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| 默认训练模式 | 单选 | local | local（本地 GPU）/ cloud（云端服务器） |
| 默认模型 | 下拉选择 | yolo11s.pt | 新任务的默认模型 |
| 默认增强强度 | 下拉选择 | medium | light / medium / heavy |
| 完成后删除原图 | 开关 | 关闭 | 任务完成后自动删除原始上传图片 |

### 14.2 图片清理策略

清理执行时机：阶段四「训练完成」或「交付页面点击完成」时执行。
清理范围：原始上传的样板图和批量图片目录，不清理标注结果和模型文件。

### 14.3 训练模式与 GPU 选择

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| 训练模式 | 单选 | local | local（本地 GPU）/ cloud（云端 SSH 服务器） |
| 本地训练 GPU | 自动检测 | 系统第一个 GPU | 本地训练时使用的 GPU 设备 |
| 云端 GPU 设备 | 文本输入 | 0 | 云端服务器上 GPU 编号（0/1/2/...） |
| 训练时允许打标 | 开关 | 关闭 | 本地训练期间是否允许同时进行新任务打标（关闭则打标排队等待） |

**本地训练适用场景：**
- 数据集规模 < 10,000 张图片
- 用户机器有 8GB 以上可用 GPU 显存
- 免上传数据集，训练启动更快

**云端训练适用场景（通用 SSH / AutoDL 均适用）：**
- 超大规模数据集（> 10,000 张）
- 用户机器 GPU 显存 < 8GB（本地训练 batch_size 受限）
- 需要 4090/A100 顶级算力加速
- 已有云服务器（阿里云/腾讯云/AWS/自有服务器），不想用 AutoDL

> **通用 SSH vs AutoDL 专用**：有云服务器的用户推荐「通用 SSH」；不想管理服务器、只想按需租用 GPU 的用户用「AutoDL」（系统自动创建/销毁实例）。

> **注意**：本地训练和打标不能同时进行（显存隔离），系统通过 `gpu_stage` 上下文管理器确保打标阶段完全释放显存后才启动训练子进程。

---

## 15. 技术栈汇总

### 15.1 依赖版本约束

**前端：**
```
react >= 18.3.0
react-dom >= 18.3.0
typescript >= 5.4.0
vite >= 5.4.0
zustand >= 4.5.0
```

**后端：**
```
fastapi >= 0.115.0
uvicorn[standard] >= 0.32.0
sqlalchemy >= 2.0.0
pydantic >= 2.0.0
httpx >= 0.27.0
paramiko >= 3.4.0
pyyaml >= 6.0
jsonschema >= 4.0
python-multipart >= 0.0.9
cryptography >= 41.0
```

**本地 Worker：**
```
torch >= 2.0.0
ultralytics >= 8.3.0
transformers >= 4.36.0
albumentations >= 1.4.0
opencv-python >= 4.8.0
psutil >= 5.9.0          # 本地训练子进程管理
```

**告警通知（可选）：**
```
dingtalk-chatbot >= 1.5.0   # 钉钉告警
httpx >= 0.27.0             # 飞书/邮件 HTTP 通知
```

### 15.2 关键版本说明

| 依赖 | 版本约束 | 说明 |
|------|----------|------|
| ultralytics | `>=8.3.0` | 内置 ByteTrack，支持 YOLO11 和 RT-DETR |
| albumentations | `>=1.4.0` | `BboxParams clip` 参数稳定 |
| transformers | `>=4.36.0` | Moondream2 所需 |
| torch | `>=2.0.0` | FP16 推理稳定版本 |
| Python | `>=3.10` | 使用了 `X | Y` 类型注解语法 |

### 15.3 启动顺序

```bash
# 终端 1：后端 API
cd backend
uvicorn main:app --reload --port 8000

# 终端 2：本地 Worker（必须单独进程，因为会加载 GPU 模型）
cd worker
python main.py

# 终端 3：前端开发服务器
cd frontend
pnpm dev
```

---

## v2 预留特性（不在本次开发范围内）

- **复合算法编排**：多模型串联（primary_detector + secondary_classifier）+ 规则引擎（ZoneDwell）
- **Prompt 模板管理**：将 VLM 解析结果保存为可复用的 few-shot 模板
- **用户账号体系**：多租户、团队协作、权限管理
- **云端打标备选**：无本地 GPU 时切换到云端 Worker 处理

---

*文档版本：v7.0 | 最后更新：2026-04-03 | 状态：可直接开始开发*
