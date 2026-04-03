# CV 自动化训练中台 — 项目架构说明书

> **文档定位**：完整的开发交付物，开发者/AI 可直接依据本文档进行实现，无需额外询问细节。
> **版本**：v5.0 | **状态**：可直接开始开发

---

## 目录

1. [项目概述与技术选型](#1-项目概述与技术选型)
2. [整体系统架构](#2-整体系统架构)
3. [目录结构](#3-目录结构)
4. [阶段一：意图解析（云端 VLM）](#4-阶段一意图解析云端-vlm)
5. [阶段二：两段式打标与清洗（本地 GPU）](#5-阶段二两段式打标与清洗本地-gpu)
6. [阶段二点五：离线数据增强（本地 CPU/GPU）](#6-阶段二点五离线数据增强本地-cpugpu)
7. [阶段三：数据集打包与训练配置（Web UI）](#7-阶段三数据集打包与训练配置web-ui)
8. [阶段四：云端调度与模型交付（AutoDL）](#8-阶段四云端调度与模型交付autodl)
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

面向「一人公司」的**零代码 CV 模型训练平台**。用户上传几张手动画好框的「样板图」+ 口语描述，平台自动完成从「意图理解」→「海量打标」→「数据增强」→「云端训练」→「模型交付」的全流程。

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| 算力极致分离 | 云端大脑（VLM 意图解析）+ 本地小脑（GPU 打标）+ 云端超算（模型训练） |
| 显存安全第一 | 本地 GPU 最低适配 GTX 1650 4GB，所有本地推理严格两段式串行，绝不并发加载模型 |
| 零 API 成本增强 | 数据增强完全在本地用 Albumentations 实现，不依赖任何云端 API |
| 幂等与可恢复 | 所有阶段支持中断续跑，云端训练支持从 checkpoint 恢复 |
| 兜底关机 | 无论训练成功还是异常，AutoDL 实例必须被关闭，防止扣费 |
| 无账号体系 | 单人单机器使用，本地存储所有数据，无需登录 |
| 图片清理可控 | 用户可配置任务完成后是否自动删除原图 |

### 1.3 关键约束（必须严格遵守）

> **⚠️ 两段式打标是绝对核心**。阶段二的两段之间必须彻底释放显存（`del model` + `torch.cuda.empty_cache()` + `gc.collect()`），绝不能同时驻留两个模型。

> **⚠️ 兜底关机是最高优先级**。AutoDL 状态机的 `finally` 块必须覆盖所有退出路径，无论成功、异常、还是被中断。

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
| 数据增强 | Albumentations ≥ 1.4.0 | 纯本地，支持 bbox 同步变换 |
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
│                      │    │                                  │
│  · VLM Adapter      │    │  阶段二：YOLO-World → 显存释放 →    │
│  · AutoDL 调度      │    │          Moondream VQA 质检        │
│  · 任务状态管理      │    │  阶段二点五：Albumentations 增强    │
│  · 文件管理          │    │  · GPU 显存安全管理（gpu_stage）   │
│  · 设置管理          │    │  · 进度 WebSocket 推送             │
└──────────┬───────────┘    └─────────────────────────────────┘
           │ SSH + AutoDL API
           ▼
┌──────────────────────────────────────────────────────────────┐
│                     AutoDL 云端 GPU 实例                       │
│     4090 / A100 · Ultralytics 训练 · 训练结束自动关机           │
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
        ▼ 阶段四（AutoDL 云端）
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
│   │   │   └── SettingsPanel.tsx       # 全局设置面板
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
│   │   ├── autodl.py                  # AutoDL 调度接口
│   │   ├── files.py                   # 文件上传/下载接口
│   │   └── settings.py                # 用户设置接口
│   ├── services/
│   │   ├── vlm_adapter.py              # VLM 多厂商适配器
│   │   ├── autodl_scheduler.py        # AutoDL SSH 状态机
│   │   ├── dataset_packer.py           # 数据集打包 + data.yaml 生成
│   │   └── model_exporter.py           # 模型导出调度
│   └── models/
│       └── db.py                      # SQLAlchemy 数据库模型
│
├── worker/                            # 本地 Worker（用户电脑运行）
│   ├── main.py                        # FastAPI + WebSocket 服务入口
│   ├── pipeline/
│   │   ├── stage2_labeler.py          # 阶段二：两段式打标
│   │   ├── stage25_augmentor.py       # 阶段二点五：数据增强
│   │   └── gpu_manager.py             # 显存安全管理（gpu_stage）
│   └── utils/
│       ├── yolo_io.py                 # YOLO .txt 格式读写
│       └── dataset_splitter.py        # train/val/test 分层分割
│
└── cloud_scripts/                     # 上传至 AutoDL 实例执行的脚本
    ├── train.py                       # 统一训练入口（支持断点续训）
    ├── export.py                      # 模型导出
    └── health_check.py                # 训练状态心跳
```

---

## 4. 阶段一：意图解析（云端 VLM）

### 4.1 用户操作

1. 用户上传 1~3 张「样板图」（已手动画框的参考图，支持 JPG/PNG，单张 ≤ 10MB）
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
| gemini | gemini-1.5-pro | OpenAI-compatible（via proxy）或 Google原生 |

用户在前端「设置面板」中配置：provider、base_url、api_key。

---

## 5. 阶段二：两段式打标与清洗（本地 GPU）

### 5.1 核心约束（必须严格遵守）

> **⚠️ GTX 1650 仅有 4GB 显存。两段之间必须彻底释放显存，绝不能同时驻留两个模型。**

```
第一段：加载 YOLO-World → 推理全量图片 → 保存 raw_boxes.json → del model → empty_cache → gc.collect
第二段：加载 Moondream   → VQA 质检      → 过滤低质量框 → del model → empty_cache → gc.collect
```

### 5.2 显存安全管理器

```python
# worker/pipeline/gpu_manager.py

import torch
import gc
import psutil
from contextlib import contextmanager

@contextmanager
def gpu_stage(stage_name: str, required_gb: float = 2.0):
    """
    显存安全上下文管理器。
    进入时检查显存是否充足，退出时强制释放。
    如果显存不足，自动将 batch_size 减半重试（最多 3 次）。
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
    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
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

### 5.4 第二段：Moondream VQA 质检

采用**三维度 VQA** 而非单一问题，提升质检准确率：

```python
def run_quality_check(
    raw_boxes_path: str,
    min_confidence: float = 0.5,   # 置信度阈值（高于此值保留）
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
            f"Does this cropped image show a complete or mostly complete object "
            f"(not severely cropped or truncated)? "
            f"Answer with a number from 0.0 to 1.0, where 1.0 is complete.",
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
    格式支持：0.85 / 0.85 Yes / 0.85, yes / Yes 0.85 / 85%
    """
    import re

    answer_clean = answer.strip()

    # 优先匹配 "0.85" 或 "1.0" 格式（最常见）
    match = re.search(r"(0(?:\.\d+|\.0)|1(?:\.0)?)", answer_clean)
    if match:
        return float(match.group(1))

    # 尝试匹配 "85%" 格式
    match = re.search(r"(\d{1,3})%", answer_clean)
    if match:
        return float(match.group(1)) / 100.0

    # 尝试匹配 "Yes" / "No"（降级方案）
    lower = answer_clean.lower()
    if lower.startswith("yes"):
        # 尝试从后面提取数字
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

    # D. 天气模拟（可选，户外场景）
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

    # 按主类别分组（文件中出现次数最多的类别）
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
    # 写入训练配置
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
  totalImages: number        // 总图片数
  classDistribution: {        // 各类别 bbox 数量分布
    className: string
    boxCount: number
    avgBoxesPerImage: number
  }[]
  avgBoxesPerImage: number     // 平均每张图 bbox 数
  warnings: string[]          // 警告信息（如某类别 < 50 bbox）
}
```

**警告规则**：
- 某类别 bbox 总数 < 50 → 建议补充更多样板图
- 某类别 bbox 总数 < 200 → 建议增加增强强度
- 某类别 bbox 总数 > 5000 → 建议减少该类别数据（可能类别不平衡）

---

## 8. 阶段四：云端调度与模型交付（AutoDL）

### 8.1 AutoDL SSH 状态机

```python
# backend/services/autodl_scheduler.py

import time
import os
import json
import requests
import paramiko
from enum import Enum
from typing import Optional, Callable


class TrainState(Enum):
    IDLE = "idle"
    CREATING = "creating"
    UPLOADING = "uploading"
    TRAINING = "training"
    PULLING = "pulling"
    SHUTTING_DOWN = "shutting_down"
    DONE = "done"
    ERROR = "error"


class AutoDLScheduler:
    def __init__(self, autodl_token: str):
        self.token = autodl_token
        self.api_base = "https://www.autodl.com/api/v1"
        self.state = TrainState.IDLE
        self._instance_id: Optional[str] = None

    def run_full_pipeline(
        self,
        zip_path: str,
        train_config: dict,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        完整训练流水线，无论成功失败都保证关机。
        返回：{best_pt_path, last_pt_path, metrics, export_paths}
        """
        try:
            # 1. 创建实例
            self.state = TrainState.CREATING
            self._instance_id = self._create_instance(train_config["gpu_type"])
            self._wait_for_running(self._instance_id)

            # 2. 上传数据集
            self.state = TrainState.UPLOADING
            self._upload_dataset(self._instance_id, zip_path)

            # 3. 执行训练（支持断点续训）
            self.state = TrainState.TRAINING
            self._run_training(self._instance_id, train_config, progress_callback)

            # 4. 拉取产物
            self.state = TrainState.PULLING
            artifacts = self._pull_artifacts(self._instance_id, train_config)

            self.state = TrainState.DONE
            return artifacts

        except Exception as e:
            self.state = TrainState.ERROR
            raise

        finally:
            # 无论如何都关机（核心保障）
            if self._instance_id:
                self.state = TrainState.SHUTTING_DOWN
                self._shutdown_instance(self._instance_id)

    def _run_training(
        self,
        instance_id: str,
        cfg: dict,
        progress_callback: Optional[Callable] = None
    ):
        ssh = self._get_ssh(instance_id)
        train_cmd = self._build_train_command(cfg)

        # screen 后台运行训练，通过轮询 results.csv 监控进度
        ssh.exec_command(f"screen -dmS train bash -c '{train_cmd}'")

        while True:
            time.sleep(30)  # 每 30 秒轮询一次
            status = self._check_training_status(ssh, cfg)
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

        # 如果存在 last.pt，使用断点续训
        resume_str = (
            f", resume='/root/training_output/exp/weights/last.pt'"
            if cfg.get("resume_last", False) else ""
        )

        base_cmd = (
            f"cd /root && python -c \""
            f"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='dataset/data.yaml', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device=0{resume_str})"
            f"\""
        )
        return base_cmd

    def _check_training_status(self, ssh, cfg: dict) -> dict:
        """通过读取 results.csv 和 last.pt 的存在性获取训练状态"""
        _, stdout, _ = ssh.exec_command(
            "tail -1 /root/training_output/exp/results.csv 2>/dev/null"
        )
        line = stdout.read().decode().strip()

        # 检查 best.pt 是否存在（训练完成）
        _, stdout2, _ = ssh.exec_command(
            "[ -f /root/training_output/exp/weights/best.pt ] && echo done"
        )
        is_done = "done" in stdout2.read().decode()

        # 检查是否有错误日志
        _, stdout3, _ = ssh.exec_command(
            "tail -5 /root/training_output/exp/train.log 2>/dev/null | grep -i error"
        )
        error_line = stdout3.read().decode().strip()

        # 尝试从 results.csv 解析当前 epoch 和 mAP
        current_epoch = 0
        current_map = 0.0
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
            "last_csv_line": line
        }

    def _pull_artifacts(self, instance_id: str, cfg: dict) -> dict:
        """拉取训练产物（权重文件 + 训练报告）"""
        ssh = self._get_ssh(instance_id)
        sftp = ssh.open_sftp()
        local_dir = f"/tmp/artifacts/{instance_id}"
        os.makedirs(local_dir, exist_ok=True)

        artifacts: dict = {}
        files_to_pull = [
            "/root/training_output/exp/weights/best.pt",
            "/root/training_output/exp/weights/last.pt",
            "/root/training_output/exp/results.csv",
            "/root/training_output/exp/confusion_matrix.png",
            "/root/training_output/exp/PR_curve.png",
            "/root/training_output/exp/F1_curve.png",
            "/root/training_output/exp/results.png",
        ]

        for remote_path in files_to_pull:
            filename = os.path.basename(remote_path)
            local_path = f"{local_dir}/{filename}"
            try:
                sftp.get(remote_path, local_path)
                artifacts[filename] = local_path
            except FileNotFoundError:
                pass  # 非必须文件缺失不中断

        # 触发可选导出（ONNX / TensorRT 等）
        for fmt in cfg.get("export_formats", []):
            export_local = self._export_model(ssh, sftp, fmt, local_dir)
            if export_local:
                artifacts[f"model.{fmt}"] = export_local

        return artifacts

    def _export_model(
        self, ssh, sftp, fmt: str, local_dir: str
    ) -> Optional[str]:
        """在云端执行模型导出后拉取"""
        export_cmd = (
            f"python -c \"from ultralytics import YOLO; "
            f"YOLO('/root/training_output/exp/weights/best.pt')"
            f".export(format='{fmt}')\""
        )
        ssh.exec_command(export_cmd)
        time.sleep(60)  # 导出需要等待
        remote = f"/root/training_output/exp/weights/best.{fmt}"
        local = f"{local_dir}/best.{fmt}"
        try:
            sftp.get(remote, local)
            return local
        except FileNotFoundError:
            return None

    def _shutdown_instance(self, instance_id: str):
        """关闭实例（最多重试 3 次），失败后记录 CRITICAL 日志"""
        for attempt in range(3):
            try:
                resp = requests.post(
                    f"{self.api_base}/instance/shutdown",
                    headers={"Authorization": f"Bearer {self.token}"},
                    json={"instance_id": instance_id},
                    timeout=30
                )
                resp.raise_for_status()
                return
            except Exception:
                time.sleep(5 * (attempt + 1))
        # 3 次失败后记录 CRITICAL 日志（用户需要手动处理）
        print(f"[CRITICAL] 实例 {instance_id} 关机失败，请手动在 AutoDL 控制台关闭！")

    def _create_instance(self, gpu_type: str = "RTX 4090") -> str:
        resp = requests.post(
            f"{self.api_base}/instance/create",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"gpu_type": gpu_type, "image": "pytorch:2.1.0-cuda11.8"},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()["data"]["instance_id"]

    def _wait_for_running(self, instance_id: str, timeout: int = 300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = requests.get(
                f"{self.api_base}/instance/status/{instance_id}",
                headers={"Authorization": f"Bearer {self.token}"}
            )
            if resp.json()["data"]["status"] == "running":
                return
            time.sleep(5)
        raise TimeoutError(f"实例 {instance_id} 启动超时（{timeout}s）")

    def _get_ssh(self, instance_id: str) -> paramiko.SSHClient:
        info = self._get_instance_info(instance_id)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            info["host"],
            port=info["port"],
            username="root",
            password=info["password"],
            timeout=30
        )
        return ssh

    def _get_instance_info(self, instance_id: str) -> dict:
        resp = requests.get(
            f"{self.api_base}/instance/info/{instance_id}",
            headers={"Authorization": f"Bearer {self.token}"}
        )
        resp.raise_for_status()
        return resp.json()["data"]

    def _upload_dataset(self, instance_id: str, zip_path: str):
        ssh = self._get_ssh(instance_id)
        sftp = ssh.open_sftp()
        sftp.put(zip_path, "/root/dataset.zip")
        ssh.exec_command("cd /root && unzip -q dataset.zip -d dataset")
        time.sleep(10)  # 等待解压完成
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
TrainConfig (阶段三，模型选型 + 超参数)
    ↓ [点击「开始云端训练」]
TrainingMonitor (阶段四，实时进度，WebSocket)
    ↓ [训练完成]
Delivery (阶段四，下载 best.pt / ONNX / 报告)
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
  labelingProgress: { current: number; total: number; phase: "detection" | "quality_check" }
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
  // ⚑ 图片清理策略
  deleteOriginalImages: boolean  // 任务完成后删除原始上传图片
}

interface TrainConfig {
  model: string           // "yolo11s.pt" 等
  epochs: number
  imgsz: number
  lr0: number
  patience: number
  conf: number
  iou: number
  exportFormats: ("onnx" | "engine" | "coreml" | "openvino")[]
  gpuType: string        // "RTX 4090" / "A100" 等
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

```typescript
// 连接地址：ws://localhost:7860/ws

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
| POST | `/api/training/start` | 提交训练任务至 AutoDL |
| GET | `/api/training/{task_id}/status` | 查询训练状态 |
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
  "vlm_api_key": "sk-xxx",           // 加密存储
  "autodl_token": "xxx",              // 加密存储
  "default_model": "yolo11s.pt",
  "default_augment_strength": "medium",
  "default_delete_original": false    // 默认不删除原图
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
  "gpu_type": "RTX 4090",
  "resume_last": false
}

// Response
{
  "code": 0,
  "data": {
    "instance_id": "autodl-instance-xxx",
    "estimated_cost": "约 ¥15-25（仅供参考）"
  }
}
```

---

## 11. 本地 Worker 进程规范

### 11.1 启动与端口

Worker 监听 `http://127.0.0.1:7860`，前端通过本地 HTTP/WS 与其通信。

### 11.2 完整 Worker 入口

```python
# worker/main.py

import uvicorn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="CV Auto Trainer Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket 端点 ────────────────────────────────────────────────

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
    """处理前端命令，返回实时进度"""
    cmd = data.get("type")
    payload = data.get("payload", {})

    if cmd == "start_detection":
        from .pipeline.stage2_labeler import run_detection, run_quality_check
        from .utils.yolo_io import save_yolo_labels
        import json

        classes = payload["classes"]
        image_dir = payload["image_dir"]

        def make_progress(current, total, phase):
            # 推送 GPU 信息 + 进度
            gpu_msg = _build_gpu_info_msg()
            ws.send_json(gpu_msg)
            ws.send_json({
                "type": "progress",
                "stage": phase,
                "current": current,
                "total": total
            })

        # 第一段：YOLO-World
        raw_boxes = run_detection(
            image_dir=image_dir,
            classes=classes,
            output_raw_dir=payload["output_raw_dir"],
            conf_threshold=payload.get("conf_threshold", 0.25),
            iou_threshold=payload.get("iou_threshold", 0.45),
            batch_size=payload.get("batch_size", 4),
            progress_callback=make_progress
        )

        # 第二段：Moondream VQA
        passed = run_quality_check(
            raw_boxes_path=f"{payload['output_raw_dir']}/raw_boxes.json",
            min_confidence=payload.get("qa_threshold", 0.5),
            progress_callback=make_progress
        )

        # 保存 YOLO 标注
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


def _build_gpu_info_msg() -> dict:
    """构建 GPU 状态信息推送"""
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


# ── REST 端点 ─────────────────────────────────────────────────────

@app.get("/gpu-info")
def get_gpu_info():
    """返回本地 GPU 信息供前端展示"""
    return _build_gpu_info_msg()


@app.post("/check-health")
def check_health():
    """健康检查，验证 Worker 是否正常运行"""
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

    # 基本信息
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 状态
    status = Column(String, default="created")   # created / intent_confirmed /
                                                 # labeling / labeling_done /
                                                 # augmenting / augmenting_done /
                                                 # training / done / error

    # 阶段一
    vlm_result = Column(JSON)      # VLM 解析结果

    # 阶段二统计
    raw_image_count = Column(Integer, default=0)
    labeled_image_count = Column(Integer, default=0)

    # 阶段二点五
    augment_config = Column(JSON)  # 增强配置

    # 阶段三统计
    total_image_count = Column(Integer, default=0)
    train_split_count = Column(Integer, default=0)
    val_split_count = Column(Integer, default=0)
    test_split_count = Column(Integer, default=0)

    # 训练配置
    train_config = Column(JSON)    # 模型选型 + 超参数

    # 图片清理策略
    delete_original_images = Column(Boolean, default=False)

    # 阶段四结果
    autodl_instance_id = Column(String)
    best_map50 = Column(Float)
    best_map50_95 = Column(Float)
    artifact_paths = Column(JSON)   # {"best.pt": "/path/...", "results.csv": "..."}
    error_message = Column(Text)

    # 文件路径
    image_dir = Column(String)       # 原始图片目录
    label_dir = Column(String)      # YOLO 标注目录
    dataset_dir = Column(String)    # 打包后数据集目录


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, default=1)

    # VLM 配置
    vlm_provider = Column(String, default="openai")
    vlm_base_url = Column(String, default="https://api.openai.com/v1")
    vlm_api_key_encrypted = Column(String)   # AES 加密存储

    # AutoDL 配置
    autodl_token_encrypted = Column(String)  # AES 加密存储

    # 默认值
    default_model = Column(String, default="yolo11s.pt")
    default_augment_strength = Column(String, default="medium")
    default_delete_original = Column(Boolean, default=False)
    default_gpu_type = Column(String, default="RTX 4090")
```

---

## 13. 错误处理与异常边界

### 13.1 错误分级

| 级别 | 触发条件 | 处理方式 |
|------|----------|----------|
| FATAL | AutoDL 实例未关机 | 记录 CRITICAL 日志，打印实例 ID 提示用户手动关闭，最多重试 3 次 API |
| ERROR | VLM 解析失败 | 自动重试 3 次，仍失败则提示用户检查 API Key |
| ERROR | 训练脚本崩溃 | 触发兜底关机，保存 last.pt（若存在），提示用户可续训 |
| ERROR | AutoDL 实例启动超时（> 5 分钟） | 重试一次，失败后提示用户选择其他 GPU 类型 |
| WARN | 显存不足 | 自动将 batch_size 减半重试，最多 3 次 |
| WARN | 某张图片增强失败 | 跳过该图片，继续处理，统计失败数量 |
| WARN | 某个框质检失败（VQA 超时） | 丢弃该框，继续处理 |
| INFO | GPU 信息 | 定期推送给前端，供用户监控 |

### 13.2 各阶段异常处理要点

**阶段二（本地打标）：**
- GPU 显存不足 → 自动将 `batch_size` 减半重试，最多 3 次，仍不足则抛出 `MemoryError`
- 单张图片读取失败 → 跳过，记录到 `failed_images.log`
- Moondream 无法启动 → 检查 transformers 版本，提示用户更新依赖
- 任务中断 → 下次启动时读取已存在的 `raw_boxes.json`，跳过已完成图片

**阶段二点五（数据增强）：**
- 增强后所有 bbox 被 clip 消失 → 丢弃该增强样本，重新抽取原图增强
- 目标数量设置过高（> 原始数据 × 50）→ 弹出警告，但仍允许继续
- 任务中断 → 下次启动时从已存在的 output 目录继续（幂等设计）

**阶段四（云端训练）：**
- SSH 连接断开 → 重连最多 3 次，每次等待 30s
- 训练过程中实例被抢占 → 检测到 `last.pt` 存在则下次可从断点续训
- 拉取产物失败 → 保存实例 ID，允许用户手动触发重新拉取
- **任何异常（包括 KeyboardInterrupt）** → `finally` 块必须触发关机

---

## 14. 配置与可选项

### 14.1 用户设置（SettingsPanel）

用户可在任意页面右上角打开设置面板，配置以下内容：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| VLM Provider | 下拉选择 | openai | openai / kimi / gemini |
| Base URL | 文本输入 | https://api.openai.com/v1 | API 端点 |
| API Key | 密码输入 | — | 加密存储在本地 SQLite |
| AutoDL Token | 密码输入 | — | 加密存储 |
| 默认模型 | 下拉选择 | yolo11s.pt | 新任务的默认模型 |
| 默认增强强度 | 下拉选择 | medium | light / medium / heavy |
| 完成后删除原图 | 开关 | 关闭 | 任务完成后自动删除原始上传图片 |
| 默认 GPU 类型 | 下拉选择 | RTX 4090 | AutoDL 实例 GPU 规格 |

### 14.2 图片清理策略

用户可在两个位置配置：

1. **AugmentConfig 页面**：`deleteOriginalImages` 开关，控制当前任务
2. **全局设置**：`default_delete_original`，设置新任务的默认行为

清理执行时机：阶段四「训练完成」或「交付页面点击完成」时执行。
清理范围：原始上传的样板图和批量图片目录，不清理标注结果和模型文件。

---

## 15. 技术栈汇总

### 15.1 依赖版本约束

**前端：**
```bash
react >= 18.3.0
react-dom >= 18.3.0
typescript >= 5.4.0
vite >= 5.4.0
zustand >= 4.5.0
```

**后端：**
```bash
fastapi >= 0.115.0
uvicorn[standard] >= 0.32.0
sqlalchemy >= 2.0.0
pydantic >= 2.0.0
httpx >= 0.27.0
paramiko >= 3.4.0
pyyaml >= 6.0
jsonschema >= 4.0
python-multipart >= 0.0.9
cryptography >= 41.0   # 用于 API Key 加密存储
```

**本地 Worker：**
```bash
torch >= 2.0.0
ultralytics >= 8.3.0
transformers >= 4.36.0
albumentations >= 1.4.0
opencv-python >= 4.8.0
```

**关键版本说明：**

| 依赖 | 版本约束 | 说明 |
|------|----------|------|
| ultralytics | `>=8.3.0` | 内置 ByteTrack，支持 YOLO11 和 RT-DETR |
| albumentations | `>=1.4.0` | `BboxParams clip` 参数在此版本稳定 |
| transformers | `>=4.36.0` | Moondream2 所需 |
| torch | `>=2.0.0` | FP16 推理稳定版本 |
| Python | `>=3.10` | 使用了 `X | Y` 类型注解语法 |

### 15.2 模型文件预下载

```bash
# YOLO-World（本地打标，第一段）
# 首次运行 ultralytics 时自动下载，也可以手动下载：
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt

# Moondream2（本地质检，第二段）
# 首次运行 transformers 时自动从 HuggingFace 下载（约 1.7GB）

# 云端训练基础模型（在 AutoDL 实例上自动下载）
# ultralytics 在训练时自动下载对应 .pt 文件
```

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

以下特性已在 v4 文档中删除，如有需要可后续迭代：

- **复合算法编排**：多模型串联（primary_detector + secondary_classifier）+ 规则引擎（ZoneDwell）
- **Prompt 模板管理**：将 VLM 解析结果保存为可复用的 few-shot 模板
- **用户账号体系**：多租户、团队协作、权限管理
- **云端打标备选**：无本地 GPU 时切换到云端 Worker 处理
- **钉钉/飞书告警**：AutoDL 实例关机失败时自动通知

---

*文档版本：v5.0 | 最后更新：2026-04-03 | 状态：可直接开始开发*
