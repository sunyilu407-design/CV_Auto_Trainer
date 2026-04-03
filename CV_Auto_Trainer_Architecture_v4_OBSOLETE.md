# CV 自动化训练中台 — 项目架构说明书 v4.0

> **文档定位**：本文档为完整的开发交付物，覆盖系统架构、模块边界、接口协议、关键算法、数据结构和异常处理。开发者/AI 可直接依据本文档进行实现，无需额外询问细节。

---

## 目录

1. [项目概述与技术选型](#1-项目概述与技术选型)
2. [整体系统架构](#2-整体系统架构)
3. [模块目录结构](#3-模块目录结构)
4. [阶段一：意图解析（Cloud VLM）](#4-阶段一意图解析cloud-vlm)
5. [阶段二：两段式打标与清洗（Local GPU）](#5-阶段二两段式打标与清洗local-gpu)
6. [阶段二点五：离线数据增强（Local CPU/GPU）](#6-阶段二点五离线数据增强local-cpugpu)
7. [阶段三：数据集打包与训练配置（Web UI）](#7-阶段三数据集打包与训练配置web-ui)
8. [阶段四：云端调度与模型交付（AutoDL）](#8-阶段四云端调度与模型交付autodl)
9. [复合算法编排模块](#9-复合算法编排模块)
10. [前端页面与状态机](#10-前端页面与状态机)
11. [后端 API 接口规范](#11-后端-api-接口规范)
12. [本地 Worker 进程规范](#12-本地-worker-进程规范)
13. [数据库模型](#13-数据库模型)
14. [错误处理与异常边界](#14-错误处理与异常边界)
15. [技术栈汇总](#15-技术栈汇总)

---

## 1. 项目概述与技术选型

### 1.1 定位

面向「一人公司」的**零代码 CV 模型训练 SaaS 平台**。用户无需了解深度学习知识，只需上传几张样板图 + 口语描述，平台自动完成从「意图理解」→「海量打标」→「数据增强」→「云端训练」→「模型交付」的全流程。

### 1.2 核心设计原则

| 原则 | 说明 |
|------|------|
| 算力极致分离 | 云端大脑（VLM 理解意图）+ 本地小脑（GPU 打标）+ 云端超算（模型训练） |
| 显存安全第一 | 本地 GPU 最低适配 GTX 1650 4GB，所有本地推理严格两段式串行，绝不并发加载模型 |
| 零 API 成本增强 | 数据增强完全在本地用 Albumentations 实现，不依赖任何云端 API |
| 幂等与可恢复 | 所有阶段支持中断续跑，云端训练支持从 checkpoint 恢复 |
| 兜底关机 | 无论训练成功还是异常，AutoDL 实例必须被关闭，防止扣费 |

### 1.3 技术选型速查

| 层级 | 技术 | 版本/说明 |
|------|------|-----------|
| 前端 | React + TypeScript + Vite | Canvas API 用于标注框可视化 |
| 前端状态 | Zustand | 轻量，适合一人项目 |
| 实时通信 | WebSocket（前端 ↔ 本地 Worker） | 训练进度推送 |
| 后端 API | FastAPI（Python） | 处理 VLM 调用、AutoDL 调度、文件管理 |
| 本地 Worker | Python 独立进程 | 运行打标、增强，通过 HTTP/WS 与前端通信 |
| 目标检测（打标） | Ultralytics YOLO-World | 本地画框 |
| VQA 质检 | Moondream2 | 本地质检，显存占用低 |
| 数据增强 | Albumentations ≥ 1.3 | 纯本地，支持 bbox 同步变换 |
| 目标跟踪 | ByteTrack（ultralytics 内置） | 可选，复合算法场景 |
| 云端训练 | Ultralytics CLI（YOLO/RT-DETR 统一） | 统一 API，降低运维复杂度 |
| 云端调度 | AutoDL OpenAPI | SSH + API 双通道 |
| 模型导出 | Ultralytics export | ONNX / TensorRT / CoreML / OpenVINO |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） | SQLAlchemy ORM |
| 文件存储 | 本地文件系统 + 阿里云 OSS（可选） | 训练数据集和模型文件持久化 |

---

## 2. 整体系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器（前端）                        │
│  React SPA：上传 → 确认 → 监控 → 下载                           │
└──────────┬──────────────────────────┬───────────────────────┘
           │ REST / WebSocket          │ REST / WebSocket
           ▼                          ▼
┌──────────────────┐      ┌───────────────────────────────────┐
│   后端 API 服务    │      │       本地 Worker 进程（用户电脑）    │
│   FastAPI         │      │  Python 独立进程，监听 localhost     │
│                  │      │                                   │
│  · VLM Adapter   │      │  阶段二：YOLO-World → Moondream    │
│  · AutoDL 调度   │      │  阶段二点五：Albumentations 增强     │
│  · 任务状态管理   │      │  · GPU 显存安全管理                  │
│  · 文件管理       │      │  · 进度 WebSocket 推送              │
└──────────┬───────┘      └───────────────────────────────────┘
           │ SSH + AutoDL API
           ▼
┌─────────────────────────────────────────────────────────────┐
│                    AutoDL 云端 GPU 实例                        │
│  4090 / A100 · Ultralytics 训练 · 训练结束自动关机              │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 数据流总览

```
用户上传样板图 + 描述
        │
        ▼ 阶段一（云端 VLM）
  JSON 检测任务书
  [{class, prompt, color_hint}]
        │
        ▼ 阶段二（本地 GPU）
  两段式：画框 → 释放显存 → 质检 → 释放显存
  输出：images/ + labels/（YOLO .txt 格式）
        │
        ▼ 阶段二点五（本地 CPU/GPU）
  用户设定目标数量 → Albumentations 增强
  输出：images_aug/ + labels_aug/
        │
        ▼ 阶段三（Web UI 配置）
  train/val/test 分层分割（8:1:1）
  生成 data.yaml
  用户配置：模型选型 + 超参数 + 复合算法编排（可选）
        │
        ▼ 阶段四（AutoDL 云端）
  打包 dataset.zip → 上传 → 启动实例 → SSH 训练
  → 拉取 best.pt + 报告 → 关机
        │
        ▼
  交付物：best.pt + ONNX/TensorRT（可选）+ 训练报告
```

---

## 3. 模块目录结构

```
project-root/
├── frontend/                          # React 前端
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Upload.tsx             # 阶段一：上传样板图 + 描述
│   │   │   ├── IntentConfirm.tsx      # 阶段一：确认 VLM 解析结果
│   │   │   ├── LabelingProgress.tsx   # 阶段二：打标进度监控
│   │   │   ├── AugmentConfig.tsx      # 阶段二点五：增强配置
│   │   │   ├── ReviewSamples.tsx      # 阶段三：抽样查看
│   │   │   ├── TrainConfig.tsx        # 阶段三：模型选型 + 超参数
│   │   │   ├── TrainingMonitor.tsx    # 阶段四：训练进度实时监控
│   │   │   └── Delivery.tsx           # 阶段四：交付物下载
│   │   ├── components/
│   │   │   ├── AnnotationCanvas.tsx   # Canvas 标注框可视化组件
│   │   │   ├── AugPreview.tsx         # 增强效果实时预览
│   │   │   ├── CompoundEditor.tsx     # 复合算法编排面板
│   │   │   └── MetricsChart.tsx       # mAP/loss 实时曲线
│   │   ├── store/                     # Zustand 状态
│   │   │   ├── taskStore.ts           # 任务全局状态
│   │   │   └── workerStore.ts         # 本地 Worker 连接状态
│   │   └── api/
│   │       ├── backend.ts             # 后端 API 调用封装
│   │       └── worker.ts              # 本地 Worker API 调用封装
│
├── backend/                           # FastAPI 后端
│   ├── main.py
│   ├── routers/
│   │   ├── vlm.py                     # VLM 意图解析接口
│   │   ├── tasks.py                   # 任务管理 CRUD
│   │   ├── autodl.py                  # AutoDL 调度接口
│   │   └── files.py                   # 文件上传/下载接口
│   ├── services/
│   │   ├── vlm_adapter.py             # VLM 多厂商适配器
│   │   ├── autodl_scheduler.py        # AutoDL SSH 状态机
│   │   ├── dataset_packer.py          # 数据集打包 + data.yaml 生成
│   │   └── model_exporter.py          # 模型导出调度
│   └── models/
│       └── db.py                      # SQLAlchemy 数据库模型
│
├── worker/                            # 本地 Worker（用户电脑运行）
│   ├── main.py                        # FastAPI + WebSocket 服务入口
│   ├── pipeline/
│   │   ├── stage2_labeler.py          # 阶段二：两段式打标
│   │   ├── stage25_augmentor.py       # 阶段二点五：数据增强
│   │   └── gpu_manager.py             # 显存安全管理
│   └── utils/
│       ├── yolo_io.py                 # YOLO .txt 格式读写
│       └── dataset_splitter.py        # train/val/test 分层分割
│
└── cloud_scripts/                     # 上传至 AutoDL 实例执行的脚本
    ├── train.py                       # 统一训练入口
    ├── export.py                      # 模型导出
    └── health_check.py                # 训练状态心跳
```

---

## 4. 阶段一：意图解析（Cloud VLM）

### 4.1 用户操作

1. 上传 1~3 张「样板图」（已手动画框的参考图，支持 JPG/PNG，单张 ≤ 10MB）
2. 输入口语化需求文本（如："把戴红帽子的人框出来"） 系统可以进行内容的补足如： 戴的叫helmet，没戴的叫 no_helmet
3. 系统调用 VLM API 返回结构化任务书
4. 前端展示解析结果，用户可手动微调每个 class 的 prompt 后确认

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

import json, re
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
        self.base_url = base_url
        self.api_key = api_key

    def parse_intent(self, images_base64: list[str], user_text: str,
                     max_retry: int = 3) -> dict:
        """调用 VLM 解析用户意图，失败自动重试"""
        last_err = None
        for attempt in range(max_retry):
            try:
                raw = self._call_api(images_base64, user_text)
                result = self._parse_and_validate(raw)
                return result
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = e
                continue
        raise RuntimeError(f"VLM 解析失败，已重试 {max_retry} 次: {last_err}")

    def _call_api(self, images_base64: list[str], user_text: str) -> str:
        """统一调用接口，根据 provider 构造不同请求体"""
        import httpx
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
        match = re.search(r'```json\s*([\s\S]*?)```', raw)
        json_str = match.group(1) if match else raw.strip()
        data = json.loads(json_str)
        validate(instance=data, schema=TASK_SCHEMA)
        return data

    def _get_model_name(self) -> str:
        mapping = {
            "openai": "gpt-4o",
            "kimi": "moonshot-v1-8k",
            "gemini": "gemini-1.5-pro"
        }
        return mapping.get(self.provider, "gpt-4o")
```

### 4.4 前端 Few-shot 模板机制

- 用户确认后的解析结果可保存为「Prompt 模板」（存入 localStorage + 后端数据库）
- 下次新任务时可选择复用历史模板，VLM 调用时将历史模板作为 few-shot 示例注入 system prompt
- 模板支持导入/导出为 JSON 文件

---

## 5. 阶段二：两段式打标与清洗（Local GPU）

### 5.1 核心约束（必须严格遵守）

> ⚠️ GTX 1650 仅有 4GB 显存。**两段之间必须彻底释放显存，绝不能同时驻留两个模型。**

```
第一段：加载 YOLO-World → 推理全量图片 → 保存框 JSON → del model → empty_cache → gc.collect
第二段：加载 Moondream  → VQA 质检      → 过滤框     → del model → empty_cache → gc.collect
```

### 5.2 第一段：YOLO-World 画框

```python
# worker/pipeline/stage2_labeler.py

import torch, gc, json, os
from pathlib import Path
from ultralytics import YOLOWorld

def run_stage1_detection(
    image_dir: str,
    classes: list[dict],          # [{class_name, prompt, ...}]
    output_raw_dir: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    batch_size: int = 4,
    progress_callback=None
) -> dict:
    """
    第一段：使用 YOLO-World 对全量图片进行目标检测，输出原始框 JSON。
    返回：{image_path: [{class_idx, class_name, bbox_xywhn, conf}]}
    """
    model = None
    try:
        model = YOLOWorld("yolov8s-world.pt")
        model.half()  # FP16 半精度，降低约 40% 显存占用
        model.set_classes([c["prompt"] for c in classes])

        image_paths = list(Path(image_dir).glob("*.jpg")) + \
                      list(Path(image_dir).glob("*.png"))
        results_map = {}

        for i in range(0, len(image_paths), batch_size):
            batch = [str(p) for p in image_paths[i:i+batch_size]]
            results = model.predict(
                batch,
                conf=conf_threshold,
                iou=iou_threshold,  # NMS IoU 阈值，去除重叠框
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
                results_map[img_path] = boxes

            if progress_callback:
                progress_callback(min(i + batch_size, len(image_paths)),
                                   len(image_paths), "detection")

        # 保存原始框到磁盘（供第二段读取）
        os.makedirs(output_raw_dir, exist_ok=True)
        with open(f"{output_raw_dir}/raw_boxes.json", "w") as f:
            json.dump(results_map, f)

        return results_map

    finally:
        # 无论是否异常，必须释放显存
        if model is not None:
            del model
        torch.cuda.empty_cache()
        gc.collect()
```

### 5.3 第二段：Moondream VQA 质检

```python
def run_stage2_quality_check(
    raw_boxes_path: str,
    min_confidence: float = 0.5,      # Moondream 置信度阈值（0~1）
    progress_callback=None
) -> dict:
    """
    第二段：使用 Moondream 对每个裁剪框进行 VQA 质检。
    min_confidence: 高于此分数的框保留（0.5 = 中等严格）
    返回：{image_path: [通过质检的 box]}
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import cv2, numpy as np

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float16  # FP16
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2",
                                                   trust_remote_code=True)

        with open(raw_boxes_path) as f:
            raw_boxes = json.load(f)

        passed_boxes = {}
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
                # 转为像素坐标并裁剪
                x1 = max(0, int((cx - bw/2) * w))
                y1 = max(0, int((cy - bh/2) * h))
                x2 = min(w, int((cx + bw/2) * w))
                y2 = min(h, int((cy + bh/2) * h))

                if (x2 - x1) < 10 or (y2 - y1) < 10:
                    processed += 1
                    continue  # 过小的框直接丢弃

                crop = img[y1:y2, x1:x2]
                question = (f"Does this image clearly show: {box['prompt']}? "
                            f"Answer with a confidence score from 0.0 to 1.0, "
                            f"then a space, then Yes or No.")

                enc_img = model.encode_image(crop)
                answer = model.answer_question(enc_img, question, tokenizer)

                # 解析置信度（"0.85 Yes" 或 "0.2 No"）
                score = _parse_confidence(answer)
                if score >= min_confidence:
                    box["qa_score"] = score
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

def _parse_confidence(answer: str) -> float:
    """从 Moondream 回答中提取置信度分数，降级处理各种格式"""
    import re
    # 尝试解析 "0.85 Yes" 格式
    match = re.search(r'(0\.\d+|1\.0)', answer)
    if match:
        return float(match.group(1))
    # 降级：Yes = 0.8, No = 0.1
    if "yes" in answer.lower():
        return 0.8
    return 0.1
```

### 5.4 清洗结果转 YOLO 标注格式

```python
# worker/utils/yolo_io.py

def save_yolo_labels(passed_boxes: dict, output_label_dir: str,
                     output_image_dir: str):
    """
    将质检通过的框保存为 YOLO .txt 格式。
    每行格式：class_idx cx cy w h
    """
    import shutil
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)

    for img_path, boxes in passed_boxes.items():
        if not boxes:
            continue  # 无有效框的图片不写入数据集
        stem = Path(img_path).stem
        label_path = f"{output_label_dir}/{stem}.txt"
        with open(label_path, "w") as f:
            for box in boxes:
                cx, cy, w, h = box["bbox_xywhn"]
                f.write(f"{box['class_idx']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        shutil.copy(img_path, f"{output_image_dir}/{Path(img_path).name}")
```

---

## 6. 阶段二点五：离线数据增强（Local CPU/GPU）

### 6.1 模块定位

在阶段二清洗完成后、打包上传前执行。用户指定「目标图片总数」，系统从清洗后干净数据自动扩充，**完全基于 Albumentations，零 Token 消耗，无需任何 API**。

### 6.2 UI 配置项

| 控件 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| 目标总图片数 | 数字输入 | 当前数量 × 5 | 系统将清洗后数据扩充至此数量 |
| 增强强度 | 单选（轻/中/重） | 中 | 轻=几何+色彩，中+噪声模糊，重+天气遮挡 |
| 几何变换 | 多选勾选 | 全选 | 翻转/旋转/缩放/透视/平移 |
| 色彩扰动 | 多选勾选 | 全选 | 亮度/饱和度/Gamma/CLAHE |
| 噪声与模糊 | 多选勾选 | 全选 | 高斯噪声/运动模糊/JPEG 压缩 |
| 天气模拟 | 多选勾选 | 不选 | 雨/雾/阳光光斑 |
| 遮挡模拟 | 多选勾选 | 不选 | Cutout/Mosaic |
| 预览按钮 | 按钮 | — | 随机抽取 3 张原图，实时展示增强效果 |

### 6.3 完整实现

```python
# worker/pipeline/stage25_augmentor.py

import albumentations as A
import cv2, math, os, shutil, random
from pathlib import Path

def build_pipeline(strength: str = "medium",
                   enabled: dict = None) -> A.Compose:
    """
    构建增强 pipeline。
    enabled 示例：{"geometric": True, "color": True, "noise": True,
                   "weather": False, "occlusion": False}
    """
    if enabled is None:
        enabled = {k: True for k in
                   ["geometric", "color", "noise", "weather", "occlusion"]}

    transforms = []

    # ── A. 几何变换（自动同步 bbox 坐标）──────────────────────────────
    if enabled.get("geometric", True):
        transforms += [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.3,
                rotate_limit=15 if strength != "light" else 5,
                border_mode=cv2.BORDER_CONSTANT, p=0.5
            ),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ]

    # ── B. 色彩与亮度扰动 ─────────────────────────────────────────────
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

    # ── C. 噪声与模糊 ─────────────────────────────────────────────────
    if enabled.get("noise", True) and strength != "light":
        transforms += [
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=95, p=0.2),
        ]

    # ── D. 天气模拟（可选，户外场景） ────────────────────────────────
    if enabled.get("weather", False) and strength == "heavy":
        transforms += [
            A.RandomRain(slant_lower=-10, slant_upper=10,
                         drop_length=20, p=0.2),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.15),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
        ]

    # ── E. 遮挡模拟 ───────────────────────────────────────────────────
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
    enabled: dict = None,
    progress_callback=None
) -> int:
    """
    将数据集从 N 张扩充至 target_count 张。
    同时保留原始数据（原图直接复制到 output 目录）。
    返回：实际生成的图片数量
    """
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    src_images = list(Path(src_image_dir).glob("*.jpg")) + \
                 list(Path(src_image_dir).glob("*.png"))

    if not src_images:
        raise ValueError(f"源目录无图片: {src_image_dir}")

    pipeline = build_pipeline(strength, enabled)

    # 1. 先复制原始数据
    for img_path in src_images:
        label_path = Path(src_label_dir) / (img_path.stem + ".txt")
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

        label_path = Path(src_label_dir) / (img_path.stem + ".txt")
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
                cv2.imwrite(f"{output_image_dir}/{out_stem}.jpg", result["image"])
                _save_yolo_label(
                    f"{output_label_dir}/{out_stem}.txt",
                    result["bboxes"],
                    result["labels"]
                )
                generated += 1

                if progress_callback:
                    progress_callback(generated, needed, "augmentation")

            except Exception as e:
                # 单张增强失败不中断整体流程
                continue

    return existing + generated

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
    with open(path, "w") as f:
        for label, bbox in zip(labels, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{label} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
```

### 6.4 Mosaic 增强（大幅提升小目标检测能力）

当图片数量 ≥ 100 张且用户启用遮挡模拟时，额外启用 Mosaic 4 合 1 拼接：

```python
def create_mosaic(image_paths: list, label_paths: list,
                  output_size: int = 640) -> tuple:
    """
    将 4 张图片拼接为 Mosaic 大图，同步变换所有 bbox。
    随机选取 4 张，拼接在 2×2 网格中。
    """
    s = output_size
    mosaic_img = np.zeros((s * 2, s * 2, 3), dtype=np.uint8)
    all_bboxes, all_labels = [], []

    for i, (img_path, lbl_path) in enumerate(
            zip(random.sample(list(zip(image_paths, label_paths)), 4), range(4))):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (s, s))
        row, col = divmod(i, 2)
        x_offset, y_offset = col * s, row * s
        mosaic_img[y_offset:y_offset+s, x_offset:x_offset+s] = img

        bboxes, labels = _load_yolo_label(lbl_path)
        for bbox, label in zip(bboxes, labels):
            cx, cy, w, h = bbox
            # 调整坐标到大图空间（归一化）
            new_cx = (cx * s + x_offset) / (s * 2)
            new_cy = (cy * s + y_offset) / (s * 2)
            new_w  = w * s / (s * 2)
            new_h  = h * s / (s * 2)
            all_bboxes.append([new_cx, new_cy, new_w, new_h])
            all_labels.append(label)

    final = cv2.resize(mosaic_img, (s, s))
    # 调整坐标回 640×640 空间（已是归一化，不需再变换）
    return final, all_bboxes, all_labels
```

---

## 7. 阶段三：数据集打包与训练配置（Web UI）

### 7.1 数据集分层分割

```python
# worker/utils/dataset_splitter.py

import random, shutil, os
from pathlib import Path
from collections import defaultdict

def split_dataset(
    image_dir: str,
    label_dir: str,
    output_root: str,
    ratios: tuple = (0.8, 0.1, 0.1),  # train / val / test
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

    # 按主类别（文件中出现次数最多的类别）分组
    class_groups = defaultdict(list)
    for lbl_path in Path(label_dir).glob("*.txt"):
        img_path = Path(image_dir) / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = Path(image_dir) / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue
        dominant_class = _get_dominant_class(lbl_path)
        class_groups[dominant_class].append((img_path, lbl_path))

    splits = {"train": [], "val": [], "test": []}

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        splits["train"] += items[:n_train]
        splits["val"]   += items[n_train:n_train+n_val]
        splits["test"]  += items[n_train+n_val:]

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
    from collections import Counter
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

import yaml, os
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
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names
    }
    with open(output_path, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    return output_path

def pack_dataset_zip(dataset_root: str, output_zip_path: str,
                     train_config: dict) -> str:
    """
    打包数据集 + 训练配置为 zip 文件。
    train_config 包含：model_name, epochs, imgsz, lr0, conf, iou,
                       export_formats, compound_config
    """
    import zipfile, json
    # 写入训练配置
    config_path = f"{dataset_root}/train_config.json"
    with open(config_path, "w") as f:
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

---

## 8. 阶段四：云端调度与模型交付（AutoDL）

### 8.1 AutoDL SSH 状态机

```python
# backend/services/autodl_scheduler.py

import paramiko, time, os, json, requests
from enum import Enum

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

    def run_full_pipeline(
        self,
        zip_path: str,
        train_config: dict,
        progress_callback=None
    ) -> dict:
        """
        完整训练流水线，无论成功失败都保证关机。
        返回：{best_pt_path, metrics, export_paths}
        """
        instance_id = None
        try:
            # 1. 创建实例
            self.state = TrainState.CREATING
            instance_id = self._create_instance(train_config["gpu_type"])
            self._wait_for_running(instance_id)

            # 2. 上传数据集
            self.state = TrainState.UPLOADING
            self._upload_dataset(instance_id, zip_path)

            # 3. 执行训练
            self.state = TrainState.TRAINING
            self._run_training(instance_id, train_config, progress_callback)

            # 4. 拉取产物
            self.state = TrainState.PULLING
            artifacts = self._pull_artifacts(instance_id, train_config)

            self.state = TrainState.DONE
            return artifacts

        except Exception as e:
            self.state = TrainState.ERROR
            raise
        finally:
            # 无论如何都关机（核心保障）
            if instance_id:
                self.state = TrainState.SHUTTING_DOWN
                self._shutdown_instance(instance_id)

    def _run_training(self, instance_id: str, cfg: dict,
                      progress_callback=None):
        ssh = self._get_ssh(instance_id)
        train_cmd = self._build_train_command(cfg)

        # 使用 screen/nohup 后台运行，通过轮询 results.csv 监控进度
        ssh.exec_command(f"screen -dmS train bash -c '{train_cmd}'")

        # 轮询训练进度
        while True:
            time.sleep(10)
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

        base_cmd = (
            f"cd /root && python -c \""
            f"from ultralytics import YOLO; "
            f"model = YOLO('{model}'); "
            f"model.train(data='dataset/data.yaml', "
            f"epochs={epochs}, imgsz={imgsz}, lr0={lr0}, "
            f"patience={patience}, project='{project}', "
            f"name='exp', exist_ok=True, device=0)"
            f"\""
        )
        return base_cmd

    def _check_training_status(self, ssh, cfg: dict) -> dict:
        """通过读取 results.csv 获取训练状态"""
        _, stdout, _ = ssh.exec_command(
            "tail -1 /root/training_output/exp/results.csv 2>/dev/null"
        )
        line = stdout.read().decode().strip()
        if not line:
            return {"done": False, "error": False, "epoch": 0, "mAP": 0}

        # 检查是否训练完成
        _, stdout2, _ = ssh.exec_command(
            "[ -f /root/training_output/exp/weights/best.pt ] && echo done"
        )
        is_done = "done" in stdout2.read().decode()

        # 检查是否有错误日志
        _, stdout3, _ = ssh.exec_command(
            "tail -5 /root/training_output/exp/train.log 2>/dev/null | grep -i error"
        )
        error_line = stdout3.read().decode().strip()

        return {
            "done": is_done,
            "error": bool(error_line),
            "error_msg": error_line,
            "last_csv_line": line
        }

    def _pull_artifacts(self, instance_id: str, cfg: dict) -> dict:
        """拉取训练产物（权重文件 + 训练报告）"""
        ssh = self._get_ssh(instance_id)
        sftp = ssh.open_sftp()
        local_dir = f"/tmp/artifacts/{instance_id}"
        os.makedirs(local_dir, exist_ok=True)

        artifacts = {}
        files_to_pull = [
            "/root/training_output/exp/weights/best.pt",
            "/root/training_output/exp/weights/last.pt",
            "/root/training_output/exp/results.csv",
            "/root/training_output/exp/confusion_matrix.png",
            "/root/training_output/exp/PR_curve.png",
            "/root/training_output/exp/F1_curve.png",
        ]
        for remote_path in files_to_pull:
            filename = os.path.basename(remote_path)
            local_path = f"{local_dir}/{filename}"
            try:
                sftp.get(remote_path, local_path)
                artifacts[filename] = local_path
            except FileNotFoundError:
                pass  # 非必须文件缺失不中断

        # 触发可选导出
        for fmt in cfg.get("export_formats", []):
            export_path = self._export_model(ssh, sftp, fmt, local_dir)
            if export_path:
                artifacts[f"model.{fmt}"] = export_path

        return artifacts

    def _export_model(self, ssh, sftp, fmt: str, local_dir: str):
        """在云端执行模型导出后拉取"""
        ssh.exec_command(
            f"python -c \"from ultralytics import YOLO; "
            f"YOLO('/root/training_output/exp/weights/best.pt')"
            f".export(format='{fmt}')\" 2>&1"
        )
        time.sleep(30)  # 等待导出完成
        remote = f"/root/training_output/exp/weights/best.{fmt}"
        local = f"{local_dir}/best.{fmt}"
        try:
            sftp.get(remote, local)
            return local
        except FileNotFoundError:
            return None

    def _shutdown_instance(self, instance_id: str):
        """关闭实例（最多重试 3 次）"""
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
        # 3 次失败后记录告警日志（此处可接入钉钉/飞书告警）
        print(f"[CRITICAL] 实例 {instance_id} 关机失败，请手动处理！")

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
        ssh.connect(info["host"], port=info["port"],
                    username="root", password=info["password"],
                    timeout=30)
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
        time.sleep(5)
```

---

## 9. 复合算法编排模块

### 9.1 适用场景

当用户的业务需要多层判断逻辑时（如：违规停车 = 车辆检测 + 跨帧跟踪 + 区域规则判定），通过本模块配置，**无需额外训练**，仅需组合已有模型 + 纯代码规则引擎。

### 9.2 编排配置数据结构

```json
{
  "compound_enabled": true,
  "models": [
    {
      "id": "m1",
      "model_path": "vehicle_detector.pt",
      "classes": ["car", "truck", "bus"],
      "conf": 0.4,
      "role": "primary_detector"
    },
    {
      "id": "m2",
      "model_path": "plate_recognizer.pt",
      "classes": ["license_plate"],
      "conf": 0.6,
      "role": "secondary_classifier",
      "trigger": "on_alert"
    }
  ],
  "tracker": {
    "enabled": true,
    "algorithm": "bytetrack",
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    "track_buffer": 30
  },
  "rules": [
    {
      "id": "r1",
      "type": "zone_dwell",
      "name": "禁停区域告警",
      "zone_polygon": [[0.1, 0.2], [0.6, 0.2], [0.6, 0.8], [0.1, 0.8]],
      "target_classes": ["car", "truck"],
      "dwell_frames": 30,
      "action": "alert"
    }
  ]
}
```

### 9.3 规则引擎核心逻辑

```python
# worker/pipeline/rule_engine.py

from shapely.geometry import Point, Polygon

class ZoneDwellRule:
    """判断目标在指定区域内的停留时长"""
    def __init__(self, config: dict):
        pts = config["zone_polygon"]
        self.zone = Polygon([(p[0], p[1]) for p in pts])
        self.target_classes = set(config["target_classes"])
        self.dwell_frames = config["dwell_frames"]
        self.dwell_counter = {}  # {track_id: frame_count}

    def update(self, tracked_objects: list) -> list:
        """
        tracked_objects: [{track_id, class_name, bbox_xywhn}]
        返回：触发告警的 track_id 列表
        """
        alerts = []
        current_ids = set()

        for obj in tracked_objects:
            if obj["class_name"] not in self.target_classes:
                continue
            track_id = obj["track_id"]
            cx, cy = obj["bbox_xywhn"][0], obj["bbox_xywhn"][1]
            current_ids.add(track_id)

            if self.zone.contains(Point(cx, cy)):
                self.dwell_counter[track_id] = \
                    self.dwell_counter.get(track_id, 0) + 1
                if self.dwell_counter[track_id] >= self.dwell_frames:
                    alerts.append({
                        "track_id": track_id,
                        "rule": "zone_dwell",
                        "dwell_frames": self.dwell_counter[track_id],
                        "class_name": obj["class_name"]
                    })
            else:
                self.dwell_counter.pop(track_id, None)

        # 清理消失的目标
        for tid in list(self.dwell_counter.keys()):
            if tid not in current_ids:
                del self.dwell_counter[tid]

        return alerts
```

---

## 10. 前端页面与状态机

### 10.1 任务全局状态（Zustand）

```typescript
// frontend/src/store/taskStore.ts

interface TaskState {
  taskId: string | null
  stage: 'upload' | 'intent_confirm' | 'labeling' | 'augment' |
         'review' | 'train_config' | 'training' | 'delivery'

  // 阶段一
  sampleImages: File[]
  userDescription: string
  vlmResult: VLMResult | null

  // 阶段二
  labelingProgress: { current: number; total: number; phase: string }
  passedBoxCount: number

  // 阶段二点五
  augConfig: AugmentConfig
  augProgress: { current: number; total: number }
  totalImageCount: number

  // 阶段三
  trainConfig: TrainConfig
  splitStats: { train: number; val: number; test: number }

  // 阶段四
  trainingState: TrainState
  trainingMetrics: TrainingMetrics[]
  artifacts: ArtifactMap

  // Actions
  setStage: (stage: TaskState['stage']) => void
  setVlmResult: (result: VLMResult) => void
  setLabelingProgress: (p: LabelingProgress) => void
  setAugConfig: (config: AugmentConfig) => void
  setTrainConfig: (config: TrainConfig) => void
  setTrainingMetrics: (metrics: TrainingMetrics[]) => void
}

interface AugmentConfig {
  targetCount: number
  strength: 'light' | 'medium' | 'heavy'
  enabled: {
    geometric: boolean
    color: boolean
    noise: boolean
    weather: boolean
    occlusion: boolean
  }
}

interface TrainConfig {
  model: string           // "yolo11s.pt" 等
  epochs: number
  imgsz: number
  lr0: number
  patience: number
  conf: number
  iou: number
  exportFormats: string[] // ["onnx", "engine"] 等
  compoundConfig?: CompoundConfig
}
```

### 10.2 本地 Worker WebSocket 协议

Worker 监听 `ws://localhost:7860/ws`，消息格式：

```typescript
// 进度推送（Worker → 前端）
interface ProgressMessage {
  type: 'progress'
  stage: 'detection' | 'quality_check' | 'augmentation'
  current: number
  total: number
  extra?: Record<string, unknown>
}

// 阶段完成（Worker → 前端）
interface StageCompleteMessage {
  type: 'stage_complete'
  stage: string
  result: Record<string, unknown>
}

// 错误（Worker → 前端）
interface ErrorMessage {
  type: 'error'
  stage: string
  message: string
  recoverable: boolean
}

// 前端指令（前端 → Worker）
interface CommandMessage {
  type: 'start' | 'pause' | 'cancel'
  payload?: Record<string, unknown>
}
```

---

## 11. 后端 API 接口规范

所有接口返回格式：

```json
{ "code": 0, "msg": "ok", "data": {} }
```

错误时：`{ "code": 非0, "msg": "错误描述", "data": null }`

### 11.1 接口列表

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/tasks` | 创建新任务 |
| GET | `/api/tasks/{task_id}` | 获取任务详情与状态 |
| POST | `/api/vlm/parse` | 调用 VLM 解析意图 |
| PUT | `/api/vlm/result/{task_id}` | 用户确认/修改 VLM 结果 |
| POST | `/api/training/start` | 提交训练任务至 AutoDL |
| GET | `/api/training/{task_id}/status` | 查询训练状态 |
| GET | `/api/training/{task_id}/metrics` | 获取实时训练指标 |
| POST | `/api/files/upload` | 上传数据集 zip |
| GET | `/api/files/{task_id}/artifacts` | 获取交付物下载链接 |
| GET | `/api/settings/vlm` | 获取用户 VLM API 配置 |
| PUT | `/api/settings/vlm` | 保存 VLM API 配置 |
| GET | `/api/templates` | 获取 Prompt 模板列表 |
| POST | `/api/templates` | 保存 Prompt 模板 |

### 11.2 关键接口详情

**POST /api/vlm/parse**

```json
// Request
{
  "task_id": "uuid",
  "images_base64": ["base64str1", "base64str2"],
  "user_text": "把戴红帽子的人框出来",
  "template_id": null
}

// Response
{
  "code": 0,
  "data": {
    "classes": [
      {
        "class_name": "helmet",
        "prompt": "construction worker wearing a red hard hat",
        "negative_prompt": "person without hat",
        "color_hint": "red"
      }
    ],
    "confidence": 0.94
  }
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
  "compound_config": null
}
```

---

## 12. 本地 Worker 进程规范

### 12.1 Worker 启动方式

Worker 以独立 Python 进程运行在用户电脑上，前端通过本地 HTTP/WS 与其通信：

```python
# worker/main.py
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # 处理前端指令，推送进度

@app.post("/start-labeling")
async def start_labeling(config: LabelingConfig):
    # 异步启动打标 pipeline
    ...

@app.post("/start-augmentation")
async def start_augmentation(config: AugConfig):
    ...

@app.get("/gpu-info")
def get_gpu_info():
    """返回本地 GPU 信息供前端展示"""
    import torch
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "free_memory_gb": (torch.cuda.get_device_properties(0).total_memory
                           - torch.cuda.memory_allocated(0)) / 1e9
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7860)
```

### 12.2 GPU 显存安全管理器

```python
# worker/pipeline/gpu_manager.py

import torch, gc, psutil
from contextlib import contextmanager

@contextmanager
def gpu_stage(stage_name: str, required_gb: float = 2.0):
    """
    显存安全上下文管理器。
    进入时检查显存是否充足，退出时强制释放。
    """
    if torch.cuda.is_available():
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated(0)) / 1e9
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

# 使用示例：
# with gpu_stage("YOLO-World 画框", required_gb=2.5):
#     run_stage1_detection(...)
# with gpu_stage("Moondream 质检", required_gb=1.5):
#     run_stage2_quality_check(...)
```

---

## 13. 数据库模型

```python
# backend/models/db.py

from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base
import enum, uuid
from datetime import datetime

Base = declarative_base()

class TaskStatus(str, enum.Enum):
    CREATED = "created"
    INTENT_CONFIRMED = "intent_confirmed"
    LABELING = "labeling"
    LABELING_DONE = "labeling_done"
    AUGMENTING = "augmenting"
    AUGMENTING_DONE = "augmenting_done"
    TRAINING = "training"
    DONE = "done"
    ERROR = "error"

class Task(Base):
    __tablename__ = "tasks"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name         = Column(String, nullable=False)
    status       = Column(Enum(TaskStatus), default=TaskStatus.CREATED)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    vlm_result      = Column(JSON)       # VLM 解析结果
    augment_config  = Column(JSON)       # 增强配置
    train_config    = Column(JSON)       # 训练配置
    compound_config = Column(JSON)       # 复合算法编排配置

    # 统计数据
    raw_image_count    = Column(Integer, default=0)
    labeled_image_count = Column(Integer, default=0)
    augmented_image_count = Column(Integer, default=0)
    train_split_count  = Column(Integer, default=0)
    val_split_count    = Column(Integer, default=0)
    test_split_count   = Column(Integer, default=0)

    # 训练结果
    autodl_instance_id = Column(String)
    best_map50         = Column(Float)
    best_map50_95      = Column(Float)
    artifact_paths     = Column(JSON)   # {best_pt, onnx, ...}
    error_message      = Column(String)

class PromptTemplate(Base):
    __tablename__ = "prompt_templates"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name        = Column(String, nullable=False)
    description = Column(String)
    classes     = Column(JSON)   # [{class_name, prompt, negative_prompt}]
    created_at  = Column(DateTime, default=datetime.utcnow)
    use_count   = Column(Integer, default=0)

class UserSettings(Base):
    __tablename__ = "user_settings"
    id          = Column(Integer, primary_key=True, default=1)
    vlm_provider  = Column(String, default="openai")
    vlm_base_url  = Column(String)
    vlm_api_key   = Column(String)  # 加密存储
    autodl_token  = Column(String)  # 加密存储
    default_augment_strength = Column(String, default="medium")
    default_model = Column(String, default="yolo11s.pt")
```

---

## 14. 错误处理与异常边界

### 14.1 错误分级

| 级别 | 类型 | 处理方式 |
|------|------|----------|
| FATAL | AutoDL 实例未关机 | 记录告警日志 + 钉钉/飞书通知 + 最多重试 3 次 |
| ERROR | VLM 解析失败 | 自动重试 3 次，仍失败则提示用户检查 API Key |
| ERROR | 训练脚本崩溃 | 触发兜底关机，保存 last.pt（若存在），提示用户续训 |
| WARN | 某张图片增强失败 | 跳过该图片，继续处理，统计失败数量 |
| WARN | 某个框质检失败 | 丢弃该框，继续处理 |
| INFO | 显存不足 | 自动降低 batch size，提示用户 |

### 14.2 各阶段异常处理要点

**阶段二（本地打标）：**
- GPU 显存不足 → 自动将 batch_size 减半重试
- 单张图片读取失败 → 跳过，记录到 failed_images.log
- Moondream 无法启动 → 检查 transformers 版本，提示用户更新依赖

**阶段二点五（数据增强）：**
- 增强后 bbox 全部被 clip 消失 → 丢弃该增强样本，重新抽取原图增强
- 目标数量设置过高（> 原始数据 × 50）→ 弹出警告提示，但允许继续

**阶段四（云端训练）：**
- 实例启动超时（> 5 分钟）→ 重试一次，失败后提示用户选择其他 GPU 类型
- SSH 连接断开 → 重连最多 3 次，每次等待 30s
- 训练过程中实例被抢占 → 检测到 last.pt 存在则自动续训
- 拉取产物失败 → 保存实例 ID，允许用户手动触发重新拉取

---

## 15. 技术栈汇总

### 15.1 依赖安装

**前端：**
```bash
pnpm add react react-dom typescript vite zustand
pnpm add @tanstack/react-query axios
# 注意：不需要额外安装 canvas 相关库，使用浏览器原生 Canvas API
```

**后端：**
```bash
pip install fastapi uvicorn sqlalchemy pydantic
pip install httpx paramiko pyyaml jsonschema
pip install python-multipart python-jose  # 文件上传 + JWT
```

**本地 Worker：**
```bash
pip install fastapi uvicorn websockets
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics>=8.3.0           # 包含 YOLO-World + ByteTrack
pip install transformers>=4.36.0          # Moondream2
pip install albumentations>=1.3.0
pip install shapely                        # 规则引擎区域判断
pip install opencv-python
```

### 15.2 关键版本约束

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| ultralytics | 8.3.0 | 内置 ByteTrack，支持 YOLO11 和 RT-DETR |
| albumentations | 1.3.0 | BboxParams clip 参数在此版本引入 |
| transformers | 4.36.0 | Moondream2 所需 |
| torch | 2.0.0 | FP16 推理稳定版本 |
| Python | 3.10+ | 使用了 X | Y 类型注解语法 |

### 15.3 模型文件下载

```bash
# YOLO-World（本地打标）
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt

# Moondream2（本地质检）- 通过 transformers 自动下载
# 首次运行时自动从 HuggingFace 下载，约 1.7GB

# 云端训练基础模型（在 AutoDL 实例上执行）
# ultralytics 在训练时自动下载对应 .pt 文件
```

---

*文档版本：v4.0 | 最后更新：2026-03 | 状态：可交付开发*
