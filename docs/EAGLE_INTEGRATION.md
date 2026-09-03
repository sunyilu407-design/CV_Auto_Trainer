# Eagle/LocateAnything 整合方案与替换重构计划

> 创建日期：2026-08-13
> 更新日期：2026-08-13
> 项目：CV_Auto_Trainer × NVlabs/Eagle

---

## 开发状态

| 状态 | 里程碑 | 完成日期 |
|------|---------|----------|
| ✅ | 代码开发完成 | 2026-08-13 |
| ✅ | 测试通过 | 2026-08-13 |
| ✅ | 依赖文件更新 | 2026-08-13 |
| ⏳ | 等待 GPU 部署 | - |

### 已创建的文件

```
worker/
├── pipeline/
│   ├── locate_anything_adapter.py   ✅ LocateAnything 封装器
│   ├── eagle_vqa_adapter.py          ✅ Eagle2.5 VQA 封装器
│   └── output_converter.py           ✅ 输出格式转换工具
├── config.py                         ✅ 配置管理
└── tests/
    └── test_eagle_engine.py         ✅ 测试脚本

docs/
├── EAGLE_INTEGRATION.md             ✅ 整合方案文档
└── EAGLE_DEPLOYMENT.md              ✅ 部署指南
```

### 已修改的文件

- `worker/pipeline/engine_router.py` — 添加 Eagle 引擎路由
- `worker/pipeline/stage2_labeler.py` — 集成新模型
- `backend/models/db.py` — 添加配置字段
- `frontend/src/store/settingsStore.ts` — 前端配置
- `worker/requirements.txt` — 添加 Eagle 依赖说明
- `worker/requirements-eagle.txt` — **新建** Eagle 专用依赖文件

---

## 一、项目现状分析

### 1.1 当前 CV_Auto_Trainer 架构

```
阶段二（本地 GPU，两段式）：
┌─────────────────────────────────────────────────────────────────┐
│  第一段：YOLO-World                                              │
│  ├── 模型：yolov8s-world.pt                                      │
│  ├── 功能：开放词汇目标检测                                       │
│  ├── 显存需求：~3GB                                              │
│  └── 输出：raw_boxes.json (xywh + class_idx + conf)              │
├─────────────────────────────────────────────────────────────────┤
│  显存清空：del model → empty_cache → gc.collect()                │
├─────────────────────────────────────────────────────────────────┤
│  第二段：Moondream2                                              │
│  ├── 模型：vikhyatk/moondream2                                   │
│  ├── 功能：VQA 质检（清晰度/完整性/目标一致性）                  │
│  ├── 显存需求：~4GB                                              │
│  └── 输出：YOLO .txt 标注文件                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 当前模型局限性

| 方面 | YOLO-World | Moondream2 |
|------|------------|------------|
| 速度 | 较慢（逐框解码） | 中等 |
| 精度 | 一般 | 基础 |
| 任务类型 | 仅检测 | 仅 VQA |
| OCR 支持 | 无 | 无 |
| 长上下文 | 无 | 有限 |
| 多模态 | 无 | 有限 |

### 1.3 相关文件

- `worker/pipeline/stage2_labeler.py` — 两段式打标核心逻辑
- `worker/pipeline/gpu_manager.py` — 显存管理与设备选择
- `worker/pipeline/engine_router.py` — 检测引擎路由

---

## 二、Eagle 项目概述

### 2.1 Eagle VLM 家族

| 版本 | 会议/年份 | 主要特性 | 适用场景 |
|------|-----------|----------|----------|
| **Eagle** | ICLR 2025 | 混合编码器，多视觉专家融合 | 高分辨率图像理解 |
| **Eagle 2** | arXiv 2025 | 后训练数据策略 | 图像理解 SOTA |
| **Eagle 2.5** | NeurIPS 2025 | 长上下文，128K token，512 帧视频 | 视频理解 + 长文档 |
| **LocateAnything** | ECCV 2026 | Parallel Box Decoding，12.7 BPS | 目标检测 + 定位 + OCR |

### 2.2 模型规模

```
LocateAnything-3B    → Qwen2.5-3B + MoonViT-SO-400M
Eagle2.5-8B          → Qwen2.5-7B + SigLIP2-SO400M
Eagle2-1B/2B/9B      → Qwen2.5 + SigLIP
```

### 2.3 性能对比

| 指标 | YOLO-World | LocateAnything-3B | 提升 |
|------|------------|-------------------|------|
| 检测速度 (H100) | ~1.2 BPS | 12.7 BPS | **10x** |
| LVIS F1 | - | 50.7 | SOTA |
| COCO F1 | - | 54.7 | SOTA |
| OCR 支持 | 无 | 有 | - |
| GUI 定位 | 无 | 有 | - |

---

## 三、替换方案

### 3.1 引擎选择策略（重要）

**不是直接替换，而是作为可选引擎添加！**

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户可选择的引擎配置                           │
├─────────────────────────────────────────────────────────────────┤
│  检测引擎选项:                                                    │
│    ├── auto              → 自动选择（优先 LocateAnything）       │
│    ├── yolo_world        → 强制使用 YOLO-World                  │
│    ├── locate_anything   → 强制使用 LocateAnything              │
│    └── grounding_dino   → 强制使用 Grounding DINO              │
├─────────────────────────────────────────────────────────────────┤
│  VQA 引擎选项:                                                   │
│    ├── auto              → 自动选择（优先 Eagle2.5）            │
│    ├── moondream         → 强制使用 Moondream2                  │
│    └── eagle_vqa         → 强制使用 Eagle2.5                    │
└─────────────────────────────────────────────────────────────────┘

前端设置面板将提供引擎选择下拉框，用户可根据硬件配置自由选择。

显存自动降级规则:
  - LocateAnything 需要 6GB，如果显存不足 → 自动降级到 YOLO-World (3GB)
  - Eagle2.5 需要 16GB，如果显存不足 → 自动降级到 Moondream2 (4GB)
```

### 3.2 显存需求对比

| 引擎 | 显存需求 | 适用场景 | 推荐硬件 |
|------|----------|----------|----------|
| **YOLO-World** | ~3GB | GTX 1650 / 集成显卡 | 入门级 GPU |
| **Moondream2** | ~4GB | GTX 1650 / 入门级 GPU | 入门级 GPU |
| **LocateAnything** | ~6GB | RTX 3060+ | 中端 GPU |
| **Eagle2.5** | ~16GB | RTX 4090+ / A100 | 高端 GPU |

### 3.3 引擎对照表

| 当前组件 | 可选升级 | 模型大小 | 显存需求 | 状态 |
|----------|----------|----------|----------|------|
| **YOLO-World** | **LocateAnything-3B** | 3B | ~6GB | ✅ 可选 |
| **Moondream2** | **Eagle2.5-8B** | 8B | ~16GB | ✅ 可选 |

> **保留原因**：原有引擎继续可用，用户可根据硬件配置自由选择

### 3.2 架构对比

```
替换后架构：
┌─────────────────────────────────────────────────────────────────┐
│  第一段：LocateAnything (替代 YOLO-World)                        │
│  ├── 模型：nvidia/LocateAnything-3B                              │
│  ├── 功能：检测 + 定位 + OCR + GUI + 点定位                      │
│  ├── 速度：12.7 BPS (H100)                                      │
│  ├── 显存需求：~6GB (FP16)                                      │
│  └── 输出：结构化 JSON / YOLO 格式                               │
├─────────────────────────────────────────────────────────────────┤
│  显存清空：del model → empty_cache → gc.collect()                │
├─────────────────────────────────────────────────────────────────┤
│  第二段：Eagle2.5 (替代 Moondream2)                              │
│  ├── 模型：nvidia/Eagle2.5-8B                                   │
│  ├── 功能：高级 VQA, 长视频理解, 复杂质检                        │
│  ├── 上下文：128K token                                         │
│  ├── 显存需求：~16GB (FP16)                                     │
│  └── 输出：质检报告 + 质量评分                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 LocateAnything API 映射

| YOLO-World | LocateAnything | 说明 |
|------------|----------------|------|
| `model.predict()` | `worker.detect()` | 目标检测 |
| `model.set_classes()` | 直接传入类别列表 | 开放词汇 |
| - | `worker.ground_multi()` | 短语定位 |
| - | `worker.detect_text()` | OCR 检测 |
| - | `worker.ground_gui()` | GUI 元素定位 |
| - | `worker.point()` | 点定位 |

### 3.4 输出格式转换

LocateAnything 输出格式：
```
<ref>label</ref><box><x1><y1><x2><y2></box>
```

转换为 YOLO xywhn 格式：
```python
def parse_locate_output(answer: str, image_size: tuple) -> list[dict]:
    import re
    w, h = image_size
    boxes = []
    
    for match in re.finditer(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        answer
    ):
        x1, y1, x2, y2 = [int(g) / 1000 for g in match.groups()]
        # 转换为 xywhn (中心点 + 宽高，相对坐标)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        boxes.append({
            "bbox_xywhn": [cx, cy, bw, bh],
            "x1": x1 * w, "y1": y1 * h,
            "x2": x2 * w, "y2": y2 * h,
        })
    return boxes
```

---

## 四、模型管理

### 4.1 模型检查机制

**Worker 启动时会自动检查模型缓存状态：**

```
Worker 启动流程：
┌─────────────────────────────────────────────────────────────────┐
│  1. get_model_cache_status()                                     │
│     ├── 检查 YOLO-World (yolov8s-world.pt)                        │
│     ├── 检查 CLIP (ViT-B-32)                                      │
│     ├── 检查 Moondream2 (vikhyatk/moondream2)                    │
│     ├── 检查 LocateAnything (nvidia/LocateAnything-3B)            │
│     └── 检查 Eagle2.5 (nvidia/Eagle2.5-8B)                      │
├─────────────────────────────────────────────────────────────────┤
│  2. 如果模型不存在                                               │
│     ├── 必需模型 (YOLO-World + CLIP) → 自动下载                  │
│     └── 可选模型 (Moondream2 / LocateAnything / Eagle2.5)        │
│         → 需要用户手动触发下载                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 模型缓存位置

| 模型 | 缓存目录 | 约大小 |
|------|----------|--------|
| YOLO-World | `worker/yolov8s-world.pt` | 75 MB |
| CLIP | `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32` | 1.5 GB |
| Moondream2 | `~/.cache/huggingface/hub/models--vikhyatk--moondream2` | 4 GB |
| **LocateAnything** | `~/.cache/huggingface/hub/models--nvidia--LocateAnything-3B` | 6 GB |
| **Eagle2.5** | `~/.cache/huggingface/hub/models--nvidia--Eagle2.5-8B` | 16 GB |

### 4.3 API 接口

**获取模型状态：**

```bash
GET /model-status
```

响应：

```json
{
  "running": false,
  "status": {
    "yolo_world": { "installed": true },
    "clip": { "installed": true },
    "moondream": { "installed": false },
    "locate_anything": { "installed": false, "size_bytes": 0 },
    "eagle_vqa": { "installed": false, "size_bytes": 0 }
  }
}
```

**准备模型：**

```bash
POST /prepare-models
{
  "include_moondream": false,
  "include_locate_anything": true,
  "include_eagle_vqa": false
}
```

### 4.4 前端交互

前端 `EnvironmentPrep` 页面提供：
1. 模型状态展示（已就绪 / 等待中 / 可选）
2. 一键准备按钮（只下载必需模型）
3. 高级选项（可选下载 Eagle 引擎）

---

## 五、重构实施计划

### 阶段一：基础设施准备

| 任务 | 说明 | 预计时间 |
|------|------|----------|
| 1.1 | 创建 conda 环境：`conda create -n eagle python=3.10 -y` | 5 分钟 |
| 1.2 | 安装 PyTorch (CUDA 11.8+) | 10 分钟 |
| 1.3 | 安装 transformers >= 4.57.1 | 5 分钟 |
| 1.4 | 安装 accelerate >= 1.5.2, deepspeed >= 0.15.4 | 5 分钟 |
| 1.5 | 下载 LocateAnything-3B 模型 | 30 分钟 |
| 1.6 | 下载 Eagle2.5-8B 模型 | 1 小时 |
| 1.7 | 安装项目：`pip install -e ./Embodied` | 10 分钟 |

### 阶段二：核心模块开发

| 任务 | 位置 | 说明 |
|------|------|------|
| 2.1 | `worker/pipeline/locate_anything_adapter.py` | LocateAnything 封装器 |
| 2.2 | `worker/pipeline/eagle_vqa_adapter.py` | Eagle2.5 VQA 封装器 |
| 2.3 | `worker/pipeline/output_converter.py` | 输出格式转换工具 |
| 2.4 | `worker/pipeline/quality_evaluator.py` | 质检评分系统 |

### 阶段三：集成适配

| 任务 | 位置 | 说明 |
|------|------|------|
| 3.1 | `worker/pipeline/engine_router.py` | 添加模型路由 |
| 3.2 | `worker/pipeline/stage2_labeler.py` | 集成新模型 |
| 3.3 | `worker/config.py` | 添加配置项 |
| 3.4 | `frontend/src/store/taskStore.ts` | 状态管理更新 |
| 3.5 | `frontend/src/pages/LabelingProgress.tsx` | UI 更新 |

### 阶段四：测试验证

| 任务 | 说明 |
|------|------|
| 4.1 | LocateAnything Adapter 单元测试 |
| 4.2 | Eagle2.5 VQA Adapter 单元测试 |
| 4.3 | 完整流程集成测试 |
| 4.4 | 显存释放验证 |
| 4.5 | 输出格式验证 |
| 4.6 | 性能基准测试 |
| 4.7 | 回归测试（YOLO-World/Moondream2 兼容性） |

### 阶段五：文档与部署

| 任务 | 说明 |
|------|------|
| 5.1 | 更新 README 集成说明 |
| 5.2 | 编写使用指南 |
| 5.3 | 云端部署脚本（AutoDL） |

---

## 六、核心代码

### 5.1 LocateAnything Adapter

文件：`worker/pipeline/locate_anything_adapter.py`

```python
"""
LocateAnything 封装器，兼容现有 stage2_labeler 接口
"""

import gc
import re
from pathlib import Path
from typing import Optional, Callable

import torch
from PIL import Image


class LocateAnythingAdapter:
    """
    LocateAnything 模型封装，提供与 YOLO-World 兼容的接口
    """
    
    MODEL_ID = "nvidia/LocateAnything-3B"
    REQUIRED_VRAM_GB = 6.0  # FP16 推理显存需求
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.worker = None
        self._load_model()
    
    def _load_model(self):
        from locateanything_worker import LocateAnythingWorker
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.worker = LocateAnythingWorker(
            self.MODEL_ID,
            torch_dtype=dtype,
        )
    
    def detect(
        self,
        image: Image.Image,
        classes: list[str],
        conf_threshold: float = 0.25,
    ) -> list[dict]:
        """
        目标检测，返回 YOLO 格式 bbox
        
        Args:
            image: 输入图像
            classes: 检测类别列表
            conf_threshold: 置信度阈值
        
        Returns:
            list[dict]: [{"bbox_xywhn": [cx, cy, w, h], ...}, ...]
        """
        result = self.worker.detect(image, classes)
        return self._parse_locate_output(result, image.size)
    
    def ground_multi(
        self,
        image: Image.Image,
        query: str,
    ) -> list[dict]:
        """
        短语定位
        
        Args:
            image: 输入图像
            query: 自然语言查询
        
        Returns:
            list[dict]: 边界框列表
        """
        result = self.worker.ground_multi(image, query)
        return self._parse_locate_output(result, image.size)
    
    def detect_text(self, image: Image.Image) -> list[dict]:
        """
        OCR 文本检测
        
        Args:
            image: 输入图像
        
        Returns:
            list[dict]: 文本区域列表
        """
        result = self.worker.detect_text(image)
        return self._parse_locate_output(result, image.size)
    
    def ground_gui(
        self,
        image: Image.Image,
        query: str,
        output_type: str = "box",
    ) -> list[dict]:
        """
        GUI 元素定位
        
        Args:
            image: 输入图像
            query: 元素描述
            output_type: "box" 或 "point"
        
        Returns:
            list[dict]: GUI 元素位置
        """
        result = self.worker.ground_gui(image, query, output_type=output_type)
        return self._parse_locate_output(result, image.size)
    
    def point(self, image: Image.Image, query: str) -> list[dict]:
        """
        点定位
        
        Args:
            image: 输入图像
            query: 目标描述
        
        Returns:
            list[dict]: 点坐标列表
        """
        result = self.worker.point(image, query)
        return self._parse_locate_output(result, image.size)
    
    def _parse_locate_output(self, result: dict, image_size: tuple) -> list[dict]:
        """
        解析 LocateAnything 输出为统一格式
        
        输出格式：<box><x1><y1><x2><y2></box> (坐标范围 0-1000)
        """
        w, h = image_size
        boxes = []
        
        for match in re.finditer(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
            result.get("answer", "")
        ):
            x1, y1, x2, y2 = [int(g) for g in match.groups()]
            
            # 转换为相对坐标 (0-1)
            x1_r, y1_r = x1 / 1000, y1 / 1000
            x2_r, y2_r = x2 / 1000, y2 / 1000
            
            # 计算 xywhn 格式 (中心点 + 宽高)
            cx = (x1_r + x2_r) / 2
            cy = (y1_r + y2_r) / 2
            bw = x2_r - x1_r
            bh = y2_r - y1_r
            
            boxes.append({
                "x1": x1_r * w,
                "y1": y1_r * h,
                "x2": x2_r * w,
                "y2": y2_r * h,
                "bbox_xywhn": [cx, cy, bw, bh],
                "conf": 1.0,  # LocateAnything 不输出置信度
            })
        
        return boxes
    
    def unload(self):
        """释放模型显存"""
        del self.worker
        self.worker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 5.2 Eagle2.5 VQA Adapter

文件：`worker/pipeline/eagle_vqa_adapter.py`

```python
"""
Eagle2.5 VQA 封装器，替代 Moondream2
"""

import gc
import json
import re
from typing import Optional, Callable

import torch
from PIL import Image


class EagleVQAAdapter:
    """
    Eagle2.5 VQA 模型封装，提供高级图像理解能力
    """
    
    MODEL_ID = "nvidia/Eagle2.5-8B"
    REQUIRED_VRAM_GB = 16.0  # FP16 推理显存需求
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.processor = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForVision2Seq
        
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID,
            trust_remote_code=True,
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
    
    def quality_check(
        self,
        image: Image.Image,
        detected_objects: list[dict],
        quality_threshold: float = 0.5,
    ) -> dict:
        """
        VQA 质检，检查检测质量
        
        Args:
            image: 输入图像
            detected_objects: 检测到的对象列表
            quality_threshold: 质量阈值
        
        Returns:
            dict: {
                "passed": bool,
                "scores": {"clarity": float, "completeness": float, "accuracy": float},
                "rejected": bool,
                "reason": str
            }
        """
        prompt = self._build_quality_prompt(image, detected_objects)
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        return self._parse_quality_response(response, quality_threshold)
    
    def describe(self, image: Image.Image, question: str = None) -> str:
        """
        图像描述
        
        Args:
            image: 输入图像
            question: 可选的提问
        
        Returns:
            str: 描述文本
        """
        if question:
            prompt = question
        else:
            prompt = "请详细描述这张图像的内容。"
        
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
            )
        
        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def _build_quality_prompt(self, image: Image.Image, objects: list) -> str:
        """构建质检提示词"""
        if not objects:
            obj_list = "无"
        else:
            obj_list = ", ".join([
                f"{o.get('class_name', o.get('prompt', '未知'))}"
                for o in objects
            ])
        
        return f"""请检查图像中的检测质量。

检测到的对象：{obj_list}

请从以下维度评分（0-1）：
1. 清晰度：图像是否清晰可辨
2. 完整性：是否所有目标都被检测到
3. 准确性：检测框是否准确

请以JSON格式输出：{{"clarity": 0.9, "completeness": 0.8, "accuracy": 0.85}}"""
    
    def _parse_quality_response(self, response: str, threshold: float) -> dict:
        """解析质检响应"""
        # 提取 JSON
        match = re.search(r'\{[^}]+\}', response)
        if match:
            try:
                scores = json.loads(match.group())
                passed = all(v >= threshold for v in scores.values())
                return {
                    "passed": passed,
                    "scores": scores,
                    "rejected": not passed,
                    "reason": self._get_rejection_reason(scores, threshold),
                }
            except json.JSONDecodeError:
                pass
        
        return {
            "passed": False,
            "scores": {},
            "rejected": True,
            "reason": f"无法解析响应: {response[:100]}..."
        }
    
    def _get_rejection_reason(self, scores: dict, threshold: float) -> str:
        """生成拒绝原因"""
        failed = [
            {"维度": k, "分数": v, "要求": threshold}
            for k, v in scores.items()
            if v < threshold
        ]
        if not failed:
            return "通过"
        
        reasons = [f"{f['维度']}={f['分数']:.2f}(要求>{threshold})" for f in failed]
        return f"以下维度不达标：{', '.join(reasons)}"
    
    def unload(self):
        """释放模型显存"""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

### 5.3 引擎路由更新

文件：`worker/pipeline/engine_router.py`（修改）

```python
"""
检测引擎路由，支持 YOLO-World/Moondream2 或 LocateAnything/Eagle2.5
"""

from enum import Enum
from typing import Optional


class DetectionEngine(str, Enum):
    """检测引擎选项"""
    # 传统方案
    YOLO_WORLD = "yolo_world"      # YOLO-World + Moondream2
    # 升级方案
    LOCATE_ANYTHING = "locate_anything"  # LocateAnything + Eagle2.5


class EngineRouter:
    """
    检测引擎路由，根据配置选择合适的模型
    """
    
    def __init__(self, engine: DetectionEngine = DetectionEngine.YOLO_WORLD):
        self.engine = engine
        self._detector = None
        self._vqa = None
    
    @property
    def detector(self):
        """获取检测器实例（懒加载）"""
        if self._detector is None:
            if self.engine == DetectionEngine.LOCATE_ANYTHING:
                from .locate_anything_adapter import LocateAnythingAdapter
                self._detector = LocateAnythingAdapter()
            else:
                # 保留原有 YOLO-World 逻辑
                from ultralytics import YOLOWorld
                self._detector = YOLOWorld("yolov8s-world.pt")
        return self._detector
    
    @property
    def vqa(self):
        """获取 VQA 质检器实例（懒加载）"""
        if self._vqa is None:
            if self.engine == DetectionEngine.LOCATE_ANYTHING:
                from .eagle_vqa_adapter import EagleVQAAdapter
                self._vqa = EagleVQAAdapter()
            else:
                # 保留原有 Moondream2 逻辑
                from transformers import pipeline
                self._vqa = pipeline(
                    "zero-shot-image-classification",
                    model="vikhyatk/moondream2"
                )
        return self._vqa
    
    def release(self):
        """释放所有模型显存"""
        if self._detector is not None:
            if hasattr(self._detector, 'unload'):
                self._detector.unload()
            else:
                del self._detector
            self._detector = None
        
        if self._vqa is not None:
            if hasattr(self._vqa, 'unload'):
                self._vqa.unload()
            else:
                del self._vqa
            self._vqa = None
        
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

---

## 七、配置文件更新

### 6.1 Worker 配置

文件：`worker/config.py`

```python
# 检测引擎选择
DETECTION_ENGINE: str = "yolo_world"  # "yolo_world" | "locate_anything"

# LocateAnything 配置
LOCATE_ANYTHING: dict = {
    "model_id": "nvidia/LocateAnything-3B",
    "device": "cuda",  # "cuda" | "cpu"
    "torch_dtype": "float16",  # "float16" | "float32"
    "conf_threshold": 0.25,
    "batch_size": 4,
}

# Eagle2.5 VQA 配置
EAGLE_VQA: dict = {
    "model_id": "nvidia/Eagle2.5-8B",
    "device": "cuda",  # "cuda" | "cpu"
    "torch_dtype": "float16",
    "quality_threshold": 0.5,
}

# 显存管理
VRAM_CONFIG: dict = {
    "reserve_gb": 2.0,  # 保留显存
    "per_model": {
        "yolo_world": 3.0,
        "moondream2": 4.0,
        "locate_anything": 6.0,
        "eagle2.5": 16.0,
    }
}
```

### 6.2 前端配置面板

在 `frontend/src/pages/Settings.tsx` 添加：

```typescript
interface EngineConfig {
  detectionEngine: 'yolo_world' | 'locate_anything';
  locateAnything?: {
    device: 'cuda' | 'cpu';
    confThreshold: number;
  };
  eagleVQA?: {
    device: 'cuda' | 'cpu';
    qualityThreshold: number;
  };
}
```

---

## 八、显存需求对比

| 方案 | GPU 显存需求 | 适用硬件 |
|------|-------------|----------|
| **当前方案** (YOLO + Moondream) | ~7GB | GTX 1650 4GB 勉强支持 |
| **升级方案** (Locate + Eagle2.5) | ~22GB | 需要 RTX 3090+ |

### 显存优化建议

1. **使用 INT8 量化**：可将 Eagle2.5 显存需求降至 ~10GB
2. **分时加载**：严格遵守两段式设计，确保显存清空
3. **梯度检查点**：训练时使用，推理时不需要

---

## 九、硬件要求

### 8.1 最低要求（传统方案）

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU, >= 4GB VRAM |
| CUDA | >= 11.8 |
| RAM | >= 8GB |
| Python | >= 3.10 |

### 8.2 推荐要求（升级方案）

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA RTX 3090 24GB 或更高 |
| CUDA | >= 12.1 |
| RAM | >= 32GB |
| Python | >= 3.10 |
| 存储 | >= 50GB (模型缓存) |

### 8.3 云端方案

| 平台 | GPU | 价格 | 特点 |
|------|-----|------|------|
| AutoDL | RTX 4090 | ~¥2/小时 | 国内访问快 |
| Lambda Labs | A100 | ~$0.60/小时 | 稳定性高 |
| RunPod | RTX 4090 | ~$0.35/小时 | 性价比高 |
| Google Colab | T4 (免费) | 免费 | 适合测试 |

---

## 十、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 模型下载失败 | 项目无法启动 | 使用 HF Mirror + 预下载 |
| 显存不足 | 程序崩溃 | 严格显存检查 + 分级降级 |
| 精度下降 | 标注质量差 | 保留 YOLO-World 作为备选 |
| API 不兼容 | 集成失败 | Adapter 模式解耦 |
| 推理速度慢 | 用户体验差 | 批处理 + 异步加载 |

---

## 十一、里程碑

| 阶段 | 里程碑 | 验收标准 |
|------|--------|----------|
| M1 | 环境搭建完成 | conda 环境可用，模型可下载 |
| M2 | LocateAnything Adapter | 检测功能正常，输出格式正确 |
| M3 | Eagle2.5 VQA Adapter | 质检功能正常，评分合理 |
| M4 | 集成测试通过 | 完整流程可运行 |
| M5 | 性能达标 | 检测速度 >= 10 BPS |
| M6 | 文档完成 | README + 使用指南 |

---

## 十二、参考资源

- [Eagle 官方仓库](https://github.com/NVlabs/Eagle)
- [LocateAnything HuggingFace](https://huggingface.co/nvidia/LocateAnything-3B)
- [Eagle2.5 HuggingFace](https://huggingface.co/nvidia/Eagle2.5-8B)
- [LocateAnything 论文](https://research.nvidia.com/labs/lpr/locate-anything/LocateAnything.pdf)

---

## 附录 A：环境安装脚本

```bash
# 创建 conda 环境
conda create -n eagle python=3.10 -y
conda activate eagle

# 安装 PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 transformers 和相关依赖
pip install transformers>=4.57.1 tokenizers>=0.22.0
pip install accelerate>=1.5.2 deepspeed>=0.15.4
pip install timm>=1.0.11 liger_kernel>=0.3.1 peft>=0.12.0 decord

# 安装 FlashAttention (可选，用于加速)
pip install flash-attn --no-build-isolation

# 克隆 Eagle 仓库
git clone https://github.com/NVlabs/Eagle.git
cd Eagle/Embodied
pip install -e .

# 下载模型 (设置 HF_TOKEN)
export HF_TOKEN=your_hf_token
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/LocateAnything-3B'); \
    snapshot_download('nvidia/Eagle2.5-8B')"
```

---

## 附录 B：快速测试脚本

```python
"""
快速测试 LocateAnything
"""
from PIL import Image
import torch
from locateanything_worker import LocateAnythingWorker

# 初始化
worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
img = Image.open("test.jpg").convert("RGB")

# 检测
result = worker.detect(img, ["person", "car", "bicycle"])
print(result["answer"])

# 短语定位
result = worker.ground_multi(img, "people wearing red shirts")
print(result["answer"])

# OCR
result = worker.detect_text(img)
print(result["answer"])
```
