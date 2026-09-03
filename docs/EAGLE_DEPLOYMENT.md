# Eagle/LocateAnything 部署指南

> 本文档说明如何在有 GPU 的机器上部署和运行 Eagle 引擎

---

## 硬件要求

| 组件 | LocateAnything | Eagle2.5 VQA | 推荐配置 |
|------|---------------|---------------|----------|
| GPU | >= 6GB VRAM | >= 16GB VRAM | RTX 4090 24GB |
| CUDA | >= 11.8 | >= 11.8 | CUDA 12.1+ |
| RAM | >= 16GB | >= 32GB | 64GB |
| 存储 | >= 20GB | >= 50GB | 100GB SSD |

### 推荐配置

1. **入门级** (仅 LocateAnything)
   - RTX 3060 12GB
   - CUDA 11.8

2. **标准级** (LocateAnything + Eagle2.5)
   - RTX 4090 24GB
   - CUDA 12.1

3. **专业级**
   - RTX 3090 24GB × 2
   - A100 40GB
   - CUDA 12.1

---

## 安装步骤

### 方式一：从 requirements-eagle.txt 安装

```bash
# 1. 激活 conda 环境 (或创建新环境)
conda activate cv_trainer
# 或创建新环境
conda create -n cv_eagle python=3.10 -y
conda activate cv_eagle

# 2. 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 Eagle 依赖
cd D:/myProjects/CV_Auto_Trainer/worker
pip install -r requirements-eagle.txt

# 4. 安装 FlashAttention (可选，显著加速)
pip install flash-attn --no-build-isolation

# 5. 验证安装
python tests/test_eagle_engine.py
```

### 方式二：从本地 Eagle 仓库安装

如果你有 `Eagle-main` 目录在同级别目录：

```bash
# 1. 安装 LocateAnything (Embodied)
cd ../Eagle-main/Embodied
pip install -e .

# 2. 安装 Eagle2.5 VQA
cd ../Eagle2_5
pip install -e .

# 3. 验证安装
cd D:/myProjects/CV_Auto_Trainer/worker
python tests/test_eagle_engine.py
```

### 方式三：使用云 GPU (AutoDL)

```bash
# 1. 登录 AutoDL，选择镜像
# 推荐镜像: PyTorch 2.1 + CUDA 11.8

# 2. 克隆项目
git clone https://github.com/your-repo/CV_Auto_Trainer.git
cd CV_Auto_Trainer

# 3. 安装依赖
cd worker
pip install -r requirements-eagle.txt

# 4. 下载模型
export HF_TOKEN=your_hf_token
export HF_ENDPOINT=https://hf-mirror.com
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/LocateAnything-3B'); \
    snapshot_download('nvidia/Eagle2.5-8B')"
```

---

## 模型下载

### HuggingFace Token

1. 注册 HuggingFace 账号: https://huggingface.co
2. 申请访问权限 (某些模型需要)
3. 创建 Access Token: https://huggingface.co/settings/tokens

### 下载脚本

```bash
# 设置环境变量
export HF_TOKEN=hf_xxxxxxxxxxxxx
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像

# 下载 LocateAnything-3B
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/LocateAnything-3B')"

# 下载 Eagle2.5-8B
python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('nvidia/Eagle2.5-8B')"

# 验证下载
ls ~/.cache/huggingface/hub/
```

---

## 配置

### 环境变量

```bash
# GPU 配置
export CUDA_VISIBLE_DEVICES=0

# HuggingFace 配置
export HF_TOKEN=hf_xxxxxxxxxxxxx
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=600

# 日志
export LOG_LEVEL=INFO
```

### Worker 配置

编辑 `worker/config.py` 或设置环境变量：

```python
# 强制使用 Eagle 引擎
export CV_ENGINE_PREFERENCE=force_eagle

# 或自动选择
export CV_ENGINE_PREFERENCE=auto
```

---

## 启动

### 启动 Worker

```bash
cd D:/myProjects/CV_Auto_Trainer/worker

# 启动 Worker (会自动选择引擎)
python main.py

# 或指定引擎
python main.py --engine locate_anything
python main.py --engine eagle_vqa
```

### 查看引擎状态

```bash
cd D:/myProjects/CV_Auto_Trainer/worker
python -c "from config import print_engine_status; print_engine_status()"
```

预期输出：

```
============================================================
CV Auto Trainer - 引擎状态
============================================================

系统信息:
  CUDA 可用: 是
  GPU: NVIDIA GeForce RTX 4090
  总显存: 23.7 GB
  可用显存: 21.3 GB

检测引擎:
  YOLO-World: 可用 (需要 3GB)
    特性: 目标检测, 开放词汇
  Grounding DINO: 可用 (需要 4GB)
    特性: 目标检测, 开放词汇, 零样本
  LocateAnything: 可用 (需要 6GB)
    特性: 目标检测, 开放词汇, OCR, GUI定位, 点定位, 12.7 BPS

VQA 引擎:
  Moondream2: 可用 (需要 4GB)
    特性: VQA质检, 图像描述
  Eagle2.5 VQA: 可用 (需要 16GB)
    特性: VQA质检, 图像描述, 长上下文(128K), 高分辨率(4K)

============================================================
```

---

## 使用示例

### Python API

```python
from pipeline.stage2_labeler import (
    run_detection_with_engine,
    run_quality_check_with_engine,
)

# 自动选择引擎
results = run_detection_with_engine(
    image_dir="./images",
    classes=[{"class_name": "person"}, {"class_name": "car"}],
    output_raw_dir="./output",
    engine="auto",
)

# 强制使用 LocateAnything
results = run_detection_with_engine(
    image_dir="./images",
    classes=[{"class_name": "person"}, {"class_name": "car"}],
    output_raw_dir="./output",
    engine="locate_anything",
)

# 使用 Eagle2.5 VQA 质检
passed_boxes, stats = run_quality_check_with_engine(
    raw_boxes_path="./output/raw_boxes.json",
    engine="eagle_vqa",
    min_confidence=0.5,
)
```

### 命令行

```bash
# 运行完整打标流程
python -m pipeline.stage2_labeler \
    --image-dir ./images \
    --classes person,car,bicycle \
    --output-dir ./output \
    --engine auto

# 仅检测
python -m pipeline.locate_anything_adapter \
    --image-dir ./images \
    --classes person,car \
    --output raw_boxes.json
```

---

## 故障排除

### 显存不足

```
MemoryError: 阶段 [locate_anything] 需要 6.0GB 显存，当前仅剩 2.1GB
```

解决：
1. 关闭其他占用 GPU 的程序
2. 使用更小的 batch size
3. 回退到 YOLO-World：`engine="yolo_world"`

### 模型下载失败

```
OSError: 无法下载 nvidia/LocateAnything-3B
```

解决：
1. 检查 HF_TOKEN 是否正确设置
2. 使用国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`
3. 手动下载模型文件到 `~/.cache/huggingface/hub/`

### ImportError

```
ImportError: cannot import name 'LocateAnythingWorker' from 'locateanything_worker'
```

解决：
1. 确保已正确安装 Eagle 依赖
2. 检查 `PYTHONPATH` 包含 `Eagle-main/Embodied`

---

## 性能基准

### LocateAnything vs YOLO-World

| 指标 | YOLO-World | LocateAnything | 提升 |
|------|-----------|----------------|------|
| 检测速度 (H100) | ~1.2 BPS | 12.7 BPS | **10x** |
| LVIS F1 | - | 50.7 | SOTA |
| COCO F1 | - | 54.7 | SOTA |
| 显存占用 | 3GB | 6GB | +3GB |

### Eagle2.5 vs Moondream2

| 指标 | Moondream2 | Eagle2.5 | 提升 |
|------|------------|----------|------|
| 上下文长度 | 2K | 128K | **64x** |
| 视觉理解 | 基础 | 高级 | 显著 |
| 质检精度 | 一般 | 高 | 显著 |
| 显存占用 | 4GB | 16GB | +12GB |

---

## 注意事项

1. **显存安全**：严格遵守两段式设计，确保模型切换时显存完全释放
2. **量化优化**：如显存不足，可使用 INT8 量化 (牺牲精度换取显存)
3. **Mac 支持**：Apple Silicon Mac 可使用 MPS，但性能低于 CUDA
4. **Windows**：主要支持 Linux/macOS，Windows 需要 WSL2

---

## 相关文档

- [EAGLE_INTEGRATION.md](./EAGLE_INTEGRATION.md) - 整合方案详细文档
- [worker/requirements-eagle.txt](./worker/requirements-eagle.txt) - Eagle 依赖清单
- [worker/config.py](./worker/config.py) - 配置模块
- [worker/tests/test_eagle_engine.py](./worker/tests/test_eagle_engine.py) - 测试脚本
