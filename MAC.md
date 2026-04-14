# Mac（M1/M2/M3）安装与使用指南

本项目支持在 Mac Apple Silicon（M 系列芯片）上运行，主要用于**画框打标**阶段。云端训练和 VLM 解析不依赖本地硬件，可正常配合使用。

---

## 支持的功能

| 功能 | Mac M1/M2/M3 支持 | 备注 |
|------|:-----------------:|------|
| 阶段二·第一段 YOLO-World 画框 | ✅ | 使用 MPS 加速（FP16） |
| 阶段二·第二段 Moondream2 VQA 质检 | ✅ | 使用 MPS 加速 |
| 阶段二点五 Albumentations 数据增强 | ✅ | 纯 CPU |
| 阶段三 数据集打包 | ✅ | 纯 CPU |
| 阶段四 云端训练 | ✅ | 不占用本地算力 |
| VLM 意图解析 | ✅ | 调用远程 API |

> **注意**：Mac 显存与内存共用，8GB 内存跑 Moondream2 质检时建议 batch_size=1 或跳过质检直接输出 YOLO 标注。

---

## 系统要求

- macOS 12.3+（ Monterey 或更高，必须开启 Rosetta 或使用原生 ARM Python）
- Apple M1 / M2 / M3 / M4 系列芯片
- 推荐 16GB 统一内存，8GB 最低可用但较紧张
- Homebrew（推荐，用于安装依赖）

---

## 安装步骤

### 1. 确认是 ARM64 原生 Python

```bash
# 检查是否为 Apple Silicon
uname -m
# 输出应为 arm64

# 检查 Python 架构（必须是 arm64，不是 x86_64）
python3 -c "import platform; print(platform.machine())"
# 应输出 arm64
```

> 如果输出 `x86_64`，说明你用的是 Rosetta 转译的 Python，需要重装原生 ARM 版本。
> Homebrew 默认安装的就是 ARM64 版本。

### 2. 安装依赖管理器

```bash
# 如果没有 homebrew，先安装
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 验证 homebrew 是 ARM64 版
brew config | grep Architecture
# 应输出：Architecture: arm64
```

### 3. 安装 PyTorch（MPS 支持）

```bash
pip3 install torch torchvision torchaudio
```

验证 MPS 可用：

```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 4. 克隆项目

```bash
git clone https://github.com/sunyilu407-design/CV_Auto_Trainer.git
cd CV_Auto_Trainer
```

### 5. 安装后端依赖

```bash
cd backend
pip3 install -r requirements.txt
```

### 6. 安装 Worker 依赖

```bash
cd worker
pip3 install -r requirements.txt
```

> 注意：如果内存紧张（8GB），可以跳过安装 Moondream2（transformers），仅用于 YOLO 画框场景。
> 如需跳过，在 requirements.txt 中注释掉 `transformers>=4.36.0`。

### 7. 安装前端依赖

```bash
cd frontend
npm install
```

### 8. 验证 GPU 调度代码已适配

项目代码已包含 MPS 支持，检查以下文件是否存在：

```bash
ls worker/pipeline/gpu_manager.py
# 确认 get_device() 函数支持 MPS
grep -n "mps" worker/pipeline/gpu_manager.py
```

---

## 启动服务

### 终端 1：后端 API

```bash
cd backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 终端 2：本地 Worker（GPU 打标）

```bash
cd worker
python3 main.py
```

### 终端 3：前端

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

访问 `http://localhost:5173`

---

## 显存/内存优化建议

### 8GB 统一内存配置

编辑 `worker/pipeline/stage2_labeler.py`，降低并发：

```python
# run_detection 中
batch_size = 1  # 原来是 4

# run_quality_check 中 required_gb 可降至 1.5
with gpu_stage("moondream_qa", required_gb=1.5):
```

### 跳过 Moondream2 质检（推荐 8GB 用户）

如果内存紧张，可以跳过质检阶段，让 YOLO-World 直接输出标注：

编辑 `worker/main.py`，注释掉 `run_quality_check` 调用，只保留检测结果转 YOLO 格式的逻辑。

### 查看 MPS 内存使用

```bash
python3 -c "
import torch
if torch.backends.mps.is_available():
    # Mac 不提供精确显存查询，用系统内存代替
    import psutil
    mem = psutil.virtual_memory()
    print(f'系统内存: {mem.total / 1e9:.1f} GB')
    print(f'可用内存: {mem.available / 1e9:.1f} GB')
"
```

---

## 已知限制

1. **两段之间必须清缓存**：代码已处理，YOLO-World 跑完后会调用 `torch.mps.empty_cache()` 再加载 Moondream2。
2. **Mac 不支持 CUDA 版 Ultralytics**：所有模型自动降级到 MPS/FP16 或 CPU。
3. **部分模型权重需要下载**：`yolov8s-world.pt` 首次运行时会自动从 HuggingFace 下载，需要网络连接。
4. **云端训练在 AutoDL/Linux 上**：Mac 本地只能打标和增强，不能训练 YOLO（显存不够）。

---

## SiliconCloud API 配置（画框打标用）

如果你使用 SiliconCloud 的 VLM API，在 Settings 中配置：

| 配置项 | 值 |
|--------|-----|
| VLM Provider | `custom` |
| Base URL | `http://tokenapi.boundlessai.tech/v1` |
| Model | `Pro/zai-org/GLM-4.7`（或列表中其他模型） |
| API Format | `openai` |
| Temperature | `0.7` |
| Top P | `0.7` |

---

## 快速检查清单

```bash
# 1. 检查 MPS 可用
python3 -c "import torch; print(torch.backends.mps.is_available())"

# 2. 检查 YOLO 可导入
python3 -c "from ultralytics import YOLOWorld; print('YOLO OK')"

# 3. 检查 Moondream 可导入（可选）
python3 -c "from transformers import AutoModelForCausalLM; print('Transformers OK')"

# 4. 检查项目代码（确认 MPS 分支存在）
grep -c "mps" worker/pipeline/gpu_manager.py
# 应输出 > 0
```

---

## 常见问题

### Q: `torch.mps.empty_cache()` 报错怎么办？

确保 macOS 版本 >= 12.3，PyTorch 版本 >= 2.0。

```bash
pip3 install --upgrade torch
```

### Q: YOLO-World 加载失败，提示 `Module not found 'ultralytics'`？

```bash
pip3 install ultralytics>=8.3.0
```

### Q: Moondream2 加载时内存爆了？

8GB Mac 上 Moondream2 建议：
- 关闭其他应用
- 把 `batch_size` 设为 1
- 或者跳过质检阶段

### Q: 如何确认跑的是 MPS 而不是 CPU？

```python
import torch
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-world.pt")
print(next(model.model.parameters()).device)
# 输出应为：mps:0
```
