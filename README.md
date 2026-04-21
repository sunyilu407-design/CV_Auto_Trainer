# CV 自动化训练中台

面向「一人公司」的零代码 CV 模型训练平台。用户上传手动画框的样板图 + 口语描述，系统自动完成从意图理解 → 海量打标 → 数据增强 → 模型训练 → 交付的全流程。

## 核心流程

```
用户上传样板图 + 文字描述
    ↓
[阶段一] VLM 解析意图 → 结构化检测任务书（OpenAI / Kimi / Gemini）
    ↓
[阶段二] 本地 GPU 两段式打标
    ├── 第一段：YOLO-World 画框（FP16，4GB 显存即可）
    └── 第二段：Moondream2 VQA 质检（清晰度/完整性/一致性）
    ↓
[阶段二点五] Albumentations 数据增强（零 API 成本）
    ↓
[阶段三] 数据集分层分割（8:1:1 训练/验证/测试）
    ↓
[阶段四] 本地或云端训练（本地 subprocess / 通用 SSH / AutoDL）
    ↓
交付 best.pt + ONNX + 训练报告
```

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React + TypeScript + Vite + Zustand |
| 实时通信 | WebSocket（前端 ↔ 本地 Worker） |
| 后端 API | FastAPI + SQLAlchemy |
| 本地 Worker | Python 独立进程 |
| 目标检测（打标） | Ultralytics YOLO-World（FP16） |
| VQA 质检 | Moondream2 |
| 数据增强 | Albumentations >= 1.4.0 |
| 云端训练 | Ultralytics YOLO CLI |
| 云端调度 | 通用 SSH（任意 GPU 服务器）/ AutoDL API |
| 数据库 | SQLite（开发）/ PostgreSQL（生产） |

## 目录结构

```
CV_Auto_Trainer/
├── frontend/                 # React 前端（Vite + TypeScript + Zustand）
│   ├── src/
│   │   ├── pages/            # Upload / IntentConfirm / LabelingProgress /
│   │   │                    #   AugmentConfig / ReviewSamples / TrainConfig /
│   │   │                    #   TrainingMonitor / Delivery
│   │   ├── components/       # AnnotationCanvas / AugPreview / MetricsChart / GpuMonitor
│   │   ├── store/            # taskStore.ts / settingsStore.ts（Zustand）
│   │   └── api/              # backend.ts / worker.ts
│   └── package.json
│
├── backend/                  # FastAPI 后端
│   ├── routers/              # tasks / vlm / files / settings / training
│   ├── services/             # vlm_adapter / train_dispatcher / cloud_trainer /
│   │                        #   generic_ssh_trainer / autodl_trainer / alert_manager
│   └── models/               # db.py（SQLAlchemy 模型）
│
├── worker/                   # 本地 Worker（独立进程，port 7860）
│   ├── pipeline/             # stage2_labeler / stage25_augmentor / gpu_manager
│   └── utils/                # yolo_io / dataset_splitter
│
├── cloud_scripts/            # 上传至云端实例的训练脚本
│   ├── train.py
│   ├── export.py
│   └── health_check.py
│
└── tests/
    └── test_integration.py   # 集成测试
```

## 快速开始

### 环境要求

- Python >= 3.10
- Node.js >= 18
- **NVIDIA GPU**（Linux/Windows，CUDA >= 11.8）
- **或 Apple M1/M2/M3/M4 Mac**（使用 MPS 加速，见 [MAC.md](MAC.md)）

> Mac 支持用于画框打标阶段。云端训练和 VLM 解析不依赖本地 GPU。

### 1. 克隆项目

```bash
git clone https://github.com/<your-username>/CV_Auto_Trainer.git
cd CV_Auto_Trainer
```

### 2. 安装后端依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 安装 Worker 依赖（GPU 打标）

```bash
cd worker
pip install -r requirements.txt
```

### 4. 安装前端依赖

```bash
cd frontend
npm install
```

### 5. 启动服务

```bash
# 终端 1：后端 API
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# 终端 2：本地 Worker（GPU 打标 + 本地训练）
cd worker
python main.py

# 终端 3：前端开发服务器
cd frontend
npm run dev -- --host 0.0.0.0
```

访问 `http://localhost:5173`

### Mac（M1/M2/M3）用户

请参考 [MAC.md](MAC.md) 获取完整的 Mac 安装与配置指南。

## 正式部署（单机）

适用场景：

- 本地画框、质检、增强
- 系统内部发起云端训练（SSH / AutoDL）
- 训练完成后直接下载模型产物和算法工程包

正式部署不再依赖 Vite 开发服务器。推荐使用：

- PostgreSQL 作为数据库
- 后端托管 `frontend/dist`
- Worker 仅监听本机端口

### 必备环境变量

```bash
CV_AUTO_TRAINER_DB_URL=postgresql://postgres:postgres@127.0.0.1:5432/cv_auto_trainer
CV_AUTO_TRAINER_SECRET_KEY=replace-this-with-a-stable-secret
CV_AUTO_TRAINER_ADMIN_USERNAME=admin
CV_AUTO_TRAINER_ADMIN_PASSWORD=change-me
CV_AUTO_TRAINER_FRONTEND_DIST=/absolute/path/to/frontend/dist
CV_AUTO_TRAINER_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

### 单机正式部署步骤

```bash
# 1. 构建前端
cd frontend
npm install
npm run build

# 2. 启动后端（托管前端 dist）
cd ../backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 3. 启动本地 Worker（画框 / 质检 / 增强）
cd ../worker
python main.py
```

部署完成后，直接访问：

```text
http://127.0.0.1:8000/
```

平台细节请参考：

- [MAC.md](MAC.md)
- [WINDOWS.md](WINDOWS.md)
- [MANUAL_CLOUD_TRAINING.md](MANUAL_CLOUD_TRAINING.md)

### PostgreSQL 初始化

如果你已经安装好了 PostgreSQL，macOS 下可以直接运行：

```bash
./scripts/init_postgres_macos.sh
```

这个脚本会：

- 创建数据库用户（默认 `cv_auto_trainer`）
- 创建数据库（默认 `cv_auto_trainer`）
- 打印 `CV_AUTO_TRAINER_DB_URL`

表结构不需要单独跑 SQL 文件。后端启动时会自动执行：

- `Base.metadata.create_all(...)`
- 轻量级 `ALTER TABLE` 补列逻辑

也就是说：**你只需要先把数据库创建出来，表会在后端第一次启动时自动建好。**

### 云训练连接失败时的手动兜底

如果系统一时无法直接连上用户租用的云训练环境，不建议让用户先盲目开机再排查。

推荐做法：

1. 本地先完成数据整理
2. 先生成“手动云训练包”
3. 确认 SSH / SCP 可用
4. 再让用户启动付费云实例

生成命令：

```bash
python scripts/prepare_manual_cloud_training.py \
  --dataset-dir backend/uploads/<task_id>/dataset \
  --output-dir /tmp/cv_manual_training
```

完整教程见：

- [MANUAL_CLOUD_TRAINING.md](MANUAL_CLOUD_TRAINING.md)

### 6. 配置

在 Settings 面板中配置：

| 配置项 | 说明 |
|--------|------|
| VLM Provider | OpenAI / Kimi / Gemini |
| API Key | 对应 provider 的密钥 |
| 训练模式 | 本地（subprocess）/ 云端（SSH / AutoDL） |

## 训练模式

### 本地训练

使用本机 NVIDIA GPU，通过 subprocess.Popen 启动独立训练子进程，显存与 Worker 主进程隔离。

### 云端训练（通用 SSH）

连接任意提供 SSH 访问的 GPU 服务器（阿里云 / 腾讯云 / AWS / 自有服务器）。需要配置 SSH 连接信息。

### 云端训练（AutoDL）

通过 AutoDL OpenAPI 自动创建和销毁 GPU 实例，用户只需提供 Token。

## 运行测试

```bash
# 全部测试（需要 GPU）
python tests/test_integration.py

# 后端导入测试
python tests/test_integration.py backend-imports

# 前端构建测试
python tests/test_integration.py frontend-build
```

## 架构设计要点

### 两段式打标

阶段二使用两个独立模型，显存必须分阶段释放：

```
第一段：YOLO-World 推理（FP16）
  → 输出 raw_boxes.json
  → del model → torch.cuda.empty_cache() → gc.collect()

第二段：Moondream2 VQA 质检
  → 三维度评分（清晰度/完整性/一致性）
  → 任一维度 < 0.4 丢弃，< 0.5 中性
  → 输出最终 YOLO .txt 标注
```

### 显存安全

GTX 1650 最低仅 4GB 显存。两段之间必须彻底释放显存，绝不能同时驻留两个模型。

### 兜底关机（云端）

AutoDL 状态机的 `finally` 块覆盖所有退出路径，无论成功、异常、还是被中断，最多重试 3 次。

## License

MIT
