# 开发计划文档

> 记录每次开发任务的进度与完成情况。每次完成一项后更新状态。

---

## 一、架构补充决策

| # | 问题 | 决策 |
|---|------|------|
| 1 | 样板图标注框数据传递 | 阶段一 VLM 解析时，Canvas 框坐标以 JSON 传给 VLM，辅助理解用户意图 |
| 2 | 样板图是否进入训练集 | 是，样板图打完标后原图+标注一并进入最终数据集 |
| 3 | WebSocket 心跳保活 | 增加 ping/pong 心跳，30s 超时判定断连，前端自动重连 |
| 4 | 云端训练方案 | 不局限于 AutoDL，做成通用 SSH 云服务器方案（阿里云/腾讯云/AWS/自有服务器均可） |
| 5 | AutoDL 专用支持 | AutoDL 降级为云端提供商之一，提供 AutoDLCloudTrainer 实现（自动创建/销毁实例） |
| 6 | 关机失败告警 | 增加钉钉/飞书/邮件告警，关机失败时自动发送通知 |
| 7 | 任务取消流程 | Worker 端支持 cancel 命令，GPU 推理可中断，SSH 训练发 SIGTERM |
| 8 | 本地训练方案 | 打标完成后释放 GPU → subprocess.Popen 启动独立训练子进程 → 显存隔离 |

---

## 二、开发任务清单

### 阶段一：项目脚手架

- [D] **1.1** 初始化前端项目（Vite + React + TypeScript + Zustand）
- [D] **1.2** 初始化后端项目（FastAPI + SQLAlchemy）
- [D] **1.3** 初始化 Worker 项目（FastAPI 独立进程）
- [D] **1.4** 目录结构创建（三端各自创建对应目录）
- [D] **1.5** 基础依赖安装并验证三端均可正常启动（依赖文件已创建，验证需本地 Python 环境）

### 阶段二：前端页面与状态机

- [D] **2.1** 任务状态管理（taskStore.ts / settingsStore.ts）
- [D] **2.2** Upload 页面（上传样板图 + 描述文本 + Canvas 画框）
- [D] **2.3** IntentConfirm 页面（展示/微调 VLM 解析结果）
- [D] **2.4** LabelingProgress 页面（WebSocket 实时进度）
- [D] **2.5** AugmentConfig 页面（增强配置 UI + 预览）
- [D] **2.6** ReviewSamples 页面（数据质量报告 + 抽样查看）
- [D] **2.7** TrainConfig 页面（模型选型 + 超参数 + **训练模式选择：本地/云端**）
- [D] **2.8** TrainingMonitor 页面（训练进度实时曲线 + WebSocket，共用，本地/云端通用）
- [D] **2.9** Delivery 页面（下载 best.pt / ONNX / 报告）
- [D] **2.10** SettingsPanel 组件（VLM / AutoDL / **本地训练 GPU** / 图片清理配置）
- [D] **2.11** GpuMonitor 组件（本地 GPU 显存实时监控）
- [D] **2.12** WebSocket 心跳机制（ping/pong + 30s 超时重连）

### 阶段三：后端 API

- [D] **3.1** 数据库模型（Task / UserSettings，SQLAlchemy，UserSettings 含通用 SSH + AutoDL 双配置）
- [D] **3.2** 任务管理 CRUD（tasks.py）
- [D] **3.3** VLM Adapter 多厂商支持（openai / kimi / gemini）
- [D] **3.4** VLM 意图解析接口（含重试逻辑 + schema 校验）
- [D] **3.5** 文件上传/下载接口（files.py）
- [D] **3.6** 用户设置接口（settings.py，含加密存储，通用 SSH / AutoDL 配置）
- [D] **3.7** 训练分发器（train_dispatcher.py，根据 train_mode 路由到本地或云端）
- [D] **3.8** 云端训练抽象基类（cloud_trainer.py，CloudTrainer ABC + for_provider 工厂方法）
- [D] **3.9** 通用 SSH 训练器（generic_ssh_trainer.py，GenericSSHCloudTrainer）
- [D] **3.10** AutoDL 专用训练器（autodl_trainer.py，AutoDLCloudTrainer，extends CloudTrainer）
- [D] **3.11** 钉钉/飞书/邮件告警管理器（alert_manager.py，关机失败时自动通知）
- [D] **3.12** 训练取消接口（POST /api/training/{task_id}/cancel）

### 阶段四：Worker 本地推理

- [D] **4.1** Worker 入口（FastAPI + WebSocket，localhost:7860）
- [D] **4.2** GPU 显存安全管理器（gpu_manager.py，gpu_stage 上下文管理器 + cancel_current_stage）
- [D] **4.3** 第一段：YOLO-World 打标（stage2_labeler.py）
- [D] **4.4** 第二段：Moondream2 VQA 质检（stage2_labeler.py）
- [D] **4.5** YOLO 标注格式读写（yolo_io.py）
- [D] **4.6** WebSocket 进度推送 + 心跳响应（ping/pong）
- [D] **4.7** 任务取消命令处理（cancel 命令 + is_cancelled 检查点）
- [D] **4.8** 数据集分层分割（dataset_splitter.py，8:1:1 分层抽样）
- [D] **4.9** 阶段二点五：数据增强（stage25_augmentor.py，Albumentations）
- [D] **4.10** 图片清理策略实现（deleteOriginalImages）

### 阶段五：本地 GPU 训练（Worker 子进程）

- [D] **5.1** 本地训练器（local_trainer.py，subprocess.Popen + 进度轮询线程）
- [D] **5.2** 本地训练 cancel 处理（SIGTERM/CTRL_BREAK_EVENT，30s 超时 kill）
- [D] **5.3** 本地训练断点续训（last.pt 检测 + --resume 参数）
- [D] **5.4** 本地训练产物收集（best.pt / last.pt / results.csv）
- [D] **5.5** 模型导出（本地：ONNX / TensorRT / CoreML / OpenVINO）
- [D] **5.6** Worker 端本地训练 WebSocket 命令处理（start_local_training / cancel）

### 阶段六：云端训练（通用 SSH + AutoDL 两种模式）

- [D] **6.1** 通用 SSH 云端训练器（阿里云/腾讯云/AWS/自有服务器均可连接）
- [D] **6.2** AutoDL 专用云端训练器（实例自动创建/销毁，用户只需提供 Token）
- [D] **6.3** 统一关机 finally 块（通用 SSH 发 shutdown 命令，AutoDL API 关机）
- [D] **6.4** 断点续训支持（last.pt 检测 + --resume 参数）
- [D] **6.5** 训练产物拉取（SFTP 拉回 best.pt / last.pt / results.csv / curves）
- [D] **6.6** 模型导出（ONNX / TensorRT / CoreML / OpenVINO）
- [D] **6.7** 关机失败告警（钉钉/飞书/邮件，通用 SSH 和 AutoDL 均适用）
- [D] **6.8** cloud_scripts/train.py（上传至云端实例）
- [D] **6.9** cloud_scripts/export.py
- [D] **6.10** cloud_scripts/health_check.py

### 阶段七：集成与测试

- [P] **7.0** 集成测试脚本（tests/test_integration.py，含后端导入/Worker 导入/显存释放/子进程测试）
- [ ] **7.1** 端到端本地训练流程测试（上传样板图 → 打标 → 增强 → 本地训练 → 下载模型）
- [ ] **7.2** 端到端云端训练流程测试（上传样板图 → 打标 → 增强 → AutoDL 训练 → 下载模型）
- [ ] **7.3** 显存泄漏测试（两段式打标记忙后显存是否正常释放）
- [ ] **7.4** 任务取消测试（中途取消，本地训练子进程和 GPU 资源是否正确释放）
- [ ] **7.5** 云端关机失败告警测试（模拟关机 API 失败，验证钉钉/飞书/邮件告警）
- [ ] **7.6** 数据增强质量验证（bbox 坐标是否正确同步变换）
- [ ] **7.7** 训练模式切换测试（本地/云端配置是否正确路由）
- [ ] **7.8** WebSocket 心跳断连重连测试

---

## 三、任务状态说明

- `[ ]` = 未开始
- `[P]` = 进行中
- `[D]` = 已完成

---

## 四、本地训练 vs 云端训练对比

| 维度 | 本地训练（阶段五） | 云端训练（阶段六） |
|------|-------------------|-------------------|
| 启动方式 | subprocess.Popen 子进程 | SSH 连接（通用）/ AutoDL API（实例自动创建） |
| 显存隔离 | 子进程独立显存空间 | 云端实例独立 GPU |
| 进度推送 | results.csv 轮询（10s间隔）→ WebSocket | SSH exec → results.csv 轮询 → WebSocket |
| 取消方式 | SIGTERM/CTRL_BREAK_EVENT | SSH 发 kill 信号（通用）/ AutoDL API shutdown |
| 产物拉取 | 直接本地文件系统 | SFTP 拉回本地 |
| 关机保活 | 进程退出即释放 | 必须 finally 关机防扣费 |
| 云端提供商 | — | 通用 SSH（任意云服务器）/ AutoDL（专用 API） |
| 适用场景 | 中小数据集，用户有足够 GPU | 超大规模数据，顶级算力需求 |

---

*文档创建：2026-04-03 | 最后更新：2026-04-03（v3：云端训练从 AutoDL 专有改为通用 SSH 云服务器方案）*
