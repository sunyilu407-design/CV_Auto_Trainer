# Single Machine Production Deployment Design

## 背景

当前系统已经具备完整的产品主流程：

- 登录、任务管理、上传、VLM 意图解析
- 算法规划、预演、确认
- 本地 Worker 负责画框、质检、增强
- 训练配置与交付页
- 算法工程包导出

集成测试已经证明这套链路在开发模式下可运行，但它还没有收口到“正式部署可用”的单机形态。当前主要问题是：

- 前端仍依赖 Vite 开发服务器
- Worker 地址和后端 CORS 仍是开发环境写死
- 云训练状态流转没有真正闭环
- 训练状态存在后端内存中，重启后会丢
- 交付页默认按固定文件名下载，未基于真实产物清单
- 缺少面向 macOS / Windows 的正式部署说明和启动脚本

## 目标

把现有系统收口成单机正式部署版本，满足以下使用方式：

- 用户在 macOS 或 Windows 单机上安装并启动服务
- 前端使用构建产物，由后端统一托管
- 本地 Worker 仅负责画框、质检、增强
- 模型训练由系统内部发起云训练（SSH / AutoDL）
- 云训练完成后，系统能稳定展示并下载模型产物与算法 bundle

## 非目标

本轮明确不做：

- 本地最终模型训练
- Docker 化部署
- Nginx / Redis / 多机分布式部署
- 视频直读推理交付
- 新增算法能力

## 方案选择

采用单机正式版方案：

- 前端静态资源由 FastAPI 托管
- Worker 仍为本机独立进程
- 数据库存储使用 PostgreSQL，保留 SQLite 兼容作为开发回退
- 云训练继续通过后端统一编排

这样可以同时兼容 macOS 与 Windows，而不会引入桌面场景下过重的基础设施。

## 核心设计

### 1. 前端生产托管

后端在配置了前端构建产物目录时：

- 直接提供 `frontend/dist` 下的静态文件
- 对非 `/api` 路由回退到 `index.html`
- 保留开发模式下现有 API 行为

目标是消除“正式部署还要额外启动 Vite”的要求。

### 2. 运行地址配置化

当前部署相关地址统一配置化：

- 后端允许通过环境变量配置 CORS 来源
- 前端 Worker WebSocket 地址通过构建时环境变量注入
- 前端 API 继续使用相对路径 `/api`

这样单机部署时既可继续使用 `localhost`，也可兼容本机域名或自定义端口。

### 3. 云训练状态闭环

训练流程拆成两条：

- 本地模式：保留前端直连 Worker 的现有行为，但不作为本轮正式部署目标
- 云端模式：前端通过后端启动训练，后端持续写入数据库状态，前端轮询真实 `task_id`

云训练状态至少包含：

- `queued`
- `running`
- `done`
- `error`
- `cancelled`

并持久化当前 epoch、总 epoch、当前 map、错误信息与训练模式。

### 4. 训练状态持久化

不引入 Redis，直接把训练状态写入 `Task` 模型字段，原因是：

- 单机场景足够
- 减少额外组件
- 可直接支撑重启后状态查看

持久化字段包括：

- `training_state`
- `training_progress`
- `training_started_at`
- `training_finished_at`

### 5. 交付页按真实产物展示

交付页不再假定固定文件一定存在，而是：

- 优先从任务的 `artifact_paths` / 文件列表读取真实产物
- 对模型权重、导出格式、训练报告和算法 bundle 分类展示
- 如果算法 bundle 尚未生成，仍允许单独触发导出

这样“训练完成后直接集成使用”才成立，因为页面反映的是实际可下载产物，而不是静态按钮。

### 6. 平台部署收口

文档和脚本覆盖两套单机平台：

- macOS：后端、Worker、PostgreSQL、前端构建与启动说明；给出 `launchd` 推荐托管方式
- Windows：PowerShell / bat 启动脚本；给出 NSSM 或任务计划程序托管说明

业务代码尽量保持一套，只在启动脚本和文档层处理平台差异。

## 涉及文件

- `backend/main.py`
- `backend/models/db.py`
- `backend/models/database.py`
- `backend/routers/training.py`
- `backend/services/train_dispatcher.py`
- `frontend/vite.config.ts`
- `frontend/src/api/worker.ts`
- `frontend/src/pages/TrainConfig.tsx`
- `frontend/src/pages/TrainingMonitor.tsx`
- `frontend/src/pages/Delivery.tsx`
- `frontend/src/store/taskStore.ts`
- `tests/test_integration.py`
- `README.md`
- `MAC.md`
- 新增 Windows 部署文档与启动脚本

## 测试策略

本轮采用集成测试优先的 TDD：

1. 先补后端训练状态持久化与云训练状态查询测试
2. 再补生产静态托管测试
3. 再补交付页依赖的产物列表行为测试
4. 实现最小代码让测试转绿
5. 最终回跑 `backend` 与 `full`

## 风险与取舍

### 风险 1：云训练适配差异较大

取舍：本轮只保证系统内发起、状态更新和产物回收闭环，不扩新的云服务商。

### 风险 2：单机正式部署仍可能受本地 Python / Node / GPU 环境影响

取舍：通过脚本与文档把部署步骤稳定下来，而不是引入 Docker。

### 风险 3：本地训练相关路径仍保留在代码里

取舍：不删除本地训练分支，但正式部署文档和默认流程以云训练为主。
