# Algorithm Pipeline Product Design

## 背景

当前项目已经具备以下能力：

- 用户上传样板图和文字描述
- VLM 将需求解析为检测类别与英文 prompt
- Worker 使用 YOLO-World 自动打框
- Worker 使用 Moondream2 进行框级质检
- 平台完成增强、训练、导出

这套能力可以稳定产出一个检测模型，但交付物仍然是 `best.pt` 为核心的训练结果，而不是用户真正要的“视觉算法”。

对于“违规停车”“仓位识别”“区域占位”“超时滞留”这类业务需求，单一检测模型并不够。用户还需要：

- 多目标跟踪
- 区域定义
- 时序约束
- 规则判定
- 统一输出事件结果

因此项目需要从“检测训练平台”升级为“视觉需求拆解与算法流水线生成平台”。

## 产品目标

`v1` 的目标不是覆盖所有 CV 能力，而是支持一类可稳定落地的组合式业务算法：

- 输入：业务描述 + 少量图片/视频示例
- 自动拆解：检测目标、跟踪对象、区域、时序条件、事件规则
- 自动补全：训练计划、推理流水线、导出配置
- 交付：配置驱动的算法工程包

## v1 边界

### 支持的能力

- 检测
- 跟踪
- 区域
- 时序
- 规则

### 暂不纳入 v1

- OCR
- 分类专用分支
- 分割
- 姿态
- ReID
- 多相机关联
- 在线托管服务

## 核心设计原则

1. 算法交付优先，不再只交付模型
2. 配置驱动优先，不生成大量不可维护的定制代码
3. 离线与实时共用同一份算法定义
4. VLM 负责需求拆解，但最终输出必须是结构化、可验证、可编辑的中间表示
5. 现有训练链路保留，作为 `detector` 节点生成器，而不是整个系统的唯一中心

## 用户流程

### 新流程

1. 用户上传样板图、示例视频片段或业务截图
2. 用户输入业务描述，例如“识别仓位是否被货箱占用，并在持续 10 秒后输出占位事件”
3. 系统调用 VLM 生成两类结果：
   - 检测训练所需类别定义
   - 业务算法草案
4. 用户在“算法规划”页面确认系统推断出的：
   - 监测对象
   - 事件定义
   - 区域依赖
   - 时间阈值
   - 输出结果
5. 平台继续执行自动标注、增强、训练
6. 平台生成算法流水线配置并导出工程包

### 工程包内容

- `pipeline.json`
- 模型权重引用信息
- 区域模板
- 运行说明
- 推理入口脚本
- 示例输入输出结构

## 系统抽象

### 1. Algorithm Plan

这是用户需求到算法流水线之间的第一层结构化产物。它描述“系统认为用户想要什么算法”。

建议结构：

- `summary`
- `scenario_type`
- `targets`
- `regions`
- `temporal_constraints`
- `events`
- `training_requirements`
- `runtime_modes`
- `confidence`

### 2. Algorithm IR

这是真正驱动训练后编排与导出的统一中间表示。`v1` 用 `pipeline.json` 表示。

建议结构：

- `metadata`
- `inputs`
- `detectors`
- `trackers`
- `regions`
- `temporal_windows`
- `rules`
- `outputs`
- `packaging`

### 3. Runtime Contract

同一份 `pipeline.json` 同时服务：

- 离线图片/视频批处理
- 实时视频流处理

运行时差异只体现在输入适配和状态保持策略，不体现在业务规则定义。

## 新增模块

### Backend

- `backend/services/algorithm_planner.py`
  - 将 `user_description + vlm_result` 转为结构化算法草案
- `backend/routers/algorithm.py`
  - 提供算法规划的生成、读取、确认接口
- `backend/schemas/algorithm_pipeline.py`
  - 定义算法规划与算法 IR 的 Pydantic 模型

### Worker

- `worker/pipeline/tracking_runtime.py`
  - 目标跟踪运行时
- `worker/pipeline/event_engine.py`
  - 区域、时序、规则的事件计算
- `worker/pipeline/package_exporter.py`
  - 生成工程包

### Frontend

- 新增 `algorithm_plan` 阶段
- 新增“算法规划”页面
- 在 Intent Confirm 与 Labeling 之间插入算法规划确认
- 页面风格保持现有 Vercel/Geist 风格，不引入全新视觉语言

## 数据模型变更

`Task` 需要补充以下字段：

- `algorithm_plan`
- `algorithm_plan_status`
- `pipeline_config`

其中：

- `algorithm_plan` 存用户确认前后的规划结果
- `algorithm_plan_status` 标识 `draft | confirmed`
- `pipeline_config` 存最终导出的流水线配置

## v1 分阶段实施

### Phase 1

把“算法规划”接入现有产品流：

- 后端生成算法草案
- 前端展示与确认
- 数据库存储规划结果

### Phase 2

把规划结果与训练链路联通：

- 从算法草案中抽取 detector 训练目标
- 补充训练需求判断

### Phase 3

实现规则运行时骨架：

- 跟踪
- 区域
- 时序
- 规则

### Phase 4

导出工程包：

- `pipeline.json`
- 运行入口
- 示例配置

## Phase 1 实施结果（2026-04-15）

### 本轮已经落地

- `algorithm_plan` 已插入 `intent_confirm -> labeling` 之间，前端具备生成、读取、预演、确认的完整页面流
- 后端已提供算法规划生成、读取、确认、预演、工程包导出接口，并把 `algorithm_plan / algorithm_plan_status / pipeline_config` 持久化到 `Task`
- `pipeline.json` 编译器已经把规划草案转成 detector / tracker / region / rule / output / packaging 结构
- 训练配置页会读取 `pipeline_config.training_recommendation`，把算法规划生成的推荐模型和导出格式展示给用户
- Worker 侧已经补入最小可验证骨架：`tracking_runtime`、`event_engine`、`package_exporter`

### 当前刻意留在边界之外的内容

- `training_recommendation` 仍是启发式结果，当前只根据 target 数量和布尔开关给出默认模型、训练模式、导出格式；它还没有成为训练调度器的强约束决策器
- `tracking_runtime` 目前是预演与规则验证使用的极简 track-state 聚合器，不是完整 ByteTrack / DeepSORT 在线跟踪实现
- `event_engine` 当前只覆盖 `region_presence_duration` 这一类规则，用于验证“区域内持续出现”的事件链路，未覆盖多规则组合、跨目标关联、复杂时序窗口
- `package_exporter` 当前导出的是 Phase 1 工程包骨架：`pipeline.json`、`manifest.json`、`README.md`、占位 `run_pipeline.py`；它还不是可直接部署到生产环境的完整运行包

### 对后续阶段的影响

- Phase 2 需要把训练需求判断从“展示推荐”推进到“约束训练编排”，让 detector 训练、追踪依赖、导出格式真正参与训练流水线决策
- Phase 3 需要把预演骨架升级为真实运行时，把跟踪状态、规则状态和流式输入接到统一 runtime contract
- Phase 4 需要把工程包从“配置骨架”升级为“可交付运行产物”，补齐模型依赖、启动脚本、示例输入输出和部署说明

## UI 约束

前端继续沿用 [vercel.DESIGN.md](/Users/ahs/work/CV_Auto_Trainer/CV_Auto_Trainer/vercel.DESIGN.md) 描述的风格：

- 黑白灰为主的静态基础色
- `#0a72ef / #de1d8d / #ff5b4f` 用于工作流阶段强调
- 阴影代替边框的卡片体系
- 紧凑标题与较大留白
- 组件新增尽量基于现有 token，而不是另起一套 UI

## v1 成功标准

- 用户能在训练前看到系统自动生成的算法规划
- 用户能确认一个完整业务算法，而不是只确认检测类别
- 平台能保留现有检测训练能力
- 新增规划阶段不破坏现有上传、打标、训练主流程
- 为后续 `pipeline.json`、规则运行时和工程包导出留出稳定接口
