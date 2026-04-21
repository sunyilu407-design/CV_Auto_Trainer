# Phase 4 Bundle Input And README Design

## 背景

Phase 4 第一轮已经把算法工程包从占位骨架推进到可运行 bundle：

- 导出 `pipeline.json`、`manifest.json`、`README.md`、`run_pipeline.py`
- 导出 `sample_input.json`、`sample_output.json`
- 随包复制最小 runtime 实现
- 集成测试会真实执行导出的 `run_pipeline.py`

当前剩余缺口主要有两个：

1. `run_pipeline.py` 只支持单个 observation frame JSON 输入，不适合业务方用本地样例快速验证。
2. bundle `README` 仍是英文、偏开发占位文案，示例不够贴近当前“仓位占用 / 区域滞留”类业务。

## 本轮目标

在不扩大 Phase 4 边界的前提下，把 bundle 提升到“更容易交付演示和本地验证”的状态。

本轮完成后，导出的 bundle 应支持：

- 读取单个 JSON 输入文件
- 读取单个 JSONL 输入文件
- 读取一个本地目录，并聚合目录内多个 JSON / JSONL 文件
- 输出中文 README，说明输入格式、运行方式和业务化示例

## 非目标

以下内容明确不在本轮范围内：

- 直接读取视频文件
- 引入新依赖
- 修改后端导出 API 结构
- 修改 `manifest.json` 主体结构
- 重构 bundle runtime 为可热更新同步机制

## 设计方案

### 方案选择

本轮采用“最小改动”方案：

- 继续由 `run_pipeline.py` 承担 bundle 入口职责
- 在入口脚本内部补充输入路径识别与 frame 装载逻辑
- 不新增额外 bundle helper 文件

这样可以保持导出结构稳定，避免在当前 Phase 4 节点引入新的 bundle 模块边界。

### 输入契约

`run_pipeline.py --input <path>` 支持三类输入：

#### 1. JSON 文件

兼容两种形态：

- `{"observation_frames": [...]}`
- `[{frame...}, {frame...}]`

读取后统一归一化为 `observation_frames` 列表。

#### 2. JSONL 文件

每行一个 frame JSON 对象。入口按行顺序读取并拼接为 `observation_frames`。

适用场景：

- 从日志或批处理脚本导出的逐帧观测结果
- 本地快速拼接短样本进行 runtime 验证

#### 3. 目录

入口按文件名排序读取目录下的：

- `*.json`
- `*.jsonl`

非递归扫描，忽略其它文件类型。每个文件解析出的 frame 结果按排序后的文件顺序拼接。

### 归一化规则

最终仍复用现有 `normalize_observation_frames()` 进行统一归一化，保证：

- 缺省 `frame_index` 时自动补齐
- 缺省 `timestamp_ms` 时自动按顺序补齐
- detection 字段结构与当前 runtime 兼容

本轮不改变 runtime 输出结构。

### README 中文化

README 模板改为中文，并明确包含：

- bundle 内容说明
- 三类输入方式说明
- 典型命令示例
- 输出结果字段说明
- 业务化样例描述，优先使用“仓位占用 / 区域持续出现事件”语境

README 目标不是覆盖所有部署细节，而是让业务方或算法同学拿到 bundle 后能在本地直接跑通示例。

## 涉及文件

- 修改 `worker/pipeline/package_exporter.py`
  - 更新 README 模板
  - 更新导出入口脚本文本
- 修改 `tests/test_integration.py`
  - 为导出 bundle 增加 JSONL 输入验证
  - 为导出 bundle 增加目录输入验证

## 测试策略

遵循现有 Phase 4 集成测试风格，在 bundle exporter 集成测试中补两轮红绿验证：

1. 新增 JSONL 输入测试
2. 运行测试，确认在实现前失败
3. 实现最小代码让 JSONL 测试通过
4. 新增目录输入测试
5. 运行测试，确认在目录支持实现前失败
6. 完成最小实现并回归 exporter 测试

最终至少回跑：

- `./.venv/bin/python tests/test_integration.py package-exporter`
- `./.venv/bin/python tests/test_integration.py algorithm-package-api`

如无额外回归成本，再补：

- `./.venv/bin/python tests/test_integration.py backend`
- `./.venv/bin/python tests/test_integration.py full`

## 风险与取舍

### 风险 1：目录输入的拼接顺序不符合外部预期

取舍：本轮统一使用“按文件名排序”这一简单规则，避免引入额外元数据协议。

### 风险 2：JSON 与 JSONL 同时混放时样本时间轴可能不连续

取舍：继续交由 `normalize_observation_frames()` 根据已有字段和默认补齐策略处理，不新增复杂时间轴合并规则。

### 风险 3：README 更贴业务后，示例可能与部分场景不完全匹配

取舍：优先让 README 可用、可理解，而不是追求场景全覆盖；后续再按真实业务案例迭代。
