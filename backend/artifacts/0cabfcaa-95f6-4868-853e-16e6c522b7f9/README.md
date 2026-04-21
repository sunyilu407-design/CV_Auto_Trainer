# 算法工程包：0cabfcaa-95f6-4868-853e-16e6c522b7f9

## 内容说明

- `pipeline.json`：编译后的算法流水线配置
- `manifest.json`：bundle 文件清单与产物映射
- `sample_input.json`：示例 observation frame 输入
- `sample_output.json`：随包 runtime 生成的示例输出
- `run_pipeline.py`：本地运行入口脚本
- `pipeline/`：随包导出的最小 runtime 实现

## 使用方式

支持以下输入形式：
- 单个 JSON 文件：兼容 `{"observation_frames": [...]}` 或直接传 frame 列表
- 单个 JSONL 文件：每行一个 frame JSON 对象
- 本地目录：按文件名顺序聚合目录内的 JSON / JSONL 文件

业务示例：仓位占用、区域持续出现、滞留检测等规则型算法。

`python run_pipeline.py --input sample_input.json --output output.json`
`python run_pipeline.py --input sample_input.jsonl --output output.json`
`python run_pipeline.py --input ./input_frames --output output.json`

## 输出说明

- `frame_count`：本次处理的 frame 数量
- `track_states`：最终跟踪状态快照
- `events`：命中的业务事件列表

## 关联产物

