# Phase 3 设计：统一规则运行时骨架

## 背景

Phase 1 和 Phase 2 已经把以下链路接通：

- 算法规划可以从业务描述生成结构化草案
- `pipeline.json` 可以从规划结果编译得到 detector / tracker / rule / output 结构
- 训练推荐已经能读取算法复杂度并输出更可解释的配置建议
- Worker 已经有 `tracking_runtime` 与 `event_engine` 的最小骨架，可用于非常粗粒度的规则预演

当前问题在于，运行时骨架还不能承接真正的 Phase 3 目标：

- `tracking_runtime` 仍按 `class_name` 聚合，不具备跨帧稳定 `track_id`
- `event_engine` 只支持单一 `region_presence_duration`
- 离线与流式输入尚未收敛到同一份 runtime contract
- 预演 API 还没有统一的 session 层来串联帧输入、跟踪状态和事件输出

因此本轮需要把“可预演的骨架”升级为“可验证的统一运行时骨架”。

## 目标

本轮只落地可验证的运行时核心，不追求完整生产级跟踪器。

Phase 3 目标：

1. 建立离线与流式共用的运行时核心
2. 以检测结果 JSON 序列作为第一验证输入
3. 支持跨帧稳定的轻量跟踪状态
4. 支持多种基础事件类型
5. 给后续真实视频帧接入保留薄适配层

## 非目标

本轮明确不做：

- 不接入真实 ByteTrack / DeepSORT / ReID
- 不实现视频解码、摄像头采集或长连接服务
- 不引入复杂轨迹重识别
- 不实现多目标关系推理
- 不做生产级事件去重、回放恢复、分布式状态同步

## 设计原则

1. 运行时统一优先：离线与流式只在输入驱动方式上不同，核心状态机一致
2. 输入标准化优先：先把 observation frame 结构固定下来
3. 轻量可替换优先：当前跟踪逻辑可以简单，但边界必须独立，后续可替换
4. 事件标准输出优先：所有规则事件走统一事件结构
5. 预演链路可复用优先：后端 preview 直接复用同一 runtime session

## 统一输入结构

首轮标准 observation frame 结构固定为：

```json
{
  "frame_index": 12,
  "timestamp_ms": 400,
  "detections": [
    {
      "class_name": "person",
      "bbox_xywhn": [0.52, 0.48, 0.18, 0.30],
      "confidence": 0.91
    }
  ]
}
```

字段约束：

- `frame_index`：整数，单调递增
- `timestamp_ms`：整数，表示该帧相对会话起点的时间戳
- `detections`：当前帧检测结果列表
- `bbox_xywhn`：归一化 `xywh`
- `confidence`：检测置信度

兼容性策略：

- 预演 API 允许继续接收当前 `sample_boxes` 输入
- 预演层把 `sample_boxes` 包装成单帧 observation，再交给 runtime session
- 后续真实图片/视频输入也只负责生成 observation frame，不直接参与规则判定

## 运行时架构

本轮运行时分成四层：

### 1. Frame Adapter

负责把不同输入来源转换为统一 observation frame。

首轮至少包含：

- JSON 序列输入适配
- 预演 `sample_boxes -> observation frame` 适配

后续真实图像/视频接入时，只新增 adapter，不改规则核心。

### 2. Tracking Runtime

负责把逐帧 detection 更新为稳定的 `track_state`。

首轮使用轻量中心点/距离匹配策略，保证：

- 同类目标在连续帧中尽量保持同一 `track_id`
- 丢失少量帧时允许短暂保留轨迹
- 跟踪状态字段稳定，供事件引擎消费

该层必须保留可替换边界，后续可以直接替换为更强的 ByteTrack 类实现。

### 3. Event Runtime

负责基于 `track_state + pipeline_config` 计算统一事件。

首轮支持以下事件类型：

- `region_enter`
- `region_exit`
- `region_presence_duration`
- `cross_region_transition`

### 4. Runtime Session

负责维护一个会话级状态：

- 当前帧索引
- 当前时间戳
- 轨迹集合
- 已发出事件的去重状态
- 上一帧与当前帧的 region 变化

离线批处理与实时流式调用都复用 `RuntimeSession.process_frame(frame)`。

## Track State 契约

统一轨迹状态至少包含：

```json
{
  "track_id": "track-1",
  "class_name": "person",
  "bbox_xywhn": [0.52, 0.48, 0.18, 0.30],
  "confidence": 0.91,
  "first_seen_frame": 12,
  "last_seen_frame": 18,
  "age_frames": 7,
  "hit_streak": 7,
  "lost_frames": 0,
  "present_duration_ms": 600,
  "regions_inside": ["zone_a"],
  "entered_region_at": {
    "zone_a": 400
  },
  "last_event_frame": {
    "rule_presence_a": 18
  }
}
```

字段说明：

- `track_id`：运行时稳定 ID
- `first_seen_frame / last_seen_frame`：轨迹出现时间边界
- `age_frames`：从首次出现到当前的生命周期长度
- `hit_streak`：连续命中帧数
- `lost_frames`：连续未命中帧数
- `present_duration_ms`：当前轨迹累计存在时长
- `regions_inside`：当前帧位于哪些区域内
- `entered_region_at`：进入某区域的时间戳
- `last_event_frame`：某规则上次触发帧号，用于去重

## 事件输出契约

所有事件统一输出以下结构：

```json
{
  "event_code": "occupancy_detected",
  "rule_id": "rule_presence_a",
  "track_id": "track-1",
  "frame_index": 18,
  "timestamp_ms": 10000,
  "region_id": "zone_a",
  "bbox_xywhn": [0.52, 0.48, 0.18, 0.30],
  "payload": {
    "duration_ms": 10000
  }
}
```

约束：

- `event_code` 与 `rule_id` 必填
- `track_id / frame_index / timestamp_ms` 必填
- `region_id` 在区域相关事件中必填
- `payload` 用于承载事件特定字段，例如持续时长、来源区域、目标区域

## 规则支持范围

### region_enter

当轨迹当前帧进入指定区域，且上一帧不在该区域时触发。

### region_exit

当轨迹上一帧在指定区域，当前帧不在该区域时触发。

### region_presence_duration

当轨迹在指定区域内停留达到阈值时触发。

要求：

- 首次达到阈值时触发
- 同一轨迹在同一区域持续停留时，不在每一帧重复发同一事件
- 离开区域后再次进入并重新达到阈值，可以重新触发

### cross_region_transition

当轨迹从 `from_region_id` 转移到 `to_region_id` 时触发。

要求：

- 先在 `from_region_id` 内，再进入 `to_region_id`
- 事件 payload 需要包含 `from_region_id` 与 `to_region_id`

## 与 pipeline.json 的关系

本轮保留现有 `pipeline.json` 结构，但对 rule 语义做扩展：

- `rule_type` 不再只限于 `region_presence_duration`
- 各规则可包含：
  - `target_class`
  - `region_id`
  - `from_region_id`
  - `to_region_id`
  - `duration_ms`
  - `event_code`

兼容策略：

- 若旧规则仍只提供 `duration_seconds`，运行时层负责兼容换算为毫秒
- 现有 `region_presence_duration` 预演能力不能被破坏

## 预演链路调整

`algorithm_preview_service` 在本轮改为：

1. 合并区域覆盖
2. 编译 `pipeline_config`
3. 把 `sample_boxes` 包装成 observation frame
4. 通过统一 `RuntimeSession` 执行
5. 返回：
   - `pipeline_config`
   - `track_states`
   - `events`

这样预演 API 和后续真实运行入口共享一套 runtime 核心。

## 测试策略

本轮必须用 JSON 帧序列驱动测试，而不是只喂单次聚合结果。

核心验证点：

1. 同一目标跨帧保持稳定 `track_id`
2. 进入区域时生成 `region_enter`
3. 离开区域时生成 `region_exit`
4. 持续停留达到阈值时生成 `region_presence_duration`
5. 从 A 区进入 B 区时生成 `cross_region_transition`
6. 预演 API 继续可用，并复用 runtime session

## 影响文件

预计本轮主要修改：

- `worker/pipeline/tracking_runtime.py`
- `worker/pipeline/event_engine.py`
- `worker/pipeline/runtime_session.py`
- `backend/services/algorithm_preview_service.py`
- `tests/test_integration.py`

## 成功标准

Phase 3 第一轮完成后，应满足：

- 统一 runtime session 可消费 observation frame 序列
- 轻量 tracking 能输出稳定 track state 契约
- 四类基础区域/时序事件可被验证
- 预演 API 走统一 runtime，而非手工拼装临时状态
- 现有 Phase 1 / Phase 2 链路不回退
