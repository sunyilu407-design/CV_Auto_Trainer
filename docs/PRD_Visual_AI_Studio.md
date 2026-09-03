# Visual AI Studio — 产品需求文档（PRD）

> **文档状态**：v1.2（修订版）
> **修订说明**：根据产品工程评审反馈修订优先级、策略和技术路线；新增 Hermes Agent 源码分析
> **编写人**：产品团队
> **版本目标**：从「CV 训练平台」升级为「工业视觉 AI 算法平台」
> **适用对象**：全栈开发、后端开发、前端开发、算法工程师
> **基准日期**：2026-07-09

---

## 修订记录

| 版本 | 日期 | 修订内容 |
|------|------|---------|
| v1.0 | 2026-07-09 | 初稿 |
| v1.1 | 2026-07-09 | 根据评审反馈修订：<br>1. 调整优先级顺序（P1a 规则引擎先于 AI 判断）<br>2. 数据闭环升为 P1（原 P2）<br>3. 规则引擎改为 YAML 配置先行策略<br>4. 部署格式收敛到 PT→ONNX→TensorRT 主链路<br>5. 低代码编排改为先集成 Node-RED<br>6. 补充卡车项目经验复用说明<br>7. 增加数据闭环（误报回流）详细设计 |
| v1.2 | 2026-07-09 | 新增 Hermes Agent 源码分析（hermes-agent/ 目录），结论：<br>1. 不能直接复用（技术栈/执行模型/进程模型不匹配）<br>2. 但架构设计高度可参考<br>3. 最终方案：Pipeline Agent 架构（PipelineProvider/StageExecutor/PipelineOrchestrator）<br>4. 详细映射表：Hermes 组件 → 本平台对应设计 |

---

## 一、产品愿景与定位

### 1.1 核心定位

**Visual AI Studio** — 面向工业视觉场景的零代码算法平台。

用户不再需要理解 YOLO、ByteTrack、ROI、MQTT 等技术概念，只需用自然语言描述业务需求（如"检测油库装车区是否有人抽烟"），平台自动完成：

```
自然语言需求 → 算法选型（AI判断是否需要训练）→ 规则/模型配置 → 流程编排 → 边缘部署 → 系统集成
```

### 1.2 目标用户

| 用户类型 | 核心诉求 | 关键痛点 |
|---------|---------|---------|
| 工厂 IT / 设备工程师 | 快速上线视觉算法 | 不懂深度学习，不想写代码 |
| 系统集成商 | 一套方案交付多个工厂 | 每家工厂场景不同，改造成本高 |
| 算法工程师 | 快速验证想法 | 不想从零搭环境、绑数据 |
| 一人公司 / 小团队 | 低成本搞定视觉需求 | 没有 GPU 服务器，没有 AI 团队 |

### 1.3 差异化定位

| 维度 | 传统训练平台 | 本平台 |
|------|------------|--------|
| 是否需要训练 | 必须训练 | **可选**——规则算法免训练直接用 |
| AI 判断 | 无 | **智能判断：能免训练就免训练** |
| 交付物 | 一个 .pt 文件 | **完整算法包**：规则配置 + 模型 + 流程 + 集成配置 |
| 输出方式 | API 调用 | **MQTT/HTTP** 直连 MES/SCADA/PLC |
| 算法组合 | 单模型 | **多模型 + 跟踪 + 规则** 串联 |
| 持续运营 | 无 | **误报回流 → 增量训练** 数据闭环 |

---

## 二、现有能力盘点

### 2.1 已有且完善的功能

```
✅ 完整训练流程：样板图 → VLM意图解析 → YOLO-World打标 → Moondream VQA质检
                  → 数据增强 → 训练配置 → 本地/云端训练 → 模型交付

✅ 模型仓库（Model Registry）
   - 内置 30+ 预训练模型：YOLOv5~11、RT-DETR、SAM-2、RF-DETR、OCR
   - 支持按设备等级筛选

✅ 算法规划（Algorithm Planner）
   - VLM 驱动生成算法方案（含事件、区域、规则）
   - 方案协商对话（Negotiation Agent）
   - 方案回滚（revision_snapshots）
   - 已有"是否需要训练"的判断逻辑，但未作为前端核心卖点暴露

✅ Pipeline 编译
   - 自动编译为 detectors / trackers / regions / rules / outputs 结构

✅ 多种导出格式
   - best.pt / best.onnx / best.engine / best.coreml / best.openvino

✅ 流式推理运行时
   - frame_adapter.py、event_engine.py、tracking_runtime.py

✅ 增量训练（Snowball）
   - 种子训练 + 自动标注 + 合并数据集 + 增量微调

✅ 设备适配（7档）
   - Jetson Nano → Orin → 桌面GPU → Mac → 云端GPU

✅ 质量门控
   - 视频离线验证 + 训练评分 + 4条优化路径

✅ 卡车项目经验复用
   - Modbus/MQTT 集成踩坑经验
   - 多目标状态机设计（可复用为通用规则引擎）

✅ Hermes Agent 架构参考
   - `hermes-agent/` 源码已纳入项目
   - 分析结论：不能直接复用，但 Pipeline Agent 架构高度可参考
   - 核心可迁移设计：PipelineProvider / StageExecutor / SQLite FTS5 持久化
```

### 2.2 已有但不完善的功能

```
⚠️  规则引擎
    - 后端：pipeline_compiler.py 生成了 rule 结构
    - 前端：无配置 UI，无 YAML 配置生成
    - 缺失：并发状态管理、时序窗口、状态持久化

⚠️  ROI 标注
    - AnnotationCanvas 只有目标检测框
    - 无多边形 ROI 绘制能力

⚠️  事件输出
    - event_engine.py 生成事件
    - 无统一 Schema（各算法输出格式不一致）
    - 无 MQTT/HTTP Adapter 层

⚠️  Pipeline 可视化
    - 无流程图编辑器

⚠️  数据闭环
    - 无误报标记 UI
    - 无 badcase 自动收集流程
```

### 2.3 完全缺失的功能

```
🔴 算法市场（Algorithm Hub）
🔴 AI 判断"是否需要训练"（需升级为前端核心交互）
🔴 规则引擎（YAML 配置版）
🔴 数据闭环（误报回流 → 增量训练）
🔴 统一事件总线（Output Adapter）
🔴 ROI 标注增强（多边形绘制）
🔴 行业算法模板
🔴 边缘部署中心（Docker 部署包）
🔴 低代码编排（先集成 Node-RED）
🔴 AI Agent 多角色协作
🔴 AI 运维中心
```

---

## 三、修订后的优先级策略

### 3.1 修订说明

本次修订基于以下核心原则：

1. **集中资源做 P0**：不在八个中心同时开工，先跑通商业价值最高的路径
2. **规则引擎先于 AI 判断**：AI 判断"不需要训练"之后，必须有规则引擎落地，否则闭环断掉
3. **数据闭环升为 P1**：持续运营能力是长期护城河，技术不复杂但战略价值极高
4. **借助现成工具**：Node-RED 集成优于自研，TensorRT 主链路优于六格式全覆盖
5. **复用已有积累**：卡车项目状态机可直接泛化为规则引擎

### 3.2 优先级矩阵

| 优先级 | 模块 | 工作量 | 商业价值 | 依赖关系 |
|--------|------|--------|---------|---------|
| **P0a** | 算法市场 | 中 | ★★★★★ | 无 |
| **P0b** | 统一事件 Schema + MQTT/HTTP 输出 | 中 | ★★★★★ | 无（但后续所有功能依赖它） |
| **P1a** | 规则引擎（YAML 配置版） | 高 | ★★★★☆ | 依赖 P0b |
| **P1b** | 数据闭环（误报回流 → 增量训练） | 低 | ★★★★☆ | 依赖训练流程（已有） |
| **P1c** | AI 需求分析 + 是否训练判断 | 中 | ★★★☆☆ | 依赖 P1a |
| **P2a** | 边缘部署中心（Docker + x86 先行） | 中 | ★★★☆☆ | 依赖 P0b + P1a |
| **P2b** | ROI 标注增强 | 低 | ★★★☆☆ | 独立 |
| **P3a** | 低代码编排（先集成 Node-RED） | 高 | ★★★☆☆ | 依赖 P1a |
| **P3b** | AI Agent 多角色协作 | 中 | ★★★☆☆ | 依赖 Hermes 框架确认 |
| **P3c** | AI 运维中心 | 中 | ★★☆☆☆ | 依赖 P2a |

### 3.3 P0 核心路径（必须先跑通）

```
用户打开平台
    │
    ├── 进入算法市场（P0a）
    │       │
    │       └── 选择算法 → 直接部署
    │                   │
    │                   ├── 算法运行 → 产生事件
    │                   │           │
    │                   │           └── 统一事件 Schema（P0b）
    │                   │                   │
    │                   │                   └── MQTT/HTTP 输出到 MES/SCADA
    │                   │
    │                   └── 用户看到效果 ✓（P0 里程碑达成）
    │
    └── 进入训练流程（原有）
            │
            └── 训练完成 → Delivery → 部署
```

P0 跑通后，用户立刻能用上平台的核心价值，不需要理解任何技术细节。

---

## 四、功能需求详述

### 4.0 算法市场 ⭐ P0a（最高优先级）

#### 4.0.1 定位说明

**最快的商业价值路径**。现有 Model Registry + 训练流程 + 事件引擎已有积累，只需封装成可浏览/可部署的"算法商品"，不需要任何训练成本，用户 3 分钟内看到效果。

#### 4.0.2 算法分类

```python
# backend/services/algorithm_hub_data.py

ALGORITHM_CATALOG = {
    # ── 需要训练的检测算法 ─────────────────────────────────────
    "detection": {
        "helmet_detection": {
            "name": "安全帽检测",
            "needs_training": True,
            "description": "检测作业人员是否佩戴安全帽，支持红/黄/白/蓝多种颜色",
            "model": "yolo11s.pt",
            "classes": ["helmet", "no_helmet"],
            "fps_hint": 25,
            "needs_gpu": True,
        },
        "fire_detection": {
            "name": "火焰检测",
            "needs_training": True,
            "model": "yolo11s.pt",
            "classes": ["fire"],
            "fps_hint": 30,
            "needs_gpu": True,
        },
        "smoke_detection": {
            "name": "烟雾检测",
            "needs_training": True,
            "model": "yolo11s.pt",
            "classes": ["smoke"],
            "fps_hint": 28,
            "needs_gpu": True,
        },
        "person_detection": {
            "name": "人员检测",
            "needs_training": False,        # YOLO11n 预训练模型直接可用
            "model": "yolo11n.pt",
            "classes": ["person"],
            "fps_hint": 40,
            "needs_gpu": True,
        },
        "vehicle_detection": {
            "name": "车辆检测",
            "needs_training": False,
            "model": "yolo11s.pt",
            "classes": ["car", "truck", "bus"],
            "fps_hint": 35,
            "needs_gpu": True,
        },
    },
    # ── 不需要训练的规则算法 ─────────────────────────────────
    "rule": {
        "region_intrusion": {
            "name": "区域入侵检测",
            "needs_training": False,
            "description": "目标进入指定 ROI 区域时触发告警",
            "requires": ["person_detection"],
            "config_fields": ["roi_id", "alarm_level", "track_required"],
        },
        "line_crossing": {
            "name": "越线检测",
            "needs_training": False,
            "description": "目标穿越指定界线时触发告警",
            "requires": ["person_detection"],
            "config_fields": ["line_start", "line_end", "direction", "alarm_level"],
        },
        "dwell_detection": {
            "name": "区域停留检测",
            "needs_training": False,
            "description": "目标在指定区域停留超过设定时长时触发告警",
            "requires": ["person_detection"],
            "config_fields": ["roi_id", "duration_sec", "alarm_level"],
        },
        "crowd_gathering": {
            "name": "人员聚集检测",
            "needs_training": False,
            "description": "区域内人数超过阈值时触发告警",
            "requires": ["person_detection"],
            "config_fields": ["roi_id", "threshold", "alarm_level"],
        },
        "counting": {
            "name": "人数统计",
            "needs_training": False,
            "description": "统计进出区域的人数",
            "requires": ["person_detection"],
            "config_fields": ["roi_id", "count_type"],
        },
        "color_classification": {
            "name": "颜色分类",
            "needs_training": False,
            "description": "对检测框内区域做 HSV 颜色分类（红/黄/蓝/白）",
            "config_fields": ["target_class", "colors"],
        },
    },
    # ── 不需要训练的 OCR 算法 ────────────────────────────────
    "ocr": {
        "gauge_reading": {
            "name": "仪表读数识别",
            "needs_training": False,
            "engine": "PaddleOCR",
            "description": "识别指针式/数字式仪表读数，直接使用 PaddleOCR",
            "needs_gpu": True,
        },
        "text_recognition": {
            "name": "通用文字识别",
            "needs_training": False,
            "engine": "PaddleOCR",
            "description": "通用文字识别，支持多语言",
            "needs_gpu": True,
        },
    },
}
```

#### 4.0.3 算法市场前端页面

**路由**：`/algorithm-hub`

```
┌──────────────────────────────────────────────────────────────────────┐
│  Visual AI Studio  │  🔍 算法市场  │  我的任务  │  训练记录  │  ⚙️  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🔍 搜索算法...                   [全部] [检测] [规则] [OCR] [分类]    │
│                                                                      │
│  ━━━━━━━━━━━━━━━━━━━━ 需要训练的算法 ━━━━━━━━━━━━━━━━━━━━━━          │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ 🪖 安全帽检测  │  │ 🔥 火焰检测    │  │ 🚬 烟雾检测    │      │
│  │                 │  │                │  │                 │      │
│  │ 需要训练         │  │ 需要训练       │  │ 需要训练        │      │
│  │ YOLO11 │ 25FPS │  │ YOLO11 │ 30FPS │  │ YOLO11 │ 28FPS │      │
│  │ 需GPU │ 9.4MB  │  │ 需GPU │ 9.4MB │  │ 需GPU │ 9.4MB │      │
│  │                 │  │                │  │                 │      │
│  │ [ 查看详情 ]     │  │ [ 查看详情 ]    │  │ [ 查看详情 ]    │      │
│  │ [ 🚀 训练新模型 ]│  │ [ 🚀 训练新模型 ]│  │ [ 🚀 训练新模型 ]│      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│                                                                      │
│  ━━━━━━━━━━━━━━━━━━━━ 免训练算法 ⭐ ━━━━━━━━━━━━━━━━━━━━━          │
│                                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ ⛔ 区域入侵     │  │ ➡️ 越线检测    │  │ ⏱️ 停留检测    │      │
│  │                 │  │                │  │                 │      │
│  │ 规则算法 ✨     │  │ 规则算法 ✨    │  │ 规则算法 ✨     │      │
│  │ 无需训练       │  │ 无需训练       │  │ 无需训练        │      │
│  │ CPU 即可       │  │ CPU 即可       │  │ CPU 即可        │      │
│  │                 │  │                │  │                 │      │
│  │ [ 查看详情 ]    │  │ [ 查看详情 ]    │  │ [ 查看详情 ]    │      │
│  │ [ 🚀 直接部署 ] │  │ [ 🚀 直接部署 ] │  │ [ 🚀 直接部署 ] │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4.0.4 直接部署流程

```
用户点击「🚀 直接部署」

    ↓

Step 1：选择视频源
    [ webcam ] [ 上传视频 ] [ RTSP 流 ]

    ↓

Step 2：配置检测区域（ROI 标注）
    AnnotationCanvas 打开，用户绘制多边形 ROI
    [ 跳过（全图检测）]

    ↓

Step 3：配置规则参数（如停留检测）
    停留时长（秒）：[ 30 ]
    告警级别：[ warning ] [ critical ]

    ↓

Step 4：配置输出集成
    ☑ MQTT（Broker: ___________ Topic: ___________）
    ☐ HTTP Push（URL: ___________）

    ↓

Step 5：预览运行
    [ 📺 打开实时预览 ]

    ↓

Step 6：生成部署包
    [ 📦 下载 Docker 包 ]
    [ 🚀 推送到边缘盒子 ]
```

---

### 4.1 统一事件 Schema + Output Adapter ⭐ P0b

#### 4.1.1 定位说明

**这是整个平台的"出口"**。无论用户用算法市场的免训练算法，还是自己训练了模型，所有算法输出必须通过统一事件 Schema + Output Adapter 分发到企业系统。这个模块是后续所有功能（P1 规则引擎、P2 边缘部署、P3 运维）的基础。

#### 4.1.2 统一事件 Schema

```python
# worker/pipeline/event_engine.py

@dataclass
class UnifiedEvent:
    """所有算法输出的统一事件格式"""
    device_id: str
    model_name: str
    timestamp: str                      # ISO 8601

    event: EventBody
    objects: List[DetectedObject]
    metadata: EventMetadata

@dataclass
class EventBody:
    event_id: str                      # UUID
    type: str                           # "helmet_alarm" / "region_intrusion" / "dwell_violation"
    level: str                         # "info" / "warning" / "critical"
    score: float                       # 置信度 0.0~1.0
    frame_id: int
    roi_id: Optional[str]             # 触发 ROI ID
    track_id: Optional[int]            # ByteTrack 跟踪 ID

@dataclass
class DetectedObject:
    id: int
    class_name: str                    # "person" / "helmet" / "fire"
    track_id: Optional[int]
    confidence: float
    bbox: Tuple[float, float, float, float]  # xywh 归一化

@dataclass
class EventMetadata:
    camera_url: Optional[str]
    pipeline_version: str
    inference_time_ms: float
    fps: float
    gpu_memory_mb: Optional[float]
```

**JSON 示例**：

```json
{
  "deviceId": "camera_01",
  "modelName": "helmet_detection",
  "timestamp": "2026-07-09T10:11:22Z",
  "event": {
    "eventId": "uuid-xxx",
    "type": "no_helmet_alarm",
    "level": "critical",
    "score": 0.91,
    "frameId": 1234,
    "roiId": "ROI-01",
    "trackId": 7
  },
  "objects": [
    {
      "id": 1,
      "className": "person",
      "trackId": 7,
      "confidence": 0.98,
      "bbox": [0.45, 0.32, 0.12, 0.55]
    },
    {
      "id": 2,
      "className": "no_helmet",
      "trackId": 7,
      "confidence": 0.91,
      "bbox": [0.48, 0.25, 0.08, 0.10]
    }
  ],
  "metadata": {
    "cameraUrl": "rtsp://192.168.1.100:554/stream",
    "pipelineVersion": "v1.0.0",
    "inferenceTimeMs": 38.5,
    "fps": 25.0,
    "gpuMemoryMb": 1204.0
  }
}
```

#### 4.1.3 Output Adapter 架构

```
事件生成（event_engine.py）
         │
         ▼
┌─────────────────────────┐
│    Event Router         │  ← 统一分发层
│  event → adapters[]     │
└───────────┬─────────────┘
            │
    ┌───────┼────────┬─────────────┐
    ▼       ▼         ▼             ▼
┌──────┐ ┌──────┐ ┌────────┐ ┌──────────┐
│ MQTT │ │ HTTP │ │WebSocket│ │ Kafka    │
│Adapter│ │Adapter│ │ Adapter │ │(Phase 3)│
└──┬───┘ └──┬──┘ └───┬────┘ └────┬─────┘
   │        │         │            │
   ▼        ▼         ▼            ▼
 MES    企业API   前端Dashboard  数据湖
 SCADA  数据库   监控大屏
```

#### 4.1.4 MQTT Adapter 实现

```python
# worker/pipeline/output_adapters/mqtt_adapter.py

import json
import paho.mqtt.client as mqtt
from typing import List
from .base_adapter import BaseOutputAdapter

class MQTTAdapter(BaseOutputAdapter):
    def __init__(self, config: dict):
        self.broker = config.get("broker", "localhost")
        self.port = config.get("port", 1883)
        self.topic_template = config.get("topic", "vision/{model}/{device_id}")
        self.qos = config.get("qos", 1)
        self._client = mqtt.Client()
        self._client.connect(self.broker, self.port, keepalive=60)

    def publish(self, event: UnifiedEvent) -> None:
        topic = self.topic_template.format(
            model=event.model_name,
            device_id=event.device_id
        )
        payload = json.dumps(asdict(event), ensure_ascii=False)
        self._client.publish(topic, payload, qos=self.qos)
```

#### 4.1.5 HTTP Adapter 实现

```python
# worker/pipeline/output_adapters/http_adapter.py

import httpx
from .base_adapter import BaseOutputAdapter

class HTTPAdapter(BaseOutputAdapter):
    def __init__(self, config: dict):
        self.endpoint = config.get("endpoint")
        self.method = config.get("method", "POST")
        self.auth = config.get("auth", "none")
        self.auth_token = config.get("auth_token", "")
        self._client = httpx.AsyncClient(timeout=10.0)

    async def send(self, event: UnifiedEvent) -> None:
        headers = {"Content-Type": "application/json"}
        if self.auth == "bearer":
            headers["Authorization"] = f"Bearer {self.auth_token}"
        elif self.auth == "basic":
            headers["Authorization"] = self.auth_token

        await self._client.request(
            method=self.method,
            url=self.endpoint,
            json=asdict(event),
            headers=headers
        )
```

#### 4.1.6 Event Router 实现

```python
# worker/pipeline/event_router.py

class EventRouter:
    """
    统一事件分发路由。
    根据配置将事件同时分发给多个 Adapter。
    """
    def __init__(self, adapter_configs: List[dict]):
        self._adapters: List[BaseOutputAdapter] = []
        for cfg in adapter_configs:
            adapter_type = cfg.get("type")
            if adapter_type == "mqtt":
                self._adapters.append(MQTTAdapter(cfg))
            elif adapter_type == "http":
                self._adapters.append(HTTPAdapter(cfg))
            elif adapter_type == "websocket":
                self._adapters.append(WebSocketAdapter(cfg))

    def emit(self, event: UnifiedEvent) -> None:
        """将事件分发给所有已注册的 Adapter"""
        for adapter in self._adapters:
            try:
                adapter.publish(event)
            except Exception as e:
                logger.warning(f"Adapter {type(adapter).__name__} failed: {e}")

    def close_all(self) -> None:
        for adapter in self._adapters:
            try:
                adapter.close()
            except Exception:
                pass
```

---

### 4.2 规则引擎（YAML 配置版）⭐ P1a

#### 4.2.1 定位说明

**这是 P0 路径的落地出口，也是整个平台工程复杂度最高的模块**。

> ⚠️ **重要工程提示**：规则引擎的实现难度被严重低估。它不是一个 CRUD 页面，而是一个**并发有限状态机**。具体可参考卡车项目中每个装卸臂独立状态机 + 臂间联动的设计——将其泛化即可。

#### 4.2.2 为什么不能一上来做拖拽 UI

参考卡车项目的经验，一个生产级规则引擎必须解决：

| 问题 | 描述 | 卡车项目中的体现 |
|------|------|----------------|
| 并发状态 | 10 个目标同时在 5 个 ROI 里，每个独立计时，离开清零 | 每个装卸臂独立状态 |
| 时序窗口 | "连续 5 帧"需要帧缓存 + 窗口管理 | 臂间协调需要等待超时 |
| 状态持久化 | 掉电重启后不能丢失"已盯了 28 秒"的状态 | 无（边缘场景可接受不持久化） |
| 抑制重复报警 | 同一事件 10 秒内不能重复报 | 臂间互斥锁 |
| 多路并发 | 多个摄像头同时跑，每路独立状态 | 多臂并发 |

**建议策略**：YAML 配置先行，验证稳定后再逐步增加可视化 UI 层。

#### 4.2.3 规则原语定义

```yaml
# config/rules/helmet_monitoring.yaml

task_name: helmet_monitoring
version: "1.0"

# ── 依赖的检测算法 ──────────────────────────────────────────
detectors:
  - id: person_detector
    type: yolo
    model: yolo11n.pt
    classes: [person]
    conf_threshold: 0.35

  - id: helmet_detector
    type: yolo
    model: best.onnx
    classes: [helmet, no_helmet]
    conf_threshold: 0.40

# ── 跟踪器 ─────────────────────────────────────────────────
tracker:
  type: bytetrack
  source_detector: person_detector
  track_thresh: 0.5
  track_buffer: 30

# ── ROI 区域定义 ──────────────────────────────────────────
regions:
  - id: ROI-01
    name: 装车作业区
    type: polygon
    points: [[0.2, 0.3], [0.8, 0.3], [0.8, 0.7], [0.2, 0.7]]
    color: "#ff0000"

  - id: ROI-02
    name: 安全通道
    type: polygon
    points: [[0.0, 0.0], [0.3, 0.0], [0.3, 0.2], [0.0, 0.2]]
    color: "#00ff00"

# ── 规则定义 ──────────────────────────────────────────────
rules:
  # 规则 1：装车区无人佩戴安全帽
  - id: rule_no_helmet_in_loading_area
    type: composite
    description: "装车作业区内如有人员，务必佩戴安全帽"
    trigger:
      detector: person_detector
      roi: ROI-01
      condition: inside        # 人员进入 ROI-01

    conditions:
      - type: associated_class_count
        detector: helmet_detector
        track_id_ref: trigger.track_id  # 与触发目标关联（同 track_id）
        class_name: helmet
        operator: eq
        value: 0                # 没有佩戴安全帽

    action:
      event_type: no_helmet_violation
      alarm_level: critical
      suppress_seconds: 30    # 30 秒内不重复报警

  # 规则 2：安全通道越线
  - id: rule_unauthorized_access
    type: region_enter
    description: "人员进入安全通道"
    trigger:
      detector: person_detector
      roi: ROI-02
      condition: inside

    action:
      event_type: unauthorized_access
      alarm_level: warning
      suppress_seconds: 60

  # 规则 3：装车区停留超时
  - id: rule_loading_area_dwell
    type: dwell
    description: "人员在装车区停留超过 5 分钟"
    trigger:
      detector: person_detector
      roi: ROI-01
      condition: inside

    temporal:
      duration_seconds: 300    # 5 分钟
      check_interval_frames: 30  # 每 30 帧检查一次

    action:
      event_type: dwell_timeout
      alarm_level: info
      suppress_seconds: 600

# ── 输出配置 ──────────────────────────────────────────────
outputs:
  mqtt:
    enabled: true
    broker: ${MQTT_BROKER}
    port: 1883
    topic: "vision/{task_name}/{device_id}"
    qos: 1

  http:
    enabled: true
    endpoint: ${HTTP_ENDPOINT}
    method: POST

  websocket:
    enabled: true
    port: 7860
```

#### 4.2.4 规则引擎核心实现

```python
# worker/pipeline/rule_engine.py

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict
import threading

@dataclass
class TrackState:
    """单个跟踪目标的状态"""
    track_id: int
    class_name: str
    roi_states: Dict[str, "ROIState"] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

@dataclass
class ROIState:
    """单个 ROI 区域内的目标状态"""
    track_id: int
    roi_id: str
    entered_at: float           # 进入时间
    is_currently_inside: bool = True
    dwell_started_at: Optional[float] = None

class RuleEngine:
    """
    通用规则引擎。
    基于 YAML 配置驱动，处理多目标并发状态。

    设计要点（参考卡车项目状态机）：
    - 每个 track_id 独立维护 TrackState
    - 每个 ROI 区域内独立计时，离开清除状态
    - 事件抑制：suppress_seconds 防止重复报警
    - 线程安全：状态修改需加锁
    """

    def __init__(self, rules_config: dict):
        self.rules = rules_config.get("rules", [])
        self.regions = {r["id"]: r for r in rules_config.get("regions", [])}
        self._track_states: Dict[int, TrackState] = {}
        self._suppression_cache: Dict[str, float] = {}  # rule_id -> last_fired
        self._lock = threading.RLock()

    def process_frame(
        self,
        tracked_objects: List[TrackedObject],
        timestamp: float
    ) -> List[RuleEvent]:
        """
        每帧处理入口。

        Args:
            tracked_objects: ByteTrack 输出的跟踪目标列表
            timestamp: 当前帧时间戳

        Returns:
            触发的事件列表
        """
        events = []
        active_track_ids: Set[int] = set()

        with self._lock:
            # 1. 更新跟踪目标状态
            for obj in tracked_objects:
                active_track_ids.add(obj.track_id)
                self._update_track_state(obj, timestamp)

            # 2. 清理消失的目标
            self._cleanup_disappeared_tracked(timestamp)

            # 3. 逐条规则评估
            for rule in self.rules:
                event = self._evaluate_rule(rule, timestamp)
                if event:
                    events.append(event)

        return events

    def _update_track_state(self, obj: TrackedObject, timestamp: float):
        """更新单个目标的跟踪状态"""
        if obj.track_id not in self._track_states:
            self._track_states[obj.track_id] = TrackState(
                track_id=obj.track_id,
                class_name=obj.class_name,
            )

        state = self._track_states[obj.track_id]
        state.last_seen = timestamp

        # 更新 ROI 状态
        for roi_id, roi_def in self.regions.items():
            is_inside = self._point_in_polygon(obj.bbox_center, roi_def["points"])

            if roi_id not in state.roi_states:
                state.roi_states[roi_id] = ROIState(
                    track_id=obj.track_id,
                    roi_id=roi_id,
                    entered_at=timestamp,
                )

            roi_state = state.roi_states[roi_id]
            was_inside = roi_state.is_currently_inside
            roi_state.is_currently_inside = is_inside

            # 进入 ROI
            if is_inside and not was_inside:
                roi_state.entered_at = timestamp
                roi_state.dwell_started_at = None

            # 离开 ROI：清除停留计时
            if not is_inside and was_inside:
                roi_state.dwell_started_at = None

    def _evaluate_rule(self, rule: dict, timestamp: float) -> Optional[RuleEvent]:
        """评估单条规则"""
        rule_id = rule["id"]
        rule_type = rule["type"]

        # 抑制检查
        suppress_seconds = rule.get("action", {}).get("suppress_seconds", 0)
        if self._is_suppressed(rule_id, suppress_seconds, timestamp):
            return None

        # 根据规则类型评估
        if rule_type == "composite":
            event = self._evaluate_composite_rule(rule, timestamp)
        elif rule_type == "region_enter":
            event = self._evaluate_region_enter_rule(rule, timestamp)
        elif rule_type == "dwell":
            event = self._evaluate_dwell_rule(rule, timestamp)
        else:
            return None

        if event:
            self._suppression_cache[rule_id] = timestamp
            return event

        return None

    def _evaluate_dwell_rule(
        self, rule: dict, timestamp: float
    ) -> Optional[RuleEvent]:
        """
        评估停留规则。
        核心逻辑：
        1. 找到 ROI 内的目标
        2. 检查是否超过设定的停留时长
        3. 仅首次超时时触发
        """
        trigger = rule.get("trigger", {})
        roi_id = trigger.get("roi")
        duration = rule.get("temporal", {}).get("duration_seconds", 30)

        with self._lock:
            for track_id, state in self._track_states.items():
                roi_state = state.roi_states.get(roi_id)
                if not roi_state or not roi_state.is_currently_inside:
                    continue

                elapsed = timestamp - roi_state.entered_at
                if elapsed >= duration:
                    return RuleEvent(
                        rule_id=rule["id"],
                        event_type=rule["action"]["event_type"],
                        alarm_level=rule["action"]["alarm_level"],
                        track_id=track_id,
                        roi_id=roi_id,
                        duration_seconds=elapsed,
                    )

        return None

    def _is_suppressed(
        self, rule_id: str, suppress_seconds: float, timestamp: float
    ) -> bool:
        """检查是否在抑制期内"""
        if suppress_seconds <= 0:
            return False
        last_fired = self._suppression_cache.get(rule_id, 0)
        return (timestamp - last_fired) < suppress_seconds
```

#### 4.2.5 规则 Builder UI（Phase 2+）

YAML 配置稳定后，逐步增加可视化 UI 层：

```
┌──────────────────────────────────────────────────────────────┐
│  规则配置                                                    │
│                                                              │
│  ┌─ 规则 1：装车区安全帽检测 ─────────────────────────┐  │
│  │                                                       │  │
│  │  触发条件：                                           │  │
│  │  ├─ 检测器：[person_detector ▼]                    │  │
│  │  ├─ 条件：☑ 进入 ROI  ☑ 离开 ROI ☑ 停留          │  │
│  │  └─ ROI 区域：[ROI-01: 装车作业区 ▼]              │  │
│  │                                                       │  │
│  │  伴随条件：                                           │  │
│  │  ├─ 检测器：[helmet_detector ▼]                   │  │
│  │  ├─ 类别：[no_helmet ▼]                         │  │
│  │  └─ 与触发目标关联：☑ 同 track_id                │  │
│  │                                                       │  │
│  │  动作：                                               │  │
│  │  ├─ 告警级别：[critical ▼]                       │  │
│  │  └─ 抑制时长：30 秒                               │  │
│  │                                                       │  │
│  │  [ 删除规则 ]                                       │  │
│  └────────────────────────────────────────────────────┘  │
│                                                              │
│  [ + 新增规则 ]                                             │
│                                                              │
│  ┌─ ROI 区域配置 ─────────────────────────────────────┐  │
│  │  ROI-01: 装车作业区  [ 📐 在图上编辑 ] [ 删除 ]    │  │
│  │  ROI-02: 安全通道    [ 📐 在图上编辑 ] [ 删除 ]    │  │
│  │  [ + 新增区域 ]                                   │  │
│  └────────────────────────────────────────────────────┘  │
│                                                              │
│  [ 预览 YAML ]  [ 保存规则 ]                                │
└──────────────────────────────────────────────────────────────┘
```

---

### 4.3 数据闭环（误报回流 → 增量训练）⭐ P1b

#### 4.3.1 定位说明

**长期护城河，最容易被低估的能力**。一个能"越用越准"的系统，客户续费和扩场景的意愿会显著增强。技术实现不复杂，复用现有 `incremental_merger.py` + `local_trainer.py` 即可。

#### 4.3.2 数据闭环流程

```
误报事件出现
     │
     ├── 用户在告警列表看到这条事件
     │       │
     │       └── 点击 [❌ 误报]
     │               │
     │               └── 弹窗备注（可选）
     │                       │
     │                       ▼
     │               自动收集：
     │               - 当前帧截图
     │               - 检测框坐标（xywh）
     │               - 事件类型
     │               - 用户备注
     │                       │
     │                       ▼
     │               存入 badcase_pool/
     │               ├── images/
     │               └── metadata.jsonl
     │                       │
     │                       ▼
     │               达到阈值（默认 50 张）
     │                       │
     │                       ▼
     │               自动触发增量训练流程
     │               ├── 用当前 best.pt 做预训练权重
     │               ├── 合并 badcase_pool + 历史数据集
     │               ├── 增量微调（epochs=30, lr=0.005）
     │               └── 生成新版本 best_v2.pt
     │                       │
     │                       ▼
     │               用户在 Delivery 页面看到新版本
     │                       │
     │                       ▼
     │               [ 🔄 热更新到运行中的 Pipeline]
```

#### 4.3.3 误报标记前端实现

```typescript
// frontend/src/components/FalseAlarmMarker.tsx

interface FalseAlarmEvent {
  eventId: string
  timestamp: string
  eventType: string
  alarmLevel: string
  frameUrl: string        // 当前帧截图 URL
  bbox: [number, number, number, number]  // 归一化坐标
  trackId?: number
  userNote?: string
}

const FalseAlarmMarker: React.FC<{
  event: FalseAlarmEvent
  onMarked: () => void
}> = ({ event, onMarked }) => {
  const [note, setNote] = useState("")
  const [loading, setLoading] = useState(false)

  const handleMark = async () => {
    setLoading(true)
    try {
      await backend.post(`/api/feedback/false-alarm`, {
        event_id: event.eventId,
        frame_url: event.frameUrl,
        bbox: event.bbox,
        track_id: event.trackId,
        user_note: note,
        timestamp: event.timestamp,
      })
      onMarked()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="false-alarm-marker">
      <img src={event.frameUrl} className="alarm-frame" />
      <div className="marker-form">
        <div className="event-info">
          <span className={`level ${event.alarmLevel}`}>
            {event.eventType}
          </span>
          <span className="time">{event.timestamp}</span>
        </div>
        <textarea
          placeholder="备注（可选）：为什么是误报？"
          value={note}
          onChange={e => setNote(e.target.value)}
        />
        <div className="actions">
          <button
            onClick={handleMark}
            disabled={loading}
            className="mark-false-btn"
          >
            {loading ? "提交中..." : "✅ 标记为误报"}
          </button>
          <button onClick={() => history.back()}>取消</button>
        </div>
      </div>
    </div>
  )
}
```

#### 4.3.4 后端误报收集服务

```python
# backend/services/false_alarm_collector.py

import json
import time
from pathlib import Path
from datetime import datetime

class FalseAlarmCollector:
    """
    误报收集器。
    将误报事件收集到 badcase_pool，供后续增量训练使用。
    """

    def __init__(self, task_id: str, pool_dir: Optional[str] = None):
        self.task_id = task_id
        self.pool_dir = Path(pool_dir or f"uploads/{task_id}/badcase_pool")
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.pool_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_file = self.pool_dir / "metadata.jsonl"

    def collect(
        self,
        event_id: str,
        frame_url: str,
        bbox: List[float],
        track_id: Optional[int],
        user_note: str,
        timestamp: str,
    ) -> dict:
        """收集一条误报"""
        # 下载并保存帧截图
        frame_filename = f"{event_id}_{int(time.time()*1000)}.jpg"
        frame_path = self.images_dir / frame_filename
        self._download_frame(frame_url, frame_path)

        # 写入 metadata
        meta = {
            "event_id": event_id,
            "frame_path": str(frame_path),
            "bbox": bbox,           # 归一化 xywh
            "track_id": track_id,
            "user_note": user_note,
            "timestamp": timestamp,
            "collected_at": datetime.now().isoformat(),
        }
        with open(self.metadata_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        # 更新计数
        count = self._get_pending_count()
        return {
            "collected": True,
            "pending_count": count,
            "threshold": self.threshold,
            "auto_retrain_triggered": count >= self.threshold,
        }

    def check_retrain_trigger(self) -> dict:
        """
        检查是否达到自动重训练阈值。
        返回是否应触发增量训练。
        """
        count = self._get_pending_count()
        if count >= self.threshold:
            return {
                "should_trigger": True,
                "pending_count": count,
                "message": f"已积累 {count} 条误报，达到阈值 {self.threshold}，建议触发增量训练"
            }
        return {
            "should_trigger": False,
            "pending_count": count,
            "message": f"还需 {self.threshold - count} 条触发自动训练"
        }

    @property
    def threshold(self) -> int:
        return 50  # 可配置
```

---

### 4.4 AI 需求分析 + 是否需要训练判断 ⭐ P1c

#### 4.4.1 定位说明

**差异化用户体验的核心**，将后端已有的"是否需要训练"判断逻辑升级为前端可见的核心交互。

现有 `algorithm_planner.py` 已有相关逻辑，但没有作为独立步骤展示给用户。

#### 4.4.2 判断流程

```
用户输入需求
     │
     ├── 涉及 OCR / 仪表读数 ─────────→ 不需要训练
     │
     ├── 涉及颜色分类 ─────────────────→ 不需要训练
     │
     ├── 涉及区域入侵 / 越线 / 停留 ──→ 不需要训练
     │
     ├── 涉及特定类别检测
     │        │
     │        └── 查询 Model Registry
     │              ├── 有可覆盖模型 → 推荐免训练 + 用户确认
     │              └── 无覆盖 → 需要训练
     │
     └── 其他复杂场景 ─────────────────→ 需要训练
```

#### 4.4.3 前端交互设计

```
┌─────────────────────────────────────────────────────────────┐
│  请描述您的检测需求：                                        │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ 油库装车区有没有人抽烟                                  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  [ 提交需求 ]                                              │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  ✅ AI 分析结果：                                           │
│                                                             │
│  检测意图：抽烟行为识别                                       │
│  推荐方案：人员检测 + 烟雾检测 + 规则判断                     │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  🎯 是否需要重新训练模型？                             │  │
│  │                                                     │  │
│  │  判断结果：❌ 不需要训练                                │  │
│  │                                                     │  │
│  │  理由：                                              │  │
│  │  抽烟行为可通过"人员检测 + 烟雾检测 + 规则引擎"组合    │  │
│  │  判断，无需自定义训练模型。                            │  │
│  │                                                     │  │
│  │  推荐方案预览：                                       │  │
│  │  Camera → Person Detection → Smoke Detection           │  │
│  │           → Rule Engine → Alarm → MQTT               │  │
│  │                                                     │  │
│  │  如仍需自定义训练（如检测特定品牌香烟）：                │  │
│  │  → 点击"需要自定义训练"可上传样本开始训练               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  [ 使用此方案 ]  [ 需要自定义训练 ]                          │
└─────────────────────────────────────────────────────────────┘
```

#### 4.4.4 实现位置

```python
# backend/services/training_necessity.py 新增

def analyze_training_necessity(
    user_description: str,
    vlm_result: Optional[dict],
    model_registry: ModelRegistry,
) -> TrainingNecessityResult:
    """
    判断用户需求是否需要训练。
    返回结果包含：needs_training + reason + suggested_approach
    """

    # 1. 规则匹配：OCR / 仪表 / 颜色 / 规则类需求 → 不需要训练
    if _matches_no_training_patterns(user_description):
        return TrainingNecessityResult(
            needs_training=False,
            confidence=0.95,
            reason="该需求属于规则类/OCR类，无需自定义训练",
            suggested_approach=_build_rule_based_approach(user_description),
            fallback_plan=None,
        )

    # 2. 检测类需求：查询 Model Registry 是否有可覆盖模型
    classes = vlm_result.get("classes", []) if vlm_result else []
    reusable = model_registry.find_reusable_model(
        required_classes=[c["class_name"] for c in classes],
        min_map50=0.7,
    )

    if reusable:
        return TrainingNecessanceResult(
            needs_training=False,
            confidence=0.85,
            reason=f"模型仓库中有可覆盖需求的已训练模型（{reusable.cache_id}），可直接使用",
            suggested_approach=_build_model_reuse_approach(reusable),
            fallback_plan=_build_training_plan(classes),
        )

    # 3. 无法覆盖：需要训练
    return TrainingNecessityResult(
        needs_training=True,
        confidence=0.9,
        reason="模型仓库中无完全覆盖该需求的模型，建议训练自定义模型",
        suggested_approach=_build_training_plan(classes),
        fallback_plan=None,
    )
```

---

### 4.5 边缘部署中心 ⭐ P2a

#### 4.5.1 部署格式策略（修订）

> ⚠️ **修订说明**：原方案承诺 PT/ONNX/Engine/RKNN/OM/MNN/NCNN 六种格式自动转换。
> 经评估，RKNN 只能在 RK 芯片上编译，OM 只能在昇腾上跑，维护成本过高。
> **第一阶段只做主链路**：`PT → ONNX → TensorRT Engine`
> RKNN / OM 作为后续"硬件合作伙伴适配"单独推进。

#### 4.5.2 部署包结构（第一阶段）

```
{task_id}/
├── models/
│   ├── best.pt                    # PyTorch 原生权重
│   ├── best.onnx                 # ONNX 跨平台（主链路）
│   └── best.engine               # TensorRT 高性能（需目标设备编译）
│
├── pipeline.json                  # 算法流程配置
├── rules.yaml                     # 规则引擎配置
├── config.yaml                   # 运行时配置
│
├── docker/
│   ├── Dockerfile                 # 多阶段构建，体积 < 2GB
│   ├── docker-compose.yml        # MQTT Broker + 算法服务一键启动
│   └── .env.example
│
├── demos/
│   ├── python/
│   │   ├── demo_inference.py
│   │   └── requirements.txt
│   └── cpp/
│       └── demo.cpp
│
├── sdk/
│   └── rest_api_client.py
│
└── README.md
```

#### 4.5.3 Dockerfile 模板

```dockerfile
# docker/Dockerfile

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    libgl1 libglib2.0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── 生产镜像 ────────────────────────────────────────────────
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 libgl1 libglib2.0 libsm6 libxext6 libxrender-dev \
    libmosquitto1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY models/ /app/models/
COPY pipeline.json /app/pipeline.json
COPY rules.yaml /app/rules.yaml
COPY config.yaml /app/config.yaml
COPY run_pipeline.py /app/run_pipeline.py

ENV PYTHONUNBUFFERED=1
EXPOSE 7860 1883

CMD ["python", "run_pipeline.py", "--mode", "production"]
```

---

### 4.6 ROI 标注增强 ⭐ P2b

#### 4.6.1 功能描述

AnnotationCanvas 扩展为同时支持目标检测框（BBox）+ ROI 区域（多边形）两种标注模式。

#### 4.6.2 实现要点

```typescript
// frontend/src/components/AnnotationCanvas.tsx

type AnnotationMode = 'bbox' | 'roi' | 'mixed'

interface ROIAnnotation {
  id: string
  name: string           // "ROI-01"
  label: string           // "装车区"
  polygon: number[][]    // [[x1,y1], [x2,y2], ...] 归一化坐标
  color: string
}

// ROI 多边形绘制逻辑
const handleROICreate = (points: number[][]) => {
  // 1. 至少 3 个点
  // 2. 自动闭合多边形
  // 3. 分配 ROI-ID（ROI-01, ROI-02...）
  // 4. 保存到 state
}
```

---

### 4.7 低代码编排（P3a）

> ⚠️ **修订说明**：自研 Node-RED 风格编辑器工作量极大，相当于一个独立产品。
> **建议：先集成 Node-RED，自定义节点接入我们的算法生态。**

#### 4.7.1 集成策略

```
Phase 3a（立即可做）：
  - 直接集成 Node-RED Web UI
  - 编写自定义节点：
      node-red-contrib-yolo-detect     → YOLO 检测
      node-red-contrib-vision-rule     → 规则引擎
      node-red-contrib-mqtt-out-vision → MQTT 输出
      node-red-contrib-roi-filter     → ROI 过滤
  - 用户通过 Node-RED 编排流程，我们提供节点包

Phase 3b（等算法市场稳定后）：
  - 自研可视化编辑器（React Flow）
  - 作为 Node-RED 的替代选项
```

#### 4.7.2 Node-RED 自定义节点示例

```javascript
// worker/nodered/nodes/vision-rule.js

module.exports = function(RED) {

    function VisionRuleNode(config) {
        RED.nodes.createNode(this, config);

        var node = this;
        this.rulesConfig = config.rulesConfig;  // YAML 规则配置路径

        // 初始化规则引擎
        var ruleEngine = new RuleEngine(this.rulesConfig);

        this.on('input', function(msg) {
            // msg.payload = ByteTrack 输出的跟踪目标列表
            var events = ruleEngine.process_frame(
                msg.payload.objects,
                msg.payload.timestamp
            );

            if (events.length > 0) {
                msg.payload = events;
                node.send(msg);
            }
        });

        this.on('close', function() {
            ruleEngine.close();
        });
    }

    RED.nodes.registerType("vision-rule", VisionRuleNode);
}
```

---

### 4.8 AI Agent 多角色协作（P3b）

> ⚠️ **修订说明**：经过对 Hermes Agent 源码的详细分析，确定不能直接复用 Hermes（技术栈不匹配、执行模型不同），但其架构设计高度可参考。
> 本节详细说明 Hermes 架构分析结论、可迁移设计、以及最终实现方案。

#### 4.8.1 Hermes Agent 源码分析结论

经过对 `hermes-agent/` 源码的全面分析，得出以下核心结论：

##### 为什么不能直接复用 Hermes

| 原因 | 说明 |
|------|------|
| **技术栈不匹配** | Hermes 是纯 Python 项目，而本平台是 React/TypeScript（前端）+ FastAPI（后端）+ Python Worker 的多语言架构 |
| **执行模型不同** | Hermes 是交互式对话 Agent，本平台是批处理式 Pipeline（上传→意图解析→打标→增强→训练→交付） |
| **GPU 依赖不同** | Hermes 假设本地有 LLM API 访问；本平台假设 AutoDL 云端 GPU + 本地 CPU |
| **进程模型不同** | Hermes AIAgent 绑定在 Python 进程中运行，无法直接嵌入 Web 应用进程 |

##### 可高度参考的 Hermes 架构设计

| Hermes 组件 | 可迁移到本平台的思路 |
|------------|---------------------|
| `TurnContext` + `build_turn_context` | Pipeline Stage 的 prologue/epilogue 分离（每阶段上下文构建 → 执行 → 收尾） |
| `MemoryManager` + `MemoryProvider` | 将 Worker 各阶段产物（raw_boxes.json / YOLO txt / 增强结果）抽象为 Pipeline Provider |
| `delegate_task` 子 Agent 隔离 | Worker 内阶段二的"两段式"隔离（YOLO-World → 显存释放 → Moondream2）可正式建模为子任务 |
| `ContextCompressor` 摘要策略 | 长任务书（phase-1 VLM 输出）超出长度时，用廉价模型摘要中间部分 |
| `hermes_state.py` SQLite FTS5 | 为 Worker 任务引入 SQLite 持久化 + 搜索，追踪阶段状态和文件路径 |
| `toolsets.py` 工具分组 | 将 Worker 工具按阶段分组（打标工具集 / 增强工具集 / 训练工具集） |
| 插件系统 + Skill | 为用户暴露 Skill 扩展点（用户可编写自己的数据增强配方 / 规则模板） |
| ACP JSON-RPC 会话管理 | 若未来需要远程 Worker 协作，可参考 ACP 的会话管理协议 |

##### Hermes 核心架构图（本平台参考版）

```
Hermes 原版架构                    本平台对应架构
─────────────────                  ───────────────────────────
AIAgent.run_conversation()          PipelineOrchestrator.run()
    │                                      │
    ├── TurnContext.build()           ├── StageContext.build()
    │   │ (系统提示词构建)              │   (阶段上下文构建)
    │   ├── 记忆预取                    ├── 各阶段 Provider
    │   │   MemoryProvider.prefetch()      │   - 打标 Provider
    │   ├── MCP 工具刷新                │   - 增强 Provider
    │   └── FTS 搜索                   │   - 训练 Provider
    │                                      │
    ├── conversation_loop             ├── StageExecutor.execute()
    │   │ (主循环)                      │   (阶段执行循环)
    │   ├── LLM API 调用                 │   ├── VLM / YOLO / Moondream
    │   ├── 工具解析                    │   ├── 工具解析（已有）
    │   └── 工具调度                    │   └── 规则引擎执行
    │                                      │
    ├── TurnFinalizer                 ├── StageFinalizer.finalize()
    │   │ (轮次收尾)                    │   (阶段收尾)
    │   ├── 摘要压缩                    │   ├── 上下文压缩
    │   ├── 轨迹保存                    │   ├── 产物保存
    │   └── 记忆写入                    │   └── 状态持久化
    │                                      │
    ├── MemoryManager                 ├── PipelineStateManager
    │   │ (多 Provider 编排)            │   (多 Stage 编排)
    │   └── sync / prefetch             │   └── track / persist
    │                                      │
    └── herems_state.py                └── pipeline_state.db
        (SQLite + FTS5)                     (SQLite + FTS5)
```

#### 4.8.2 最终实现方案：Pipeline Agent 架构

基于 Hermes 架构分析和本平台实际情况，采用 **Pipeline Agent 架构**。

**核心原则（来自 Hermes AGENTS.md）：「核心要窄，能力在边缘」**

每新增一个核心能力，优先级为：
1. 扩展现有代码
2. CLI 命令 + Skill（用户可编写配方）
3. 服务级工具
4. 插件
5. MCP 服务
6. 新核心工具（最后手段）

##### 架构设计

```python
# backend/services/pipeline_agent_orchestrator.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json

# ── 核心概念映射 ────────────────────────────────────────────
# Hermes: TurnContext → 本平台: StageContext
# Hermes: MemoryProvider → 本平台: PipelineProvider
# Hermes: delegate_task → 本平台: StageExecutor（子任务隔离）
# Hermes: conversation_loop → 本平台: PipelineOrchestrator
# Hermes: hermes_state.py → 本平台: pipeline_state.db

class StageType(Enum):
    """Pipeline 阶段类型（对应 Hermes 的 Agent 角色分工）"""
    VLM_INTENT = "vlm_intent"              # AI 产品经理：需求理解
    DATA_LABELING = "data_labeling"         # AI 标注专家：打标管理
    DATA_ANALYSIS = "data_analysis"         # AI 数据分析师：质量评估
    MODEL_ENGINEERING = "model_engineering"  # AI 模型工程师：选型推荐
    AUGMENTATION = "augmentation"           # AI 增强专家：增强配方
    RULE_ENGINEERING = "rule_engineering"  # AI 规则工程师：规则配置
    TRAINING = "training"                   # AI 训练执行器：训练监控
    DEPLOYMENT = "deployment"              # AI 部署工程师：部署配置

@dataclass
class StageContext:
    """
    单个 Pipeline 阶段的上下文。
    对应 Hermes 的 TurnContext。
    """
    stage_type: StageType
    task_id: str
    stage_id: str                    # 如 "stage_2_yolo_world"

    # 上下文产物（对应 Hermes 的 messages 历史）
    system_prompt: str = ""          # 本阶段的系统提示词
    input_data: Dict[str, Any] = field(default_factory=dict)   # 输入数据
    output_data: Dict[str, Any] = field(default_factory=dict)  # 输出数据
    artifacts: Dict[str, Any] = field(default_factory=dict)    # 产物路径
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> str:
        """将阶段上下文压缩为摘要（对应 Hermes 的 ContextCompressor）"""
        return json.dumps({
            "stage": self.stage_type.value,
            "input_keys": list(self.input_data.keys()),
            "output_keys": list(self.output_data.keys()),
            "artifacts": self.artifacts,
            "error_count": len(self.errors),
        }, ensure_ascii=False)

# ── PipelineProvider 抽象（对应 Hermes 的 MemoryProvider）────

class PipelineProvider(ABC):
    """
    Pipeline 阶段产物提供者。
    对应 Hermes 的 MemoryProvider。

    思路：将 Worker 各阶段产物（raw_boxes.json / YOLO txt / 增强结果）
    抽象为 Provider，支持跨阶段查询和上下文构建。
    """
    name: str

    @abstractmethod
    def system_prompt_block(self) -> str:
        """返回静态提示文本块（拼接进系统提示词）"""

    @abstractmethod
    def prefetch(self, query: str, context: StageContext) -> str:
        """
        阶段执行前预取相关上下文。
        对应 Hermes: MemoryProvider.prefetch(query) -> str
        """

    @abstractmethod
    def sync_turn(self, context: StageContext) -> None:
        """阶段执行后写入产物（异步）"""

    @abstractmethod
    def get_tool_schemas(self) -> List[dict]:
        """返回暴露给模型的工具 schema（可选）"""

    def on_turn_start(self, context: StageContext) -> None:
        """生命周期钩子：阶段开始前"""
        pass

    def on_stage_complete(self, context: StageContext) -> None:
        """生命周期钩子：阶段完成后"""
        pass


# ── 内置 PipelineProvider ─────────────────────────────────────

class IntentProvider(PipelineProvider):
    """阶段一产物：VLM 意图解析结果"""
    name = "intent"

    def system_prompt_block(self) -> str:
        return """## 意图理解阶段产物（仅供参考）
已解析的需求、检测目标列表、场景类型、行业分类均存储在 intent_provider 中。
使用 /intent 查看当前任务的意图解析结果。"""

    def prefetch(self, query: str, context: StageContext) -> str:
        # 从 context.artifacts 读取意图解析结果
        intent = context.artifacts.get("vlm_intent", {})
        return f"当前意图解析结果：{json.dumps(intent, ensure_ascii=False, indent=2)}"

    def sync_turn(self, context: StageContext) -> None:
        # 写入 pipeline_state.db
        pass

    def get_tool_schemas(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "intent_view",
                    "description": "查看当前任务的意图解析结果",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]


class LabelingProvider(PipelineProvider):
    """阶段二产物：打标结果和质量报告"""
    name = "labeling"

    def prefetch(self, query: str, context: StageContext) -> str:
        # 读取打标质量报告
        return context.artifacts.get("labeling_report", "")

    def get_tool_schemas(self) -> List[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "labeling_status",
                    "description": "查看当前打标任务状态和质量问题",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]


# ── StageExecutor（对应 Hermes 的 delegate_task 子 Agent 隔离）─

class StageExecutor:
    """
    阶段执行器。
    对应 Hermes 的 delegate_task：子 Agent 在独立上下文中执行。

    用于阶段二的"两段式打标"：
    - StageExecutor("yolo_world")  → YOLO-World 推理
    - StageExecutor("moondream_qa") → Moondream VQA 质检
    - 中间执行 del model + empty_cache + gc.collect()
    """

    def __init__(self, stage_type: StageType, config: dict):
        self.stage_type = stage_type
        self.config = config
        self._model = None

    def execute(self, input_data: dict) -> dict:
        """在隔离上下文中执行阶段任务"""
        if self.stage_type == StageType.VLM_INTENT:
            return self._execute_vlm_intent(input_data)
        elif self.stage_type == StageType.DATA_LABELING:
            return self._execute_labeling(input_data)
        elif self.stage_type == StageType.DATA_ANALYSIS:
            return self._execute_analysis(input_data)
        else:
            raise ValueError(f"Unknown stage type: {self.stage_type}")

    def _execute_labeling(self, input_data: dict) -> dict:
        """
        两段式打标：
        1. YOLO-World 推理
        2. 显存释放
        3. Moondream2 VQA 质检
        """
        # 第一段：YOLO-World
        yolo_result = self._run_yolo_world(input_data)
        self._release_gpu_memory()  # del + empty_cache + gc.collect()

        # 第二段：Moondream2 VQA
        qa_result = self._run_moondream_qa(yolo_result)
        self._release_gpu_memory()

        return {
            "raw_boxes": yolo_result,
            "qa_filtered": qa_result,
        }

    def _release_gpu_memory(self):
        """显存释放（对应 Hermes 的子 Agent 隔离清理）"""
        import gc
        if self._model is not None:
            del self._model
            self._model = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── 其他阶段执行器（架构预留）───────────────────────────

    def _execute_vlm_intent(self, input_data: dict) -> dict:
        """VLM 意图解析"""
        pass

    def _execute_analysis(self, input_data: dict) -> dict:
        """数据分析"""
        pass


# ── PipelineOrchestrator（对应 Hermes 的 AIAgent.run_conversation）─

class PipelineOrchestrator:
    """
    Pipeline 编排器。
    对应 Hermes 的 AIAgent.run_conversation()。

    核心职责：
    1. 按顺序执行各 Pipeline Stage
    2. 管理 StageContext 上下文
    3. 调用 PipelineProvider 构建上下文
    4. 处理阶段间的上下文传递
    5. 管理任务持久化（SQLite + FTS5）
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.stage_contexts: Dict[StageType, StageContext] = {}
        self.providers: Dict[str, PipelineProvider] = {}

        # 注册内置 Provider
        self._register_default_providers()

    def _register_default_providers(self):
        """注册内置 PipelineProvider（对应 Hermes 的内置 MemoryProvider）"""
        self.register_provider(LabelingProvider())
        self.register_provider(LabelingProvider())

    def register_provider(self, provider: PipelineProvider):
        """注册外部 Provider（对应 Hermes 的 plugin memory provider）"""
        self.providers[provider.name] = provider

    def run(self) -> dict:
        """
        主入口：按顺序执行所有 Pipeline Stage。
        对应 Hermes: AIAgent.run_conversation()
        """
        results = {}
        for stage_type in self._get_stage_sequence():
            context = self._build_stage_context(stage_type)

            # 1. Turn Prologue（阶段上下文构建）
            self._stage_prologue(context)

            # 2. 执行阶段
            result = self._execute_stage(context)

            # 3. Turn Epilogue（阶段收尾）
            self._stage_epilogue(context, result)

            results[stage_type.value] = result

        return results

    def _build_stage_context(self, stage_type: StageType) -> StageContext:
        """构建单阶段上下文（对应 Hermes: build_turn_context）"""
        context = StageContext(
            stage_type=stage_type,
            task_id=self.task_id,
            stage_id=f"{self.task_id}_{stage_type.value}",
        )

        # 1. 系统提示词构建（来自各 Provider 的 static block）
        context.system_prompt = self._build_system_prompt(context)

        # 2. 预取外部上下文（来自各 Provider 的 prefetch）
        for provider in self.providers.values():
            context.metadata[f"prefetch_{provider.name}"] = provider.prefetch(
                query="", context=context
            )

        return context

    def _build_system_prompt(self, context: StageContext) -> str:
        """拼接系统提示词（对应 Hermes: PromptBuilder）"""
        blocks = [
            "# Pipeline 阶段系统提示词",
            f"当前任务：{self.task_id}",
            f"阶段类型：{context.stage_type.value}",
        ]

        for provider in self.providers.values():
            block = provider.system_prompt_block()
            if block:
                blocks.append(block)

        return "\n\n".join(blocks)

    def _stage_prologue(self, context: StageContext):
        """阶段开始前钩子（对应 Hermes: pre_llm_call 钩子链）"""
        for provider in self.providers.values():
            provider.on_turn_start(context)

    def _execute_stage(self, context: StageContext) -> dict:
        """执行单个阶段（对应 Hermes: conversation_loop）"""
        executor = StageExecutor(context.stage_type, {})
        return executor.execute(context.input_data)

    def _stage_epilogue(self, context: StageContext, result: dict):
        """阶段完成后收尾（对应 Hermes: TurnFinalizer）"""
        context.output_data = result
        context.artifacts.update(result)

        # 1. 写入各 Provider
        for provider in self.providers.values():
            provider.sync_turn(context)
            provider.on_stage_complete(context)

        # 2. 持久化到 pipeline_state.db（对应 Hermes: hermes_state.py）
        self._persist_stage_context(context)

    def _persist_stage_context(self, context: StageContext):
        """阶段上下文持久化（对应 Hermes: SessionDB）"""
        import sqlite3
        conn = sqlite3.connect(f"uploads/{self.task_id}/pipeline_state.db")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stage_contexts (
                stage_id TEXT PRIMARY KEY,
                stage_type TEXT,
                summary TEXT,
                artifacts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            INSERT OR REPLACE INTO stage_contexts
            (stage_id, stage_type, summary, artifacts)
            VALUES (?, ?, ?, ?)
        """, (
            context.stage_id,
            context.stage_type.value,
            context.to_summary(),
            json.dumps(context.artifacts, ensure_ascii=False),
        ))
        conn.commit()
        conn.close()

    def _get_stage_sequence(self) -> List[StageType]:
        """根据任务类型确定阶段执行顺序"""
        return [
            StageType.VLM_INTENT,
            StageType.DATA_LABELING,
            StageType.DATA_ANALYSIS,
            StageType.AUGMENTATION,
            StageType.MODEL_ENGINEERING,
            StageType.TRAINING,
        ]
```

#### 4.8.3 可落地的 Agent 分工

基于上述架构设计，各 Agent 的最终分工和实现策略如下：

| Agent | 职责 | 实现方式 | 能力边界 |
|-------|------|---------|---------|
| AI 产品经理 | 需求理解、追问、场景识别 | `VLMIntentProvider`（阶段一） | ✅ 完整实现 |
| AI 数据分析师 | 数据质量评估、建议补充方向 | `DataAnalysisProvider`（质量门控模块增强） | ✅ 完整实现 |
| AI 模型工程师 | 模型选型、训练参数推荐 | `ModelEngineeringProvider`（training_recommendation_service 升级） | ✅ 完整实现 |
| AI 标注专家 | 标注规则说明、质量反馈 | `LabelingProvider`（阶段二产物管理） | ✅ 完整实现 |
| AI 增强专家 | 增强配方推荐 | `AugmentationProvider`（Albumentations 配方库） | ✅ 完整实现 |
| AI 规则工程师 | 规则 YAML 生成、验证 | `RuleEngineeringProvider`（规则引擎配置层） | ✅ 完整实现 |
| AI 部署工程师 | 部署包配置、边缘设备适配 | `DeploymentProvider`（Docker 部署包生成） | ⚠️ 辅助建议为主 |
| AI 调参专家 | 超参推荐 | `TrainingRecommendationProvider` | ⚠️ **不作为核心卖点** |

#### 4.8.4 实现路线图

```
Phase 3（第一阶段）：
  1. 实现 PipelineProvider 抽象基类
  2. 实现 PipelineStateManager（SQLite 持久化）
  3. 重构阶段二打标为 StageExecutor（显存隔离）
  4. 实现 LabelingProvider + IntentProvider

Phase 4（第二阶段）：
  5. 实现 DataAnalysisProvider（质量报告增强）
  6. 实现 ModelEngineeringProvider（训练推荐服务）
  7. 实现 AugmentationProvider（增强配方库）
  8. 实现 RuleEngineeringProvider（规则配置层）

Phase 5（第三阶段）：
  9. 实现 DeploymentProvider（部署包生成）
  10. Plugin 系统（用户可编写自定义 Provider）
  11. 外部 Provider 集成（类似 Hermes 的 honcho/supermemory）
```

#### 4.8.5 Hermes 技术债和风险

| 风险 | 影响 | 应对 |
|------|------|------|
| Hermes 是对话 Agent，本平台是 Pipeline，二者执行模型根本不同 | 直接复用代码不可行 | 只参考架构设计，重新实现核心抽象 |
| Hermes 的上下文压缩依赖辅助 LLM API，本平台 VLM 调用有成本 | 压缩策略需要评估性价比 | Phase 3 后再评估是否引入摘要模型 |
| PipelineProvider 抽象可能过度工程化 | 简单场景反而变复杂 | 核心 Provider 先行，Plugin 机制按需引入 |
| 多 Provider 并行预取可能拖慢阶段启动 | 性能瓶颈 | 异步预取 + 缓存，减少重复查询 |

---

### 4.9 行业算法模板（P3 并行）

#### 4.9.1 模板定义（油库场景示例）

```python
# backend/services/industry_templates.py

INDUSTRY_TEMPLATES = {
    "oil_depot": {
        "name": "油库安全监控",
        "icon": "🛢️",
        "algorithms": [
            "person_detection",
            "helmet_detection",
            "fire_detection",
            "smoke_detection",
            "region_intrusion",
            "dwell_detection",
        ],
        "default_regions": [
            {"id": "ROI-01", "name": "装车作业区"},
            {"id": "ROI-02", "name": "储罐区"},
            {"id": "ROI-03", "name": "巡检通道"},
        ],
        "default_rules": [
            {"type": "dwell", "roi": "ROI-01", "duration_sec": 300, "level": "critical"},
            {"type": "region_intrusion", "roi": "ROI-02", "level": "critical"},
        ],
        "recommended_device": "Jetson Orin NX",
    }
}
```

---

## 五、实施计划（修订版）

### Phase 0：基础建设（2 周，并行推进）

**目标**：为所有后续功能打基础

| 模块 | 内容 | 产出 |
|------|------|------|
| 统一事件 Schema | `UnifiedEvent` 数据类 + JSON Schema 定义 | 文档 + Schema JSON |
| Output Adapter 基类 | `BaseOutputAdapter` + MQTT + HTTP Adapter | 可运行的 Adapter |
| Event Router | 事件分发路由 | EventRouter 可用 |
| ROI 标注 | AnnotationCanvas 多边形扩展 | 可绘制 ROI |

### Phase 1：P0 跑通（4~6 周）

**目标**：算法市场 + 直接部署，让用户立刻感受到价值

```
里程碑 1：算法市场首页上线
  - 15+ 内置算法
  - 算法卡片展示
  - 算法详情页

里程碑 2：直接部署跑通
  - 免训练算法可部署
  - 视频流 → 算法 → 事件 → MQTT/HTTP

里程碑 3：统一事件 Schema 完成
  - 所有算法输出统一 Schema
  - Event Router 分发到各 Adapter
```

**代码变更范围**：

```
frontend/src/pages/
├── AlgorithmHub.tsx          [新增]
└── AlgorithmDetail.tsx     [新增]

backend/
├── routers/
│   └── algorithm_hub.py     [新增]
├── services/
│   ├── algorithm_hub_service.py [新增]
│   ├── output_adapters/     [新增目录]
│   │   ├── base.py
│   │   ├── mqtt_adapter.py
│   │   └── http_adapter.py
│   └── event_router.py      [新增]
└── models/
    └── db.py                 [改造: AlgorithmListing 表]

worker/pipeline/
├── output_adapters/         [新增目录]
└── rule_engine.py           [新增]
```

### Phase 2：P1 完善（4~6 周）

**目标**：规则引擎 + 数据闭环 + AI 判断

```
里程碑 4：YAML 规则引擎可用
  - 支持 composite / region_enter / dwell / counting 规则
  - 事件抑制逻辑正常

里程碑 5：数据闭环跑通
  - 误报标记 UI
  - badcase_pool 自动收集
  - 达到阈值触发增量训练

里程碑 6：AI 免训练推荐展示
  - 前端可见判断结果和理由
  - 推荐方案预览
```

### Phase 3：P2 建设（4~6 周）

**目标**：Docker 部署包 + ROI 标注 + Node-RED 集成

```
里程碑 7：Docker 部署包一键生成
  - 一键打包 best.onnx + pipeline.json + rules.yaml
  - Docker Compose 启动

里程碑 8：ROI 标注集成到算法市场部署流程
```

### Phase 4：P3 演进（持续迭代）

```
里程碑 9：Node-RED 自定义节点包发布
里程碑 10：行业算法模板上线（油库/化工/制造）
里程碑 11：AI Agent 多角色协作
里程碑 12：运维中心
```

---

## 六、技术架构变更摘要（修订版）

### 6.1 数据库变更

```python
# backend/models/db.py

# 新增表
class AlgorithmListing(Base):        # 算法市场条目
    __tablename__ = "algorithm_listings"
    id = Column(String, primary_key=True)
    name = Column(String)
    algorithm_type = Column(String)   # "yolo" / "rule" / "ocr"
    needs_training = Column(Boolean)
    needs_gpu = Column(Boolean)
    model_id = Column(String)        # 所需模型
    description = Column(Text)
    category = Column(String)
    output_schema = Column(JSON)
    default_config = Column(JSON)
    is_featured = Column(Boolean, default=False)

class FalseAlarmLog(Base):           # 误报日志
    __tablename__ = "false_alarm_logs"
    id = Column(String, primary_key=True)
    task_id = Column(String)
    event_id = Column(String)
    frame_path = Column(String)
    bbox = Column(JSON)
    user_note = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PipelineTemplate(Base):         # Pipeline 模板
    __tablename__ = "pipeline_templates"
    id = Column(String, primary_key=True)
    name = Column(String)
    industry = Column(String)        # oil_depot / chemical / manufacturing
    nodes = Column(JSON)              # PipelineNode[]
    edges = Column(JSON)               # PipelineEdge[]
    rules_config = Column(JSON)       # rules.yaml 内容

# Task 表新增字段
Task.roi_annotations = Column(JSON, default=list)
Task.pipeline_template_id = Column(String)
Task.deployment_status = Column(String)
```

### 6.2 新增 API 路由

| Method | Path | 所属 Phase | 描述 |
|--------|------|-----------|------|
| GET | `/api/algorithm-hub/listings` | P0 | 获取算法列表 |
| GET | `/api/algorithm-hub/listings/{id}` | P0 | 获取算法详情 |
| POST | `/api/algorithm-hub/deploy/{id}` | P0 | 直接部署算法 |
| POST | `/api/feedback/false-alarm` | P1 | 标记误报 |
| GET | `/api/feedback/badcase-pool/{task_id}` | P1 | 获取 badcase 池状态 |
| POST | `/api/feedback/auto-retrain/{task_id}` | P1 | 触发自动重训练 |
| GET | `/api/rules/template/{task_id}` | P1 | 获取规则配置 |
| PUT | `/api/rules/template/{task_id}` | P1 | 保存规则配置 |
| GET | `/api/deployment/packages/{task_id}` | P2 | 生成部署包 |
| POST | `/api/deployment/push` | P2 | 推送到边缘盒子 |

### 6.3 新增前端页面

| 页面 | 路由 | Phase |
|------|------|-------|
| AlgorithmHub | `/algorithm-hub` | P0 |
| AlgorithmDetail | `/algorithm-hub/:id` | P0 |
| IntentAnalysis | `/intent-analysis` | P1 |
| RuleBuilder | `/rule-builder/:taskId` | P1 |
| DeploymentCenter | `/deployment` | P2 |
| OperationsCenter | `/operations-center` | P3 |

---

## 七、风险与应对

| 风险 | 影响 | 应对策略 |
|------|------|---------|
| 规则引擎并发状态管理复杂 | 工期超预期 | YAML 配置先行，卡车项目状态机复用，不追求一步到位 |
| 规则引擎稳定性验证周期长 | 上线延迟 | 与 Phase 1 算法市场并行开发，规则引擎独立模块可灰度发布 |
| Node-RED 集成遇到生态限制 | 功能受限 | 准备自研编辑器作为备选（React Flow），不影响主链路 |
| 六格式全覆盖维护成本高 | 后续拖累 | 收敛到 PT→ONNX→TensorRT 主链路，其余作为合作方适配 |
| AI 调参过度承诺 | 口碑风险 | 明确"辅助建议"定位，不作为核心卖点宣传 |

---

## 八、成功指标

| 阶段 | 指标 | 目标 |
|------|------|------|
| Phase 0 末 | 统一事件 Schema 覆盖率 | 100% 算法使用统一 Schema |
| Phase 1 末 | 算法市场算法数 | ≥ 20 个 |
| Phase 1 末 | 直接部署使用率 | 占新任务的 30%+ |
| Phase 2 末 | 免训练推荐采纳率 | 推荐免训练时 50%+ 用户采纳 |
| Phase 2 末 | 误报标记功能使用率 | 有告警的任务中 40%+ 使用 |
| Phase 3 末 | Docker 部署包生成成功率 | > 95% |

---

*文档版本：v1.1 | 最后更新：2026-07-09 | 状态：待评审*
