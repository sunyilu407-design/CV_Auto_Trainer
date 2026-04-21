from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

from pipeline.runtime_session import RuntimeSession


def _region_center(region: Dict[str, Any], fallback_x: float = 0.5) -> List[float]:
    bbox = region.get("bbox_xywhn") or [fallback_x, 0.5, 0.2, 0.2]
    if len(bbox) < 4:
        return [fallback_x, 0.5, 0.2, 0.2]
    return [float(bbox[0]), float(bbox[1]), 0.12, 0.18]


def _outside_left_bbox(region: Dict[str, Any]) -> List[float]:
    bbox = region.get("bbox_xywhn") or [0.5, 0.5, 0.2, 0.2]
    left = float(bbox[0]) - float(bbox[2]) / 2
    x = max(0.05, left - 0.12)
    return [x, float(bbox[1]), 0.12, 0.18]


def _outside_right_bbox(region: Dict[str, Any]) -> List[float]:
    bbox = region.get("bbox_xywhn") or [0.5, 0.5, 0.2, 0.2]
    right = float(bbox[0]) + float(bbox[2]) / 2
    x = min(0.95, right + 0.12)
    return [x, float(bbox[1]), 0.12, 0.18]


def _primary_target_class(pipeline_config: Dict[str, Any]) -> str:
    detectors = pipeline_config.get("detectors", [])
    if detectors and detectors[0].get("target_classes"):
        return detectors[0]["target_classes"][0]
    rules = pipeline_config.get("rules", [])
    if rules and rules[0].get("target_class"):
        return rules[0]["target_class"]
    return "target"


def _required_duration_ms(pipeline_config: Dict[str, Any]) -> int:
    duration_values = []
    for rule in pipeline_config.get("rules", []):
        if rule.get("duration_ms") is not None:
            duration_values.append(int(rule.get("duration_ms", 0)))
        elif rule.get("duration_seconds") is not None:
            duration_values.append(int(rule.get("duration_seconds", 0)) * 1000)
    return max(duration_values + [2000])


def _build_sample_frames(pipeline_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    regions = {
        region.get("region_id"): region
        for region in pipeline_config.get("regions", [])
        if region.get("region_id")
    }
    rules = pipeline_config.get("rules", [])
    target_class = _primary_target_class(pipeline_config)
    duration_ms = _required_duration_ms(pipeline_config)
    transition_rule = next(
        (rule for rule in rules if rule.get("rule_type") == "cross_region_transition"),
        None,
    )

    if transition_rule:
        from_region = regions.get(transition_rule.get("from_region_id"), {})
        to_region = regions.get(transition_rule.get("to_region_id"), {})
        return [
            {
                "frame_index": 1,
                "timestamp_ms": 0,
                "detections": [{"class_name": target_class, "bbox_xywhn": _outside_left_bbox(from_region), "confidence": 0.95}],
            },
            {
                "frame_index": 2,
                "timestamp_ms": 1000,
                "detections": [{"class_name": target_class, "bbox_xywhn": _region_center(from_region, 0.35), "confidence": 0.94}],
            },
            {
                "frame_index": 3,
                "timestamp_ms": max(1000 + duration_ms + 500, 2500),
                "detections": [{"class_name": target_class, "bbox_xywhn": _region_center(from_region, 0.35), "confidence": 0.93}],
            },
            {
                "frame_index": 4,
                "timestamp_ms": max(1000 + duration_ms + 1500, 3500),
                "detections": [{"class_name": target_class, "bbox_xywhn": _region_center(to_region, 0.65), "confidence": 0.92}],
            },
            {
                "frame_index": 5,
                "timestamp_ms": max(1000 + duration_ms + 2500, 4500),
                "detections": [{"class_name": target_class, "bbox_xywhn": _outside_right_bbox(to_region or from_region), "confidence": 0.91}],
            },
        ]

    primary_region = regions.get(rules[0].get("region_id"), {}) if rules else {}
    return [
        {
            "frame_index": 1,
            "timestamp_ms": 0,
            "detections": [{"class_name": target_class, "bbox_xywhn": _outside_left_bbox(primary_region), "confidence": 0.95}],
        },
        {
            "frame_index": 2,
            "timestamp_ms": 1000,
            "detections": [{"class_name": target_class, "bbox_xywhn": _region_center(primary_region), "confidence": 0.94}],
        },
        {
            "frame_index": 3,
            "timestamp_ms": 1000 + duration_ms + 500,
            "detections": [{"class_name": target_class, "bbox_xywhn": _region_center(primary_region), "confidence": 0.93}],
        },
        {
            "frame_index": 4,
            "timestamp_ms": 1000 + duration_ms + 1500,
            "detections": [{"class_name": target_class, "bbox_xywhn": _outside_right_bbox(primary_region), "confidence": 0.92}],
        },
    ]


def _run_sample_output(pipeline_config: Dict[str, Any], sample_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
    session = RuntimeSession(pipeline_config)
    events: List[Dict[str, Any]] = []
    track_states: List[Dict[str, Any]] = []
    for frame in sample_frames:
        result = session.process_frame(frame)
        track_states = result.get("track_states", [])
        events.extend(result.get("events", []))
    return {
        "observation_frames": sample_frames,
        "track_states": track_states,
        "events": events,
    }


def _process_multi_model_artifacts(
    artifacts: Dict[str, Any],
    pipeline_config: Dict[str, Any],
    output_dir: Path,
) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """
    从 artifacts 中提取 __multi_model__ 字段，把每个模型的权重复制到
    bundle 的 models/<step_id>/ 子目录下，并在 pipeline_config 的 model_pipeline
    中补全 bundle_weight_path 字段。

    返回:
        - 更新后的 pipeline_config
        - 拷贝进 bundle 的文件相对路径列表
        - 模型清单（供 manifest.json 使用）
    """
    multi_model = artifacts.get("__multi_model__") or {}
    copied_files: List[str] = []
    model_manifest: List[Dict[str, Any]] = []

    if not multi_model:
        return pipeline_config, copied_files, model_manifest

    # 建立 step_id → pipeline step 的引用，便于回写 bundle_weight_path
    pipeline = dict(pipeline_config)
    model_pipeline = [dict(step) for step in pipeline.get("model_pipeline", [])]
    step_id_to_step = {step.get("step_id"): step for step in model_pipeline}

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for step_id, record in multi_model.items():
        step_dir = models_dir / str(step_id)
        step_dir.mkdir(parents=True, exist_ok=True)

        entry: Dict[str, Any] = {
            "step_id": step_id,
            "role": record.get("role"),
            "model_id": record.get("model_id"),
            "source": record.get("source"),
            "files": [],
        }

        if record.get("source") == "reuse":
            weight_src = record.get("weight_path")
            if weight_src and Path(weight_src).exists():
                weight_dst = step_dir / "best.pt"
                shutil.copyfile(weight_src, weight_dst)
                rel = f"models/{step_id}/best.pt"
                copied_files.append(rel)
                entry["files"].append(rel)
                if step_id in step_id_to_step:
                    step_id_to_step[step_id]["bundle_weight_path"] = rel
            entry["cache_id"] = record.get("cache_id")
        elif record.get("source") == "trained":
            step_artifacts = record.get("artifacts") or {}
            for fname, src_path in step_artifacts.items():
                if not src_path or not Path(src_path).exists():
                    continue
                dst_path = step_dir / Path(fname).name
                try:
                    shutil.copyfile(src_path, dst_path)
                    rel = f"models/{step_id}/{Path(fname).name}"
                    copied_files.append(rel)
                    entry["files"].append(rel)
                    # best.pt 写入 pipeline step
                    if fname == "best.pt" and step_id in step_id_to_step:
                        step_id_to_step[step_id]["bundle_weight_path"] = rel
                except OSError:
                    continue

        model_manifest.append(entry)

    if model_pipeline:
        pipeline["model_pipeline"] = model_pipeline

    return pipeline, copied_files, model_manifest


def _copy_runtime_modules(output_dir: Path) -> List[str]:
    source_dir = Path(__file__).resolve().parent
    bundle_pipeline_dir = output_dir / "pipeline"
    bundle_pipeline_dir.mkdir(parents=True, exist_ok=True)

    copied_files = []
    for name in ["__init__.py", "frame_adapter.py", "tracking_runtime.py", "event_engine.py", "runtime_session.py"]:
        source_path = source_dir / name
        target_path = bundle_pipeline_dir / name
        shutil.copyfile(source_path, target_path)
        copied_files.append(f"pipeline/{name}")

    runtime_support = "\n".join(
        [
            "from pipeline.frame_adapter import normalize_observation_frames",
            "from pipeline.runtime_session import RuntimeSession",
            "",
            "__all__ = ['RuntimeSession', 'normalize_observation_frames']",
        ]
    )
    runtime_support_path = output_dir / "runtime_support.py"
    runtime_support_path.write_text(runtime_support + "\n", encoding="utf-8")
    copied_files.append("runtime_support.py")
    return copied_files


def _build_readme(
    task_id: str,
    artifacts: Dict[str, Any],
    model_manifest: List[Dict[str, Any]] | None = None,
) -> str:
    lines = [
        f"# 算法工程包：{task_id}",
        "",
        "## 内容说明",
        "",
        "- `pipeline.json`：编译后的算法流水线配置（含规则引擎、时序窗口）",
        "- `manifest.json`：bundle 文件清单与产物映射",
        "- `sample_input.json`：示例 observation frame 输入",
        "- `sample_output.json`：随包 runtime 生成的示例输出",
        "- `run_pipeline.py`：本地运行入口脚本",
        "- `pipeline/`：随包导出的最小 runtime 实现",
    ]

    if model_manifest:
        lines.append("- `models/`：各阶段训练/复用的权重文件（按 step_id 分子目录）")

    lines.extend([
        "",
        "## 使用方式",
        "",
        "支持以下输入形式：",
        "- 单个 JSON 文件：兼容 `{\"observation_frames\": [...]}` 或直接传 frame 列表",
        "- 单个 JSONL 文件：每行一个 frame JSON 对象",
        "- 本地目录：按文件名顺序聚合目录内的 JSON / JSONL 文件",
        "",
        "业务示例：仓位占用、区域持续出现、滞留检测等规则型算法。",
        "",
        "`python run_pipeline.py --input sample_input.json --output output.json`",
        "`python run_pipeline.py --input sample_input.jsonl --output output.json`",
        "`python run_pipeline.py --input ./input_frames --output output.json`",
        "",
        "## 输出说明",
        "",
        "- `frame_count`：本次处理的 frame 数量",
        "- `track_states`：最终跟踪状态快照",
        "- `events`：命中的业务事件列表",
        "",
    ])

    if model_manifest:
        lines.append("## 多模型流水线")
        lines.append("")
        lines.append("本方案由多个模型协同工作，权重文件位于 `models/` 下：")
        lines.append("")
        for entry in model_manifest:
            role = entry.get("role", "?")
            step_id = entry.get("step_id", "?")
            model_id = entry.get("model_id", "?")
            source = entry.get("source", "?")
            source_label = {
                "trained": "本次训练",
                "reuse": "复用已有模型",
                "failed": "训练失败",
            }.get(source, source)
            lines.append(f"- **{role}** (`{step_id}`) · 模型 `{model_id}` · 来源：{source_label}")
            for f in entry.get("files") or []:
                lines.append(f"  - `{f}`")
        lines.append("")
        lines.append("每个模型的 `best.pt` 路径已写入 `pipeline.json` 的 `model_pipeline[*].bundle_weight_path`，")
        lines.append("你的推理程序可直接按 step_id 查找对应权重。")
        lines.append("")

    lines.append("## 关联产物")
    lines.append("")
    lines.extend([f"- `{name}` -> `{path}`" for name, path in artifacts.items()])
    return "\n".join(lines)


def _build_entrypoint_script() -> str:
    return "\n".join(
        [
            "import argparse",
            "import json",
            "from pathlib import Path",
            "",
            "from runtime_support import RuntimeSession, normalize_observation_frames",
            "",
            "CONFIG_PATH = Path(__file__).with_name('pipeline.json')",
            "DEFAULT_INPUT_PATH = Path(__file__).with_name('sample_input.json')",
            "DEFAULT_OUTPUT_PATH = Path(__file__).with_name('sample_output.json')",
            "",
            "def _load_json_file(path: Path):",
            "    payload = json.loads(path.read_text(encoding='utf-8'))",
            "    if isinstance(payload, dict):",
            "        return payload.get('observation_frames', [])",
            "    if isinstance(payload, list):",
            "        return payload",
            "    raise ValueError(f'Unsupported JSON payload: {path}')",
            "",
            "def _load_jsonl_file(path: Path):",
            "    frames = []",
            "    for line in path.read_text(encoding='utf-8').splitlines():",
            "        line = line.strip()",
            "        if not line:",
            "            continue",
            "        frames.append(json.loads(line))",
            "    return frames",
            "",
            "def _load_input_frames(path: Path):",
            "    if path.is_dir():",
            "        frames = []",
            "        for child in sorted(path.iterdir(), key=lambda item: item.name):",
            "            suffix = child.suffix.lower()",
            "            if suffix == '.json':",
            "                frames.extend(_load_json_file(child))",
            "            elif suffix == '.jsonl':",
            "                frames.extend(_load_jsonl_file(child))",
            "        return frames",
            "    suffix = path.suffix.lower()",
            "    if suffix == '.json':",
            "        return _load_json_file(path)",
            "    if suffix == '.jsonl':",
            "        return _load_jsonl_file(path)",
            "    raise ValueError(f'Unsupported input path: {path}')",
            "",
            "def main():",
            "    parser = argparse.ArgumentParser(description='Run exported algorithm package')",
            "    parser.add_argument('--input', default=str(DEFAULT_INPUT_PATH), help='path to json/jsonl/directory input')",
            "    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='path to write runtime output json')",
            "    args = parser.parse_args()",
            "",
            "    pipeline = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))",
            "    frames = normalize_observation_frames(_load_input_frames(Path(args.input)))",
            "    session = RuntimeSession(pipeline)",
            "    events = []",
            "    track_states = []",
            "    for frame in frames:",
            "        result = session.process_frame(frame)",
            "        track_states = result.get('track_states', [])",
            "        events.extend(result.get('events', []))",
            "",
            "    output_payload = {",
            "        'metadata': pipeline.get('metadata', {}),",
            "        'frame_count': len(frames),",
            "        'track_states': track_states,",
            "        'events': events,",
            "    }",
            "    Path(args.output).write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding='utf-8')",
            "    print(f\"Generated {len(events)} events -> {args.output}\")",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
    )


def export_algorithm_package(
    task_id: str,
    pipeline_config: Dict,
    artifacts: Dict,
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_path = output_dir / "pipeline.json"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"
    entrypoint_path = output_dir / "run_pipeline.py"
    sample_input_path = output_dir / "sample_input.json"
    sample_output_path = output_dir / "sample_output.json"

    # 处理多模型权重：复制到 bundle/models/<step_id>/ 并更新 pipeline_config
    pipeline_config, multi_model_files, model_manifest = _process_multi_model_artifacts(
        artifacts=artifacts,
        pipeline_config=pipeline_config,
        output_dir=output_dir,
    )

    # 外部清单仅展示公开 artifacts（剥离内部 __multi_model__ 等内部键）
    public_artifacts = {
        k: v for k, v in artifacts.items()
        if not (isinstance(k, str) and k.startswith("__"))
    }

    sample_frames = _build_sample_frames(pipeline_config)
    sample_output = _run_sample_output(pipeline_config, sample_frames)
    bundled_runtime_files = _copy_runtime_modules(output_dir)

    pipeline_path.write_text(json.dumps(pipeline_config, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_input_path.write_text(
        json.dumps({"observation_frames": sample_frames}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sample_output_path.write_text(json.dumps(sample_output, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "task_id": task_id,
        "bundle_version": "v1",
        "artifacts": public_artifacts,
        "model_pipeline": model_manifest,
        "files": [
            "pipeline.json",
            "manifest.json",
            "README.md",
            "run_pipeline.py",
            "sample_input.json",
            "sample_output.json",
        ]
        + bundled_runtime_files
        + multi_model_files,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(_build_readme(task_id, public_artifacts, model_manifest) + "\n", encoding="utf-8")
    entrypoint_path.write_text(_build_entrypoint_script() + "\n", encoding="utf-8")

    return {
        "bundle_dir": str(output_dir),
        "pipeline_path": str(pipeline_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "entrypoint_path": str(entrypoint_path),
        "sample_input_path": str(sample_input_path),
        "sample_output_path": str(sample_output_path),
        "runtime_support_path": str(output_dir / "runtime_support.py"),
    }
