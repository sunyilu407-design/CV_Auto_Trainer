# Phase 4 Bundle Input And README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让导出的算法 bundle 支持 `JSON / JSONL / 目录` 三类输入，并输出中文 README，便于本地交付演示与快速验证。

**Architecture:** 保持现有 Phase 4 bundle 结构不变，只在 `run_pipeline.py` 导出模板里增加输入装载逻辑，并更新 README 文案模板。测试继续以集成测试脚本为主，先在 exporter 场景下锁定失败，再以最小实现转绿。

**Tech Stack:** Python 3.11, FastAPI backend bundle export path, local runtime session, integration test script

---

### Task 1: 用 JSONL 与目录输入测试锁定新行为

**Files:**
- Modify: `tests/test_integration.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing JSONL and directory assertions in exporter test**

```python
            jsonl_input = output_dir / "sample_input.jsonl"
            jsonl_frames = json.loads((output_dir / "sample_input.json").read_text(encoding="utf-8"))["observation_frames"]
            jsonl_input.write_text(
                "\n".join(json.dumps(frame, ensure_ascii=False) for frame in jsonl_frames) + "\n",
                encoding="utf-8",
            )

            jsonl_result = run_cmd(
                [
                    sys.executable,
                    str(output_dir / "run_pipeline.py"),
                    "--input",
                    str(jsonl_input),
                    "--output",
                    str(output_dir / "generated_output_jsonl.json"),
                ],
                cwd=output_dir,
                timeout=30,
            )
            if jsonl_result.returncode != 0:
                print(f"✗ JSONL 输入运行失败: {jsonl_result.returncode}")
                return False

            jsonl_output = json.loads((output_dir / "generated_output_jsonl.json").read_text(encoding="utf-8"))
            if jsonl_output.get("frame_count") != len(jsonl_frames):
                print(f"✗ JSONL frame_count 不正确: {jsonl_output}")
                return False
            if not jsonl_output.get("events"):
                print(f"✗ JSONL 运行结果缺少 events: {jsonl_output}")
                return False

            input_dir = output_dir / "input_dir"
            input_dir.mkdir()
            (input_dir / "001_frames.json").write_text(
                json.dumps({"observation_frames": jsonl_frames[:2]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (input_dir / "002_frames.jsonl").write_text(
                "\n".join(json.dumps(frame, ensure_ascii=False) for frame in jsonl_frames[2:]) + "\n",
                encoding="utf-8",
            )

            dir_result = run_cmd(
                [
                    sys.executable,
                    str(output_dir / "run_pipeline.py"),
                    "--input",
                    str(input_dir),
                    "--output",
                    str(output_dir / "generated_output_dir.json"),
                ],
                cwd=output_dir,
                timeout=30,
            )
            if dir_result.returncode != 0:
                print(f"✗ 目录输入运行失败: {dir_result.returncode}")
                return False

            dir_output = json.loads((output_dir / "generated_output_dir.json").read_text(encoding="utf-8"))
            if dir_output.get("frame_count") != len(jsonl_frames):
                print(f"✗ 目录输入 frame_count 不正确: {dir_output}")
                return False
            if not dir_output.get("events"):
                print(f"✗ 目录输入运行结果缺少 events: {dir_output}")
                return False
```

- [ ] **Step 2: Run exporter test to verify it fails before implementation**

Run: `./.venv/bin/python tests/test_integration.py package-exporter`
Expected: FAIL in `run_pipeline.py` when reading `.jsonl` or directory input

- [ ] **Step 3: Tighten README expectations in the same test**

```python
            readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
            if "## 使用方式" not in readme_text:
                print("✗ README 未中文化")
                return False
            if "JSONL" not in readme_text or "目录" not in readme_text:
                print(f"✗ README 缺少输入格式说明: {readme_text}")
                return False
            if "仓位占用" not in readme_text:
                print(f"✗ README 缺少业务化示例: {readme_text}")
                return False
```

- [ ] **Step 4: Re-run exporter test to confirm README assertions also fail before implementation**

Run: `./.venv/bin/python tests/test_integration.py package-exporter`
Expected: FAIL because exported README is still English and missing JSONL / directory usage

### Task 2: 最小实现 bundle 多输入入口与中文 README

**Files:**
- Modify: `worker/pipeline/package_exporter.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Add README template content for Chinese usage guidance**

```python
def _build_readme(task_id: str, artifacts: Dict[str, str]) -> str:
    return "\n".join(
        [
            f"# 算法工程包：{task_id}",
            "",
            "## 内容说明",
            "",
            "- `pipeline.json`：编译后的算法流水线配置",
            "- `manifest.json`：bundle 文件清单与产物映射",
            "- `sample_input.json`：示例 observation frame 输入",
            "- `sample_output.json`：示例事件输出结果",
            "- `run_pipeline.py`：本地运行入口脚本",
            "- `pipeline/`：随包导出的最小 runtime 实现",
            "",
            "## 使用方式",
            "",
            "支持以下输入形式：",
            "- 单个 JSON 文件",
            "- 单个 JSONL 文件",
            "- 包含多个 JSON / JSONL 文件的本地目录",
            "",
            "示例：仓位占用或区域持续出现事件",
            "`python run_pipeline.py --input sample_input.json --output output.json`",
            "",
            "## 输出说明",
            "",
            "- `frame_count`：本次处理的 frame 数量",
            "- `track_states`：最终跟踪状态",
            "- `events`：命中的业务事件",
            "",
            "## 关联产物",
            "",
        ]
        + [f"- `{name}` -> `{path}`" for name, path in artifacts.items()]
    )
```

- [ ] **Step 2: Implement JSON, JSONL, and directory loading inside exported entrypoint**

```python
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
            "            if child.suffix.lower() == '.json':",
            "                frames.extend(_load_json_file(child))",
            "            elif child.suffix.lower() == '.jsonl':",
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
```

- [ ] **Step 3: Run exporter test to verify the new behavior passes**

Run: `./.venv/bin/python tests/test_integration.py package-exporter`
Expected: PASS with JSON, JSONL, directory input, and Chinese README assertions all green

- [ ] **Step 4: Run algorithm package API regression**

Run: `./.venv/bin/python tests/test_integration.py algorithm-package-api`
Expected: PASS and exported package paths remain readable through API response

- [ ] **Step 5: Run broader backend regression if package tests stay green**

Run: `./.venv/bin/python tests/test_integration.py backend`
Expected: PASS with no regression in backend imports or algorithm package route wiring
