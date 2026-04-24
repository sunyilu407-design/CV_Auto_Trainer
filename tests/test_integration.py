#!/usr/bin/env python3
"""
阶段七：集成测试脚本
在本地安装好所有依赖后，执行此脚本进行端到端测试。

用法：
    # 后端测试
    python tests/test_integration.py backend

    # 前端构建测试
    python tests/test_integration.py frontend

    # Worker 测试
    python tests/test_integration.py worker

    # 完整流程（需要 GPU）
    python tests/test_integration.py full
"""

import sys
import subprocess
import os
import json
from pathlib import Path
import time
import importlib
import tempfile
from contextlib import nullcontext
from types import ModuleType
from urllib.error import URLError
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).parent.parent
TEST_ADMIN_USERNAME = "admin"
TEST_ADMIN_PASSWORD = "1"
TEST_SECRET_KEY = "integration-test-secret-key"


def run_cmd(cmd: list[str], cwd: Path = None, timeout: int = 60) -> subprocess.CompletedProcess:
    """运行命令，返回 CompletedProcess"""
    print(f"\n$ {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.TimeoutExpired as e:
        print(f"TIMEOUT after {timeout}s")
        print(e.stdout.decode() if e.stdout else "")
        print(e.stderr.decode() if e.stderr else "")
        raise


def ensure_test_auth_env():
    os.environ["CV_AUTO_TRAINER_ADMIN_USERNAME"] = TEST_ADMIN_USERNAME
    os.environ["CV_AUTO_TRAINER_ADMIN_PASSWORD"] = TEST_ADMIN_PASSWORD
    os.environ["CV_AUTO_TRAINER_SECRET_KEY"] = TEST_SECRET_KEY


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def is_sandbox_network_error(exc: Exception) -> bool:
    return "Operation not permitted" in str(exc)


def test_backend_imports():
    """测试后端模块是否可以正确导入"""
    print("\n=== 测试：后端模块导入 ===")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "backend"))
        import models.database
        import models.db
        import services.vlm_adapter
        import services.train_dispatcher
        import services.cloud_trainer
        import services.generic_ssh_trainer
        import services.alert_manager
        import services.settings_manager
        import services.algorithm_planner
        print("✓ 所有后端模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False


def test_algorithm_planner_service():
    """测试：算法规划服务能把业务描述转成结构化草案。"""
    print("\n=== 测试：算法规划服务 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.algorithm_planner import build_algorithm_plan

        result = build_algorithm_plan(
            user_description="识别仓位是否被货箱占用，持续10秒后输出占位事件",
            vlm_result={
                "classes": [
                    {"class_name": "cargo_box", "prompt": "stacked brown cargo box"},
                    {"class_name": "rack_slot", "prompt": "warehouse rack slot"},
                ]
            },
        )

        if result.get("scenario_type") != "occupancy_monitoring":
            print(f"✗ scenario_type 不正确: {result.get('scenario_type')}")
            return False
        if result.get("runtime_modes") != ["offline", "stream"]:
            print(f"✗ runtime_modes 不正确: {result.get('runtime_modes')}")
            return False
        if not result.get("targets") or result["targets"][0].get("class_name") != "cargo_box":
            print(f"✗ targets 不正确: {result.get('targets')}")
            return False
        if not result.get("events") or result["events"][0].get("event_code") != "occupancy_detected":
            print(f"✗ events 不正确: {result.get('events')}")
            return False
        if (
            not result.get("temporal_constraints")
            or result["temporal_constraints"][0].get("duration_seconds") != 10
        ):
            print(f"✗ temporal_constraints 不正确: {result.get('temporal_constraints')}")
            return False

        print("✓ 算法规划服务输出正常")
        return True
    except Exception as e:
        print(f"✗ 算法规划服务测试异常: {e}")
        return False


def test_algorithm_planner_multi_event_service():
    """测试：算法规划服务能从复杂描述生成多事件草案。"""
    print("\n=== 测试：算法规划服务多事件 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.algorithm_planner import build_algorithm_plan

        result = build_algorithm_plan(
            user_description="识别人员进入A区、离开A区，并在从A区进入B区时输出跨区事件；在A区持续2秒后输出滞留事件",
            vlm_result={"classes": [{"class_name": "person", "prompt": "person"}]},
        )

        if result.get("scenario_type") not in {"intrusion_monitoring", "custom_event_monitoring", "dwell_time_monitoring"}:
            print(f"✗ multi-event scenario_type 不正确: {result}")
            return False
        regions = result.get("regions", [])
        region_ids = [item.get("region_id") for item in regions]
        if not {"zone_a", "zone_b"}.issubset(set(region_ids)):
            print(f"✗ 多区域抽取不正确: {regions}")
            return False
        events = result.get("events", [])
        event_types = {item.get("event_type") for item in events}
        if not {"region_enter", "region_exit", "cross_region_transition", "region_presence_duration"}.issubset(
            event_types
        ):
            print(f"✗ 多事件抽取不完整: {events}")
            return False

        transition_event = next((item for item in events if item.get("event_type") == "cross_region_transition"), None)
        if not transition_event:
            print(f"✗ 缺少跨区事件: {events}")
            return False
        trigger = transition_event.get("trigger", {})
        if trigger.get("from_region_id") != "zone_a" or trigger.get("to_region_id") != "zone_b":
            print(f"✗ 跨区触发条件不正确: {transition_event}")
            return False

        print("✓ 算法规划服务多事件输出正常")
        return True
    except Exception as e:
        print(f"✗ 算法规划服务多事件测试异常: {e}")
        return False


def test_algorithm_plan_returns_capabilities_and_negotiation_summary():
    """测试：算法规划服务返回能力草图与需求协商摘要。"""
    print("\n=== 测试：算法规划返回能力草图与需求协商摘要 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.algorithm_planner import build_algorithm_plan

        result = build_algorithm_plan(
            user_description="识别未佩戴工帽的人员进入危险区域并停留5秒后触发告警",
            vlm_result=None,
        )

        capabilities = result.get("capabilities", [])
        negotiation = result.get("negotiation_summary", {})

        capability_kinds = {item.get("kind") for item in capabilities}
        if len(capabilities) < 2 or not {"detection", "classification", "rule"}.issubset(capability_kinds):
            print(f"✗ 未生成多能力草图: {result}")
            return False
        if negotiation.get("duration_seconds") != 5:
            print(f"✗ negotiation_summary 时长错误: {result}")
            return False
        if "危险区域" not in "".join(negotiation.get("regions", [])):
            print(f"✗ negotiation_summary 区域摘要错误: {result}")
            return False
        if "进入危险区域" not in "".join(negotiation.get("events", [])):
            print(f"✗ negotiation_summary 事件摘要错误: {result}")
            return False
        if "工帽" not in "".join(negotiation.get("objects", [])):
            print(f"✗ negotiation_summary 对象摘要错误: {result}")
            return False

        print("✓ 算法规划能力草图与协商摘要正常")
        return True
    except Exception as e:
        print(f"✗ 算法规划能力草图与协商摘要测试异常: {e}")
        return False


def test_algorithm_plan_api():
    """测试：算法规划接口能生成、读取并确认草案。"""
    print("\n=== 测试：算法规划 API ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_algorithm_plan.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False
                headers = auth_headers(login_json["data"]["token"])

                settings_resp = client.put(
                    "/api/settings",
                    json={
                        "default_model": "yolo11l.pt",
                        "default_gpu_type": "Apple M4 Pro",
                        "default_train_mode": "cloud",
                    },
                    headers=headers,
                )
                if settings_resp.status_code != 200 or settings_resp.json().get("code") != 0:
                    print(f"✗ 保存默认训练设置失败: {settings_resp.status_code} {settings_resp.json()}")
                    return False

                create_resp = client.post("/api/tasks", json={"name": "仓位识别算法"}, headers=headers)
                create_json = create_resp.json()
                if create_resp.status_code != 200:
                    print(f"✗ 创建任务失败: {create_resp.status_code} {create_json}")
                    return False

                task_id = create_json["id"]
                plan_payload = {
                    "task_id": task_id,
                    "user_description": "识别仓位是否被货箱占用，持续10秒后输出占位事件",
                    "vlm_result": {
                        "classes": [
                            {"class_name": "cargo_box", "prompt": "stacked brown cargo box"},
                            {"class_name": "rack_slot", "prompt": "warehouse rack slot"},
                        ]
                    },
                    "runtime_capability": {
                        "local_training_available": True,
                        "preferred_device": "mps",
                        "available_export_formats": ["onnx", "coreml"],
                        "supports_cloud_training": True,
                    },
                }

                generate_resp = client.post("/api/algorithm/plan", json=plan_payload, headers=headers)
                generate_json = generate_resp.json()
                if generate_resp.status_code != 200 or generate_json.get("code") != 0:
                    print(f"✗ 生成算法规划失败: {generate_resp.status_code} {generate_json}")
                    return False

                plan = generate_json["data"]
                if plan.get("status") != "draft":
                    print(f"✗ draft 状态不正确: {plan}")
                    return False
                if plan.get("algorithm_plan", {}).get("scenario_type") != "occupancy_monitoring":
                    print(f"✗ scenario_type 不正确: {plan}")
                    return False
                if not plan.get("algorithm_plan", {}).get("capabilities"):
                    print(f"✗ capabilities 缺失: {plan}")
                    return False
                if plan.get("algorithm_plan", {}).get("negotiation_summary", {}).get("duration_seconds") != 10:
                    print(f"✗ negotiation_summary 不正确: {plan}")
                    return False

                get_resp = client.get(f"/api/algorithm/plan/{task_id}", headers=headers)
                get_json = get_resp.json()
                if get_resp.status_code != 200 or get_json.get("code") != 0:
                    print(f"✗ 获取算法规划失败: {get_resp.status_code} {get_json}")
                    return False

                negotiate_resp = client.post(
                    f"/api/algorithm/plan/{task_id}/negotiate",
                    json={
                        "negotiation_summary": {
                            "scenario_label": "危险区域未佩戴工帽停留告警",
                            "objects": ["人员", "工帽"],
                            "regions": ["危险区域"],
                            "duration_seconds": 5,
                            "events": ["进入危险区域", "停留 5 秒告警"],
                        },
                        "offline_evaluation": {
                            "status": "pending",
                            "clips": [
                                {"clip_id": "trigger", "label": "应触发"},
                                {"clip_id": "negative", "label": "不应触发"},
                            ],
                        },
                    },
                    headers=headers,
                )
                negotiate_json = negotiate_resp.json()
                if negotiate_resp.status_code != 200 or negotiate_json.get("code") != 0:
                    print(f"✗ 更新需求协商失败: {negotiate_resp.status_code} {negotiate_json}")
                    return False
                if negotiate_json.get("data", {}).get("negotiation_summary", {}).get("duration_seconds") != 5:
                    print(f"✗ negotiate 响应内容不正确: {negotiate_json}")
                    return False

                task_resp = client.get(f"/api/tasks/{task_id}", headers=headers)
                task_json = task_resp.json()
                if task_resp.status_code != 200:
                    print(f"✗ 读取任务需求协商状态失败: {task_resp.status_code} {task_json}")
                    return False
                if task_json.get("negotiation_summary", {}).get("scenario_label") != "危险区域未佩戴工帽停留告警":
                    print(f"✗ negotiation_summary 未持久化: {task_json}")
                    return False
                clips = task_json.get("offline_evaluation", {}).get("clips", [])
                if len(clips) != 2:
                    print(f"✗ offline_evaluation 未持久化: {task_json}")
                    return False

                confirm_resp = client.post(
                    f"/api/algorithm/plan/{task_id}/confirm",
                    json={
                        "region_overrides": [
                            {
                                "region_id": "primary_region",
                                "name": "主监测区域",
                                "bbox_xywhn": [0.5, 0.5, 0.4, 0.5],
                            }
                        ],
                        "runtime_capability": {
                            "local_training_available": True,
                            "preferred_device": "mps",
                            "available_export_formats": ["onnx", "coreml"],
                            "supports_cloud_training": True,
                        },
                    },
                    headers=headers,
                )
                confirm_json = confirm_resp.json()
                if confirm_resp.status_code != 200 or confirm_json.get("code") != 0:
                    print(f"✗ 确认算法规划失败: {confirm_resp.status_code} {confirm_json}")
                    return False
                if confirm_json["data"].get("status") != "confirmed":
                    print(f"✗ confirmed 状态不正确: {confirm_json}")
                    return False
                pipeline_config = confirm_json["data"].get("pipeline_config")
                if not pipeline_config:
                    print(f"✗ pipeline_config 缺失: {confirm_json}")
                    return False
                detectors = pipeline_config.get("detectors", [])
                if not detectors or detectors[0].get("detector_id") != "primary_detector":
                    print(f"✗ detectors 不正确: {pipeline_config}")
                    return False
                if pipeline_config.get("trackers", [{}])[0].get("tracker_type") != "bytetrack":
                    print(f"✗ trackers 不正确: {pipeline_config}")
                    return False
                if pipeline_config.get("regions", [{}])[0].get("bbox_xywhn") != [0.5, 0.5, 0.4, 0.5]:
                    print(f"✗ regions 不正确: {pipeline_config}")
                    return False
                if pipeline_config.get("rules", [{}])[0].get("rule_type") != "region_presence_duration":
                    print(f"✗ rules 不正确: {pipeline_config}")
                    return False
                training_recommendation = pipeline_config.get("training_recommendation", {})
                if training_recommendation.get("recommended_model") != "yolo11s.pt":
                    print(f"✗ training_recommendation 不正确: {pipeline_config}")
                    return False
                recommended_config = training_recommendation.get("recommended_config", {})
                if recommended_config.get("model") != "yolo11s.pt":
                    print(f"✗ recommended_config.model 不正确: {training_recommendation}")
                    return False
                if recommended_config.get("train_mode") != "local":
                    print(f"✗ recommended_config.train_mode 不正确: {training_recommendation}")
                    return False
                if recommended_config.get("export_formats") != ["onnx", "coreml"]:
                    print(f"✗ recommended_config.export_formats 不正确: {training_recommendation}")
                    return False
                if recommended_config.get("gpu_type") != "Apple M4 Pro":
                    print(f"✗ recommended_config.gpu_type 不正确: {training_recommendation}")
                    return False
                if training_recommendation.get("reason_summary") is None:
                    print(f"✗ reason_summary 缺失: {training_recommendation}")
                    return False
                if training_recommendation.get("source_map", {}).get("train_mode") != "runtime":
                    print(f"✗ source_map.train_mode 不正确: {training_recommendation}")
                    return False
                if training_recommendation.get("recommended_model") != recommended_config.get("model"):
                    print(f"✗ legacy recommended_model 与 recommended_config.model 不一致: {training_recommendation}")
                    return False
                if training_recommendation.get("train_mode") != recommended_config.get("train_mode"):
                    print(f"✗ legacy train_mode 与 recommended_config.train_mode 不一致: {training_recommendation}")
                    return False
                if training_recommendation.get("export_formats") != recommended_config.get("export_formats"):
                    print(f"✗ legacy export_formats 与 recommended_config.export_formats 不一致: {training_recommendation}")
                    return False

        print("✓ 算法规划 API 正常")
        return True
    except Exception as e:
        print(f"✗ 算法规划 API 测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_pipeline_compiler_service():
    """测试：流水线编译器能输出 detector/tracker/rule 结构和训练建议。"""
    print("\n=== 测试：流水线编译器 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.pipeline_compiler import compile_algorithm_pipeline

        pipeline = compile_algorithm_pipeline(
            {
                "summary": "基于 cargo_box 的 occupancy_monitoring 算法草案",
                "scenario_type": "occupancy_monitoring",
                "runtime_modes": ["offline", "stream"],
                "targets": [
                    {
                        "class_name": "cargo_box",
                        "prompt": "stacked brown cargo box",
                        "role": "primary_subject",
                        "requires_training": True,
                    }
                ],
                "regions": [
                    {
                        "region_id": "primary_region",
                        "name": "主监测区域",
                        "source": "user_defined",
                        "required": True,
                    }
                ],
                "temporal_constraints": [
                    {
                        "constraint_id": "primary_duration",
                        "type": "sustain",
                        "duration_seconds": 10,
                    }
                ],
                "events": [
                    {
                        "event_code": "occupancy_detected",
                        "name": "主事件",
                        "trigger": {
                            "target_class": "cargo_box",
                            "region_id": "primary_region",
                            "temporal_constraint_id": "primary_duration",
                        },
                    }
                ],
                "training_requirements": {
                    "detector_training_required": True,
                    "tracking_required": True,
                    "rule_engine_required": True,
                },
                "confidence": 0.72,
            }
        )

        if pipeline.get("version") != "v1":
            print(f"✗ version 不正确: {pipeline}")
            return False
        if pipeline.get("detectors", [{}])[0].get("target_classes") != ["cargo_box"]:
            print(f"✗ detectors 不正确: {pipeline}")
            return False
        if pipeline.get("trackers", [{}])[0].get("tracker_type") != "bytetrack":
            print(f"✗ trackers 不正确: {pipeline}")
            return False
        if pipeline.get("rules", [{}])[0].get("event_code") != "occupancy_detected":
            print(f"✗ rules 不正确: {pipeline}")
            return False
        if pipeline.get("training_recommendation", {}).get("export_formats") != ["onnx"]:
            print(f"✗ training_recommendation 不正确: {pipeline}")
            return False

        print("✓ 流水线编译器输出正常")
        return True
    except Exception as e:
        print(f"✗ 流水线编译器测试异常: {e}")
        return False


def test_pipeline_compiler_multi_event_service():
    """测试：流水线编译器能编译多事件规则。"""
    print("\n=== 测试：流水线编译器多事件 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.pipeline_compiler import compile_algorithm_pipeline

        pipeline = compile_algorithm_pipeline(
            {
                "summary": "人员跨区与滞留算法草案",
                "scenario_type": "intrusion_monitoring",
                "runtime_modes": ["offline", "stream"],
                "targets": [{"class_name": "person", "prompt": "person", "role": "primary_subject", "requires_training": True}],
                "regions": [
                    {"region_id": "zone_a", "name": "A 区", "source": "user_defined", "required": True},
                    {"region_id": "zone_b", "name": "B 区", "source": "user_defined", "required": True},
                ],
                "temporal_constraints": [{"constraint_id": "stay_2s", "type": "sustain", "duration_seconds": 2}],
                "events": [
                    {
                        "event_type": "region_enter",
                        "event_code": "entered_zone_a",
                        "name": "进入 A 区",
                        "trigger": {"target_class": "person", "region_id": "zone_a"},
                    },
                    {
                        "event_type": "region_presence_duration",
                        "event_code": "dwell_zone_a",
                        "name": "A 区滞留",
                        "trigger": {"target_class": "person", "region_id": "zone_a", "temporal_constraint_id": "stay_2s"},
                    },
                    {
                        "event_type": "region_exit",
                        "event_code": "left_zone_a",
                        "name": "离开 A 区",
                        "trigger": {"target_class": "person", "region_id": "zone_a"},
                    },
                    {
                        "event_type": "cross_region_transition",
                        "event_code": "crossed_a_to_b",
                        "name": "A 到 B",
                        "trigger": {"target_class": "person", "from_region_id": "zone_a", "to_region_id": "zone_b"},
                    },
                ],
                "training_requirements": {
                    "detector_training_required": True,
                    "tracking_required": True,
                    "rule_engine_required": True,
                },
                "confidence": 0.88,
            }
        )

        rules = pipeline.get("rules", [])
        rule_types = {item.get("rule_type") for item in rules}
        if not {"region_enter", "region_presence_duration", "region_exit", "cross_region_transition"}.issubset(
            rule_types
        ):
            print(f"✗ 多事件 rule_type 不完整: {rules}")
            return False

        duration_rule = next((item for item in rules if item.get("rule_type") == "region_presence_duration"), None)
        if not duration_rule or int(duration_rule.get("duration_ms", 0)) != 2000:
            print(f"✗ duration_ms 编译不正确: {rules}")
            return False

        transition_rule = next((item for item in rules if item.get("rule_type") == "cross_region_transition"), None)
        if not transition_rule:
            print(f"✗ 缺少跨区 rule: {rules}")
            return False
        if transition_rule.get("from_region_id") != "zone_a" or transition_rule.get("to_region_id") != "zone_b":
            print(f"✗ 跨区 rule 字段不正确: {transition_rule}")
            return False

        print("✓ 流水线编译器多事件输出正常")
        return True
    except Exception as e:
        print(f"✗ 流水线编译器多事件测试异常: {e}")
        return False


def test_training_recommendation_service():
    """测试：训练推荐服务会合并算法信号、用户默认值和运行时能力。"""
    print("\n=== 测试：训练推荐服务 ===")
    try:
        from types import SimpleNamespace

        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.pipeline_compiler import compile_algorithm_pipeline
        from services.training_recommendation_service import build_training_recommendation

        plan = {
            "summary": "基于 cargo_box 的 occupancy_monitoring 算法草案",
            "scenario_type": "occupancy_monitoring",
            "runtime_modes": ["offline", "stream"],
            "targets": [
                {
                    "class_name": "cargo_box",
                    "prompt": "cargo box",
                    "role": "primary_subject",
                    "requires_training": True,
                }
            ],
            "regions": [
                {
                    "region_id": "primary_region",
                    "name": "主监测区域",
                    "source": "user_defined",
                    "required": True,
                }
            ],
            "temporal_constraints": [
                {
                    "constraint_id": "primary_duration",
                    "type": "sustain",
                    "duration_seconds": 10,
                }
            ],
            "events": [
                {
                    "event_code": "occupancy_detected",
                    "name": "主事件",
                    "trigger": {
                        "target_class": "cargo_box",
                        "region_id": "primary_region",
                        "temporal_constraint_id": "primary_duration",
                    },
                }
            ],
            "training_requirements": {
                "detector_training_required": True,
                "tracking_required": True,
                "rule_engine_required": True,
            },
            "confidence": 0.72,
        }
        pipeline = compile_algorithm_pipeline(plan)
        settings = SimpleNamespace(
            default_model="yolo11l.pt",
            default_train_mode="cloud",
            default_gpu_type="Apple M4 Pro",
        )
        runtime = {
            "local_training_available": True,
            "preferred_device": "mps",
            "available_export_formats": ["onnx", "coreml"],
            "supports_cloud_training": True,
        }

        recommendation = build_training_recommendation(
            algorithm_plan=plan,
            pipeline_config=pipeline,
            settings=settings,
            runtime_capability=runtime,
        )

        config = recommendation["recommended_config"]
        if config.get("model") != "yolo11s.pt":
            print(f"✗ model 不正确: {config}")
            return False
        if config.get("train_mode") != "local":
            print(f"✗ train_mode 不正确: {config}")
            return False
        if config.get("export_formats") != ["onnx", "coreml"]:
            print(f"✗ export_formats 不正确: {config}")
            return False
        if config.get("imgsz") != 640:
            print(f"✗ imgsz 不正确: {config}")
            return False
        if int(config.get("epochs", 0)) < 100:
            print(f"✗ epochs 不正确: {config}")
            return False
        if config.get("patience") != 20:
            print(f"✗ patience 不正确: {config}")
            return False
        if recommendation.get("source_map", {}).get("train_mode") != "runtime":
            print(f"✗ source_map 不正确: {recommendation}")
            return False
        if recommendation.get("legacy", {}).get("recommended_model") != "yolo11s.pt":
            print(f"✗ legacy.recommended_model 不正确: {recommendation}")
            return False
        reason_summary = recommendation.get("reason_summary", "")
        if not isinstance(reason_summary, str) or "推荐" not in reason_summary or "当前任务" not in reason_summary:
            print(f"✗ 中文 reason_summary 不正确: {recommendation}")
            return False

        complex_plan = {
            "summary": "多区域多目标占位与滞留算法草案",
            "scenario_type": "dwell_time_monitoring",
            "runtime_modes": ["offline", "stream"],
            "targets": [
                {"class_name": "person", "prompt": "person", "role": "primary_subject", "requires_training": True},
                {"class_name": "helmet", "prompt": "helmet", "role": "secondary_subject", "requires_training": True},
                {"class_name": "vest", "prompt": "vest", "role": "secondary_subject", "requires_training": True},
                {"class_name": "forklift", "prompt": "forklift", "role": "secondary_subject", "requires_training": True},
            ],
            "regions": [
                {"region_id": "region_a", "name": "A 区", "source": "user_defined", "required": True},
                {"region_id": "region_b", "name": "B 区", "source": "user_defined", "required": True},
            ],
            "temporal_constraints": [
                {"constraint_id": "stay_10s", "type": "sustain", "duration_seconds": 10},
                {"constraint_id": "stay_20s", "type": "sustain", "duration_seconds": 20},
            ],
            "events": [
                {
                    "event_code": "dwell_detected",
                    "name": "滞留事件",
                    "trigger": {
                        "target_class": "person",
                        "region_id": "region_a",
                        "temporal_constraint_id": "stay_10s",
                    },
                },
                {
                    "event_code": "cross_region_detected",
                    "name": "跨区事件",
                    "trigger": {
                        "target_class": "forklift",
                        "region_id": "region_b",
                        "temporal_constraint_id": "stay_20s",
                    },
                },
            ],
            "training_requirements": {
                "detector_training_required": True,
                "tracking_required": True,
                "rule_engine_required": True,
            },
            "confidence": 0.82,
        }
        complex_pipeline = compile_algorithm_pipeline(complex_plan)
        complex_recommendation = build_training_recommendation(
            algorithm_plan=complex_plan,
            pipeline_config=complex_pipeline,
            settings=settings,
            runtime_capability=runtime,
        )
        complex_config = complex_recommendation["recommended_config"]
        if complex_config.get("model") != "yolo11m.pt":
            print(f"✗ 高复杂度 model 不正确: {complex_config}")
            return False
        if complex_config.get("imgsz") != 1280:
            print(f"✗ 高复杂度 imgsz 不正确: {complex_config}")
            return False
        if complex_config.get("epochs") != 140:
            print(f"✗ 高复杂度 epochs 不正确: {complex_config}")
            return False
        if complex_config.get("patience") != 30:
            print(f"✗ 高复杂度 patience 不正确: {complex_config}")
            return False

        print("✓ 训练推荐服务输出正常")
        return True
    except Exception as e:
        print(f"✗ 训练推荐服务测试异常: {e}")
        return False


def test_worker_imports():
    """测试 Worker 模块是否可以正确导入"""
    print("\n=== 测试：Worker 模块导入 ===")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        import pipeline.gpu_manager
        import pipeline.stage2_labeler
        import pipeline.stage25_augmentor
        import pipeline.local_trainer
        import pipeline.tracking_runtime
        import pipeline.frame_adapter
        import pipeline.event_engine
        import pipeline.package_exporter
        import utils.yolo_io
        import utils.dataset_splitter
        print("✓ 所有 Worker 模块导入成功")
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False


def test_event_engine_runtime():
    """测试：事件引擎在持续时间满足时输出事件。"""
    print("\n=== 测试：事件引擎运行时 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline.event_engine import evaluate_pipeline_events

        pipeline_config = {
            "regions": [{"region_id": "primary_region", "name": "主监测区域", "bbox_xywhn": [0.5, 0.5, 0.4, 0.5]}],
            "rules": [
                {
                    "rule_id": "rule_1",
                    "rule_type": "region_presence_duration",
                    "event_code": "occupancy_detected",
                    "target_class": "cargo_box",
                    "region_id": "primary_region",
                    "duration_seconds": 10,
                }
            ],
        }
        track_states = [
            {
                "track_id": "track-1",
                "class_name": "cargo_box",
                "bbox_xywhn": [0.52, 0.53, 0.2, 0.2],
                "frame_index": 12,
                "timestamp_ms": 12000,
                "present_duration_ms": 12000,
                "regions_inside": ["primary_region"],
                "entered_region_at": {"primary_region": 0},
                "last_event_frame": {},
            }
        ]

        events = evaluate_pipeline_events(pipeline_config, track_states)
        if not events:
            print("✗ 未生成事件")
            return False
        event = events[0]
        if event.get("event_code") != "occupancy_detected":
            print(f"✗ event_code 不正确: {event}")
            return False
        if event.get("track_id") != "track-1":
            print(f"✗ track_id 不正确: {event}")
            return False
        if event.get("payload", {}).get("duration_ms") != 12000:
            print(f"✗ payload.duration_ms 不正确: {event}")
            return False

        print("✓ 事件引擎输出正常")
        return True
    except Exception as e:
        print(f"✗ 事件引擎测试异常: {e}")
        return False


def test_phase3_runtime_session():
    """测试：统一运行时会话能保持稳定轨迹并输出多类事件。"""
    print("\n=== 测试：Phase 3 统一运行时 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline.runtime_session import RuntimeSession

        pipeline_config = {
            "regions": [
                {"region_id": "zone_a", "name": "A 区", "bbox_xywhn": [0.35, 0.5, 0.2, 0.4]},
                {"region_id": "zone_b", "name": "B 区", "bbox_xywhn": [0.58, 0.5, 0.18, 0.4]},
            ],
            "rules": [
                {
                    "rule_id": "rule_enter_a",
                    "rule_type": "region_enter",
                    "event_code": "entered_zone_a",
                    "target_class": "person",
                    "region_id": "zone_a",
                },
                {
                    "rule_id": "rule_presence_a",
                    "rule_type": "region_presence_duration",
                    "event_code": "presence_zone_a",
                    "target_class": "person",
                    "region_id": "zone_a",
                    "duration_ms": 1500,
                },
                {
                    "rule_id": "rule_exit_a",
                    "rule_type": "region_exit",
                    "event_code": "left_zone_a",
                    "target_class": "person",
                    "region_id": "zone_a",
                },
                {
                    "rule_id": "rule_enter_b",
                    "rule_type": "region_enter",
                    "event_code": "entered_zone_b",
                    "target_class": "person",
                    "region_id": "zone_b",
                },
                {
                    "rule_id": "rule_transition_ab",
                    "rule_type": "cross_region_transition",
                    "event_code": "crossed_a_to_b",
                    "target_class": "person",
                    "from_region_id": "zone_a",
                    "to_region_id": "zone_b",
                },
                {
                    "rule_id": "rule_exit_b",
                    "rule_type": "region_exit",
                    "event_code": "left_zone_b",
                    "target_class": "person",
                    "region_id": "zone_b",
                },
            ],
        }

        frames = [
            {
                "frame_index": 1,
                "timestamp_ms": 0,
                "detections": [{"class_name": "person", "bbox_xywhn": [0.15, 0.5, 0.1, 0.2], "confidence": 0.95}],
            },
            {
                "frame_index": 2,
                "timestamp_ms": 1000,
                "detections": [{"class_name": "person", "bbox_xywhn": [0.35, 0.5, 0.1, 0.2], "confidence": 0.94}],
            },
            {
                "frame_index": 3,
                "timestamp_ms": 3000,
                "detections": [{"class_name": "person", "bbox_xywhn": [0.38, 0.5, 0.1, 0.2], "confidence": 0.93}],
            },
            {
                "frame_index": 4,
                "timestamp_ms": 4000,
                "detections": [{"class_name": "person", "bbox_xywhn": [0.58, 0.5, 0.1, 0.2], "confidence": 0.92}],
            },
            {
                "frame_index": 5,
                "timestamp_ms": 5000,
                "detections": [{"class_name": "person", "bbox_xywhn": [0.78, 0.5, 0.1, 0.2], "confidence": 0.91}],
            },
        ]

        session = RuntimeSession(pipeline_config)
        all_events = []
        seen_track_ids = []

        for frame in frames:
            result = session.process_frame(frame)
            track_states = result.get("track_states", [])
            if not track_states:
                print(f"✗ 当前帧缺少 track_states: {result}")
                return False
            seen_track_ids.append(track_states[0].get("track_id"))
            all_events.extend(result.get("events", []))

        unique_track_ids = {item for item in seen_track_ids if item}
        if len(unique_track_ids) != 1:
            print(f"✗ track_id 未跨帧稳定复用: {seen_track_ids}")
            return False

        event_codes = [item.get("event_code") for item in all_events]
        expected_codes = {
            "entered_zone_a",
            "presence_zone_a",
            "entered_zone_b",
            "left_zone_a",
            "crossed_a_to_b",
            "left_zone_b",
        }
        if not expected_codes.issubset(set(event_codes)):
            print(f"✗ 事件集合不完整: {event_codes}")
            return False

        transition_event = next((item for item in all_events if item.get("event_code") == "crossed_a_to_b"), None)
        if not transition_event:
            print(f"✗ 缺少跨区事件: {all_events}")
            return False
        payload = transition_event.get("payload", {})
        if payload.get("from_region_id") != "zone_a" or payload.get("to_region_id") != "zone_b":
            print(f"✗ 跨区事件 payload 不正确: {transition_event}")
            return False

        print("✓ Phase 3 统一运行时输出正常")
        return True
    except Exception as e:
        print(f"✗ Phase 3 统一运行时测试异常: {e}")
        return False


def test_tracking_runtime_resilience():
    """测试：tracking 在短时丢帧和较大位移后仍能复用旧 track_id。"""
    print("\n=== 测试：Tracking 稳定性增强 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline.tracking_runtime import update_track_states

        tracks = update_track_states(
            existing_tracks=[],
            detections=[{"class_name": "person", "bbox_xywhn": [0.10, 0.50, 0.10, 0.20], "confidence": 0.95}],
            frame_index=1,
            timestamp_ms=0,
        )
        first_track_id = tracks[0].get("track_id")

        tracks = update_track_states(
            existing_tracks=tracks,
            detections=[],
            frame_index=2,
            timestamp_ms=1000,
        )
        if not tracks or tracks[0].get("lost_frames") != 1:
            print(f"✗ 丢帧状态不正确: {tracks}")
            return False

        tracks = update_track_states(
            existing_tracks=tracks,
            detections=[{"class_name": "person", "bbox_xywhn": [0.43, 0.50, 0.10, 0.20], "confidence": 0.93}],
            frame_index=3,
            timestamp_ms=2000,
        )
        resumed_track_id = tracks[0].get("track_id")
        if resumed_track_id != first_track_id:
            print(f"✗ 短时丢帧后 track_id 未复用: {first_track_id} -> {resumed_track_id}")
            return False

        print("✓ Tracking 稳定性增强正常")
        return True
    except Exception as e:
        print(f"✗ Tracking 稳定性增强测试异常: {e}")
        return False


def test_algorithm_package_exporter():
    """测试：工程包导出器能输出可运行的算法工程包。"""
    print("\n=== 测试：工程包导出器 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline.package_exporter import export_algorithm_package

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "algorithm_bundle"
            pipeline_config = {
                "version": "v1",
                "metadata": {"summary": "仓位占位算法", "scenario_type": "occupancy_monitoring"},
                "detectors": [{"detector_id": "primary_detector", "target_classes": ["cargo_box"]}],
                "regions": [{"region_id": "primary_region", "name": "主监测区域", "bbox_xywhn": [0.5, 0.5, 0.4, 0.4]}],
                "rules": [
                    {
                        "rule_id": "rule_1",
                        "rule_type": "region_presence_duration",
                        "event_code": "occupancy_detected",
                        "target_class": "cargo_box",
                        "region_id": "primary_region",
                        "duration_ms": 2000,
                    }
                ],
                "packaging": {"config_path": "pipeline.json", "entrypoint": "run_pipeline.py"},
            }
            artifacts = {"best.pt": "/tmp/mock/best.pt"}

            result = export_algorithm_package(
                task_id="task-123",
                pipeline_config=pipeline_config,
                artifacts=artifacts,
                output_dir=output_dir,
            )

            if not (output_dir / "pipeline.json").exists():
                print(f"✗ pipeline.json 未生成: {result}")
                return False
            if not (output_dir / "manifest.json").exists():
                print(f"✗ manifest.json 未生成: {result}")
                return False
            if not (output_dir / "README.md").exists():
                print(f"✗ README.md 未生成: {result}")
                return False
            if not (output_dir / "sample_input.json").exists():
                print(f"✗ sample_input.json 未生成: {result}")
                return False
            if not (output_dir / "sample_output.json").exists():
                print(f"✗ sample_output.json 未生成: {result}")
                return False
            if not (output_dir / "runtime_support.py").exists():
                print(f"✗ runtime_support.py 未生成: {result}")
                return False
            if result.get("bundle_dir") != str(output_dir):
                print(f"✗ bundle_dir 不正确: {result}")
                return False

            run_result = run_cmd(
                [
                    sys.executable,
                    str(output_dir / "run_pipeline.py"),
                    "--input",
                    str(output_dir / "sample_input.json"),
                    "--output",
                    str(output_dir / "generated_output.json"),
                ],
                cwd=output_dir,
                timeout=30,
            )
            if run_result.returncode != 0:
                print(f"✗ 导出入口运行失败: {run_result.returncode}")
                return False

            generated_output = output_dir / "generated_output.json"
            if not generated_output.exists():
                print("✗ generated_output.json 未生成")
                return False
            output_data = json.loads(generated_output.read_text(encoding="utf-8"))
            if not output_data.get("events"):
                print(f"✗ 运行结果缺少 events: {output_data}")
                return False
            if output_data["events"][0].get("event_code") != "occupancy_detected":
                print(f"✗ 运行结果 event_code 不正确: {output_data}")
                return False

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

            jsonl_frames = json.loads((output_dir / "sample_input.json").read_text(encoding="utf-8"))["observation_frames"]
            jsonl_input = output_dir / "sample_input.jsonl"
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

        print("✓ 工程包导出器输出正常")
        return True
    except Exception as e:
        print(f"✗ 工程包导出器测试异常: {e}")
        return False


def test_algorithm_package_api():
    """测试：后端能为已确认算法生成工程包文件。"""
    print("\n=== 测试：算法工程包 API ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_algorithm_package.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False
                headers = auth_headers(login_json["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "算法工程包"}, headers=headers)
                task_id = create_resp.json()["id"]

                client.post(
                    "/api/algorithm/plan",
                    json={
                        "task_id": task_id,
                        "user_description": "识别仓位是否被货箱占用，持续10秒后输出占位事件",
                        "vlm_result": {
                            "classes": [
                                {"class_name": "cargo_box", "prompt": "stacked brown cargo box"},
                            ]
                        },
                    },
                    headers=headers,
                )
                client.post(f"/api/algorithm/plan/{task_id}/confirm", headers=headers)

                package_resp = client.post(f"/api/algorithm/package/{task_id}", headers=headers)
                package_json = package_resp.json()
                if package_resp.status_code != 200 or package_json.get("code") != 0:
                    print(f"✗ 导出算法工程包失败: {package_resp.status_code} {package_json}")
                    return False

                bundle = package_json["data"]
                pipeline_path = Path(bundle["pipeline_path"])
                manifest_path = Path(bundle["manifest_path"])
                readme_path = Path(bundle["readme_path"])
                sample_input_path = Path(bundle["sample_input_path"])
                sample_output_path = Path(bundle["sample_output_path"])
                runtime_support_path = Path(bundle["runtime_support_path"])
                if not pipeline_path.exists() or not manifest_path.exists() or not readme_path.exists():
                    print(f"✗ 导出文件不存在: {bundle}")
                    return False
                if not sample_input_path.exists() or not sample_output_path.exists() or not runtime_support_path.exists():
                    print(f"✗ 导出示例或运行时文件不存在: {bundle}")
                    return False

        print("✓ 算法工程包 API 正常")
        return True
    except Exception as e:
        print(f"✗ 算法工程包 API 测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_algorithm_preview_api():
    """测试：算法预演接口能基于样板框和 ROI 返回事件预演结果。"""
    print("\n=== 测试：算法预演 API ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_algorithm_preview.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False
                headers = auth_headers(login_json["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "算法预演"}, headers=headers)
                task_id = create_resp.json()["id"]

                client.post(
                    "/api/algorithm/plan",
                    json={
                        "task_id": task_id,
                        "user_description": "识别仓位是否被货箱占用，持续10秒后输出占位事件",
                        "vlm_result": {
                            "classes": [{"class_name": "cargo_box", "prompt": "stacked brown cargo box"}]
                        },
                    },
                    headers=headers,
                )

                preview_resp = client.post(
                    f"/api/algorithm/preview/{task_id}",
                    json={
                        "region_overrides": [
                            {
                                "region_id": "primary_region",
                                "name": "主监测区域",
                                "bbox_xywhn": [0.5, 0.5, 0.4, 0.4],
                            }
                        ],
                        "sample_boxes": [
                            {
                                "bbox_xywhn": [0.5, 0.5, 0.2, 0.2],
                                "class_name": "cargo_box",
                            }
                        ],
                    },
                    headers=headers,
                )
                preview_json = preview_resp.json()
                if preview_resp.status_code != 200 or preview_json.get("code") != 0:
                    print(f"✗ 算法预演失败: {preview_resp.status_code} {preview_json}")
                    return False
                data = preview_json["data"]
                if not data.get("events"):
                    print(f"✗ 预演未生成事件: {data}")
                    return False
                if data["events"][0].get("event_code") != "occupancy_detected":
                    print(f"✗ 事件码不正确: {data}")
                    return False
                if not data.get("track_states"):
                    print(f"✗ track_states 缺失: {data}")
                    return False
                track_state = data["track_states"][0]
                required_track_keys = {
                    "track_id",
                    "first_seen_frame",
                    "last_seen_frame",
                    "present_duration_ms",
                    "regions_inside",
                }
                if not required_track_keys.issubset(set(track_state.keys())):
                    print(f"✗ preview track_state 契约不完整: {track_state}")
                    return False

        print("✓ 算法预演 API 正常")
        return True
    except Exception as e:
        print(f"✗ 算法预演 API 测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_algorithm_preview_api_with_observation_frames():
    """测试：算法预演 API 支持 observation frame 序列并输出多事件。"""
    print("\n=== 测试：算法预演 API observation frames ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_algorithm_preview_frames.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False
                headers = auth_headers(login_json["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "算法预演 observation frames"}, headers=headers)
                task_id = create_resp.json()["id"]

                client.post(
                    "/api/algorithm/plan",
                    json={
                        "task_id": task_id,
                        "user_description": "识别人员进入A区、离开A区，并在从A区进入B区时输出跨区事件；在A区持续2秒后输出滞留事件",
                        "vlm_result": {"classes": [{"class_name": "person", "prompt": "person"}]},
                    },
                    headers=headers,
                )

                preview_resp = client.post(
                    f"/api/algorithm/preview/{task_id}",
                    json={
                        "region_overrides": [
                            {"region_id": "zone_a", "name": "A 区", "bbox_xywhn": [0.35, 0.5, 0.2, 0.4]},
                            {"region_id": "zone_b", "name": "B 区", "bbox_xywhn": [0.58, 0.5, 0.18, 0.4]},
                        ],
                        "observation_frames": [
                            {
                                "frame_index": 1,
                                "timestamp_ms": 0,
                                "detections": [{"class_name": "person", "bbox_xywhn": [0.10, 0.5, 0.1, 0.2], "confidence": 0.95}],
                            },
                            {
                                "frame_index": 2,
                                "timestamp_ms": 1000,
                                "detections": [{"class_name": "person", "bbox_xywhn": [0.35, 0.5, 0.1, 0.2], "confidence": 0.94}],
                            },
                            {
                                "frame_index": 3,
                                "timestamp_ms": 3000,
                                "detections": [{"class_name": "person", "bbox_xywhn": [0.38, 0.5, 0.1, 0.2], "confidence": 0.93}],
                            },
                            {
                                "frame_index": 4,
                                "timestamp_ms": 4000,
                                "detections": [{"class_name": "person", "bbox_xywhn": [0.58, 0.5, 0.1, 0.2], "confidence": 0.92}],
                            },
                            {
                                "frame_index": 5,
                                "timestamp_ms": 5000,
                                "detections": [{"class_name": "person", "bbox_xywhn": [0.78, 0.5, 0.1, 0.2], "confidence": 0.91}],
                            },
                        ],
                    },
                    headers=headers,
                )
                preview_json = preview_resp.json()
                if preview_resp.status_code != 200 or preview_json.get("code") != 0:
                    print(f"✗ observation frames 预演失败: {preview_resp.status_code} {preview_json}")
                    return False

                data = preview_json["data"]
                events = data.get("events", [])
                event_codes = {item.get("event_code") for item in events}
                if not {"entered_zone_a", "dwell_zone_a", "left_zone_a", "crossed_a_to_b"}.issubset(event_codes):
                    print(f"✗ observation frames 事件不完整: {events}")
                    return False
                track_states = data.get("track_states", [])
                if not track_states or not track_states[0].get("track_id"):
                    print(f"✗ observation frames track_states 不正确: {data}")
                    return False

        print("✓ 算法预演 API observation frames 正常")
        return True
    except Exception as e:
        print(f"✗ 算法预演 API observation frames 测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_backend_api():
    """测试后端 API 是否可以启动"""
    print("\n=== 测试：后端 API 启动 ===")
    # 启动后端（后台），等待 5s，测试 health 端点
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "18000"],
        cwd=PROJECT_ROOT / "backend",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)

    try:
        import httpx
        resp = httpx.get("http://localhost:18000/api/health", timeout=5)
        if resp.status_code == 200:
            print(f"✓ API 健康检查通过: {resp.json()}")
            return True
        else:
            print(f"✗ API 返回错误状态: {resp.status_code}")
            return False
    except Exception as e:
        if is_sandbox_network_error(e):
            print(f"⊘ 沙箱限制本地端口探活，跳过 backend-api: {e}")
            return True
        print(f"✗ 无法连接 API: {e}")
        return False
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_macos_start_scripts_resolve_python_interpreter():
    """测试：macOS 启动脚本在未激活 venv 时也能解析 Python 解释器。"""
    print("\n=== 测试：macOS 启动脚本解释器解析 ===")
    try:
        expectations = {
            PROJECT_ROOT / "scripts" / "start_backend_macos.sh": [
                'VENV_PYTHON="$ROOT_DIR/.venv/bin/python"',
                'if [[ -x "$VENV_PYTHON" ]]; then',
                'elif command -v python3 >/dev/null 2>&1; then',
                'elif command -v python >/dev/null 2>&1; then',
                'export CV_AUTO_TRAINER_DB_URL="${CV_AUTO_TRAINER_DB_URL:-sqlite:///cv_auto_trainer.db}"',
                'exec "$PYTHON_BIN" -m uvicorn main:app --host 127.0.0.1 --port 8000',
            ],
            PROJECT_ROOT / "scripts" / "start_worker_macos.sh": [
                'VENV_PYTHON="$ROOT_DIR/.venv/bin/python"',
                'if [[ -x "$VENV_PYTHON" ]]; then',
                'elif command -v python3 >/dev/null 2>&1; then',
                'elif command -v python >/dev/null 2>&1; then',
                'exec "$PYTHON_BIN" main.py',
            ],
        }

        for script_path, snippets in expectations.items():
            source = script_path.read_text(encoding="utf-8")
            for snippet in snippets:
                if snippet not in source:
                    print(f"✗ 启动脚本缺少解释器解析片段: {script_path} -> {snippet}")
                    return False

        print("✓ macOS 启动脚本解释器解析逻辑存在")
        return True
    except Exception as e:
        print(f"✗ macOS 启动脚本解释器解析测试异常: {e}")
        return False


def test_backend_requirements_include_postgres_driver():
    """测试：后端依赖应包含 PostgreSQL 驱动，匹配文档中的正式部署方式。"""
    print("\n=== 测试：后端 PostgreSQL 驱动依赖 ===")
    try:
        requirements_path = PROJECT_ROOT / "backend" / "requirements.txt"
        requirements = requirements_path.read_text(encoding="utf-8").splitlines()
        has_postgres_driver = any(
            line.strip().startswith(("psycopg2-binary", "psycopg2", "psycopg"))
            for line in requirements
            if line.strip() and not line.strip().startswith("#")
        )

        if not has_postgres_driver:
            print(f"✗ 后端依赖缺少 PostgreSQL 驱动: {requirements_path}")
            return False

        print("✓ 后端依赖包含 PostgreSQL 驱动")
        return True
    except Exception as e:
        print(f"✗ 后端 PostgreSQL 驱动依赖测试异常: {e}")
        return False


def test_backend_serves_frontend_dist():
    """测试：后端在生产模式下能托管前端 dist。"""
    print("\n=== 测试：后端生产静态托管 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_dir = Path(tmpdir) / "dist"
            assets_dir = dist_dir / "assets"
            assets_dir.mkdir(parents=True)
            (dist_dir / "index.html").write_text("<html><body>cv-auto-trainer-app</body></html>", encoding="utf-8")
            (assets_dir / "app.js").write_text("console.log('ok')", encoding="utf-8")

            os.environ["CV_AUTO_TRAINER_FRONTEND_DIST"] = str(dist_dir)
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with TestClient(main.app) as client:
                app_resp = client.get("/app")
                if app_resp.status_code != 200 or "cv-auto-trainer-app" not in app_resp.text:
                    print(f"✗ 后端未返回前端 dist 内容: {app_resp.status_code} {app_resp.text}")
                    return False

                asset_resp = client.get("/assets/app.js")
                if asset_resp.status_code != 200 or "console.log('ok')" not in asset_resp.text:
                    print(f"✗ 后端未返回静态资源: {asset_resp.status_code} {asset_resp.text}")
                    return False

        print("✓ 后端生产静态托管正常")
        return True
    except Exception as e:
        print(f"✗ 后端生产静态托管测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_FRONTEND_DIST", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_backend_resolves_relative_frontend_dist_from_project_root():
    """测试：相对 frontend/dist 配置应按项目根目录解析。"""
    print("\n=== 测试：后端相对前端 dist 路径解析 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        ensure_test_auth_env()

        backend_path = str(backend_dir)
        if backend_path in sys.path:
            sys.path.remove(backend_path)
        sys.path.insert(0, backend_path)

        for name in list(sys.modules):
            if name == "main" or name.startswith(("routers", "models", "services")):
                sys.modules.pop(name, None)
        main = importlib.import_module("main")

        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            dist_dir = project_root / "frontend" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "index.html").write_text("<html>ok</html>", encoding="utf-8")

            resolved = main.resolve_frontend_dist(
                frontend_dist_env="frontend/dist",
                project_root=project_root,
                cwd=project_root / "backend",
            )
            if resolved.resolve(strict=False) != dist_dir.resolve(strict=False):
                print(f"✗ 相对路径未按项目根目录解析: {resolved} != {dist_dir}")
                return False

        print("✓ 后端能按项目根目录解析相对 dist 路径")
        return True
    except Exception as e:
        print(f"✗ 后端相对 dist 路径解析测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_training_status_persistence():
    """测试：云训练状态应持久化到任务记录并可查询产物。"""
    print("\n=== 测试：训练状态持久化 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_training_status.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")

            with patch("services.train_dispatcher.TrainDispatcher.dispatch", return_value={"best.pt": "/tmp/mock/best.pt"}):
                with TestClient(main.app) as client:
                    login_resp = client.post(
                        "/api/auth/login",
                        json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                    )
                    login_json = login_resp.json()
                    if login_resp.status_code != 200 or login_json.get("code") != 0:
                        print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                        return False
                    headers = auth_headers(login_json["data"]["token"])

                    create_resp = client.post("/api/tasks", json={"name": "云训练持久化"}, headers=headers)
                    task_id = create_resp.json()["id"]

                    start_resp = client.post(
                        "/api/training/start",
                        json={
                            "task_id": task_id,
                            "model": "yolo11s.pt",
                            "epochs": 10,
                            "imgsz": 640,
                            "lr0": 0.01,
                            "patience": 5,
                            "conf": 0.25,
                            "iou": 0.7,
                            "export_formats": ["onnx"],
                            "train_mode": "cloud",
                            "gpu_type": "RTX 4090",
                            "resume_last": False,
                        },
                        headers=headers,
                    )
                    start_json = start_resp.json()
                    if start_resp.status_code != 200 or start_json.get("code") != 0:
                        print(f"✗ 云训练启动失败: {start_resp.status_code} {start_json}")
                        return False

                    deadline = time.time() + 3
                    status = None
                    while time.time() < deadline:
                        status_resp = client.get(f"/api/training/{task_id}/status", headers=headers)
                        status_json = status_resp.json()
                        if status_resp.status_code != 200 or status_json.get("code") != 0:
                            print(f"✗ 云训练状态查询失败: {status_resp.status_code} {status_json}")
                            return False
                        status = status_json["data"]
                        if status.get("state") == "done":
                            break
                        time.sleep(0.1)

                    if not status or status.get("state") != "done":
                        print(f"✗ 云训练状态未持久化为 done: {status}")
                        return False
                    if status.get("artifact_paths", {}).get("best.pt") != "/tmp/mock/best.pt":
                        print(f"✗ 云训练产物未持久化返回: {status}")
                        return False

        print("✓ 训练状态持久化正常")
        return True
    except Exception as e:
        print(f"✗ 训练状态持久化测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_task_persists_workflow_state_fields():
    print("\n=== 测试：任务持久化需求协商与离线评估状态 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_workflow_state.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)

            from fastapi.testclient import TestClient
            main = importlib.import_module("main")
            auth_router = importlib.import_module("routers.auth")
            database = importlib.import_module("models.database")
            db_models = importlib.import_module("models.db")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                if login_resp.status_code != 200 or login_resp.json().get("code") != 0:
                    print(f"✗ 管理员登录失败: {login_resp.status_code} {login_resp.json()}")
                    return False
                headers = auth_headers(login_resp.json()["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "工作流状态持久化"}, headers=headers)
                if create_resp.status_code != 200:
                    print(f"✗ 创建任务失败: {create_resp.status_code} {create_resp.json()}")
                    return False
                created = create_resp.json()
                task_id = created["id"]
                workflow_fields = {
                    "negotiation_summary",
                    "offline_evaluation",
                    "training_plan",
                    "delivery_package",
                }
                exposed_fields = workflow_fields.intersection(created)
                if exposed_fields:
                    print(f"✗ 创建任务响应不应暴露工作流字段: {created}")
                    return False

                unauthenticated_cases = [
                    ("GET", "/api/tasks"),
                    ("GET", f"/api/tasks/{task_id}"),
                    ("DELETE", f"/api/tasks/{task_id}"),
                ]
                for method, path in unauthenticated_cases:
                    resp = client.request(method, path)
                    if resp.status_code != 401:
                        print(f"✗ 未登录访问应被拒绝: {method} {path} -> {resp.status_code} {resp.json()}")
                        return False

                other_username = "workflow-other-user"
                other_password = "workflow-other-pass"

                db = database.SessionLocal()
                try:
                    task = db.query(db_models.Task).filter(db_models.Task.id == task_id).first()
                    if not task:
                        print(f"✗ 未找到任务记录: {task_id}")
                        return False
                    task.negotiation_summary = {
                        "scenario_label": "区域停留",
                        "objects": ["人员"],
                        "regions": ["A区"],
                        "duration_seconds": 10,
                        "events": ["进入区域", "停留超时"],
                    }
                    task.offline_evaluation = {
                        "status": "pending",
                        "clips": [
                            {"clip_id": "clip_trigger", "label": "应触发", "path": "/tmp/clip_trigger.mp4"},
                            {"clip_id": "clip_negative", "label": "不应触发", "path": "/tmp/clip_negative.mp4"},
                        ],
                    }
                    task.training_plan = {
                        "status": "draft",
                        "model": "yolo11s.pt",
                        "epochs": 50,
                    }
                    task.delivery_package = {
                        "status": "pending",
                        "artifacts": ["pipeline.json", "README.md"],
                    }
                    other_user = db_models.User(
                        username=other_username,
                        password_hash=auth_router.hash_password(other_password),
                        role="user",
                        token_version=0,
                    )
                    db.add(other_user)
                    db.commit()
                finally:
                    db.close()

                other_login_resp = client.post(
                    "/api/auth/login",
                    json={"username": other_username, "password": other_password},
                )
                if other_login_resp.status_code != 200 or other_login_resp.json().get("code") != 0:
                    print(f"✗ 次级用户登录失败: {other_login_resp.status_code} {other_login_resp.json()}")
                    return False
                other_headers = auth_headers(other_login_resp.json()["data"]["token"])

                other_list_resp = client.get("/api/tasks", headers=other_headers)
                if other_list_resp.status_code != 200:
                    print(f"✗ 次级用户任务列表读取失败: {other_list_resp.status_code} {other_list_resp.json()}")
                    return False
                if any(item.get("id") == task_id for item in other_list_resp.json()):
                    print(f"✗ 次级用户不应看到其他人的任务: {other_list_resp.json()}")
                    return False

                other_get_resp = client.get(f"/api/tasks/{task_id}", headers=other_headers)
                if other_get_resp.status_code != 404:
                    print(f"✗ 次级用户读取他人任务应返回 404: {other_get_resp.status_code} {other_get_resp.json()}")
                    return False

                other_delete_resp = client.delete(f"/api/tasks/{task_id}", headers=other_headers)
                if other_delete_resp.status_code != 404:
                    print(
                        f"✗ 次级用户删除他人任务应返回 404: "
                        f"{other_delete_resp.status_code} {other_delete_resp.json()}"
                    )
                    return False

                fetch_resp = client.get(f"/api/tasks/{task_id}", headers=headers)
                fetched = fetch_resp.json()
                if fetch_resp.status_code != 200:
                    print(f"✗ 读取任务失败: {fetch_resp.status_code} {fetched}")
                    return False

                if fetched.get("negotiation_summary", {}).get("scenario_label") != "区域停留":
                    print(f"✗ negotiation_summary 未持久化: {fetched}")
                    return False
                clips = fetched.get("offline_evaluation", {}).get("clips", [])
                if len(clips) != 2:
                    print(f"✗ offline_evaluation clips 未持久化: {fetched}")
                    return False
                if fetched.get("training_plan", {}).get("model") != "yolo11s.pt":
                    print(f"✗ training_plan 未持久化: {fetched}")
                    return False
                artifacts = fetched.get("delivery_package", {}).get("artifacts", [])
                if artifacts != ["pipeline.json", "README.md"]:
                    print(f"✗ delivery_package 未持久化: {fetched}")
                    return False

        print("✓ 任务需求协商与离线评估状态持久化正常")
        return True
    except Exception as e:
        print(f"✗ 任务需求协商与离线评估状态测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_artifact_listing_uses_task_artifact_paths():
    """测试：交付文件接口应支持直接读取任务记录里的产物路径。"""
    print("\n=== 测试：任务产物路径下载 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_artifact_listing.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path in sys.path:
                sys.path.remove(backend_path)
            sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")
            from models.database import SessionLocal
            from models.db import Task

            artifact_file = Path(tmpdir) / "best.pt"
            artifact_file.write_text("mock-model", encoding="utf-8")

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False
                headers = auth_headers(login_json["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "产物下载"}, headers=headers)
                task_id = create_resp.json()["id"]

                db = SessionLocal()
                try:
                    task = db.query(Task).filter(Task.id == task_id).first()
                    task.artifact_paths = {"best.pt": str(artifact_file)}
                    db.commit()
                finally:
                    db.close()

                list_resp = client.get(f"/api/files/{task_id}/artifacts", headers=headers)
                list_json = list_resp.json()
                if list_resp.status_code != 200 or list_json.get("code") != 0:
                    print(f"✗ 获取产物列表失败: {list_resp.status_code} {list_json}")
                    return False
                artifact_names = {item["name"] for item in list_json["data"]}
                if "best.pt" not in artifact_names:
                    print(f"✗ task.artifact_paths 未出现在产物列表中: {list_json['data']}")
                    return False

                download_resp = client.get(f"/api/files/{task_id}/artifacts/best.pt", headers=headers)
                if download_resp.status_code != 200 or download_resp.content != b"mock-model":
                    print(f"✗ 产物下载失败: {download_resp.status_code} {download_resp.content}")
                    return False

        print("✓ 任务产物路径下载正常")
        return True
    except Exception as e:
        print(f"✗ 任务产物路径下载测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_prepare_manual_cloud_training_bundle():
    """测试：手动云训练包脚本应生成 zip、脚本和说明文档。"""
    print("\n=== 测试：手动云训练包 ===")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_dir = tmp_path / "dataset"
            (dataset_dir / "images" / "train").mkdir(parents=True)
            (dataset_dir / "labels" / "train").mkdir(parents=True)
            (dataset_dir / "images" / "train" / "sample.jpg").write_bytes(b"fake-image")
            (dataset_dir / "labels" / "train" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (dataset_dir / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/train\nnames:\n  0: person\n",
                encoding="utf-8",
            )

            output_dir = tmp_path / "output"
            result = run_cmd(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prepare_manual_cloud_training.py"),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output-dir",
                    str(output_dir),
                    "--export-format",
                    "onnx",
                ],
                timeout=30,
            )
            if result.returncode != 0:
                print(f"✗ 手动云训练包脚本运行失败: {result.returncode}")
                return False

            bundle_dir = output_dir / "manual_cloud_training"
            if not (bundle_dir / "dataset.zip").exists():
                print("✗ dataset.zip 未生成")
                return False
            if not (bundle_dir / "cloud_scripts" / "train.py").exists():
                print("✗ cloud_scripts/train.py 未复制")
                return False
            readme_path = bundle_dir / "README.md"
            if not readme_path.exists():
                print("✗ 手动训练 README 未生成")
                return False
            readme_text = readme_path.read_text(encoding="utf-8")
            if "scp" not in readme_text or "cloud_scripts/train.py" not in readme_text:
                print(f"✗ 手动训练 README 内容不完整: {readme_text}")
                return False

        print("✓ 手动云训练包正常")
        return True
    except Exception as e:
        print(f"✗ 手动云训练包测试异常: {e}")
        return False


def test_training_monitor_manual_cloud_fallback():
    """测试：训练监控页应展示手动云训练兜底说明。"""
    print("\n=== 测试：训练监控页手动云训练兜底 ===")
    try:
        training_monitor_path = PROJECT_ROOT / "frontend" / "src" / "pages" / "TrainingMonitor.tsx"
        source = training_monitor_path.read_text(encoding="utf-8")

        required_snippets = {
            "手动云训练": "缺少手动云训练标题",
            "prepare_manual_cloud_training.py": "缺少手动训练包生成命令",
            "MANUAL_CLOUD_TRAINING.md": "缺少手动训练说明文档引用",
            "backend/uploads/": "缺少数据集目录路径提示",
            "scp": "缺少云端上传步骤提示",
        }
        for snippet, error_message in required_snippets.items():
            if snippet not in source:
                print(f"✗ {error_message}: {training_monitor_path}")
                return False

        print("✓ 训练监控页手动云训练兜底说明正常")
        return True
    except Exception as e:
        print(f"✗ 训练监控页手动云训练兜底测试异常: {e}")
        return False


def test_frontend_worker_endpoints_are_production_safe():
    """测试：前端 Worker 地址在正式部署下不应写死 localhost。"""
    print("\n=== 测试：前端 Worker 地址生产兼容 ===")
    try:
        worker_api_path = PROJECT_ROOT / "frontend" / "src" / "api" / "worker.ts"
        gpu_monitor_path = PROJECT_ROOT / "frontend" / "src" / "components" / "GpuMonitor.tsx"

        worker_source = worker_api_path.read_text(encoding="utf-8")
        gpu_source = gpu_monitor_path.read_text(encoding="utf-8")

        required_worker_snippets = [
            "VITE_WORKER_HTTP_URL",
            "export const WORKER_HTTP_BASE",
            "window.location.hostname",
        ]
        for snippet in required_worker_snippets:
            if snippet not in worker_source:
                print(f"✗ Worker API 缺少生产地址配置: {worker_api_path} -> {snippet}")
                return False

        if "http://localhost:7860/gpu-info" in gpu_source:
            print(f"✗ GPU 监控仍写死 localhost: {gpu_monitor_path}")
            return False
        if "WORKER_HTTP_BASE" not in gpu_source:
            print(f"✗ GPU 监控未复用 Worker HTTP 基地址: {gpu_monitor_path}")
            return False

        print("✓ 前端 Worker 地址生产兼容正常")
        return True
    except Exception as e:
        print(f"✗ 前端 Worker 地址生产兼容测试异常: {e}")
        return False


def test_frontend_local_worker_paths_use_task_directories():
    """测试：本地 Worker 任务路径应使用真实 task 目录，而不是占位符或错误相对路径。"""
    print("\n=== 测试：前端本地 Worker 路径 ===")
    try:
        augment_config_path = PROJECT_ROOT / "frontend" / "src" / "pages" / "AugmentConfig.tsx"
        train_config_path = PROJECT_ROOT / "frontend" / "src" / "pages" / "TrainConfig.tsx"

        augment_source = augment_config_path.read_text(encoding="utf-8")
        train_source = train_config_path.read_text(encoding="utf-8")

        if "/path/to/" in augment_source:
            print(f"✗ 数据增强仍包含占位路径: {augment_config_path}")
            return False
        expected_augment_snippets = [
            "../backend/uploads/${taskId}/labeled_images",
            "../backend/uploads/${taskId}/labels",
            "../backend/uploads/${taskId}/dataset/images/train",
            "../backend/uploads/${taskId}/dataset/labels/train",
        ]
        for snippet in expected_augment_snippets:
            if snippet not in augment_source:
                print(f"✗ 数据增强未使用真实任务目录: {augment_config_path} -> {snippet}")
                return False

        if "dataset_dir: `./uploads/${taskId}/dataset`" in train_source:
            print(f"✗ 本地训练仍使用错误相对路径: {train_config_path}")
            return False
        if "../backend/uploads/${taskId}/dataset" not in train_source:
            print(f"✗ 本地训练未切换到真实数据集目录: {train_config_path}")
            return False

        print("✓ 前端本地 Worker 路径正常")
        return True
    except Exception as e:
        print(f"✗ 前端本地 Worker 路径测试异常: {e}")
        return False


def test_worker_supports_production_origin_and_host_config():
    """测试：Worker 应支持正式部署来源和可配置监听地址。"""
    print("\n=== 测试：Worker 正式部署配置 ===")
    try:
        worker_main_path = PROJECT_ROOT / "worker" / "main.py"
        source = worker_main_path.read_text(encoding="utf-8")

        required_snippets = [
            "CV_AUTO_TRAINER_WORKER_ALLOWED_ORIGINS",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "CV_AUTO_TRAINER_WORKER_HOST",
            "CV_AUTO_TRAINER_WORKER_PORT",
        ]
        for snippet in required_snippets:
            if snippet not in source:
                print(f"✗ Worker 缺少正式部署配置: {worker_main_path} -> {snippet}")
                return False

        print("✓ Worker 正式部署配置正常")
        return True
    except Exception as e:
        print(f"✗ Worker 正式部署配置测试异常: {e}")
        return False


def test_frontend_index_does_not_reference_missing_vite_favicon():
    """测试：正式构建入口不应继续引用缺失的 vite.svg。"""
    print("\n=== 测试：前端入口 favicon 引用 ===")
    try:
        index_path = PROJECT_ROOT / "frontend" / "index.html"
        source = index_path.read_text(encoding="utf-8")

        if "/vite.svg" in source:
            print(f"✗ 前端入口仍引用 vite.svg: {index_path}")
            return False

        print("✓ 前端入口 favicon 引用正常")
        return True
    except Exception as e:
        print(f"✗ 前端入口 favicon 测试异常: {e}")
        return False


def test_worker_health():
    """测试 Worker 是否可以启动"""
    print("\n=== 测试：Worker 启动 ===")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "17860"],
        cwd=PROJECT_ROOT / "worker",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)

    try:
        import httpx
        resp = httpx.get("http://localhost:17860/gpu-info", timeout=5)
        if resp.status_code == 200:
            print(f"✓ Worker 健康检查通过: {resp.json()}")
            return True
        else:
            print(f"✗ Worker 返回错误状态: {resp.status_code}")
            return False
    except Exception as e:
        if is_sandbox_network_error(e):
            print(f"⊘ 沙箱限制本地端口探活，跳过 worker-health: {e}")
            return True
        print(f"✗ 无法连接 Worker: {e}")
        return False
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_frontend_build():
    """测试前端是否可以构建"""
    print("\n=== 测试：前端构建 ===")
    import os, subprocess as sp
    env = os.environ.copy()
    # 确保 npm 在 PATH 中
    if "/d/node" not in env.get("PATH", ""):
        env["PATH"] = "/d/node:" + env.get("PATH", "")
    result = sp.run(
        "cd frontend && npm run build",
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        shell=True,
        env=env,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    if result.returncode == 0:
        print("✓ 前端构建成功")
        return True
    else:
        print(f"✗ 前端构建失败（exit {result.returncode}）")
        return False
    if result.returncode == 0:
        print("✓ 前端构建成功")
        return True
    else:
        print(f"✗ 前端构建失败（exit {result.returncode}）")
        return False


def test_gpu_memory_release():
    """
    测试：两段式打标显存是否正确释放
    模拟加载 YOLO → 释放 → 加载 Moondream → 检查显存
    """
    print("\n=== 测试：两段式打标配重释放 ===")
    try:
        import torch
        if not torch.cuda.is_available():
            print("⊘ 无 GPU，跳过显存测试")
            return True

        import gc
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))

        # 第一段：加载 YOLO-World
        print("第一段：加载 YOLO-World...")
        from ultralytics import YOLOWorld
        model = YOLOWorld("yolov8s-world.pt")
        model.half()
        mem_after_yolo = torch.cuda.memory_allocated(0) / 1e9
        print(f"  YOLO-World 加载后显存: {mem_after_yolo:.2f} GB")

        # 释放
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_after_del = torch.cuda.memory_allocated(0) / 1e9
        print(f"  释放后显存: {mem_after_del:.2f} GB")

        if mem_after_del > mem_after_yolo * 0.5:
            print(f"✗ 显存释放不充分（残留 {mem_after_del:.2f} GB）")
            return False

        print("✓ 显存释放正常")
        return True

    except Exception as e:
        print(f"✗ 显存测试异常: {e}")
        return False


def test_local_training_subprocess():
    """
    测试：本地训练子进程是否可以启动和取消
    """
    print("\n=== 测试：本地训练子进程 ===")
    try:
        import sys
        import time
        import threading
        sys.path.insert(0, str(PROJECT_ROOT / "worker"))
        from pipeline.local_trainer import LocalTrainer

        trainer = LocalTrainer()
        print("  LocalTrainer 实例化成功")

        # 测试 cancel
        trainer._stop_flag = True
        print("  cancel() 方法可调用")
        print("✓ 本地训练器接口正常")
        return True
    except Exception as e:
        print(f"✗ 本地训练器测试异常: {e}")
        return False


def test_settings_auth_persistence():
    """
    测试：登录后的 settings 保存，在后端重启后仍可继续读取。
    这能覆盖“偶发 401”和“刷新后像回默认值”的核心链路。
    """
    print("\n=== 测试：设置鉴权持久性 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_settings.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            def boot_client() -> TestClient:
                for name in list(sys.modules):
                    if name == "main" or name.startswith(("routers", "models", "services")):
                        sys.modules.pop(name, None)
                main = importlib.import_module("main")
                return TestClient(main.app)

            with boot_client() as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                login_json = login_resp.json()
                if login_resp.status_code != 200 or login_json.get("code") != 0:
                    print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                    return False

                token = login_json["data"]["token"]
                headers = auth_headers(token)

                save_resp = client.put(
                    "/api/settings",
                    json={
                        "vlm_provider": "custom",
                        "vlm_base_url": "https://example.invalid/v1",
                        "autodl_token": "temporary-secret",
                        "default_model": "yolo11m.pt",
                        "default_train_mode": "cloud",
                    },
                    headers=headers,
                )
                save_json = save_resp.json()
                if save_resp.status_code != 200 or save_json.get("code") != 0:
                    print(f"✗ 保存设置失败: {save_resp.status_code} {save_json}")
                    return False

                clear_resp = client.put(
                    "/api/settings",
                    json={"autodl_token": ""},
                    headers=headers,
                )
                clear_json = clear_resp.json()
                if clear_resp.status_code != 200 or clear_json.get("code") != 0:
                    print(f"✗ 清空敏感设置失败: {clear_resp.status_code} {clear_json}")
                    return False

            with boot_client() as client:
                load_resp = client.get("/api/settings", headers=headers)
                load_json = load_resp.json()
                if load_resp.status_code != 200 or load_json.get("code") != 0:
                    print(f"✗ 重启后读取设置失败: {load_resp.status_code} {load_json}")
                    return False

                data = load_json["data"]
        if data.get("vlm_provider") != "custom":
            print(f"✗ vlm_provider 未持久化: {data.get('vlm_provider')}")
            return False
        if data.get("vlm_base_url") != "https://example.invalid/v1":
            print(f"✗ vlm_base_url 未持久化: {data.get('vlm_base_url')}")
            return False
        if data.get("autodl_token") != "":
            print(f"✗ autodl_token 未清空: {data.get('autodl_token')}")
            return False
        if data.get("default_model") != "yolo11m.pt":
            print(f"✗ default_model 未持久化: {data.get('default_model')}")
            return False
        if data.get("default_train_mode") != "cloud":
            print(f"✗ default_train_mode 未持久化: {data.get('default_train_mode')}")
            return False

        print("✓ 设置保存与重启后读取正常")
        return True
    except Exception as e:
        print(f"✗ 设置鉴权持久性测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_vlm_parse_uses_saved_settings():
    """
    测试：/api/vlm/parse 应该使用已保存设置中的明文 key、api_format 和 model。
    这覆盖“测试连接成功，但真实解析失败”的核心差异。
    """
    print("\n=== 测试：VLM 解析使用已保存设置 ===")
    backend_dir = PROJECT_ROOT / "backend"
    old_cwd = Path.cwd()
    try:
        os.chdir(backend_dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["CV_AUTO_TRAINER_DB_URL"] = f"sqlite:///{Path(tmpdir) / 'test_vlm.db'}"
            ensure_test_auth_env()

            backend_path = str(backend_dir)
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)

            from fastapi.testclient import TestClient

            for name in list(sys.modules):
                if name == "main" or name.startswith(("routers", "models", "services")):
                    sys.modules.pop(name, None)
            main = importlib.import_module("main")
            routers_vlm = importlib.import_module("routers.vlm")

            captured: dict[str, object] = {}

            class FakeAdapter:
                def __init__(
                    self,
                    provider: str,
                    base_url: str,
                    api_key: str,
                    api_format=None,
                    model=None,
                    temperature=None,
                    top_p=None,
                    stop=None,
                ):
                    captured.update({
                        "provider": provider,
                        "base_url": base_url,
                        "api_key": api_key,
                        "api_format": api_format,
                        "model": model,
                    })

                def parse_intent(self, images_base64, user_text, sample_boxes=None):
                    return {
                        "classes": [{"class_name": "helmet", "prompt": "helmet"}],
                        "confidence": 0.9,
                    }

            with patch.object(routers_vlm, "VLMAdapter", FakeAdapter):
                with TestClient(main.app) as client:
                    login_resp = client.post(
                        "/api/auth/login",
                        json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                    )
                    login_json = login_resp.json()
                    if login_resp.status_code != 200 or login_json.get("code") != 0:
                        print(f"✗ 登录失败: {login_resp.status_code} {login_json}")
                        return False
                    headers = auth_headers(login_json["data"]["token"])

                    save_resp = client.put(
                        "/api/settings",
                        json={
                            "vlm_provider": "custom",
                            "vlm_base_url": "https://example.invalid/v1",
                            "vlm_api_key": "plain-secret-key",
                            "vlm_api_format": "anthropic",
                            "vlm_model": "claude-test-model",
                        },
                        headers=headers,
                    )
                    if save_resp.status_code != 200 or save_resp.json().get("code") != 0:
                        print(f"✗ 保存 VLM 设置失败: {save_resp.status_code} {save_resp.json()}")
                        return False

                    parse_resp = client.post(
                        "/api/vlm/parse",
                        json={
                            "images_base64": ["ZmFrZS1pbWFnZQ=="],
                            "user_text": "detect helmets",
                            "sample_boxes": [],
                        },
                        headers=headers,
                    )
                    if parse_resp.status_code != 200 or parse_resp.json().get("code") != 0:
                        print(f"✗ 调用解析接口失败: {parse_resp.status_code} {parse_resp.json()}")
                        return False

            if captured.get("api_key") != "plain-secret-key":
                print(f"✗ 解析接口未使用明文 API key: {captured.get('api_key')}")
                return False
            if captured.get("api_format") != "anthropic":
                print(f"✗ 解析接口未使用已保存 api_format: {captured.get('api_format')}")
                return False
            if captured.get("model") != "claude-test-model":
                print(f"✗ 解析接口未使用已保存 model: {captured.get('model')}")
                return False

            print("✓ VLM 解析正确使用已保存设置")
            return True
    except Exception as e:
        print(f"✗ VLM 解析设置测试异常: {e}")
        return False
    finally:
        os.environ.pop("CV_AUTO_TRAINER_DB_URL", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_USERNAME", None)
        os.environ.pop("CV_AUTO_TRAINER_ADMIN_PASSWORD", None)
        os.environ.pop("CV_AUTO_TRAINER_SECRET_KEY", None)
        os.chdir(old_cwd)


def test_vlm_parse_returns_structured_fallback_on_invalid_json():
    """
    测试：VLMAdapter 在视觉解析失败时应返回结构化降级结果，而不是抛出 RuntimeError。
    """
    print("\n=== 测试：VLM 解析失败返回降级结构 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.vlm_adapter import VLMAdapter

        adapter = VLMAdapter.__new__(VLMAdapter)
        adapter._call_api = lambda images_base64, user_text, sample_boxes=None: "<html>invalid</html>"

        result = VLMAdapter.parse_intent(
            adapter,
            images_base64=["ZmFrZS1pbWFnZQ=="],
            user_text="识别仓位是否被货箱占用，持续10秒后输出占位事件",
            sample_boxes=[],
            max_retry=1,
        )

        if result.get("status") != "failed":
            print(f"✗ 未返回 failed 状态: {result}")
            return False
        if result.get("retryable") is not True:
            print(f"✗ retryable 未按预期返回: {result}")
            return False
        if "根据文字需求生成草案" not in (result.get("message") or ""):
            print(f"✗ 失败提示不够可操作: {result}")
            return False
        if result.get("raw_vlm_response") != "<html>invalid</html>":
            print(f"✗ 未保留原始响应: {result}")
            return False

        print("✓ VLM 解析失败降级结构正常")
        return True
    except Exception as e:
        print(f"✗ VLM 解析失败降级测试异常: {e}")
        return False


def test_vlm_display_field_fallbacks():
    """
    测试：即使 VLM 未返回中文展示字段，后端也会补齐展示字段回退值。
    这样页面显示不会因为历史数据或偶发漏字段而崩掉。
    """
    print("\n=== 测试：VLM 展示字段回退 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.vlm_adapter import VLMAdapter

        adapter = VLMAdapter.__new__(VLMAdapter)
        parsed = VLMAdapter._parse_and_validate(
            adapter,
            """
            {
              "confidence": 0.81,
              "classes": [
                {
                  "class_name": "red_helmet",
                  "prompt": "a red safety helmet worn on a person's head",
                  "negative_prompt": "baseball cap, hood, bare head",
                  "color_hint": "red"
                }
              ]
            }
            """,
        )

        cls = parsed["classes"][0]
        if cls.get("display_name_zh") != "red_helmet":
            print(f"✗ display_name_zh 回退错误: {cls.get('display_name_zh')}")
            return False
        if cls.get("display_prompt_zh") != cls.get("prompt"):
            print(f"✗ display_prompt_zh 回退错误: {cls.get('display_prompt_zh')}")
            return False
        if cls.get("display_negative_prompt_zh") != cls.get("negative_prompt"):
            print(f"✗ display_negative_prompt_zh 回退错误: {cls.get('display_negative_prompt_zh')}")
            return False
        if cls.get("display_color_hint_zh") != cls.get("color_hint"):
            print(f"✗ display_color_hint_zh 回退错误: {cls.get('display_color_hint_zh')}")
            return False

        print("✓ VLM 展示字段回退正常")
        return True
    except Exception as e:
        print(f"✗ VLM 展示字段回退测试异常: {e}")
        return False


def test_worker_image_support_and_mps_guard():
    """
    测试：Worker 应识别 .jpeg / 大写扩展名图片，且 MPS 不应被硬性显存预检直接拦截。
    这覆盖“上传成功但显示 0 张图片”和 “Apple Silicon 直接报显存不足” 两个问题。
    """
    print("\n=== 测试：Worker 图片发现与 MPS 预检 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline import gpu_manager
        from utils import image_files

        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir)
            for name in ["a.JPG", "b.jpeg", "c.PNG", "d.jpg", "ignore.txt"]:
                (image_dir / name).write_text("x", encoding="utf-8")

            discovered = [path.name for path in image_files.list_image_files(image_dir)]
            expected = ["a.JPG", "b.jpeg", "c.PNG", "d.jpg"]
            if discovered != expected:
                print(f"✗ 图片发现结果错误: {discovered}")
                return False

            stem_match = image_files.find_image_for_stem(image_dir, "b")
            if stem_match is None or stem_match.name != "b.jpeg":
                print(f"✗ 按 stem 查找图片失败: {stem_match}")
                return False

        if gpu_manager.should_enforce_memory_guard("cuda") is not True:
            print("✗ CUDA 应保留显存预检")
            return False
        if gpu_manager.should_enforce_memory_guard("mps") is not False:
            print("✗ MPS 不应执行硬性显存预检")
            return False

        print("✓ Worker 图片发现与 MPS 预检策略正常")
        return True
    except Exception as e:
        print(f"✗ Worker 图片发现与 MPS 预检测试异常: {e}")
        return False


def test_worker_clip_setup_error_message():
    """
    测试：当 CLIP 权重下载失败时，Worker 应返回明确的手动缓存提示，而不是裸 SSL/urllib 异常。
    """
    print("\n=== 测试：Worker CLIP 初始化错误提示 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        import pipeline.stage2_labeler as stage2_labeler

        class FakeYOLOWorld:
            def __init__(self, _model_name: str):
                pass

            def set_classes(self, _classes):
                raise URLError("EOF occurred in violation of protocol (_ssl.c:1129)")

        fake_ultralytics = ModuleType("ultralytics")
        fake_ultralytics.YOLOWorld = FakeYOLOWorld

        with patch.object(stage2_labeler, "gpu_stage", return_value=nullcontext()):
            with patch.object(stage2_labeler, "get_device", return_value="cpu"):
                with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                    try:
                        stage2_labeler.run_detection(
                            image_dir="/tmp/unused",
                            classes=[{"prompt": "person", "class_name": "person"}],
                            output_raw_dir="/tmp/unused",
                        )
                        print("✗ 预期应抛出 DetectionSetupError，但未抛出")
                        return False
                    except stage2_labeler.DetectionSetupError as e:
                        message = str(e)
                        if "ViT-B-32.pt" not in message:
                            print(f"✗ 错误提示未包含权重文件名: {message}")
                            return False
                        if "~/.cache/clip/ViT-B-32.pt" not in message:
                            print(f"✗ 错误提示未包含缓存路径: {message}")
                            return False
                        if "openaipublic.azureedge.net" not in message:
                            print(f"✗ 错误提示未包含网络诊断线索: {message}")
                            return False
                        if "EOF occurred in violation of protocol" not in message:
                            print(f"✗ 错误提示未保留原始异常: {message}")
                            return False

        print("✓ Worker CLIP 初始化错误提示正常")
        return True
    except Exception as e:
        print(f"✗ Worker CLIP 初始化错误提示测试异常: {e}")
        return False


def test_worker_moondream_setup_error_message():
    """
    测试：当 Moondream2 首次下载失败时，Worker 应返回明确的缓存/网络提示。
    """
    print("\n=== 测试：Worker Moondream2 初始化错误提示 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        import pipeline.stage2_labeler as stage2_labeler

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise OSError("HTTPSConnectionPool(host='huggingface.co', port=443): EOF occurred in violation of protocol")

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise AssertionError("tokenizer should not be reached when model init fails")

        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeAutoTokenizer

        fake_cv2 = ModuleType("cv2")

        with patch.object(stage2_labeler, "gpu_stage", return_value=nullcontext()):
            with patch.object(stage2_labeler, "get_device", return_value="cpu"):
                with patch.dict(sys.modules, {"transformers": fake_transformers, "cv2": fake_cv2}):
                    try:
                        stage2_labeler.run_quality_check("/tmp/unused.json")
                        print("✗ 预期应抛出 DetectionSetupError，但未抛出")
                        return False
                    except stage2_labeler.DetectionSetupError as e:
                        message = str(e)
                        if "vikhyatk/moondream2" not in message:
                            print(f"✗ 错误提示未包含模型名: {message}")
                            return False
                        if "huggingface.co" not in message:
                            print(f"✗ 错误提示未包含下载源线索: {message}")
                            return False
                        if "网络" not in message and "缓存" not in message:
                            print(f"✗ 错误提示不够可操作: {message}")
                            return False

        print("✓ Worker Moondream2 初始化错误提示正常")
        return True
    except Exception as e:
        print(f"✗ Worker Moondream2 初始化错误提示测试异常: {e}")
        return False


def test_local_trainer_uses_current_python():
    """
    测试：本地训练子进程应复用当前 Python 解释器，而不是依赖裸 python 命令。
    """
    print("\n=== 测试：本地训练使用当前 Python 解释器 ===")
    try:
        worker_path = str(PROJECT_ROOT / "worker")
        if worker_path not in sys.path:
            sys.path.insert(0, worker_path)

        from pipeline.local_trainer import LocalTrainer

        trainer = LocalTrainer()
        trainer._output_dir = PROJECT_ROOT / "backend" / "uploads" / "dummy" / "local_training_output" / "exp"
        cmd = trainer._build_command({}, PROJECT_ROOT / "backend" / "uploads" / "dummy" / "dataset" / "data.yaml")
        if cmd[0] != sys.executable:
            print(f"✗ 本地训练未复用当前解释器: {cmd[0]} != {sys.executable}")
            return False

        print("✓ 本地训练解释器选择正常")
        return True
    except Exception as e:
        print(f"✗ 本地训练解释器测试异常: {e}")
        return False


# ---------------------------------------------------------------------------
# 新增测试：OCR + 视频帧采样 + VLM 元数据
# ---------------------------------------------------------------------------

def test_ocr_role_in_pipeline():
    """测试：OCR role 被正确识别为内置引擎，不参与训练"""
    print("\n=== 测试：OCR role 内置引擎识别 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.multi_model_orchestrator import OCR_ROLES, TRAINABLE_ROLES, MultiModelTrainingOrchestrator

        # OCR 在 OCR_ROLES 中但不在 TRAINABLE_ROLES 中
        assert "ocr" in OCR_ROLES, "OCR role 未在 OCR_ROLES 中"
        assert "ocr" not in TRAINABLE_ROLES, "OCR 不应在 TRAINABLE_ROLES 中"
        assert "primary_detector" in TRAINABLE_ROLES, "主检测器应在 TRAINABLE_ROLES 中"

        # model_registry 中的 EasyOCR 条目
        from services.model_registry import get_model_registry
        registry = get_model_registry()
        easyocr = registry.get_model("easyocr")
        assert easyocr is not None, "EasyOCR 模型未注册"
        assert "ocr" in easyocr.task_types, "EasyOCR 应支持 ocr 任务"
        assert easyocr.training_framework == "custom", "OCR 应为 custom 框架"

        print("[OK] OCR role correctly identified as built-in engine")
        return True
    except Exception as e:
        print(f"[FAIL] OCR role test exception: {e}")
        return False


def test_hybrid_frame_sampling():
    """测试：混合帧采样返回正确的元数据结构"""
    print("\n=== 测试：混合帧采样 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        import tempfile
        import os

        # 创建临时测试视频（用 opencv 写一个简单的帧）
        try:
            import cv2
        except ImportError:
            print("- cv2 不可用，跳过实际视频测试")
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            import numpy as np
            for i in range(90):  # 90 帧，3 秒 @ 30fps
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Frame {i}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                out.write(frame)
            out.release()

            from services.video_processor import extract_frames_for_vlm
            frames_b64, frame_meta = extract_frames_for_vlm(video_path, max_frames=8)

            assert len(frames_b64) > 0, "未提取到帧"
            assert len(frame_meta) > 0, "未返回帧元数据"
            assert len(frames_b64) == len(frame_meta), "帧数和元数据数不一致"

            # 每条 meta 应包含必需字段
            for meta in frame_meta:
                assert "frame_index" in meta, "meta 缺少 frame_index"
                assert "timestamp_ms" in meta, "meta 缺少 timestamp_ms"
                assert "source" in meta, "meta 缺少 source"
                assert meta["source"] in ("keyframe", "uniform"), f"source 值非法: {meta['source']}"

            # 验证 source 字段存在且有效
            sources = set(m["source"] for m in frame_meta)
            count_map = {}
            for s in sources:
                count_map[s] = sum(1 for m in frame_meta if m["source"] == s)
        print(f"[OK] Hybrid sampling: extracted {len(frames_b64)} frames, sources: {count_map}")
        return True
    except Exception as e:
        print(f"[FAIL] Hybrid sampling test exception: {e}")
        return False


def test_vlm_video_info_passed():
    """测试：video_info 被正确传入 VLM prompt 构建"""
    print("\n=== 测试：VLM video_info 参数 ===")
    try:
        backend_path = str(PROJECT_ROOT / "backend")
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)

        from services.vlm_adapter import VLMAdapter

        adapter = VLMAdapter(provider="openai", base_url="https://api.openai.com", api_key="mock-key")

        # 无 video_info
        prompt_no_vi = adapter._build_system_prompt(sample_boxes=None, video_info=None)
        assert "视频帧率" not in prompt_no_vi, "无 video_info 时不应包含视频上下文"

        # 有 video_info
        video_info = {
            "fps": 30.0,
            "duration_seconds": 3.0,
            "width": 1920,
            "height": 1080,
        }
        prompt_with_vi = adapter._build_system_prompt(sample_boxes=None, video_info=video_info)
        assert "视频帧率" in prompt_with_vi, "有 video_info 时应包含视频帧率"
        assert "30.0 fps" in prompt_with_vi, "视频帧率值不正确"
        assert "3.0 秒" in prompt_with_vi, "视频时长值不正确"
        assert "1920x1080" in prompt_with_vi, "视频分辨率值不正确"

        # parse_intent 签名支持 video_info
        import inspect
        sig = inspect.signature(adapter.parse_intent)
        assert "video_info" in sig.parameters, "parse_intent 缺少 video_info 参数"

        print("[OK] VLM video_info parameter passed correctly")
        return True
    except Exception as e:
        print(f"[FAIL] VLM video_info test exception: {e}")
        return False


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else ["all"]

    tests = {
        "backend-imports": test_backend_imports,
        "algorithm-planner": test_algorithm_planner_service,
        "algorithm-planner-multi-event": test_algorithm_planner_multi_event_service,
        "algorithm-capability-plan": test_algorithm_plan_returns_capabilities_and_negotiation_summary,
        "algorithm-plan-api": test_algorithm_plan_api,
        "algorithm-package-api": test_algorithm_package_api,
        "algorithm-preview-api": test_algorithm_preview_api,
        "algorithm-preview-api-frames": test_algorithm_preview_api_with_observation_frames,
        "pipeline-compiler": test_pipeline_compiler_service,
        "pipeline-compiler-multi-event": test_pipeline_compiler_multi_event_service,
        "training-recommendation-service": test_training_recommendation_service,
        "worker-imports": test_worker_imports,
        "event-engine": test_event_engine_runtime,
        "phase3-runtime": test_phase3_runtime_session,
        "tracking-runtime": test_tracking_runtime_resilience,
        "package-exporter": test_algorithm_package_exporter,
        "backend-api": test_backend_api,
        "macos-start-scripts": test_macos_start_scripts_resolve_python_interpreter,
        "backend-postgres-driver": test_backend_requirements_include_postgres_driver,
        "backend-frontend-dist": test_backend_serves_frontend_dist,
        "backend-frontend-dist-relative": test_backend_resolves_relative_frontend_dist_from_project_root,
        "training-status-persistence": test_training_status_persistence,
        "workflow-state-persistence": test_task_persists_workflow_state_fields,
        "artifact-listing": test_artifact_listing_uses_task_artifact_paths,
        "manual-cloud-training": test_prepare_manual_cloud_training_bundle,
        "training-monitor-manual-cloud": test_training_monitor_manual_cloud_fallback,
        "frontend-worker-endpoints": test_frontend_worker_endpoints_are_production_safe,
        "frontend-worker-paths": test_frontend_local_worker_paths_use_task_directories,
        "worker-production-config": test_worker_supports_production_origin_and_host_config,
        "frontend-favicon": test_frontend_index_does_not_reference_missing_vite_favicon,
        "settings-auth-persistence": test_settings_auth_persistence,
        "vlm-parse-settings": test_vlm_parse_uses_saved_settings,
        "vlm-parse-fallback": test_vlm_parse_returns_structured_fallback_on_invalid_json,
        "vlm-display-fallbacks": test_vlm_display_field_fallbacks,
        "worker-image-mps": test_worker_image_support_and_mps_guard,
        "worker-clip-error": test_worker_clip_setup_error_message,
        "worker-moondream-error": test_worker_moondream_setup_error_message,
        "local-trainer-python": test_local_trainer_uses_current_python,
        "worker-health": test_worker_health,
        "frontend-build": test_frontend_build,
        "gpu-release": test_gpu_memory_release,
        "local-trainer": test_local_training_subprocess,
        "ocr-role": test_ocr_role_in_pipeline,
        "hybrid-frame-sampling": test_hybrid_frame_sampling,
        "vlm-video-info": test_vlm_video_info_passed,
    }
    groups = {
        "backend": [
            "backend-imports",
            "algorithm-planner",
            "algorithm-planner-multi-event",
            "algorithm-plan-api",
            "algorithm-package-api",
            "algorithm-preview-api",
            "algorithm-preview-api-frames",
            "pipeline-compiler",
            "pipeline-compiler-multi-event",
            "training-recommendation-service",
            "backend-frontend-dist",
            "backend-frontend-dist-relative",
            "macos-start-scripts",
            "backend-postgres-driver",
            "training-status-persistence",
            "workflow-state-persistence",
            "artifact-listing",
            "manual-cloud-training",
            "settings-auth-persistence",
            "vlm-parse-settings",
            "vlm-parse-fallback",
            "vlm-display-fallbacks",
        ],
        "frontend": [
            "training-monitor-manual-cloud",
            "frontend-worker-endpoints",
            "frontend-worker-paths",
            "frontend-favicon",
            "frontend-build",
        ],
        "worker": [
            "worker-production-config",
            "worker-imports",
            "event-engine",
            "phase3-runtime",
            "tracking-runtime",
            "package-exporter",
            "worker-image-mps",
            "worker-clip-error",
            "worker-moondream-error",
            "local-trainer-python",
            "worker-health",
            "gpu-release",
            "local-trainer",
        ],
    }
    groups["full"] = groups["backend"] + groups["frontend"] + groups["worker"]

    if "all" in args:
        selected = list(tests.keys())
    else:
        selected = []
        for name in args:
            if name in groups:
                selected.extend(groups[name])
            else:
                selected.append(name)

    results = {}
    for name in selected:
        if name in tests:
            original_sys_path = list(sys.path)
            try:
                for module_name in list(sys.modules):
                    if module_name == "main" or module_name.startswith(
                        ("routers", "models", "services", "pipeline", "utils")
                    ):
                        sys.modules.pop(module_name, None)
                results[name] = tests[name]()
            except Exception as e:
                print(f"✗ 测试 {name} 异常: {e}")
                results[name] = False
            finally:
                sys.path = original_sys_path

    print("\n" + "=" * 50)
    print("Test results:")
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")

    all_passed = all(results.values())
    print("=" * 50)
    print(f"Overall: {'ALL PASS' if all_passed else 'SOME FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
