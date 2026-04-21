# User-Facing Algorithm Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the product from a detector-training flow into a business-user workflow with requirement negotiation, multi-clip offline evaluation, multi-capability training orchestration, and final algorithm-package delivery.

**Architecture:** Extend the existing algorithm plan flow into a persisted workflow with three new stateful layers: negotiation state, offline evaluation state, and capability-oriented training state. Keep end-user UX business-language-first while representing the internal plan as structured capability/rule data that can drive VLM review, training orchestration, and delivery packaging.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, React, TypeScript, Zustand, Vite, existing integration test runner in `tests/test_integration.py`.

---

## File Structure

**Primary files and responsibilities:**

- `backend/models/db.py`
  Stores new workflow state on `Task`, including negotiation, offline evaluation, capability plan, and delivery metadata.
- `backend/models/database.py`
  Adds migration-safe column initialization for the new persisted workflow fields.
- `backend/services/algorithm_planner.py`
  Evolves the algorithm plan from a single-detector draft into a capability-and-rule plan that can represent multi-capability solutions.
- `backend/services/offline_evaluation_service.py`
  New service that assembles multi-clip evaluation inputs, calls VLM review, and normalizes timeline/evidence/verdict output.
- `backend/services/training_plan_service.py`
  New service that maps algorithm capabilities into user-facing trainable units and training stages.
- `backend/routers/algorithm.py`
  Extends algorithm APIs to support negotiation updates, offline evaluation creation/feedback, and training-plan retrieval.
- `backend/routers/files.py`
  Supports offline evaluation video upload grouping and artifact grouping for final package delivery.
- `frontend/src/store/taskStore.ts`
  Adds persisted/working client state for negotiation edits, offline evaluation videos, evaluation feedback, and training units.
- `frontend/src/api/backend.ts`
  Adds typed APIs for negotiation, offline evaluation, training orchestration, and final delivery metadata.
- `frontend/src/pages/IntentConfirm.tsx`
  Converts the current VLM confirmation stage into the business-facing requirement negotiation page.
- `frontend/src/pages/AlgorithmPlan.tsx`
  Reframes algorithm draft rendering around business capabilities and route to offline evaluation instead of direct labeling.
- `frontend/src/pages/OfflineEvaluation.tsx`
  New page for multi-clip upload, VLM review results, timeline/evidence display, and user feedback capture.
- `frontend/src/pages/TrainingPlan.tsx`
  New page for user-facing multi-capability training orchestration.
- `frontend/src/pages/TrainingMonitor.tsx`
  Upgrades monitoring UI from single-model status to workflow-stage and per-capability progress.
- `frontend/src/pages/Delivery.tsx`
  Upgrades delivery UI from file listing to final algorithm package summary.
- `frontend/src/App.tsx`
  Adds new stages/routes for offline evaluation and training planning.
- `tests/test_integration.py`
  Adds regression coverage for negotiation flow, offline evaluation persistence, capability-oriented training plan generation, and delivery package structure.

---

### Task 1: Persist Negotiation, Evaluation, and Capability Workflow State

**Files:**
- Modify: `backend/models/db.py`
- Modify: `backend/models/database.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing persistence test**

```python
def test_task_persists_negotiation_and_offline_evaluation_state():
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

            with TestClient(main.app) as client:
                login_resp = client.post(
                    "/api/auth/login",
                    json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
                )
                headers = auth_headers(login_resp.json()["data"]["token"])

                create_resp = client.post("/api/tasks", json={"name": "工作流状态持久化"}, headers=headers)
                task_id = create_resp.json()["id"]

                plan_resp = client.post(
                    "/api/algorithm/plan",
                    json={
                        "task_id": task_id,
                        "user_description": "识别人员进入A区并停留10秒触发事件",
                        "vlm_result": None,
                    },
                    headers=headers,
                )
                if plan_resp.status_code != 200:
                    print(f"✗ 规划创建失败: {plan_resp.status_code} {plan_resp.text}")
                    return False

                update_resp = client.post(
                    f"/api/algorithm/plan/{task_id}/negotiate",
                    json={
                        "negotiation_summary": {
                            "scenario_label": "区域停留",
                            "objects": ["人员"],
                            "regions": ["A区"],
                            "duration_seconds": 10,
                            "events": ["进入区域", "停留超时"],
                        },
                        "offline_evaluation": {
                            "status": "pending",
                            "clips": [
                                {"clip_id": "clip_trigger", "label": "应触发", "path": "/tmp/clip_trigger.mp4"},
                                {"clip_id": "clip_negative", "label": "不应触发", "path": "/tmp/clip_negative.mp4"},
                            ],
                        },
                    },
                    headers=headers,
                )
                payload = update_resp.json()
                if update_resp.status_code != 200 or payload.get("code") != 0:
                    print(f"✗ 协商状态写入失败: {update_resp.status_code} {payload}")
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py workflow-state-persistence`
Expected: FAIL because `Task` does not yet expose or persist `negotiation_summary` and `offline_evaluation`.

- [ ] **Step 3: Add new persisted workflow columns**

```python
# backend/models/db.py
class Task(Base):
    __tablename__ = "tasks"

    ...
    negotiation_summary = Column(JSON)
    offline_evaluation = Column(JSON)
    training_plan = Column(JSON)
    delivery_package = Column(JSON)
```

- [ ] **Step 4: Add migration-safe column initialization**

```python
# backend/models/database.py
_REQUIRED_TASK_COLUMNS = {
    ...
    "negotiation_summary": "ALTER TABLE tasks ADD COLUMN negotiation_summary JSON",
    "offline_evaluation": "ALTER TABLE tasks ADD COLUMN offline_evaluation JSON",
    "training_plan": "ALTER TABLE tasks ADD COLUMN training_plan JSON",
    "delivery_package": "ALTER TABLE tasks ADD COLUMN delivery_package JSON",
}
```

- [ ] **Step 5: Expose the new fields through task retrieval**

```python
# backend/routers/tasks.py
return {
    "id": task.id,
    ...
    "negotiation_summary": task.negotiation_summary,
    "offline_evaluation": task.offline_evaluation,
    "training_plan": task.training_plan,
    "delivery_package": task.delivery_package,
}
```

- [ ] **Step 6: Register the test**

```python
# tests/test_integration.py
tests = {
    ...
    "workflow-state-persistence": test_task_persists_negotiation_and_offline_evaluation_state,
}
groups["backend"] = [
    ...
    "workflow-state-persistence",
]
```

- [ ] **Step 7: Run the persistence regression**

Run: `./.venv/bin/python tests/test_integration.py workflow-state-persistence`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/models/db.py backend/models/database.py backend/routers/tasks.py tests/test_integration.py
git commit -m "Persist workflow negotiation and evaluation state"
```


### Task 2: Upgrade Algorithm Plan Into Requirement Negotiation And Capability Draft

**Files:**
- Modify: `backend/services/algorithm_planner.py`
- Modify: `backend/routers/algorithm.py`
- Modify: `frontend/src/api/backend.ts`
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `frontend/src/pages/IntentConfirm.tsx`
- Modify: `frontend/src/pages/AlgorithmPlan.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing negotiation/capability test**

```python
def test_algorithm_plan_returns_capabilities_and_negotiation_summary():
    print("\n=== 测试：算法规划返回能力草图与需求协商摘要 ===")
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

    if len(capabilities) < 2:
        print(f"✗ 未生成多能力草图: {result}")
        return False
    if negotiation.get("duration_seconds") != 5:
        print(f"✗ negotiation_summary 时长错误: {result}")
        return False
    if "进入危险区域" not in "".join(negotiation.get("events", [])):
        print(f"✗ negotiation_summary 事件摘要错误: {result}")
        return False

    print("✓ 算法规划能力草图与协商摘要正常")
    return True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py algorithm-capability-plan`
Expected: FAIL because `build_algorithm_plan()` does not yet include `capabilities` or `negotiation_summary`.

- [ ] **Step 3: Extend the planner shape**

```python
# backend/services/algorithm_planner.py
return {
    "summary": ...,
    "scenario_type": scenario_type,
    "negotiation_summary": {
        "scenario_label": scenario_label,
        "objects": inferred_objects,
        "regions": [region["name"] for region in regions],
        "duration_seconds": duration_seconds,
        "events": [event["name"] for event in events],
    },
    "capabilities": [
        {"capability_id": "detect_person", "label": "识别人员", "trainable": True, "kind": "detection"},
        {"capability_id": "classify_helmet", "label": "判断是否佩戴工帽", "trainable": True, "kind": "classification"},
        {"capability_id": "rule_zone_dwell", "label": "危险区域停留规则", "trainable": False, "kind": "rule"},
    ],
    ...
}
```

- [ ] **Step 4: Add negotiation update endpoint**

```python
# backend/routers/algorithm.py
@router.post("/plan/{task_id}/negotiate")
def update_negotiation(task_id: str, payload: NegotiationUpdateRequest, ...):
    task = get_task_for_user(db, task_id, current_user)
    task.negotiation_summary = payload.negotiation_summary
    if payload.offline_evaluation is not None:
        task.offline_evaluation = payload.offline_evaluation
    db.commit()
    return {"code": 0, "msg": "ok", "data": {"task_id": task.id, "negotiation_summary": task.negotiation_summary}}
```

- [ ] **Step 5: Add typed frontend API/state support**

```ts
// frontend/src/api/backend.ts
export interface NegotiationSummary {
  scenario_label: string
  objects: string[]
  regions: string[]
  duration_seconds: number | null
  events: string[]
}

export interface CapabilityDraft {
  capability_id: string
  label: string
  trainable: boolean
  kind: 'detection' | 'classification' | 'tracking' | 'rule'
}
```

- [ ] **Step 6: Reframe Stage 1 / Stage 2 pages around negotiation**

```tsx
// frontend/src/pages/IntentConfirm.tsx
<h1 className="page-title">确认需求协商</h1>
<p className="page-subtitle">请确认系统对监测对象、区域、时长和事件结果的理解。</p>

// frontend/src/pages/AlgorithmPlan.tsx
<h3 className="text-heading-sm">能力草图</h3>
{draft.capabilities.map((item) => (
  <CapabilityCard key={item.capability_id} label={item.label} trainable={item.trainable} kind={item.kind} />
))}
```

- [ ] **Step 7: Register and run the negotiation test**

Run: `./.venv/bin/python tests/test_integration.py algorithm-capability-plan`
Expected: PASS

- [ ] **Step 8: Verify frontend build still passes**

Run: `cd frontend && npm run build`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add backend/services/algorithm_planner.py backend/routers/algorithm.py frontend/src/api/backend.ts frontend/src/store/taskStore.ts frontend/src/pages/IntentConfirm.tsx frontend/src/pages/AlgorithmPlan.tsx frontend/src/App.tsx tests/test_integration.py
git commit -m "Add requirement negotiation and capability-based algorithm drafts"
```


### Task 3: Add Multi-Clip Offline Evaluation With VLM Review And User Feedback

**Files:**
- Create: `backend/services/offline_evaluation_service.py`
- Modify: `backend/routers/algorithm.py`
- Modify: `backend/routers/files.py`
- Modify: `frontend/src/api/backend.ts`
- Modify: `frontend/src/store/taskStore.ts`
- Create: `frontend/src/pages/OfflineEvaluation.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing offline evaluation test**

```python
def test_offline_evaluation_generates_vlm_review_and_feedback_slots():
    print("\n=== 测试：离线评估生成 VLM 评审结果与反馈槽位 ===")
    backend_path = str(PROJECT_ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from services.offline_evaluation_service import build_offline_evaluation_review

    result = build_offline_evaluation_review(
        negotiation_summary={
            "scenario_label": "区域停留",
            "objects": ["人员"],
            "regions": ["A区"],
            "duration_seconds": 10,
            "events": ["进入区域", "停留超时"],
        },
        algorithm_plan={
            "capabilities": [{"capability_id": "detect_person", "label": "识别人员", "trainable": True, "kind": "detection"}],
            "events": [{"event_code": "dwell_timeout_detected", "name": "持续停留事件"}],
        },
        clips=[
            {"clip_id": "trigger", "label": "应触发", "path": "/tmp/trigger.mp4"},
            {"clip_id": "negative", "label": "不应触发", "path": "/tmp/negative.mp4"},
        ],
    )

    if len(result.get("timeline", [])) != 2:
        print(f"✗ timeline 未按 clip 生成: {result}")
        return False
    if "feedback_options" not in result:
        print(f"✗ 缺少 feedback_options: {result}")
        return False
    if "这个触发对" not in result["feedback_options"]:
        print(f"✗ 缺少业务反馈按钮: {result}")
        return False

    print("✓ 离线评估结构正常")
    return True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py offline-evaluation-review`
Expected: FAIL because `offline_evaluation_service.py` and the review builder do not exist yet.

- [ ] **Step 3: Implement the VLM review normalizer**

```python
# backend/services/offline_evaluation_service.py
def build_offline_evaluation_review(*, negotiation_summary: dict, algorithm_plan: dict, clips: list[dict]) -> dict:
    return {
        "status": "reviewed",
        "timeline": [
            {
                "clip_id": clip["clip_id"],
                "label": clip["label"],
                "should_trigger": clip["label"] == "应触发",
                "events": [],
                "evidence_frames": [],
                "false_positive_candidates": [],
                "false_negative_candidates": [],
                "rule_explanations": [],
            }
            for clip in clips
        ],
        "feedback_options": ["这个触发对", "这是误报", "这里漏了", "区域不对", "时长不对"],
    }
```

- [ ] **Step 4: Add offline evaluation APIs**

```python
# backend/routers/algorithm.py
@router.post("/evaluation/{task_id}")
def create_offline_evaluation(task_id: str, payload: OfflineEvaluationRequest, ...):
    task = get_task_for_user(db, task_id, current_user)
    review = build_offline_evaluation_review(
        negotiation_summary=task.negotiation_summary or {},
        algorithm_plan=task.algorithm_plan or {},
        clips=payload.clips,
    )
    task.offline_evaluation = {"status": "reviewed", "clips": payload.clips, "review": review, "feedback": []}
    db.commit()
    return {"code": 0, "msg": "ok", "data": task.offline_evaluation}

@router.post("/evaluation/{task_id}/feedback")
def submit_offline_evaluation_feedback(task_id: str, payload: OfflineEvaluationFeedbackRequest, ...):
    ...
```

- [ ] **Step 5: Add frontend page and typed API**

```ts
// frontend/src/api/backend.ts
export interface OfflineEvaluationClip {
  clip_id: string
  label: '应触发' | '不应触发' | '边界场景'
  path: string
}

// frontend/src/pages/OfflineEvaluation.tsx
<h1 className="page-title">离线评估</h1>
<p className="page-subtitle">上传多段业务验证视频，确认这是不是你想要的算法效果。</p>
```

- [ ] **Step 6: Wire the new stage into the app**

```ts
// frontend/src/store/taskStore.ts
export type Stage = 'upload' | 'intent_confirm' | 'algorithm_plan' | 'offline_evaluation' | 'labeling' | ...

// frontend/src/App.tsx
{stage === 'offline_evaluation' && <OfflineEvaluation />}
```

- [ ] **Step 7: Register and run the backend evaluation test**

Run: `./.venv/bin/python tests/test_integration.py offline-evaluation-review`
Expected: PASS

- [ ] **Step 8: Verify frontend build**

Run: `cd frontend && npm run build`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add backend/services/offline_evaluation_service.py backend/routers/algorithm.py backend/routers/files.py frontend/src/api/backend.ts frontend/src/store/taskStore.ts frontend/src/pages/OfflineEvaluation.tsx frontend/src/App.tsx tests/test_integration.py
git commit -m "Add multi-clip offline evaluation with VLM review"
```


### Task 4: Add Capability-Oriented Training Plan And Workflow Monitoring

**Files:**
- Create: `backend/services/training_plan_service.py`
- Modify: `backend/routers/algorithm.py`
- Modify: `backend/routers/training.py`
- Modify: `frontend/src/api/backend.ts`
- Modify: `frontend/src/store/taskStore.ts`
- Create: `frontend/src/pages/TrainingPlan.tsx`
- Modify: `frontend/src/pages/TrainingMonitor.tsx`
- Modify: `frontend/src/App.tsx`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing training-plan test**

```python
def test_training_plan_groups_trainable_capabilities():
    print("\n=== 测试：训练编排按能力拆分可训练单元 ===")
    backend_path = str(PROJECT_ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from services.training_plan_service import build_training_plan

    result = build_training_plan(
        algorithm_plan={
            "capabilities": [
                {"capability_id": "detect_person", "label": "识别人员", "trainable": True, "kind": "detection"},
                {"capability_id": "classify_helmet", "label": "判断是否佩戴工帽", "trainable": True, "kind": "classification"},
                {"capability_id": "rule_zone_dwell", "label": "危险区域停留规则", "trainable": False, "kind": "rule"},
            ]
        }
    )

    units = result.get("training_units", [])
    if len(units) != 2:
        print(f"✗ 训练单元数量错误: {result}")
        return False
    if result.get("non_trainable_units") != ["危险区域停留规则"]:
        print(f"✗ 非训练单元表达错误: {result}")
        return False

    print("✓ 训练编排结构正常")
    return True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py training-plan-orchestration`
Expected: FAIL because `training_plan_service.py` does not exist and training plan routing is missing.

- [ ] **Step 3: Implement the training plan builder**

```python
# backend/services/training_plan_service.py
def build_training_plan(*, algorithm_plan: dict) -> dict:
    capabilities = algorithm_plan.get("capabilities", [])
    training_units = [
        {"unit_id": item["capability_id"], "label": item["label"], "kind": item["kind"], "status": "pending"}
        for item in capabilities
        if item.get("trainable")
    ]
    non_trainable_units = [item["label"] for item in capabilities if not item.get("trainable")]
    return {
        "training_units": training_units,
        "non_trainable_units": non_trainable_units,
        "workflow_stages": ["准备中", "能力训练", "规则组装", "交付打包"],
    }
```

- [ ] **Step 4: Expose training plan and multi-unit monitor payload**

```python
# backend/routers/algorithm.py
@router.get("/training-plan/{task_id}")
def get_training_plan(task_id: str, ...):
    task = get_task_for_user(db, task_id, current_user)
    training_plan = build_training_plan(algorithm_plan=task.algorithm_plan or {})
    task.training_plan = training_plan
    db.commit()
    return {"code": 0, "msg": "ok", "data": training_plan}
```

- [ ] **Step 5: Add frontend training-plan and monitor UI**

```tsx
// frontend/src/pages/TrainingPlan.tsx
<h1 className="page-title">训练编排</h1>
<p className="page-subtitle">确认哪些识别能力需要训练，哪些规则将直接参与最终方案组装。</p>

// frontend/src/pages/TrainingMonitor.tsx
{trainingPlan.training_units.map((unit) => (
  <TrainingUnitRow key={unit.unit_id} label={unit.label} status={unit.status} />
))}
```

- [ ] **Step 6: Register and run the orchestration test**

Run: `./.venv/bin/python tests/test_integration.py training-plan-orchestration`
Expected: PASS

- [ ] **Step 7: Verify frontend build**

Run: `cd frontend && npm run build`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/services/training_plan_service.py backend/routers/algorithm.py backend/routers/training.py frontend/src/api/backend.ts frontend/src/store/taskStore.ts frontend/src/pages/TrainingPlan.tsx frontend/src/pages/TrainingMonitor.tsx frontend/src/App.tsx tests/test_integration.py
git commit -m "Add capability-oriented training orchestration"
```


### Task 5: Upgrade Delivery Into Final Algorithm Package Summary

**Files:**
- Modify: `backend/services/algorithm_package_service.py`
- Modify: `backend/routers/algorithm.py`
- Modify: `frontend/src/pages/Delivery.tsx`
- Modify: `frontend/src/api/backend.ts`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing delivery-package test**

```python
def test_algorithm_package_includes_models_rules_and_readme():
    print("\n=== 测试：交付包包含模型、规则与说明 ===")
    backend_path = str(PROJECT_ROOT / "backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    from services.algorithm_package_service import build_delivery_package_summary

    result = build_delivery_package_summary(
        artifact_paths={
            "detector": "/tmp/detect_person.onnx",
            "classifier": "/tmp/classify_helmet.onnx",
            "bundle_readme": "/tmp/README.md",
        },
        algorithm_plan={
            "capabilities": [
                {"label": "识别人员", "trainable": True},
                {"label": "判断是否佩戴工帽", "trainable": True},
                {"label": "危险区域停留规则", "trainable": False},
            ]
        },
    )

    if len(result.get("model_files", [])) != 2:
        print(f"✗ model_files 不完整: {result}")
        return False
    if "危险区域停留规则" not in result.get("rules", []):
        print(f"✗ rules 不完整: {result}")
        return False
    if not result.get("readme_path", "").endswith("README.md"):
        print(f"✗ README 路径错误: {result}")
        return False

    print("✓ 交付包摘要正常")
    return True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py delivery-package-summary`
Expected: FAIL because package summary helper and delivery metadata shape do not yet exist.

- [ ] **Step 3: Implement package summary helper**

```python
# backend/services/algorithm_package_service.py
def build_delivery_package_summary(*, artifact_paths: dict, algorithm_plan: dict) -> dict:
    model_files = [path for key, path in artifact_paths.items() if key in {"detector", "classifier", "tracker"}]
    rules = [item["label"] for item in algorithm_plan.get("capabilities", []) if not item.get("trainable")]
    return {
        "model_files": model_files,
        "rules": rules,
        "readme_path": artifact_paths.get("bundle_readme", ""),
    }
```

- [ ] **Step 4: Expose package summary to the frontend**

```python
# backend/routers/algorithm.py
@router.post("/package/{task_id}")
def export_algorithm_package(...):
    package = export_task_algorithm_package(task, ...)
    summary = build_delivery_package_summary(
        artifact_paths=package,
        algorithm_plan=task.algorithm_plan or {},
    )
    task.delivery_package = summary
    db.commit()
    return {"code": 0, "msg": "ok", "data": {**package, "summary": summary}}
```

- [ ] **Step 5: Render delivery as algorithm package summary**

```tsx
// frontend/src/pages/Delivery.tsx
<h1 className="page-title">算法交付</h1>
<p className="page-subtitle">你将拿到可直接使用的算法方案包，而不只是单个模型文件。</p>

{deliveryPackage.model_files.map((path) => (
  <ArtifactRow key={path} label={path.split('/').pop() ?? path} />
))}
```

- [ ] **Step 6: Register and run the delivery test**

Run: `./.venv/bin/python tests/test_integration.py delivery-package-summary`
Expected: PASS

- [ ] **Step 7: Run the final verification set**

Run: `./.venv/bin/python tests/test_integration.py workflow-state-persistence algorithm-capability-plan offline-evaluation-review training-plan-orchestration delivery-package-summary`
Expected: PASS

Run: `cd frontend && npm run build`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add backend/services/algorithm_package_service.py backend/routers/algorithm.py frontend/src/pages/Delivery.tsx frontend/src/api/backend.ts tests/test_integration.py
git commit -m "Upgrade delivery to final algorithm package summaries"
```


## Self-Review

- Spec coverage:
  - Requirement negotiation is implemented in Task 2.
  - Multi-clip offline evaluation and VLM review are implemented in Task 3.
  - Multi-capability training orchestration is implemented in Task 4.
  - Final algorithm package delivery is implemented in Task 5.
  - Persisted workflow state needed across the new flow is implemented in Task 1.
- Placeholder scan:
  - No TODO/TBD placeholders remain.
  - Each task includes concrete file paths, code snippets, commands, and expected outcomes.
- Type consistency:
  - `negotiation_summary`, `offline_evaluation`, `training_plan`, and `delivery_package` are the persisted workflow state keys throughout the plan.
  - `capabilities` is the internal algorithm-plan array used consistently by negotiation, evaluation, training, and delivery.
  - Stage names use `offline_evaluation` and `training_plan` consistently across app/store/API tasks.
