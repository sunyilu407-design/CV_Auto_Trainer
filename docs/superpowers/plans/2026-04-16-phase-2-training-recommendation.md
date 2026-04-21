# Phase 2 Training Recommendation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `training_recommendation` into a richer, mostly-complete training prefill that merges algorithm signals, user defaults, and runtime capability while preserving user override priority.

**Architecture:** Keep `pipeline_compiler` focused on algorithm IR compilation and add a separate recommendation service that enriches `pipeline_config` before it is returned to the frontend. Extend the task store so recommendation-applied values and user-overridden values are tracked separately, allowing the UI to refresh recommendations without silently stomping user edits.

**Tech Stack:** FastAPI, SQLAlchemy, Python integration tests, React 18, Zustand, TypeScript, Vite

---

## File Structure

- Create: `backend/services/training_recommendation_service.py`
- Modify: `backend/routers/algorithm.py`
- Modify: `frontend/src/api/backend.ts`
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `frontend/src/pages/AlgorithmPlan.tsx`
- Modify: `frontend/src/pages/TrainConfig.tsx`
- Modify: `tests/test_integration.py`

## Task 1: Add backend recommendation service and direct tests

**Files:**
- Create: `backend/services/training_recommendation_service.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Add a direct backend test that asserts the new service emits a richer recommendation payload:

```python
def test_training_recommendation_service_merges_algorithm_settings_and_runtime():
    from types import SimpleNamespace
    from services.pipeline_compiler import compile_algorithm_pipeline
    from services.training_recommendation_service import build_training_recommendation

    plan = {
        "summary": "基于 cargo_box 的 occupancy_monitoring 算法草案",
        "scenario_type": "occupancy_monitoring",
        "runtime_modes": ["offline", "stream"],
        "targets": [{"class_name": "cargo_box", "prompt": "cargo box", "role": "primary_subject", "requires_training": True}],
        "regions": [{"region_id": "primary_region", "name": "主监测区域", "source": "user_defined", "required": True}],
        "temporal_constraints": [{"constraint_id": "primary_duration", "type": "sustain", "duration_seconds": 10}],
        "events": [{"event_code": "occupancy_detected", "name": "主事件", "trigger": {"target_class": "cargo_box", "region_id": "primary_region", "temporal_constraint_id": "primary_duration"}}],
        "training_requirements": {"detector_training_required": True, "tracking_required": True, "rule_engine_required": True},
        "confidence": 0.72,
    }
    pipeline = compile_algorithm_pipeline(plan)
    settings = SimpleNamespace(default_model="yolo11l.pt", default_train_mode="cloud", default_gpu_type="Apple M4 Pro")
    runtime = {"local_training_available": True, "preferred_device": "mps", "available_export_formats": ["onnx", "coreml"], "supports_cloud_training": True}

    recommendation = build_training_recommendation(plan, pipeline, settings, runtime)

    assert recommendation["recommended_config"]["model"] == "yolo11s.pt"
    assert recommendation["recommended_config"]["train_mode"] == "local"
    assert recommendation["recommended_config"]["export_formats"] == ["onnx", "coreml"]
    assert recommendation["recommended_config"]["imgsz"] == 640
    assert recommendation["recommended_config"]["epochs"] >= 100
    assert recommendation["source_map"]["train_mode"] == "runtime"
    assert recommendation["legacy"]["recommended_model"] == "yolo11s.pt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py training-recommendation-service`

Expected: FAIL because `services.training_recommendation_service` and the new test alias do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement a pure recommendation service that:

- derives simple complexity from plan + pipeline
- chooses model/train mode/export formats/imgsz/epochs/lr0/patience/conf/iou
- emits `recommended_config`, `reason_summary`, `source_map`
- keeps a `legacy` block for existing consumers

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py training-recommendation-service`

Expected: PASS with the richer recommendation payload.

## Task 2: Enrich algorithm plan responses with recommendation context

**Files:**
- Modify: `backend/routers/algorithm.py`
- Modify: `frontend/src/api/backend.ts`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing integration assertion**

Extend the algorithm plan API test so it submits runtime capability and expects:

- `pipeline_config.training_recommendation.recommended_config`
- `reason_summary`
- `source_map`
- backward-compatible `recommended_model / train_mode / export_formats`

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py algorithm-plan-api`

Expected: FAIL because the API still returns the shallow recommendation shape.

- [ ] **Step 3: Write minimal implementation**

Update `/api/algorithm/plan`, `/api/algorithm/plan/{id}`, and `/api/algorithm/plan/{id}/confirm` to:

- accept optional `runtime_capability`
- read current user settings
- compile algorithm pipeline
- enrich the pipeline with the new recommendation service before returning/saving it

Also update frontend API types so the richer payload is typed.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py algorithm-plan-api`

Expected: PASS with the richer recommendation contract.

## Task 3: Preserve user overrides while applying recommendations

**Files:**
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `frontend/src/pages/AlgorithmPlan.tsx`
- Modify: `frontend/src/pages/TrainConfig.tsx`

- [ ] **Step 1: Write the failing proof**

Use TypeScript compile as the failing proof by first adding store references for:

- `trainConfigOverrides`
- `applyRecommendedTrainConfig`
- a richer recommendation shape in the training page

- [ ] **Step 2: Run compile to verify failure**

Run: `cd frontend && npm run build`

Expected: FAIL because the new store fields and recommendation shape do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Extend the task store to:

- track per-field override flags
- treat `setTrainConfig` as user-owned by default
- add a recommendation-only apply action that skips overridden fields

Update the algorithm plan page to send runtime capability hints and apply recommendation defaults safely.

Update the training page to:

- display the richer recommendation summary
- prefill from `recommended_config`
- preserve user edits on subsequent recommendation application

- [ ] **Step 4: Run compile to verify it passes**

Run: `cd frontend && npm run build`

Expected: PASS with the richer recommendation wiring.

## Task 4: Full verification and compatibility sweep

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add compatibility assertions**

Keep the current old-shape assertions working by checking that:

- `recommended_model == recommended_config.model`
- `train_mode == recommended_config.train_mode`
- `export_formats == recommended_config.export_formats`

- [ ] **Step 2: Run backend and frontend verification**

Run:

```bash
./.venv/bin/python tests/test_integration.py backend
./.venv/bin/python tests/test_integration.py frontend
```

Expected: all pass.

- [ ] **Step 3: Run full regression**

Run:

```bash
./.venv/bin/python tests/test_integration.py full
```

Expected: all pass, with sandbox-only health checks reported as skips where applicable.

## Self-Review

### Spec coverage

- backend recommendation boundary: covered by Tasks 1-2
- richer response payload: covered by Task 2
- frontend prefill + override preservation: covered by Task 3
- compatibility and regression safety: covered by Task 4

### Placeholder scan

No `TODO`, `TBD`, or “similar to Task N” shortcuts remain.

### Type consistency

Use `recommended_config`, `reason_summary`, `source_map`, and `legacy` consistently across backend, API typing, and UI consumption.
