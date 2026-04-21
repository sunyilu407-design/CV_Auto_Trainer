# Phase 3 Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a shared Phase 3 runtime core that consumes detection-observation frames, keeps lightweight stable tracks across frames, and emits the first set of region/time-based events for both preview and future streaming paths.

**Architecture:** Keep runtime boundaries explicit: frame input normalization, lightweight tracking state updates, event evaluation, and a session wrapper that maintains cross-frame state. Reuse the same runtime session in preview so offline validation and future stream processing stay on one contract.

**Tech Stack:** Python 3, FastAPI service layer, worker runtime modules, integration test harness

---

## File Structure

- Modify: `worker/pipeline/tracking_runtime.py`
- Modify: `worker/pipeline/event_engine.py`
- Create: `worker/pipeline/runtime_session.py`
- Modify: `backend/services/algorithm_preview_service.py`
- Modify: `tests/test_integration.py`

### Task 1: Add failing runtime-session coverage for Phase 3 events

**Files:**
- Modify: `tests/test_integration.py`
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Add a new `test_phase3_runtime_session()` that feeds a frame sequence like:

```python
frames = [
    {
        "frame_index": 1,
        "timestamp_ms": 0,
        "detections": [{"class_name": "person", "bbox_xywhn": [0.15, 0.5, 0.1, 0.2], "confidence": 0.95}],
    },
    {
        "frame_index": 2,
        "timestamp_ms": 1000,
        "detections": [{"class_name": "person", "bbox_xywhn": [0.32, 0.5, 0.1, 0.2], "confidence": 0.94}],
    },
    {
        "frame_index": 3,
        "timestamp_ms": 3000,
        "detections": [{"class_name": "person", "bbox_xywhn": [0.52, 0.5, 0.1, 0.2], "confidence": 0.93}],
    },
    {
        "frame_index": 4,
        "timestamp_ms": 4000,
        "detections": [{"class_name": "person", "bbox_xywhn": [0.82, 0.5, 0.1, 0.2], "confidence": 0.92}],
    },
]
```

Assert:

- all per-frame track snapshots use the same `track_id`
- the emitted event codes include `entered_zone_a`, `presence_zone_a`, `entered_zone_b`, `left_zone_a`, `crossed_a_to_b`, `left_zone_b`
- the transition event payload includes `from_region_id == "zone_a"` and `to_region_id == "zone_b"`

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py phase3-runtime`

Expected: FAIL because `runtime_session` does not exist and current runtime cannot keep stable cross-frame state or emit the full event set.

- [ ] **Step 3: Write minimal implementation**

Create a session-oriented runtime that:

- accepts `frame_index / timestamp_ms / detections`
- updates lightweight tracks by class + nearest-center matching
- evaluates `region_enter`, `region_exit`, `region_presence_duration`, and `cross_region_transition`
- returns current `track_states` and frame events

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py phase3-runtime`

Expected: PASS.

### Task 2: Rewire preview service to use the shared runtime session

**Files:**
- Modify: `backend/services/algorithm_preview_service.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Extend the preview-service test expectations so the preview result proves the shared runtime is used:

- `track_states[0]` contains `first_seen_frame`, `last_seen_frame`, `present_duration_ms`, and `regions_inside`
- returned events still include the configured `event_code`

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py algorithm-preview-service`

Expected: FAIL because the current preview path still calls the old one-shot helpers.

- [ ] **Step 3: Write minimal implementation**

Update preview service to:

- wrap `sample_boxes` into a single observation frame
- instantiate `RuntimeSession`
- execute the frame
- return the session-produced `track_states` and `events`

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py algorithm-preview-service`

Expected: PASS.

### Task 3: Preserve legacy helper compatibility and rule-field compatibility

**Files:**
- Modify: `worker/pipeline/tracking_runtime.py`
- Modify: `worker/pipeline/event_engine.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Extend `test_event_engine_runtime()` so it uses the richer track contract:

```python
track_states = [{
    "track_id": "track-1",
    "class_name": "cargo_box",
    "bbox_xywhn": [0.52, 0.53, 0.2, 0.2],
    "frame_index": 12,
    "timestamp_ms": 12000,
    "present_duration_ms": 12000,
    "regions_inside": ["primary_region"],
    "entered_region_at": {"primary_region": 0},
    "last_event_frame": {},
}]
```

Also assert the engine still accepts old `duration_seconds` rule fields and emits the same configured `event_code`.

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py event-engine`

Expected: FAIL until the engine understands the new fields and compatibility conversion.

- [ ] **Step 3: Write minimal implementation**

Update runtime helpers so:

- `event_engine` reads both `duration_ms` and legacy `duration_seconds`
- tracking helper exports richer track fields even for one-shot callers
- rule evaluation is centralized around the new track contract

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py event-engine`

Expected: PASS.

### Task 4: Run regression verification

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Run focused runtime verification**

Run: `./.venv/bin/python tests/test_integration.py phase3-runtime`

Expected: PASS.

- [ ] **Step 2: Run backend integration verification**

Run: `./.venv/bin/python tests/test_integration.py backend`

Expected: PASS.

- [ ] **Step 3: Run full verification**

Run: `./.venv/bin/python tests/test_integration.py full`

Expected: PASS, while preserving existing sandbox-only skips.

## Self-Review

### Spec coverage

- shared runtime core: covered by Task 1 and Task 2
- lightweight replaceable tracking: covered by Task 1 and Task 3
- four event types: covered by Task 1
- preview reuse of runtime session: covered by Task 2
- compatibility with existing rule shape: covered by Task 3

### Placeholder scan

No `TODO`, `TBD`, or “similar to Task N” placeholders remain.

### Type consistency

Use one consistent frame shape (`frame_index`, `timestamp_ms`, `detections`), one consistent track shape (`present_duration_ms`, `regions_inside`, `entered_region_at`, `last_event_frame`), and one consistent event shape (`event_code`, `rule_id`, `track_id`, `frame_index`, `timestamp_ms`, `region_id`, `bbox_xywhn`, `payload`).
