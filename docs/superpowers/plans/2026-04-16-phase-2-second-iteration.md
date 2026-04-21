# Phase 2 Second Iteration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Phase 2 by making training recommendations easier to understand in the UI and by refining scenario-complexity-based defaults for key training fields.

**Architecture:** Keep the existing recommendation boundary intact and only refine two areas: backend complexity heuristics plus frontend provenance display. Derive source labels from `source_map` and `trainConfigOverrides` instead of adding a new persisted provenance layer.

**Tech Stack:** FastAPI, Python integration tests, React 18, Zustand, TypeScript, Vite

---

## File Structure

- Modify: `backend/services/training_recommendation_service.py`
- Modify: `frontend/src/pages/TrainConfig.tsx`
- Modify: `frontend/src/store/taskStore.ts`
- Modify: `tests/test_integration.py`

## Task 1: Refine recommendation complexity rules and Chinese explanations

**Files:**
- Modify: `backend/services/training_recommendation_service.py`
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write the failing test**

Extend `test_training_recommendation_service()` so it checks:

- low-complexity occupancy monitoring keeps `yolo11s.pt`, `640`, `100`
- a high-complexity multi-target tracking + temporal case upgrades to `yolo11m.pt`, `1280`, `140`, `30`
- `reason_summary` is natural Chinese instead of the current English fragment string

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/python tests/test_integration.py training-recommendation-service`

Expected: FAIL on the new high-complexity or Chinese summary assertions.

- [ ] **Step 3: Write minimal implementation**

Update the recommendation service to:

- derive a more explicit scenario complexity level from target count, rule count, tracking, and temporal/region intensity
- only change `model`, `imgsz`, `epochs`, and `patience`
- emit a Chinese `reason_summary`

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/python tests/test_integration.py training-recommendation-service`

Expected: PASS.

## Task 2: Add field-level provenance display in training config

**Files:**
- Modify: `frontend/src/pages/TrainConfig.tsx`
- Modify: `frontend/src/store/taskStore.ts`

- [ ] **Step 1: Write the failing proof**

Add code references in `TrainConfig.tsx` for:

- provenance label rendering based on `source_map`
- override-aware label display using `trainConfigOverrides`

- [ ] **Step 2: Run compile to verify it fails or catches missing wiring**

Run: `cd frontend && npm run build`

Expected: FAIL if helper types or store access are incomplete, otherwise use this as the first regression checkpoint before implementation.

- [ ] **Step 3: Write minimal implementation**

Update the training page to:

- show a provenance summary row in the recommendation card
- show small source badges next to `model`, `trainMode`, `exportFormats`, `imgsz`, `epochs`, and `patience`
- prioritize `已手动修改` over recommendation provenance

- [ ] **Step 4: Run compile to verify it passes**

Run: `cd frontend && npm run build`

Expected: PASS.

## Task 3: Run regression verification

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Run backend verification**

Run: `./.venv/bin/python tests/test_integration.py backend`

Expected: PASS.

- [ ] **Step 2: Run full verification**

Run: `./.venv/bin/python tests/test_integration.py full`

Expected: PASS, with existing sandbox-only skips remaining explicit.

## Self-Review

### Spec coverage

- explanation layer: covered by Task 2
- scenario complexity refinement: covered by Task 1
- regression safety: covered by Task 3

### Placeholder scan

No `TODO`, `TBD`, or “similar to Task N” shortcuts remain.

### Type consistency

Continue using `recommended_config`, `reason_summary`, `source_map`, and `trainConfigOverrides` consistently.
