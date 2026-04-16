# Phase 2 Training Recommendation Design

## Background

Phase 1 inserted an `algorithm_plan` stage into the existing workflow and proved that the product can:

- generate an algorithm draft from user intent and VLM output
- confirm the draft in the UI
- compile the draft into `pipeline_config`
- preview simple event behavior
- export a bundle skeleton

The current `training_recommendation` field is still intentionally shallow. It only exposes a small heuristic summary and is used as a display-oriented prefill, not as a real training decision layer.

Phase 2 upgrades that layer. The goal is not to lock users into algorithm-derived training configs. The goal is to generate a richer default training configuration that reflects:

- algorithm needs
- user defaults
- current runtime/device capability

The user must still be able to override any recommended field manually.

## Goal

Turn `training_recommendation` into a first-class recommendation output that can prefill most training parameters while preserving user override priority.

## Non-Goals

Phase 2 does not attempt to:

- implement a full online tracking runtime
- turn recommendations into hard execution constraints
- replace the existing training API contract
- introduce a policy DSL or a large rules engine
- make the exported package fully production-ready

## Product Rules

### Recommendation Priority

Field priority is:

1. user manual override
2. system recommendation
3. stored user defaults
4. system fallback

The recommendation layer may prefill fields, but it must not silently override a user-edited training field after the user has changed it.

### Recommendation Scope

Phase 2 recommendations should cover most train-facing fields, not just model selection:

- `model`
- `train_mode`
- `export_formats`
- `imgsz`
- `epochs`
- `lr0`
- `patience`
- `conf`
- `iou`

### Conflict Strategy

Conflicts should degrade gracefully instead of hard-failing:

- if a recommended local training mode is not suitable for the current capability context, downgrade to `cloud`
- if export formats conflict with the current capability context, filter the incompatible formats
- if a field cannot be inferred confidently, fall back to user defaults and then to stable system defaults

Phase 2 uses soft recommendations, not hard constraints.

## Architecture

### Layer 1: Algorithm Compiler

`backend/services/pipeline_compiler.py` should remain responsible for algorithm-facing compilation only:

- convert `algorithm_plan` into `pipeline_config`
- preserve detector / tracker / region / rule / output / packaging structure
- emit algorithm-side training signals such as `training_requirements`

It should not absorb user settings or runtime capability logic.

### Layer 2: Training Recommendation Service

Add a dedicated backend recommendation layer:

- `backend/services/training_recommendation_service.py`

This service merges:

- algorithm-facing signals from `algorithm_plan` and `pipeline_config`
- user defaults from settings
- runtime capability hints

and produces:

- `recommended_config`
- `reason_summary`
- `source_map`

This keeps recommendation logic separate from algorithm IR compilation.

### Layer 3: Frontend Prefill and Override Tracking

The frontend should consume the richer recommendation payload and apply it only as default values.

Once a user edits a field in training configuration, that field becomes user-owned for the current task state and should not be silently overwritten by later recommendation refreshes.

## Data Flow

### Inputs

The recommendation service should combine four inputs:

1. `algorithm_plan`
2. `pipeline_config`
3. user settings
4. runtime capability

### Runtime Capability Shape

Phase 2 should use a compact capability input rather than a large environment snapshot.

Recommended shape:

```json
{
  "local_training_available": true,
  "preferred_device": "mps",
  "available_export_formats": ["onnx", "coreml"],
  "supports_cloud_training": true
}
```

The capability payload may come from:

- backend-side inference when enough information already exists
- frontend-provided capability hints when local runtime information is only available there

If no capability hint is available, the recommendation service must fall back to safe defaults.

### Output Shape

Phase 2 should enrich `training_recommendation` so it aligns closely with the existing train config shape instead of inventing a parallel schema.

Recommended shape:

```json
{
  "recommended_config": {
    "model": "yolo11s.pt",
    "train_mode": "local",
    "export_formats": ["onnx"],
    "imgsz": 640,
    "epochs": 100,
    "lr0": 0.01,
    "patience": 20,
    "conf": 0.25,
    "iou": 0.45
  },
  "reason_summary": "Single-target occupancy monitoring with temporal rule; local training is available, so the system recommends YOLO11s, local mode, and ONNX export.",
  "source_map": {
    "model": "algorithm",
    "train_mode": "runtime",
    "export_formats": "runtime",
    "imgsz": "algorithm",
    "epochs": "algorithm",
    "lr0": "system_default",
    "patience": "algorithm",
    "conf": "user_default",
    "iou": "user_default"
  }
}
```

## Recommendation Rules

### Step 1: Derive Task Complexity

The recommendation service should first derive a compact internal signal set from `algorithm_plan` and `pipeline_config`:

- target class count
- tracking required or not
- temporal rule present or not
- region rule present or not
- scenario type
- runtime modes

This becomes the basis for all later per-field recommendations.

### Step 2: Merge Runtime and User Defaults

The service should then merge:

- capability hints
- settings defaults

into a normalized context that can answer questions such as:

- should local training be considered available
- should cloud training be preferred
- which export formats are valid in this environment
- which user defaults should be reused when algorithm signals are weak

### Step 3: Generate Per-Field Recommendations

The service should generate each field separately instead of returning a monolithic heuristic blob.

Recommended rules:

- `model`
  - prefer `yolo11s.pt` for small single-target tasks
  - prefer `yolo11m.pt` when target count or scenario complexity increases
- `train_mode`
  - prefer `local` when capability indicates it is available
  - prefer `cloud` when local capability is unavailable or unsuitable
- `export_formats`
  - start from algorithm/runtime needs
  - filter by capability support
- `imgsz`
  - use stable tiered values such as `640` for standard tasks and `1280` for denser tasks
- `epochs`
  - use higher values for more complex target/rule mixes
- `lr0`, `patience`, `conf`, `iou`
  - keep the rule set conservative and stable
  - prefer bounded adjustments rather than aggressive “auto-tuning”

## API Design

### Recommendation Generation Surface

Phase 2 should avoid introducing a large new API family. The simplest stable shape is:

- keep recommendation generation backend-owned
- enrich the existing algorithm-plan fetch / confirm responses with the richer `training_recommendation`
- optionally add a focused refresh endpoint later only if capability-dependent recomputation is needed after the initial plan confirmation

Initial recommendation refresh can therefore happen in the existing algorithm flow, and the frontend can prefill training config without a separate orchestration step.

### Capability Input

If frontend-local capability is needed, add an optional capability payload to the recommendation refresh path rather than requiring it everywhere.

That keeps existing flows backward-compatible while allowing more accurate local recommendations when the app has that information.

## Frontend Behavior

### Initial Prefill

When the task enters training configuration for the first time:

- read `training_recommendation.recommended_config`
- merge it into the existing train config state

### User Override Tracking

The task store should track which fields have been manually edited by the user for the current task.

Recommended state addition:

- `trainConfigOverrides: Record<string, boolean>`

When a user edits a field:

- mark that field as overridden

When a fresh recommendation is applied:

- only update fields that are not marked overridden

### Recommendation Explanation

The training page should display:

- a short explanation summary
- optionally a compact “source” hint for a few key fields such as `model`, `train_mode`, and `export_formats`

This improves trust without turning the page into a debugging console.

## Error Handling

### Missing Capability

If runtime capability is unavailable:

- do not block the user
- compute recommendations from algorithm signals plus settings defaults
- include a reason summary that does not overclaim runtime certainty

### Partial Recommendation Failure

If one recommendation field cannot be confidently derived:

- fall back per field
- do not fail the whole recommendation payload

### Backward Compatibility

Existing tasks with an older `pipeline_config.training_recommendation` must still load.

Frontend code should tolerate:

- missing `recommended_config`
- missing `source_map`
- older recommendation payloads

## Testing Strategy

### Backend

Add focused tests for:

- richer recommendation generation from algorithm plan complexity
- merging user defaults into weakly inferred fields
- capability-driven `train_mode` downgrade from `local` to `cloud`
- export format filtering
- stable output shape and source attribution

### Frontend

Add tests or compile-verified behavior for:

- initial prefill from recommendation
- user-edited fields remaining stable after recommendation refresh
- legacy recommendation payload compatibility

### Integration

Extend integration coverage so the end-to-end flow proves:

- algorithm confirmation returns the richer recommendation payload
- training page receives it in a store-compatible shape
- manual user overrides are preserved

## File Impact

Expected primary change surface:

- create `backend/services/training_recommendation_service.py`
- modify `backend/services/pipeline_compiler.py`
- modify `backend/routers/algorithm.py`
- modify `backend/routers/training.py` only if recommendation data needs to flow into training start payload validation
- modify `frontend/src/store/taskStore.ts`
- modify `frontend/src/pages/AlgorithmPlan.tsx`
- modify `frontend/src/pages/TrainConfig.tsx`
- modify `frontend/src/api/backend.ts`
- modify `tests/test_integration.py`

## Success Criteria

Phase 2 is complete when:

- the system emits a richer training recommendation derived from algorithm signals, settings, and capability hints
- training config is prefilled from that recommendation
- user manual edits always remain authoritative
- current Phase 1 flows continue to pass
- the design still leaves Phase 3 and Phase 4 room to evolve into stronger runtime and packaging systems without rewriting the recommendation boundary
