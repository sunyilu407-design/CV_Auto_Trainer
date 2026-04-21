import sys
from pathlib import Path
from typing import Dict, List, Optional

from services.pipeline_compiler import compile_algorithm_pipeline


def _merge_region_overrides(plan: Dict, region_overrides: Optional[List[Dict]]) -> Dict:
    merged_plan = dict(plan)
    merged_plan["regions"] = list(plan.get("regions", []))

    if not region_overrides:
        return merged_plan

    overrides_by_id = {
        item.get("region_id"): item
        for item in region_overrides
        if item.get("region_id")
    }
    regions = []
    for region in merged_plan.get("regions", []):
        override = overrides_by_id.get(region.get("region_id"))
        if override:
            next_region = dict(region)
            next_region.update(override)
            regions.append(next_region)
        else:
            regions.append(region)
    merged_plan["regions"] = regions
    return merged_plan


def preview_algorithm_events(
    algorithm_plan: Dict,
    sample_boxes: List[Dict],
    observation_frames: Optional[List[Dict]] = None,
    region_overrides: Optional[List[Dict]] = None,
) -> Dict:
    project_root = Path(__file__).resolve().parents[2]
    worker_path = project_root / "worker"
    if str(worker_path) not in sys.path:
        sys.path.insert(0, str(worker_path))

    from pipeline.frame_adapter import build_preview_frames, normalize_observation_frames
    from pipeline.runtime_session import RuntimeSession

    merged_plan = _merge_region_overrides(algorithm_plan, region_overrides)
    pipeline_config = compile_algorithm_pipeline(merged_plan)

    rules = pipeline_config.get("rules", [])
    preview_duration_ms = max(
        [
            int(rule.get("duration_ms", 0))
            if rule.get("duration_ms") is not None
            else int(rule.get("duration_seconds", 0)) * 1000
            for rule in rules
        ]
        + [1000]
    ) + 2000

    session = RuntimeSession(pipeline_config)
    frames = normalize_observation_frames(observation_frames)
    if not frames:
        frames = build_preview_frames(sample_boxes, preview_duration_ms)

    track_states: List[Dict] = []
    events: List[Dict] = []
    for frame in frames:
        result = session.process_frame(frame)
        track_states = result.get("track_states", [])
        events.extend(result.get("events", []))

    return {
        "pipeline_config": pipeline_config,
        "track_states": track_states,
        "events": events,
    }
