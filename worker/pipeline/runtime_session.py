from copy import deepcopy
from typing import Any, Dict, List

from pipeline.event_engine import evaluate_pipeline_events
from pipeline.tracking_runtime import update_track_states


class RuntimeSession:
    """统一离线/流式规则运行时会话。"""

    def __init__(self, pipeline_config: Dict[str, Any]):
        self.pipeline_config = pipeline_config or {}
        self.regions = list(self.pipeline_config.get("regions", []))
        self.track_states: List[Dict[str, Any]] = []

    def process_frame(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        frame_index = int(frame.get("frame_index", len(self.track_states) + 1))
        timestamp_ms = int(frame.get("timestamp_ms", 0))
        detections = list(frame.get("detections", []))
        previous_tracks = deepcopy(self.track_states)

        self.track_states = update_track_states(
            existing_tracks=self.track_states,
            detections=detections,
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            regions=self.regions,
        )
        events = evaluate_pipeline_events(
            pipeline_config=self.pipeline_config,
            track_states=self.track_states,
            previous_tracks=previous_tracks,
        )

        return {
            "frame_index": frame_index,
            "timestamp_ms": timestamp_ms,
            "track_states": deepcopy(self.track_states),
            "events": events,
        }
