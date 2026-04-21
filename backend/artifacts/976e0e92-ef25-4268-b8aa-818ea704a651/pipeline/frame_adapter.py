from typing import Any, Dict, List, Optional


def _normalize_detection(detection: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "class_name": detection.get("class_name", "target"),
        "bbox_xywhn": list(detection.get("bbox_xywhn", [])),
        "confidence": float(detection.get("confidence", detection.get("conf", 1.0))),
    }


def normalize_observation_frame(
    frame: Dict[str, Any],
    fallback_index: int,
    fallback_timestamp_ms: int,
) -> Dict[str, Any]:
    detections = frame.get("detections", [])
    return {
        "frame_index": int(frame.get("frame_index", fallback_index)),
        "timestamp_ms": int(frame.get("timestamp_ms", fallback_timestamp_ms)),
        "detections": [_normalize_detection(detection) for detection in detections],
    }


def normalize_observation_frames(observation_frames: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not observation_frames:
        return []

    normalized_frames: List[Dict[str, Any]] = []
    fallback_timestamp_ms = 0
    for index, frame in enumerate(observation_frames, start=1):
        normalized = normalize_observation_frame(
            frame=frame,
            fallback_index=index,
            fallback_timestamp_ms=fallback_timestamp_ms,
        )
        fallback_timestamp_ms = normalized["timestamp_ms"] + 1000
        normalized_frames.append(normalized)
    return normalized_frames


def build_preview_frames(sample_boxes: List[Dict[str, Any]], preview_duration_ms: int) -> List[Dict[str, Any]]:
    detections = [_normalize_detection(item) for item in sample_boxes]
    return [
        {
            "frame_index": 1,
            "timestamp_ms": 0,
            "detections": detections,
        },
        {
            "frame_index": 2,
            "timestamp_ms": preview_duration_ms,
            "detections": detections,
        },
    ]
