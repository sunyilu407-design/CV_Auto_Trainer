import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bbox_center(bbox_xywhn: Optional[Sequence[Any]]) -> Tuple[float, float]:
    if not bbox_xywhn or len(bbox_xywhn) < 2:
        return 0.0, 0.0
    return _to_float(bbox_xywhn[0]), _to_float(bbox_xywhn[1])


def _distance(a_bbox: Optional[Sequence[Any]], b_bbox: Optional[Sequence[Any]]) -> float:
    ax, ay = _bbox_center(a_bbox)
    bx, by = _bbox_center(b_bbox)
    return math.hypot(ax - bx, ay - by)


def _bbox_iou(a_bbox: Optional[Sequence[Any]], b_bbox: Optional[Sequence[Any]]) -> float:
    if not a_bbox or not b_bbox or len(a_bbox) < 4 or len(b_bbox) < 4:
        return 0.0

    ax, ay, aw, ah = [_to_float(item) for item in a_bbox[:4]]
    bx, by, bw, bh = [_to_float(item) for item in b_bbox[:4]]

    a_left = ax - aw / 2
    a_right = ax + aw / 2
    a_top = ay - ah / 2
    a_bottom = ay + ah / 2
    b_left = bx - bw / 2
    b_right = bx + bw / 2
    b_top = by - bh / 2
    b_bottom = by + bh / 2

    inter_left = max(a_left, b_left)
    inter_right = min(a_right, b_right)
    inter_top = max(a_top, b_top)
    inter_bottom = min(a_bottom, b_bottom)
    inter_width = max(0.0, inter_right - inter_left)
    inter_height = max(0.0, inter_bottom - inter_top)
    inter_area = inter_width * inter_height

    a_area = aw * ah
    b_area = bw * bh
    union_area = max(a_area + b_area - inter_area, 1e-9)
    return inter_area / union_area


def _center_in_region(bbox_xywhn: Optional[Sequence[Any]], region: Dict[str, Any]) -> bool:
    region_bbox = region.get("bbox_xywhn")
    if not bbox_xywhn or not region_bbox or len(region_bbox) < 4:
        return False

    cx, cy = _bbox_center(bbox_xywhn)
    rcx, rcy, rw, rh = [_to_float(item) for item in region_bbox[:4]]
    left = rcx - rw / 2
    right = rcx + rw / 2
    top = rcy - rh / 2
    bottom = rcy + rh / 2
    return left <= cx <= right and top <= cy <= bottom


def _regions_inside(
    bbox_xywhn: Optional[Sequence[Any]],
    regions: Optional[List[Dict[str, Any]]],
    detection_region_id: Optional[str],
    default_region_id: Optional[str],
) -> List[str]:
    if regions:
        region_ids = [region.get("region_id") for region in regions if _center_in_region(bbox_xywhn, region)]
        return [region_id for region_id in region_ids if region_id]

    fallback_region = detection_region_id or default_region_id
    return [fallback_region] if fallback_region else []


def _next_track_number(existing_tracks: List[Dict[str, Any]]) -> int:
    max_number = 0
    for track in existing_tracks:
        track_id = str(track.get("track_id", ""))
        if track_id.startswith("track-"):
            suffix = track_id.split("track-", 1)[1]
            if suffix.isdigit():
                max_number = max(max_number, int(suffix))
    return max_number + 1


def _match_threshold(track: Dict[str, Any], max_match_distance: float) -> float:
    bbox_xywhn = track.get("bbox_xywhn") or [0, 0, 0, 0]
    bbox_scale = max(_to_float(bbox_xywhn[2], 0.0), _to_float(bbox_xywhn[3], 0.0))
    lost_bonus = 0.12 * min(int(track.get("lost_frames", 0)), 2)
    size_bonus = min(0.06, bbox_scale * 0.25)
    return max_match_distance + lost_bonus + size_bonus


def _normalize_existing_track(
    track: Dict[str, Any],
    timestamp_ms: int,
    default_region_id: Optional[str],
) -> Dict[str, Any]:
    normalized = dict(track)
    first_seen_frame = int(normalized.get("first_seen_frame", normalized.get("last_seen_frame", 1) or 1))
    normalized["first_seen_frame"] = first_seen_frame
    normalized["last_seen_frame"] = int(normalized.get("last_seen_frame", first_seen_frame))
    normalized["age_frames"] = int(normalized.get("age_frames", normalized["last_seen_frame"] - first_seen_frame + 1))
    normalized["hit_streak"] = int(normalized.get("hit_streak", 1))
    normalized["lost_frames"] = int(normalized.get("lost_frames", 0))
    normalized["timestamp_ms"] = int(normalized.get("timestamp_ms", timestamp_ms))
    normalized["present_duration_ms"] = int(
        normalized.get(
            "present_duration_ms",
            int(_to_float(normalized.get("present_duration_seconds", 0)) * 1000),
        )
    )
    normalized["regions_inside"] = list(normalized.get("regions_inside", []))
    normalized["entered_region_at"] = dict(normalized.get("entered_region_at", {}))
    normalized["last_event_frame"] = dict(normalized.get("last_event_frame", {}))
    normalized["first_seen_timestamp_ms"] = int(
        normalized.get("first_seen_timestamp_ms", normalized["timestamp_ms"] - normalized["present_duration_ms"])
    )
    if "region_id" not in normalized and normalized["regions_inside"]:
        normalized["region_id"] = normalized["regions_inside"][0]
    elif "region_id" not in normalized and default_region_id:
        normalized["region_id"] = default_region_id
    return normalized


def _build_track_state(
    previous: Optional[Dict[str, Any]],
    detection: Dict[str, Any],
    frame_index: int,
    timestamp_ms: int,
    track_id: str,
    regions: Optional[List[Dict[str, Any]]],
    default_region_id: Optional[str],
) -> Dict[str, Any]:
    bbox_xywhn = detection.get("bbox_xywhn")
    previous_regions = set(previous.get("regions_inside", [])) if previous else set()
    previous_entered = dict(previous.get("entered_region_at", {})) if previous else {}

    current_regions = _regions_inside(
        bbox_xywhn=bbox_xywhn,
        regions=regions,
        detection_region_id=detection.get("region_id"),
        default_region_id=default_region_id,
    )
    entered_region_at = {
        region_id: previous_entered.get(region_id, timestamp_ms)
        for region_id in current_regions
    }

    first_seen_frame = int(previous.get("first_seen_frame", frame_index)) if previous else frame_index
    first_seen_timestamp_ms = (
        int(previous.get("first_seen_timestamp_ms", timestamp_ms))
        if previous
        else timestamp_ms
    )

    track_state = {
        "track_id": track_id,
        "class_name": detection.get("class_name", previous.get("class_name", "target") if previous else "target"),
        "bbox_xywhn": bbox_xywhn,
        "confidence": _to_float(detection.get("confidence", detection.get("conf", 1.0)), 1.0),
        "region_id": current_regions[0] if current_regions else detection.get("region_id"),
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "first_seen_frame": first_seen_frame,
        "last_seen_frame": frame_index,
        "age_frames": frame_index - first_seen_frame + 1,
        "hit_streak": (int(previous.get("hit_streak", 0)) + 1) if previous else 1,
        "lost_frames": 0,
        "present_duration_ms": max(timestamp_ms - first_seen_timestamp_ms, 0),
        "present_duration_seconds": max(timestamp_ms - first_seen_timestamp_ms, 0) // 1000,
        "regions_inside": current_regions,
        "entered_region_at": entered_region_at,
        "last_event_frame": dict(previous.get("last_event_frame", {})) if previous else {},
        "first_seen_timestamp_ms": first_seen_timestamp_ms,
        "previous_regions_inside": list(previous_regions),
    }
    return track_state


def update_track_states(
    existing_tracks: List[Dict[str, Any]],
    detections: List[Dict[str, Any]],
    elapsed_seconds: Optional[int] = None,
    frame_index: Optional[int] = None,
    timestamp_ms: Optional[int] = None,
    region_id: str = "primary_region",
    regions: Optional[List[Dict[str, Any]]] = None,
    max_match_distance: float = 0.28,
    max_lost_frames: int = 2,
) -> List[Dict[str, Any]]:
    """
    轻量跟踪状态更新器。

    当前实现使用同类目标的中心点距离做匹配，保证 Phase 3 的统一事件运行时可验证，
    同时保持和旧 preview 逻辑兼容的基础字段。
    """

    if frame_index is None:
        frame_index = max([int(track.get("last_seen_frame", 0)) for track in existing_tracks] + [0]) + 1

    if timestamp_ms is None:
        if elapsed_seconds is not None:
            timestamp_ms = int(elapsed_seconds) * 1000
        else:
            timestamp_ms = max([int(track.get("timestamp_ms", 0)) for track in existing_tracks] + [0])

    normalized_tracks = [
        _normalize_existing_track(track, timestamp_ms=timestamp_ms, default_region_id=region_id)
        for track in existing_tracks
    ]
    matched_track_ids: Set[str] = set()
    next_states: List[Dict[str, Any]] = []
    next_track_number = _next_track_number(normalized_tracks)

    for detection in detections:
        class_name = detection.get("class_name", "target")
        candidates = [
            track
            for track in normalized_tracks
            if track.get("track_id") not in matched_track_ids
            and track.get("class_name") == class_name
            and int(track.get("lost_frames", 0)) <= max_lost_frames
        ]

        matched_track = None
        matched_distance = None
        matched_iou = 0.0
        for candidate in candidates:
            distance = _distance(candidate.get("bbox_xywhn"), detection.get("bbox_xywhn"))
            iou = _bbox_iou(candidate.get("bbox_xywhn"), detection.get("bbox_xywhn"))
            candidate_score = distance - (iou * 0.12)
            if matched_distance is None or candidate_score < matched_distance:
                matched_track = candidate
                matched_distance = candidate_score
                matched_iou = iou

        if (
            matched_track
            and matched_distance is not None
            and (
                matched_distance <= _match_threshold(matched_track, max_match_distance)
                or matched_iou >= 0.05
            )
        ):
            track_id = matched_track.get("track_id", f"track-{next_track_number}")
            matched_track_ids.add(track_id)
            next_states.append(
                _build_track_state(
                    previous=matched_track,
                    detection=detection,
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    track_id=track_id,
                    regions=regions,
                    default_region_id=region_id,
                )
            )
            continue

        track_id = f"track-{next_track_number}"
        next_track_number += 1
        matched_track_ids.add(track_id)
        next_states.append(
            _build_track_state(
                previous=None,
                detection=detection,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                track_id=track_id,
                regions=regions,
                default_region_id=region_id,
            )
        )

    for track in normalized_tracks:
        track_id = track.get("track_id")
        if track_id in matched_track_ids:
            continue

        lost_track = dict(track)
        lost_track["lost_frames"] = int(track.get("lost_frames", 0)) + 1
        lost_track["age_frames"] = frame_index - int(track.get("first_seen_frame", frame_index)) + 1
        lost_track["frame_index"] = frame_index
        lost_track["timestamp_ms"] = timestamp_ms
        lost_track["present_duration_ms"] = max(
            timestamp_ms - int(track.get("first_seen_timestamp_ms", timestamp_ms)),
            int(track.get("present_duration_ms", 0)),
        )
        lost_track["present_duration_seconds"] = int(lost_track["present_duration_ms"]) // 1000
        lost_track["previous_regions_inside"] = list(track.get("regions_inside", []))
        if int(lost_track["lost_frames"]) <= max_lost_frames:
            next_states.append(lost_track)

    return next_states
