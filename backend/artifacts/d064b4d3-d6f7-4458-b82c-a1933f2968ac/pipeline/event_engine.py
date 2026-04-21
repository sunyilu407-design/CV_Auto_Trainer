from typing import Any, Dict, List, Optional, Set


def _center_in_region(track: Optional[Dict[str, Any]], region: Dict[str, Any]) -> bool:
    if not track:
        return False

    bbox = track.get("bbox_xywhn")
    region_bbox = region.get("bbox_xywhn")
    if not bbox or not region_bbox:
        return track.get("region_id") == region.get("region_id")

    cx, cy = bbox[0], bbox[1]
    rcx, rcy, rw, rh = region_bbox
    left = rcx - rw / 2
    right = rcx + rw / 2
    top = rcy - rh / 2
    bottom = rcy + rh / 2
    return left <= cx <= right and top <= cy <= bottom


def _regions_for_track(track: Optional[Dict[str, Any]], regions: Dict[str, Dict[str, Any]]) -> Set[str]:
    if not track:
        return set()

    declared_regions = track.get("regions_inside")
    if declared_regions is not None:
        return {region_id for region_id in declared_regions if region_id}

    if regions:
        return {
            region_id
            for region_id, region in regions.items()
            if _center_in_region(track, region)
        }

    region_id = track.get("region_id")
    return {region_id} if region_id else set()


def _duration_ms(rule: Dict[str, Any]) -> int:
    if "duration_ms" in rule and rule.get("duration_ms") is not None:
        return int(rule.get("duration_ms", 0))
    if "duration_seconds" in rule and rule.get("duration_seconds") is not None:
        return int(rule.get("duration_seconds", 0)) * 1000
    return 0


def _region_presence_ms(track: Optional[Dict[str, Any]], region_id: Optional[str]) -> int:
    if not track or not region_id:
        return 0

    entered_region_at = dict(track.get("entered_region_at", {}))
    timestamp_ms = int(track.get("timestamp_ms", 0))
    if region_id in entered_region_at:
        return max(timestamp_ms - int(entered_region_at[region_id]), 0)

    if region_id in _regions_for_track(track, {}):
        if "present_duration_ms" in track:
            return int(track.get("present_duration_ms", 0))
        if "present_duration_seconds" in track:
            return int(track.get("present_duration_seconds", 0)) * 1000
    return 0


def _base_event(
    rule: Dict[str, Any],
    track: Dict[str, Any],
    frame_index: int,
    timestamp_ms: int,
    region_id: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_code": rule.get("event_code"),
        "rule_id": rule.get("rule_id"),
        "track_id": track.get("track_id"),
        "class_name": track.get("class_name"),
        "frame_index": frame_index,
        "timestamp_ms": timestamp_ms,
        "region_id": region_id,
        "bbox_xywhn": track.get("bbox_xywhn"),
        "payload": payload or {},
    }


def evaluate_pipeline_events(
    pipeline_config: Dict[str, Any],
    track_states: List[Dict[str, Any]],
    previous_tracks: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    rules = pipeline_config.get("rules", [])
    regions = {
        region.get("region_id"): region
        for region in pipeline_config.get("regions", [])
        if region.get("region_id")
    }
    current_by_id = {
        track.get("track_id"): track
        for track in track_states
        if track.get("track_id")
    }
    previous_by_id = {
        track.get("track_id"): track
        for track in (previous_tracks or [])
        if track.get("track_id")
    }

    for rule in rules:
        rule_type = rule.get("rule_type")
        rule_id = rule.get("rule_id")
        target_class = rule.get("target_class")
        for track_id in set(current_by_id.keys()) | set(previous_by_id.keys()):
            current_track = current_by_id.get(track_id)
            previous_track = previous_by_id.get(track_id)
            anchor_track = current_track or previous_track
            if not anchor_track:
                continue
            if target_class and anchor_track.get("class_name") != target_class:
                continue

            current_regions = _regions_for_track(current_track, regions)
            previous_regions = _regions_for_track(previous_track, regions)
            frame_index = int((current_track or previous_track).get("frame_index", 0))
            timestamp_ms = int((current_track or previous_track).get("timestamp_ms", 0))

            if rule_type == "region_enter":
                region_id = rule.get("region_id")
                if current_track and region_id in current_regions and region_id not in previous_regions:
                    event = _base_event(rule, current_track, frame_index, timestamp_ms, region_id)
                    current_track.setdefault("last_event_frame", {})[rule_id] = frame_index
                    events.append(event)

            elif rule_type == "region_exit":
                region_id = rule.get("region_id")
                if previous_track and region_id in previous_regions and region_id not in current_regions:
                    base_track = current_track or previous_track
                    event = _base_event(rule, base_track, frame_index, timestamp_ms, region_id)
                    if current_track:
                        current_track.setdefault("last_event_frame", {})[rule_id] = frame_index
                    events.append(event)

            elif rule_type == "region_presence_duration":
                region_id = rule.get("region_id")
                if not current_track or region_id not in current_regions:
                    continue
                required_ms = _duration_ms(rule)
                current_duration_ms = _region_presence_ms(current_track, region_id)
                previous_duration_ms = (
                    _region_presence_ms(previous_track, region_id) if region_id in previous_regions else 0
                )
                if current_duration_ms < required_ms:
                    continue
                if previous_duration_ms >= required_ms:
                    continue

                payload = {
                    "duration_ms": current_duration_ms,
                    "duration_seconds": current_duration_ms // 1000,
                }
                event = _base_event(rule, current_track, frame_index, timestamp_ms, region_id, payload=payload)
                current_track.setdefault("last_event_frame", {})[rule_id] = frame_index
                events.append(event)

            elif rule_type == "cross_region_transition":
                from_region_id = rule.get("from_region_id")
                to_region_id = rule.get("to_region_id")
                if (
                    current_track
                    and previous_track
                    and from_region_id in previous_regions
                    and to_region_id in current_regions
                    and to_region_id not in previous_regions
                ):
                    payload = {
                        "from_region_id": from_region_id,
                        "to_region_id": to_region_id,
                    }
                    event = _base_event(rule, current_track, frame_index, timestamp_ms, to_region_id, payload=payload)
                    current_track.setdefault("last_event_frame", {})[rule_id] = frame_index
                    events.append(event)

    return events
