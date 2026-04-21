import argparse
import json
from pathlib import Path

from runtime_support import RuntimeSession, normalize_observation_frames

CONFIG_PATH = Path(__file__).with_name('pipeline.json')
DEFAULT_INPUT_PATH = Path(__file__).with_name('sample_input.json')
DEFAULT_OUTPUT_PATH = Path(__file__).with_name('sample_output.json')

def _load_json_file(path: Path):
    payload = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        return payload.get('observation_frames', [])
    if isinstance(payload, list):
        return payload
    raise ValueError(f'Unsupported JSON payload: {path}')

def _load_jsonl_file(path: Path):
    frames = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames

def _load_input_frames(path: Path):
    if path.is_dir():
        frames = []
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            suffix = child.suffix.lower()
            if suffix == '.json':
                frames.extend(_load_json_file(child))
            elif suffix == '.jsonl':
                frames.extend(_load_jsonl_file(child))
        return frames
    suffix = path.suffix.lower()
    if suffix == '.json':
        return _load_json_file(path)
    if suffix == '.jsonl':
        return _load_jsonl_file(path)
    raise ValueError(f'Unsupported input path: {path}')

def main():
    parser = argparse.ArgumentParser(description='Run exported algorithm package')
    parser.add_argument('--input', default=str(DEFAULT_INPUT_PATH), help='path to json/jsonl/directory input')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='path to write runtime output json')
    args = parser.parse_args()

    pipeline = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    frames = normalize_observation_frames(_load_input_frames(Path(args.input)))
    session = RuntimeSession(pipeline)
    events = []
    track_states = []
    for frame in frames:
        result = session.process_frame(frame)
        track_states = result.get('track_states', [])
        events.extend(result.get('events', []))

    output_payload = {
        'metadata': pipeline.get('metadata', {}),
        'frame_count': len(frames),
        'track_states': track_states,
        'events': events,
    }
    Path(args.output).write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Generated {len(events)} events -> {args.output}")

if __name__ == '__main__':
    main()
