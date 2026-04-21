import argparse
import json
from pathlib import Path

from runtime_support import RuntimeSession, normalize_observation_frames

CONFIG_PATH = Path(__file__).with_name('pipeline.json')
DEFAULT_INPUT_PATH = Path(__file__).with_name('sample_input.json')
DEFAULT_OUTPUT_PATH = Path(__file__).with_name('sample_output.json')

def main():
    parser = argparse.ArgumentParser(description='Run exported algorithm package')
    parser.add_argument('--input', default=str(DEFAULT_INPUT_PATH), help='path to observation frame json')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_PATH), help='path to write runtime output json')
    args = parser.parse_args()

    pipeline = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    input_payload = json.loads(Path(args.input).read_text(encoding='utf-8'))
    frames = normalize_observation_frames(input_payload.get('observation_frames', []))
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
