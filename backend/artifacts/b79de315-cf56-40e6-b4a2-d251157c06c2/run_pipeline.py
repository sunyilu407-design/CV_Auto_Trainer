import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name('pipeline.json')

def main():
    pipeline = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
    print('Loaded pipeline', pipeline.get('metadata', {}).get('summary', 'unknown'))

if __name__ == '__main__':
    main()
