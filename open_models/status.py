"""Print concise live status without loading models or making network calls."""
import json
from common import MODELS, RUN, read


def main():
    for name in ['supervisor_status.json', 'input_progress.json']:
        path = RUN / name
        if path.exists():
            print(name, json.dumps(read(path), ensure_ascii=False))
    for entry in read(MODELS / 'models.json'):
        from pathlib import Path
        path = Path(entry['path'])
        size = sum(p.stat().st_size for p in path.rglob('*') if p.is_file()) / 1e9
        print(path.name, f'downloaded={size:.2f} GB',
            'download_complete=' + str((path / 'DOWNLOAD_COMPLETE.json').exists()))
        progress = RUN / path.name / 'progress.json'
        if progress.exists():
            print(json.dumps(read(progress), ensure_ascii=False))
        exp2_progress = RUN / path.name / 'experiment2_progress.json'
        if exp2_progress.exists():
            print('parallel_experiment2', json.dumps(read(exp2_progress), ensure_ascii=False))


if __name__ == '__main__':
    main()
