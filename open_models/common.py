"""Shared frozen protocols for the 2026-11 open-weight baseline expansion."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'closed_models'))
import run_responses as protocol

RUN = ROOT / 'open_models/outputs/expansion_20261116'
MODELS = Path('/workspace/usr3/models/baseline_expansion_20261116')
SETTINGS = ('experiment1', 'experiment2')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def selection():
    rows = protocol.read_csv(ROOT / 'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
    ids = [r['unified_id'] for r in rows]
    assert len(ids) == len(set(ids)) == 4000
    smoke = [r['unified_id'] for r in rows if int(r['row_index']) < 10]
    assert len(smoke) == 80
    return smoke, [key for key in ids if key not in set(smoke)]


def batches():
    smoke, remaining = selection()
    result = []
    for stage, keys in [('smoke', smoke), ('full', remaining)]:
        for offset in range(0, len(keys), 16):
            result.append((stage, offset, keys[offset:offset + 16]))
    return result


def batch_name(stage, offset):
    return f'{stage}_{offset:04d}'


def write(path, value):
    protocol.write_json(Path(path), value)


def read(path):
    return json.loads(Path(path).read_text())
