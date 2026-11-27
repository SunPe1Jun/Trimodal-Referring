"""Generate one real sample per task, cache it for the full run, and release the GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

if '--family' not in sys.argv or sys.argv[sys.argv.index('--family') + 1] != 'onevision':
    sys.path.insert(0, '/workspace/usr3/models/baseline_expansion_20261116/runtime_457')

from common import MODELS, ROOT, RUN, digest, protocol, read, write
from worker import infer
from adapters import Adapter
from compatibility import update_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--family', required=True)
    args = parser.parse_args()
    import torch
    import transformers
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.92)
    out = RUN / args.model
    out.mkdir(parents=True, exist_ok=True)
    lock = {'model': args.model, 'family': args.family,
        'checkpoint': read(MODELS / args.model / 'DOWNLOAD_COMPLETE.json'),
        'dtype': 'bfloat16', 'quantization': None, 'do_sample': False,
        'output_budgets': {'experiment1': 1536, 'experiment2': 768},
        'coordinate_policy': 'paired_canvas_map', 'secondary_policy': 'source_or_canvas_map',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'source_sha256': {name: digest(Path(__file__).parent / name)
            for name in ['worker.py', 'adapters.py', 'common.py']}}
    if (out / 'protocol.json').exists():
        assert read(out / 'protocol.json') == lock
    write(out / 'protocol.json', lock)
    write(out / 'progress.json', {'status': 'VALIDATING', 'pid': os.getpid()})
    adapter = Adapter(args.family, str(MODELS / args.model))
    tasks = read(RUN / 'inputs/smoke_0000/prepared_tasks.json')
    m = protocol.modules()
    results = []
    for setting in ['experiment1', 'experiment2']:
        task = next(t for t in tasks if t['setting'] == setting)
        result = infer(task, out / 'batches/smoke_0000', adapter, lock)
        if setting == 'experiment1':
            result = update_record(task, result, m)
            write(out / 'batches/smoke_0000' / setting / 'raw' / task['scene'] / f"row_{task['row_index']}.json", result)
        elif result['response_status'] == 'unparsed':
            ok, parsed, extra = protocol.parse_task(task, result['model_raw_output'], m)
            result.update(parse_ok=ok, parsed_json=parsed, response_status='ok' if ok else 'invalid', **extra)
            if setting == 'experiment1':
                result['parsed_response'] = parsed
            write(out / 'batches/smoke_0000' / setting / 'raw' / task['scene'] / f"row_{task['row_index']}.json", result)
        results.append({k: result[k] for k in ['setting', 'key', 'parse_ok', 'elapsed_seconds', 'peak_reserved_gib']})
        print(json.dumps(results[-1]), flush=True)
        torch.cuda.empty_cache()
    write(out / 'adapter_validation.json', {'results': results,
        'generation_config': str(getattr(getattr(adapter.model, 'llm', adapter.model), 'generation_config', None)),
        'cached_for_full_run': True})
    write(out / 'progress.json', {'status': 'VALIDATED', 'completed_requests': len(results),
        'expected_requests': 8000, 'pid': os.getpid()})


if __name__ == '__main__':
    main()
