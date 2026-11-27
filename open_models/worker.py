"""Resumable local-only experiment 1/2 worker with input locks and original evaluation."""
import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# Keep the original research environment unchanged; custom checkpoints use their own runtime.
RUNTIME = '/workspace/usr3/models/baseline_expansion_20261116/runtime_457'
if '--family' not in sys.argv or sys.argv[sys.argv.index('--family') + 1] not in ('onevision', 'qwen35'):
    sys.path.insert(0, RUNTIME)

from common import MODELS, ROOT, RUN, SETTINGS, batch_name, batches, digest, protocol, read, write
from adapters import Adapter
from compatibility import VERSION as PARSER_VERSION, update_record


def infer(task, out, adapter, lock):
    import torch
    path = out / task['setting'] / 'raw' / task['scene'] / f"row_{task['row_index']}.json"
    request_hash = protocol.sha(json.dumps({'lock': lock, 'task': task}, sort_keys=True).encode())
    if path.exists():
        result = read(path)
        if result['request_sha256'] != request_hash:
            raise RuntimeError('Cached model/input differs: ' + str(path))
        if result.get('response_status') in ('ok', 'invalid', 'unparsed'):
            return result
    for image in task['image_audit']:
        if digest(image['path']) != image['sha256']:
            raise RuntimeError('Input image drift: ' + image['path'])
    budget = 1536 if task['setting'] == 'experiment1' else 768
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    raw, preprocessing = adapter.generate(task, budget)
    result = dict(task, model=lock['model'], served_model=lock['model'],
        request_sha256=request_hash, model_raw_output=raw, parsed_json={}, parse_ok=False,
        error_message='', response_status='unparsed', usage={},
        preprocessing=preprocessing, max_new_tokens=budget,
        elapsed_seconds=time.monotonic() - started,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    # Preserve generation before parsing so an evaluator exception cannot trigger regeneration.
    write(path, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--family', required=True, choices=['minicpm', 'molmo', 'onevision', 'internvl', 'qwen35'])
    p.add_argument('--memory-fraction', type=float, default=.46)
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    out = RUN / args.model
    out.mkdir(parents=True, exist_ok=True)
    identity = read(MODELS / args.model / 'DOWNLOAD_COMPLETE.json')
    import transformers
    lock = {'model': args.model, 'family': args.family, 'checkpoint': identity,
        'dtype': 'bfloat16', 'quantization': None, 'do_sample': False,
        'output_budgets': {'experiment1': 1536, 'experiment2': 768},
        'coordinate_policy': 'paired_canvas_map', 'secondary_policy': 'source_or_canvas_map',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'source_sha256': {name: digest(Path(__file__).parent / name)
            for name in ['worker.py', 'adapters.py', 'common.py']}}
    if args.family == 'qwen35':
        lock['source_sha256']['qwen35_adapter.py'] = digest(Path(__file__).parent / 'qwen35_adapter.py')
    lock_path = out / 'protocol.json'
    if lock_path.exists() and read(lock_path) != lock:
        previous = read(lock_path)
        before = {k: v for k, v in previous.items() if k != 'source_sha256'}
        after = {k: v for k, v in lock.items() if k != 'source_sha256'}
        if before != after or any(previous['source_sha256'][name] != lock['source_sha256'][name]
                                  for name in lock['source_sha256'] if name != 'worker.py'):
            raise RuntimeError('Model-visible protocol/source changed; use a new run directory')
        # Resume-policy changes do not alter images/prompts/generation. Keep the
        # original request identity, disclose both worker revisions separately.
        history_path = out / 'worker_source_history.json'
        history = read(history_path) if history_path.exists() else []
        history.append({'original_worker': previous['source_sha256']['worker.py'],
                        'current_worker': lock['source_sha256']['worker.py'],
                        'reason': 'Local runner change; model-visible adapter and protocol unchanged. Parser version recorded separately.'})
        write(history_path, history)
        lock = previous
    write(lock_path, lock)
    write(out / 'parser_protocol.json', {'version': PARSER_VERSION,
        'sha256': digest(Path(__file__).parent / 'compatibility.py'),
        'scope': 'experiment1 JSON field compatibility; experiment2 mapping unchanged',
        'generation_protocol_unchanged': True})
    write(out / 'progress.json', {'status': 'LOADING_MODEL', 'pid': os.getpid()})
    if args.family == 'qwen35':
        from qwen35_adapter import Qwen35Adapter
        adapter = Qwen35Adapter(args.family, str(MODELS / args.model))
    else:
        adapter = Adapter(args.family, str(MODELS / args.model))
    modules = protocol.modules()
    all_results = []
    for stage, offset, keys in batches():
        name = batch_name(stage, offset)
        source = RUN / 'inputs' / name
        while not (source / 'READY.json').exists():
            producer_pid = int((RUN / 'prepare.pid').read_text())
            producer = Path('/proc', str(producer_pid), 'stat')
            if not producer.exists() or producer.read_text().split()[2] == 'Z':
                raise RuntimeError('Input producer stopped before ' + name + '; inspect prepare.log')
            write(out / 'progress.json', {'status': 'WAITING_INPUTS', 'batch': name,
                'completed_requests': len(all_results), 'pid': os.getpid()})
            time.sleep(10)
        target = out / 'batches' / name
        target.mkdir(parents=True, exist_ok=True)
        for rel in ['prepared_tasks.json', 'selection.json', 'experiment2/manifest/manifest_all.csv']:
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source / rel, dest)
            assert digest(dest) == digest(source / rel)
        tasks = read(target / 'prepared_tasks.json')
        _, gt = protocol.restore_frozen_gt(target)
        results = []
        for task in tasks:
            write(out / 'progress.json', {'status': 'RUNNING', 'batch': name,
                'active_setting': task['setting'], 'active_key': task['key'],
                'completed_requests': len(all_results) + len(results), 'expected_requests': 8000,
                'pid': os.getpid(), 'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
            result = infer(task, target, adapter, lock)
            if task['setting'] == 'experiment1' and result.get('parser_version') != PARSER_VERSION:
                result = update_record(task, result, modules)
                write(target / task['setting'] / 'raw' / task['scene'] / f"row_{task['row_index']}.json", result)
            elif result['response_status'] == 'unparsed':
                ok, parsed, extra = protocol.parse_task(task, result['model_raw_output'], modules)
                result.update(parse_ok=ok, parsed_json=parsed, response_status='ok' if ok else 'invalid', **extra)
                if task['setting'] == 'experiment1':
                    result['parsed_response'] = parsed
                write(target / task['setting'] / 'raw' / task['scene'] / f"row_{task['row_index']}.json", result)
            results.append(result)
            print('[done]', args.model, task['setting'], task['key'],
                  'parse_ok=', result['parse_ok'], 'seconds=', round(result['elapsed_seconds'], 2),
                  'peak_GiB=', round(result['peak_reserved_gib'], 2), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        metrics = protocol.evaluate(results, gt, target, modules)
        write(target / 'summary.json', {'requests': len(results), 'valid_outputs': sum(r['parse_ok'] for r in results),
            'metrics': metrics, 'stage': stage})
        all_results.extend(results)
        # Gate only on engineering validity, never on task accuracy. Invalid answers stay in evaluation.
        for setting in SETTINGS:
            records = [r for r in all_results if r['setting'] == setting]
            if os.environ.get('OPEN_MODELS_STOP_ON_INVALID', '0') == '1' and len(records) >= 16 and sum(not r['parse_ok'] for r in records) / len(records) > .25:
                raise RuntimeError('More than 25% invalid JSON outputs: inspect adapter before continuing ' + setting)
        from report import report_model
        report_model(out)
    write(out / 'progress.json', {'status': 'COMPLETE', 'completed_requests': len(all_results),
        'expected_requests': 8000, 'pid': os.getpid()})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import torch
        model = sys.argv[sys.argv.index('--model') + 1]
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        write(RUN / model / 'failure.json', {'status': 'OOM' if oom else 'ERROR',
            'error': str(exc), 'traceback': traceback.format_exc(), 'pid': os.getpid()})
        write(RUN / model / 'progress.json', {'status': 'OOM' if oom else 'ERROR', 'error': str(exc)})
        traceback.print_exc()
        sys.exit(75 if oom else 1)
