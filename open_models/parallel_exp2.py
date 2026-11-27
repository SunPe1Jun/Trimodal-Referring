"""Run OneVision experiment 2 alongside the full queue, reusing its generation lock."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from common import MODELS, ROOT, RUN, batches, batch_name, digest, protocol, read, write

MODEL = 'LLaVA-OneVision-2-8B-Instruct'


def selected_tasks(tasks):
    return [t for t in tasks if t['setting'] == 'experiment2']


def main():
    # --family also selects the native Transformers runtime in worker.py.
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['onevision'], default='onevision')
    parser.parse_args()
    import torch
    from adapters import Adapter
    from worker import infer
    torch.set_num_threads(3)
    torch.cuda.set_per_process_memory_fraction(.28)
    os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']
    out = RUN / MODEL
    guard = (out / 'parallel_exp2.lock').open('w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = read(out / 'protocol.json')
    assert lock['checkpoint'] == read(MODELS / MODEL / 'DOWNLOAD_COMPLETE.json')
    for name in ['adapters.py', 'common.py']:
        assert lock['source_sha256'][name] == digest(ROOT / 'open_models' / name)
    reservation = RUN / 'parallel_exp2_reservation.json'
    write(reservation, {'model': MODEL, 'pid': os.getpid(), 'memory_fraction': .28,
        'scope': 'experiment2 only', 'status': 'RUNNING'})
    status = out / 'experiment2_progress.json'
    write(status, {'status': 'LOADING_MODEL', 'pid': os.getpid(), 'expected_requests': 4000})
    try:
        adapter = Adapter('onevision', str(MODELS / MODEL))
        m = protocol.modules()
        results = []
        ids = set()
        for stage, offset, keys in batches():
            name = batch_name(stage, offset)
            source = RUN / 'inputs' / name
            target = out / 'batches' / name
            target.mkdir(parents=True, exist_ok=True)
            for rel in ['prepared_tasks.json', 'selection.json', 'experiment2/manifest/manifest_all.csv']:
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copy2(source / rel, dest)
                assert digest(dest) == digest(source / rel)
            tasks = selected_tasks(read(target / 'prepared_tasks.json'))
            assert len(tasks) == len(keys)
            records = []
            for task in tasks:
                if (RUN / 'STOP').exists():
                    write(status, {'status': 'STOPPED', 'completed_requests': len(results), 'pid': os.getpid()})
                    return
                assert task['key'] not in ids
                ids.add(task['key'])
                write(status, {'status': 'RUNNING', 'pid': os.getpid(), 'batch': name,
                    'active_key': task['key'], 'completed_requests': len(results) + len(records),
                    'expected_requests': 4000,
                    'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
                record = infer(task, target, adapter, lock)
                if record['response_status'] == 'unparsed':
                    ok, parsed, extra = protocol.parse_task(task, record['model_raw_output'], m)
                    record.update(parse_ok=ok, parsed_json=parsed,
                        response_status='ok' if ok else 'invalid', **extra)
                    write(target / 'experiment2/raw' / task['scene'] / f"row_{task['row_index']}.json", record)
                records.append(record)
                print('[done]', MODEL, task['key'], 'parse_ok=', record['parse_ok'],
                    'peak_GiB=', round(record['peak_reserved_gib'], 2), flush=True)
                torch.cuda.empty_cache()
            _, gt = protocol.restore_frozen_gt(target)
            metrics = protocol.evaluate(records, gt, target, m)
            write(target / 'experiment2/parallel_summary.json', {
                'requests': len(records), 'valid_outputs': sum(r['parse_ok'] for r in records),
                'metrics': metrics, 'stage': stage})
            results.extend(records)
            aggregate = {}
            summaries = [read(p) for p in (out / 'batches').glob('*/experiment2/parallel_summary.json')]
            for policy in ['experiment2', 'experiment2_declared_space']:
                values = [s['metrics'][policy] for s in summaries]
                record = {}
                for kind, threshold in [('time', None), ('point', '100'), ('joint', '100'),
                                         ('point', '200'), ('joint', '200')]:
                    tp, fp, fn = [sum(v.get(kind + '_' + k + ('_' + threshold if threshold else ''), 0)
                                     for v in values) for k in ['tp', 'fp', 'fn']]
                    record[kind + ('_' + threshold if threshold else '') + '_f1'] = (
                        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
                aggregate[policy] = record
            write(out / 'experiment2_results_so_far.json', {'requests': len(results),
                'valid_outputs': sum(r['parse_ok'] for r in results), 'metrics': aggregate})
        write(status, {'status': 'COMPLETE', 'pid': os.getpid(), 'completed_requests': len(results),
            'expected_requests': 4000, 'unique_events': len(ids)})
    except Exception as exc:
        write(status, {'status': 'OOM' if isinstance(exc, torch.cuda.OutOfMemoryError) else 'ERROR',
            'pid': os.getpid(), 'error': str(exc), 'traceback': traceback.format_exc(),
            'note': 'Full queue remains running; saved raw responses reusable by full worker.'})
        raise
    finally:
        write(reservation, {'model': MODEL, 'pid': os.getpid(), 'status': 'RELEASED', 'memory_fraction': 0})


if __name__ == '__main__':
    main()
