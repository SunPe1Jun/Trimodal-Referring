"""Run two BF16 workers at most; retry memory-limited workers once with the GPU alone."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from common import MODELS, ROOT, RUN, read, write

FAMILIES = {'InternVL3_5-8B': 'internvl', 'MiniCPM-V-4_5': 'minicpm',
            'Qwen3.5-9B': 'qwen35', 'Molmo2-8B': 'molmo',
            'LLaVA-OneVision-2-8B-Instruct': 'onevision'}
BUDGETS = {'InternVL3_5-8B': .30, 'MiniCPM-V-4_5': .65,
           'Qwen3.5-9B': .65, 'Molmo2-8B': .45, 'LLaVA-OneVision-2-8B-Instruct': .92}


def interpreter(model):
    if model == 'Molmo2-8B':
        return str(MODELS / 'molmo_env/bin/python')
    return sys.executable


def runtime_ready(model):
    return model != 'Molmo2-8B' or (MODELS / 'molmo_env/READY.json').exists()


def external_reservations():
    path = RUN / 'parallel_exp2_reservation.json'
    if not path.exists():
        return []
    record = read(path)
    stat = Path('/proc', str(record.get('pid', 0)), 'stat')
    if record.get('status') == 'RUNNING' and stat.exists() and stat.read_text().split()[2] != 'Z':
        return [record]
    return []


class AdoptedProcess:
    """Reconnect to a detached worker after a supervisor-only restart."""
    def __init__(self, pid, model):
        self.pid, self.model = pid, model

    def poll(self):
        stat = Path('/proc', str(self.pid), 'stat')
        if stat.exists() and stat.read_text().split()[2] != 'Z':
            return None
        state = read(RUN / self.model / 'progress.json').get('status')
        return 0 if state == 'COMPLETE' else 75 if state == 'OOM' else 1

    def terminate(self):
        import signal
        os.kill(self.pid, signal.SIGTERM)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    guard = (RUN / 'supervisor.lock').open('w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = list(FAMILIES)
    active, finished, failed, exclusive = {}, {}, {}, []
    logs = []
    for model in list(pending):
        state_path = RUN / model / 'progress.json'
        if not state_path.exists():
            continue
        state = read(state_path)
        if state.get('status') == 'COMPLETE':
            finished[model] = 'COMPLETE'
            pending.remove(model)
            continue
        pid = state.get('pid')
        if pid and state.get('status') in ('RUNNING', 'LOADING_MODEL', 'WAITING_INPUTS'):
            process = AdoptedProcess(pid, model)
            if process.poll() is None:
                active[model] = (process, BUDGETS[model] > .9)
                pending.remove(model)
    write(RUN / 'plan.json', {'models': FAMILIES, 'settings': ['experiment1', 'experiment2'],
        'events_per_setting': 4000, 'total_generations': len(FAMILIES) * 8000,
        'smoke_events_per_setting': 80, 'max_concurrent_workers': 2,
        'memory_fractions': BUDGETS, 'max_total_reserved_fraction': .95,
        'oom_retry': 'one retry on GPU alone, same model-visible protocol',
        'experiment3': 'disabled'})
    while pending or active or exclusive:
        if (RUN / 'STOP').exists():
            for process, _ in active.values():
                process.terminate()
            write(RUN / 'supervisor_status.json', {'status': 'STOPPED', 'pid': os.getpid()})
            return
        for model, (process, solo) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            del active[model]
            if code == 0:
                finished[model] = 'COMPLETE'
            elif code == 75 and not solo:
                exclusive.append(model)
            else:
                failed[model] = {'returncode': code, 'see': str(RUN / model / 'worker.log')}
        external = external_reservations()
        external_fraction = sum(r['memory_fraction'] for r in external)
        candidates = []
        if exclusive:
            if not active and not external:
                candidates = [(exclusive.pop(0), True)]
        elif not any(solo for _, solo in active.values()):
            for model in list(pending):
                if len(active) + len(candidates) + len(external) >= 2:
                    break
                if any(r['model'] == model for r in external):
                    continue
                reserved = external_fraction + sum(BUDGETS[k] for k in active) + sum(BUDGETS[k] for k, _ in candidates)
                if BUDGETS[model] + reserved > .951:
                    continue
                if (MODELS / model / 'DOWNLOAD_COMPLETE.json').exists() and runtime_ready(model):
                    pending.remove(model)
                    candidates.append((model, BUDGETS[model] > .9))
        for model, solo in candidates:
            output = RUN / model
            output.mkdir(parents=True, exist_ok=True)
            if (output / 'progress.json').exists() and read(output / 'progress.json').get('status') == 'COMPLETE':
                finished[model] = 'COMPLETE'
                continue
            log = (output / 'worker.log').open('a')
            logs.append(log)
            env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            process = subprocess.Popen([interpreter(model), '-u', str(ROOT / 'open_models/worker.py'),
                '--model', model, '--family', FAMILIES[model],
                '--memory-fraction', str(.92 if solo else BUDGETS[model])],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            active[model] = (process, solo)
            print('[start]', model, process.pid, 'exclusive=', solo, flush=True)
        # Detect terminal downloader failures rather than waiting forever.
        downloads = read(MODELS / 'models.json')
        for item in downloads:
            model = Path(item['path']).name
            if model in pending and not Path('/proc', str(item['download_pid'])).exists():
                if not (MODELS / model / 'DOWNLOAD_COMPLETE.json').exists():
                    pending.remove(model)
                    failed[model] = {'error': 'download failed', 'see': str(MODELS / (model + '.download.log'))}
        write(RUN / 'supervisor_status.json', {'status': 'RUNNING', 'pid': os.getpid(),
            'active': {k: {'pid': v[0].pid, 'exclusive': v[1]} for k, v in active.items()},
            'parallel_experiment2': external,
            'waiting_models': pending,
            'waiting_downloads': [m for m in pending if not (MODELS / m / 'DOWNLOAD_COMPLETE.json').exists()],
            'waiting_gpu': [m for m in pending if (MODELS / m / 'DOWNLOAD_COMPLETE.json').exists()],
            'waiting_runtime': [m for m in pending if not runtime_ready(m)],
            'waiting_exclusive': exclusive, 'finished': finished, 'failed': failed,
            'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        time.sleep(10)
    write(RUN / 'supervisor_status.json', {'status': 'COMPLETE' if not failed else 'FINISHED_WITH_ERRORS',
        'finished': finished, 'failed': failed})


if __name__ == '__main__':
    main()
