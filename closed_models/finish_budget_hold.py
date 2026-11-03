#!/usr/bin/env python3
"""Let already-authorized batch workers persist results, then retire held dispatchers."""
import json
import os
from pathlib import Path
import signal
import time

from run_responses import ROOT,write_json
from run_full_baseline import aggregate

JOBS={'experiment1':'gpt61sol_anchor_full_v1','experiment2':'gpt61sol_projected2d_full_v1',
      'experiment3':'gpt61sol_world3d_legacy_full_v1'}
PARENTS=[396819,422120,422925,438873]

def live_workers():
    workers=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if any(x.endswith('/run_responses.py') for x in args) and any(name in ' '.join(args) for name in JOBS.values()):
                workers.append(int(p.name))
        except OSError:pass
    return workers

while live_workers():
    time.sleep(5)
for pid in PARENTS:
    try:
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().decode(errors='replace')
        if 'run_full_baseline.py' in cmd:
            os.kill(pid,signal.SIGTERM)
            os.kill(pid,signal.SIGCONT)
    except (OSError,ProcessLookupError):pass
for setting,name in JOBS.items():
    root=ROOT/'closed_models/outputs'/name
    write_json(root/'progress.json',{'status':'BUDGET_HOLD',**aggregate(root,[setting],4000)})
print('Current batches finished; no further requests scheduled',flush=True)
