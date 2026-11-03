#!/usr/bin/env python3
"""One frozen configuration per setting, shared stratified IDs and billing checks.

Reuses paid answers only when the exact serialized request hash matches.
No ablations or new prompt optimization. Balance checks use the relay's reported
balance and conservatively the largest of cost/account_cost/actual_cost deltas.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import time

from run_responses import (ROOT, read_csv, write_csv, write_json, restore_frozen_gt,
    modules, prepare, build_request, run_task, evaluate, _http_sessions)
import requests


def usage_snapshot(config):
    key=config.get('api_key') or os.environ[config['api_key_env']]
    endpoint=config['base_url'].rstrip('/')+'/usage'
    last=None
    for attempt in range(2):
        try:
            response=requests.get(endpoint,headers={'Authorization':'Bearer '+key},timeout=20,allow_redirects=False)
            response.raise_for_status()
            payload=response.json()
            if payload.get('unit')!='USD' or not isinstance(payload.get('balance'),(int,float)):
                raise ValueError('Relay usage response has no verified USD balance')
            return payload
        except requests.RequestException as exc:
            last=exc
    raise RuntimeError('Cannot verify billing; no further paid requests: '+str(last).replace(key,'[REDACTED]'))


def costs(payload):
    stats=payload.get('model_stats',[])
    return {field:sum(float(x.get(field,0)) for x in stats) for field in ['cost','actual_cost','account_cost']}


def remaining_budget(state,snapshot):
    initial=state['initial_usage']
    delta={k:costs(snapshot)[k]-costs(initial)[k] for k in costs(initial)}
    delta['balance_debit']=initial['balance']-snapshot['balance']
    charged=max(0,*delta.values())
    # Ignore apparent balance increases from deposits; retain the approved run cap.
    available=min(state['spend_cap_usd']-charged,snapshot['balance']-state['reserve_usd'])
    return available,charged,delta


def cache_index(config):
    index={}
    for directory in (ROOT/'closed_models/outputs').glob('**/prepared_tasks.json'):
        if 'budgeted_shared200_v1' in str(directory):continue
        try:tasks=json.loads(directory.read_text())
        except (OSError,json.JSONDecodeError):continue
        for task in tasks:
            if task.get('evidence_mode','legacy')!='legacy':continue
            raw_path=directory.parent/task['setting']/'raw'/task['scene']/f"row_{task['row_index']}.json"
            if not raw_path.exists():continue
            try:record=json.loads(raw_path.read_text())
            except (OSError,json.JSONDecodeError):continue
            if record.get('exp3_prompt_mode','legacy')!='legacy' or record.get('ablation_variant','full')!='full':continue
            if record.get('error_message') or not record.get('response'):continue
            if record.get('model')!=config['model']:continue
            # Do not redo invalid answers; they remain failures in the denominator.
            index[(task['setting'],task['key'])]=(task,raw_path,directory.parent)
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(Path.home()/'.config/trimodal-referring/taomu.json'))
    parser.add_argument('--selection',default='closed_models/outputs/budget_review/selection_200_shared.json')
    parser.add_argument('--output',default='closed_models/outputs/budgeted_shared200_v1')
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--concurrency',type=int,default=3)
    parser.add_argument('--approved-ceiling',type=float,default=300.)
    parser.add_argument('--max-request-reserve',type=float,default=1.,
                        help='Conservative billing-space reserve per in-flight request; not a claimed unit price')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1 or args.concurrency<1:raise SystemExit('Positive batch size/concurrency required')
    output=(ROOT/args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text());config['concurrency']=args.concurrency
    config['retry_errors']=False
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    selection=json.loads((ROOT/args.selection).read_text())
    target=sum(len(v) for v in selection.values())
    if not all(len(v)==len(set(v)) for v in selection.values()):raise SystemExit('Duplicate IDs')
    snapshot=usage_snapshot(config)
    state_path=output/'budget_state.json'
    if state_path.exists():
        state=json.loads(state_path.read_text())
    else:
        usable=min(args.approved_ceiling,snapshot['balance'])
        state={'initial_usage':snapshot,'approved_ceiling_usd':args.approved_ceiling,
               'reserve_usd':usable*.2,'spend_cap_usd':usable*.8,
               'cost_multiplier_user_reported':.1,'billing_policy':'max of reported cost, actual_cost, account_cost deltas and balance debit',
               'selection_count':target,'selection_path':str((ROOT/args.selection).resolve()),
               'protocol':'one model; legacy main protocol each setting; no new ablations'}
        write_json(state_path,state)
    cache=cache_index(config)
    write_json(output/'selection.json',selection)
    m=modules()
    # Shuffle once within setting to avoid spending only on the first scene if budget exhausts.
    ordering={}
    for setting,ids in selection.items():
        ids=list(ids);random.Random(20261026).shuffle(ids);ordering[setting]=ids
    batch_counts=max(math.ceil(len(v)/args.batch_size) for v in ordering.values())
    plan={'target_requests':target,'available_cached_keys':sum((s,k) in cache for s,ids in selection.items() for k in ids),
          'verified_balance_usd':snapshot['balance'],'run_cap_usd':state['spend_cap_usd'],
          'reserve_usd':state['reserve_usd'],'dry_run':args.dry_run}
    write_json(output/'plan.json',plan);print(json.dumps(plan),flush=True)
    if args.dry_run:return
    completed=[];usage_queries=0
    for batch_index in range(batch_counts):
        for setting,ids in ordering.items():
            chosen=ids[batch_index*args.batch_size:(batch_index+1)*args.batch_size]
            if not chosen:continue
            if (output/'BUDGET_HOLD').exists():
                write_json(output/'progress.json',{'status':'USER_HOLD','completed_batches':completed});return
            batch=output/setting/f'batch_{batch_index:03d}'
            if (batch/'summary.json').exists():
                completed.append(str(batch.relative_to(output)));continue
            batch.mkdir(parents=True,exist_ok=True)
            missing=[];reused=[]
            manifest2=[]
            for key in chosen:
                entry=cache.get((setting,key))
                if entry is None:
                    missing.append(key);continue
                task,path,source=entry
                _,_,digest=build_request(task,config)
                record=json.loads(path.read_text())
                if digest!=record.get('request_sha256') or record.get('endpoint')!=config['base_url'].rstrip('/')+'/responses':
                    missing.append(key);continue
                reused.append(task)
                destination=batch/setting/'raw'/task['scene']/path.name
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
                if setting=='experiment2':
                    manifest2.extend(r for r in read_csv(source/setting/'manifest/manifest_all.csv')
                                     if r['scene']==task['scene'] and int(r['row_index'])==task['row_index'])
            _,gt=restore_frozen_gt(batch)
            tasks=[]
            if missing:
                # Prepare locally before reserving a paid wave; do not consume tokens on input checks.
                tasks,_=prepare({setting:missing},batch,m)
                if setting=='experiment2':
                    manifest2.extend(read_csv(batch/setting/'manifest/manifest_all.csv'))
            tasks.extend(reused)
            if setting=='experiment2':
                m.builder2.write_manifest(batch/setting/'manifest/manifest_all.csv',manifest2)
                from run_responses import validate_projection_archive
                validate_projection_archive(batch)
            if setting=='experiment3':
                frozen={(r['scene'],r['row_index']):r for r in read_csv(ROOT/'paper_experiment_evidence/manifests/experiment3_evaluable_manifest.csv')}
                write_csv(batch/setting/'gt_manifest_eval.csv',[frozen[(t['scene'],str(t['row_index']))] for t in tasks])
            write_json(batch/'prepared_tasks.json',tasks)
            write_json(batch/'selection.json',{setting:chosen})
            # Each small wave reserves a full USD per paid call against the conservative meter.
            # Poll after it, and never count a cache hit as a new request.
            results=[]
            for start in range(0,len(tasks),args.concurrency):
                wave=tasks[start:start+args.concurrency]
                paid=sum(not (batch/setting/'raw'/t['scene']/f"row_{t['row_index']}.json").exists() for t in wave)
                if paid:
                    snapshot=usage_snapshot(config);usage_queries+=1
                    available,spent,delta=remaining_budget(state,snapshot)
                    write_json(output/'billing_latest.json',{'usage':snapshot,'conservative_spend_usd':spent,
                               'available_usd':available,'meter_deltas':delta})
                    if available<paid*args.max_request_reserve:
                        write_json(output/'progress.json',{'status':'STOPPED_BUDGET_RESERVE',
                                   'available_usd':available,'completed_batches':completed,'pending_batch':str(batch)})
                        return
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                    results.extend(pool.map(lambda t:run_task(t,batch,config,m),wave))
            metrics=evaluate(results,gt,batch,m)
            summary={'setting':setting,'n':len(results),'valid_outputs':sum(r['parse_ok'] for r in results),
                     'cached':len(reused),'new_paid_targets':len(missing),'metrics':metrics}
            write_json(batch/'summary.json',summary)
            completed.append(str(batch.relative_to(output)))
            write_json(output/'progress.json',{'status':'RUNNING','completed_batches':completed,'target_requests':target,
                                               'usage_queries':usage_queries})
            print('[subset batch]',setting,batch_index,'n',len(results),'cached',len(reused),flush=True)
    final=usage_snapshot(config)
    available,spent,delta=remaining_budget(state,final)
    write_json(output/'progress.json',{'status':'COMPLETE','completed_batches':completed,'target_requests':target,
                                      'conservative_spend_usd':spent,'ending_balance_usd':final['balance']})


if __name__=='__main__':main()
