#!/usr/bin/env python3
"""Incremental full evaluation: finish/evaluate each batch before preparing the next."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_responses import ROOT, read_csv, write_json


def aggregate(root, settings, total_expected):
    counts = {'completed_batches':0,'completed_requests':0,'valid_outputs':0,'failed_requests':0,
              'input_tokens':0,'output_tokens':0,'total_tokens':0}
    per_setting = {}
    for setting in settings:
        summaries = []
        for path in sorted((root / setting).glob('batch_*/smoke_summary.json')):
            summary = json.loads(path.read_text())
            summaries.append(summary)
            counts['completed_batches'] += 1
            counts['completed_requests'] += summary['requests']
            counts['valid_outputs'] += summary['valid_outputs']
            counts['failed_requests'] += summary['requests'] - summary['valid_outputs']
            for key in ['input_tokens','output_tokens','total_tokens']:
                counts[key] += summary['usage'][key]
        per_setting[setting] = {'n':sum(s['requests'] for s in summaries),
                                'valid':sum(s['valid_outputs'] for s in summaries)}
        if setting == 'experiment1' and summaries:
            metrics = [s['metrics'][setting] for s in summaries]
            tp = sum(m['set_true_positive_count'] for m in metrics)
            fp = sum(m['set_false_positive_count'] for m in metrics)
            fn = sum(m['set_false_negative_count'] for m in metrics)
            n = per_setting[setting]['n']
            per_setting[setting].update(micro_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                                        recall=tp/(tp+fn) if tp+fn else 0,
                                        exact=sum(m['exact_match_count_overall'] for m in metrics)/n,
                                        tp=tp,fp=fp,fn=fn)
        if setting == 'experiment2' and summaries:
            metric_keys = ['time','point_100','point_200','joint_100','joint_200']
            for protocol in ['experiment2','experiment2_declared_space']:
                available = [s['metrics'][protocol] for s in summaries if protocol in s['metrics']]
                if not available:
                    continue
                aggregate_metrics = {}
                for prefix in metric_keys:
                    tp, fp, fn = [sum(m.get(prefix+'_'+x,0) for m in available) for x in ['tp','fp','fn']]
                    if prefix == 'time':
                        tp, fp, fn = [sum(m.get('time_'+x,0) for m in available) for x in ['tp','fp','fn']]
                    elif prefix.startswith(('point_','joint_')):
                        kind,threshold = prefix.split('_')
                        tp,fp,fn=[sum(m.get(kind+'_'+x+'_'+threshold,0) for m in available) for x in ['tp','fp','fn']]
                    aggregate_metrics[prefix+'_f1'] = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
                per_setting[setting][protocol] = aggregate_metrics
        if setting == 'experiment3' and summaries:
            detail_rows=[]
            for path in sorted((root / setting).glob('batch_*/smoke_summary.json')):
                detail_path=path.parent/setting/'eval/evaluation_detail.csv'
                if detail_path.exists():
                    detail_rows.extend(read_csv(detail_path))
            tp,fp,fn=[sum(int(r['set_'+k]) for r in detail_rows) for k in ['tp','fp','fn']]
            mtp,mfp,mfn=[sum(int(r['margin_'+k+'_at_1_0']) for r in detail_rows) for k in ['tp','fp','fn']]
            if detail_rows:
                per_setting[setting].update(anchor_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
                    recall=tp/(tp+fn) if tp+fn else 0,
                    exact=sum(r['anchor_set_exact']=='True' for r in detail_rows)/len(detail_rows),
                    margin_f1_1=2*mtp/(2*mtp+mfp+mfn) if 2*mtp+mfp+mfn else 0)
    return {'expected_requests':total_expected, **counts,'per_setting':per_setting}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--settings', nargs='+', choices=['experiment1','experiment2','experiment3'], default=['experiment1'])
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--concurrency',type=int,default=6)
    parser.add_argument('--config',default=str(Path.home()/'.config/trimodal-referring/taomu.json'))
    parser.add_argument('--exp3-prompt-mode',choices=['legacy','evidence'],default='legacy')
    parser.add_argument('--exp3-evidence-mode',choices=['legacy','coverage_pts'],default='legacy')
    parser.add_argument('--stop-after-batches',type=int)
    parser.add_argument('--wallet-guard',action='store_true')
    parser.add_argument('--resume-after-pid',type=int,help='Wait for an existing orchestrator, then resume only if incomplete')
    args = parser.parse_args()
    root=(ROOT/args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'BUDGET_HOLD').exists():
        print('Budget hold is active; no API batches submitted',flush=True)
        return
    if args.resume_after_pid:
        existing=Path('/proc')/str(args.resume_after_pid)/'stat'
        original=existing.read_text().split()[21] if existing.exists() else None
        while existing.exists() and original is not None:
            fields=existing.read_text().split()
            if fields[21]!=original or fields[2]=='Z':
                break
            time.sleep(10)
        progress_path=root/'progress.json'
        if progress_path.exists() and json.loads(progress_path.read_text()).get('status')=='COMPLETE':
            print('Existing full evaluation already complete; no requests to repeat',flush=True)
            return
    rows=read_csv(ROOT/'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
    ids=[r['unified_id'] for r in rows]
    expected=len(ids)*len(args.settings)
    write_json(root/'selection.json',{s:ids for s in args.settings})
    write_json(root/'protocol.json',{'model':'gpt-6.1-sol','endpoint':'/v1/responses','n_per_setting':len(ids),
                                    'settings':args.settings,'exp3_prompt_mode':args.exp3_prompt_mode,
                                    'exp3_evidence_mode':args.exp3_evidence_mode,
                                    'batch_size':args.batch_size,'concurrency':args.concurrency,
                                    'selection':'all frozen 4000 canonical scene::row_index IDs; no performance filtering'})
    completed_this_run=0
    for setting in args.settings:
        for start in range(0,len(ids),args.batch_size):
            if (root/'BUDGET_HOLD').exists():
                write_json(root/'progress.json',{'status':'BUDGET_HOLD',**aggregate(root,args.settings,expected)})
                print('Budget hold reached; current results preserved',flush=True)
                return
            batch=root/setting/f'batch_{start:04d}'
            batch.mkdir(parents=True,exist_ok=True)
            if (batch/'smoke_summary.json').exists():
                print('[reuse completed batch]',setting,start,flush=True)
                continue
            if args.wallet_guard:
                from relay_budget import check,release
                config=json.loads(Path(args.config).read_text())
                while True:
                    try:
                        status,balance,debit=check(config,min(args.batch_size,len(ids)-start))
                    except Exception as exc:
                        write_json(root/'progress.json',{'status':'WAITING_BILLING_VERIFICATION','error':str(exc),**aggregate(root,args.settings,expected)})
                        print('Wallet verification unavailable; waiting without inference',flush=True)
                        time.sleep(30)
                        continue
                    if status=='RUNNING':break
                    write_json(root/'progress.json',{'status':status,'balance_usd':balance,'spend_usd':debit,**aggregate(root,args.settings,expected)})
                    if status=='CAP_REACHED':
                        print('Approved budget cap reached; preserving all results',flush=True)
                        return
                    print('Waiting for wallet top-up; no model requests',balance,flush=True)
                    time.sleep(30)
            selection=batch/'requested_selection.json'
            write_json(selection,{setting:ids[start:start+args.batch_size]})
            command=[sys.executable,str(ROOT/'closed_models/run_responses.py'),'--config',args.config,
                     '--settings',setting,'--selection',str(selection),'--output',str(batch),
                     '--concurrency',str(args.concurrency),'--exp3-prompt-mode',args.exp3_prompt_mode,
                     '--exp3-evidence-mode',args.exp3_evidence_mode,'--retry-errors']
            # Existing batches were prepared with the current frozen input
            # builder. Archive validation is retained as a diagnostic artifact,
            # but it must not abort a resumed paid run before model requests.
            current=aggregate(root,args.settings,expected)
            write_json(root/'progress.json',{'status':'RUNNING','active_setting':setting,'active_batch':start,
                                            'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**current})
            print(f'[batch] {setting} {start}/{len(ids)} completed={current["completed_requests"]}',flush=True)
            with (batch/'run.log').open('a') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if args.wallet_guard:
                release()
            if result.returncode not in (0,2):
                write_json(root/'progress.json',{'status':'STOPPED_BATCH_ERROR','active_batch':start,**aggregate(root,args.settings,expected)})
                raise SystemExit(result.returncode)
            current=aggregate(root,args.settings,expected)
            # Stop repeated failures early while preserving completed predictions.
            if current['failed_requests'] > max(5,current['completed_requests']*.1):
                write_json(root/'progress.json',{'status':'STOPPED_FAILURE_THRESHOLD',**current})
                raise SystemExit(2)
            completed_this_run += 1
            write_json(root/'progress.json',{'status':'RUNNING',**current})
            print(json.dumps(current,ensure_ascii=False),flush=True)
            if args.stop_after_batches and completed_this_run>=args.stop_after_batches:
                write_json(root/'progress.json',{'status':'STOPPED_REQUESTED_BATCH_LIMIT',**current})
                return
    write_json(root/'progress.json',{'status':'COMPLETE',**aggregate(root,args.settings,expected)})


if __name__=='__main__':
    main()
