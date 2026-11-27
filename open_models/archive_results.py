"""Export the completed Exp.1/2 expansion from local evaluators, without inference."""
import csv
import hashlib
import json
from pathlib import Path

from common import ROOT, RUN, batch_name, batches

ARCHIVE = ROOT / 'open_models/results/2026-11-27'
METRICS = ['time', *[f'{kind}_{radius}' for kind in ('point', 'joint') for radius in (50, 100, 150, 200)]]
POLICIES = {
    'paired_canvas_map': ('experiment2', Path('eval')),
    'source_or_canvas_map': ('experiment2_declared_space', Path('declared_space_sensitivity/eval')),
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def event_id(row):
    return f"{row['scene']}::{row['row_index']}"


def anchor_key(row):
    # Rooms in scene4 can share event_id; the raw prediction path carries the room identifier.
    scene = Path(row['prediction_json']).parent.name
    return f"{scene}::{row['row_index']}"


def index(records, key, expected):
    indexed = {key(r): r for r in records}
    assert len(indexed) == len(records) and indexed.keys() == set(expected)
    return indexed


def fields(metric):
    kind, _, radius = metric.partition('_')
    return ['_'.join(filter(None, (kind, count, radius))) for count in ('tp', 'fp', 'fn')]


def ids(value):
    return {item.strip() for item in value.split(',') if item.strip()}


def export_model(model, jobs, gt):
    folder = RUN / model
    result = read_json(folder / 'results_so_far.json')
    assert read_json(folder / 'progress.json')['status'] == 'COMPLETE'
    assert result['completed_batches'] == len(jobs) and result['completed_requests'] == 8000
    events = []
    for stage, offset, keys in jobs:
        batch = folder / 'batches' / batch_name(stage, offset)
        anchor = index(rows(batch / 'experiment1/eval/detail.csv'), anchor_key, keys)
        point = {p: index(rows(batch / 'experiment2' / subdir / '2d_eval_detail.csv'), event_id, keys)
                 for p, (_, subdir) in POLICIES.items()}
        for key in keys:
            a = anchor[key]
            targets = set(json.loads(gt[key]['gt_anchor_ids']))
            assert targets == ids(a['gt_referents_mapped'])
            predicted = ids(a['predicted_referents_mapped'])
            record = dict(key=key, gt_anchor_ids=a['gt_referents_mapped'], predicted_anchor_ids=a['predicted_referents_mapped'],
                          valid1=int(a['response_status'] == 'ok'), exact=int(a['exact_match'] == 'True'),
                          tp=len(targets & predicted), fp=len(predicted - targets), fn=len(targets - predicted))
            main = point['paired_canvas_map'][key]
            record['valid2'] = int(main['parse_ok'] == 'True')
            for label, indexed in point.items():
                row = indexed[key]
                assert all(main[f] == row[f] for f in ['parse_ok', *fields('time')])
                for metric in METRICS:
                    for f in fields(metric):
                        record[f'{label}_{f}'] = int(row[f])
            events.append(record)

    assert len(events) == len({r['key'] for r in events}) == len(gt) == 4000
    valid = {f: sum(r[f] for r in events) for f in ('valid1', 'valid2')}
    assert sum(valid.values()) == result['valid_outputs']
    a = result['metrics']['experiment1']
    for f in ('tp', 'fp', 'fn'):
        assert sum(r[f] for r in events) == a[f]
    assert sum(r['exact'] for r in events) / len(events) == a['exact']
    assert 2*a['tp'] / (2*a['tp'] + a['fp'] + a['fn']) == a['micro_f1']
    for label, (setting, _) in POLICIES.items():
        for metric in METRICS:
            tp, fp, fn = (sum(r[f'{label}_{f}'] for r in events) for f in fields(metric))
            score = 2*tp / (2*tp + fp + fn)
            dest = result['metrics'][setting]
            assert dest.get(metric + '_f1', score) == score
            dest[metric + '_f1'] = score
    result['valid_per_experiment'] = {'experiment1': valid['valid1'], 'experiment2': valid['valid2']}
    result['generation_protocol'] = read_json(folder / 'protocol.json')
    result['parser_protocol'] = read_json(folder / 'parser_protocol.json')
    path = ARCHIVE / (model + '.csv')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=events[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(events)
    result['event_table_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    ref = rows(ROOT / 'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
    gt = {r['unified_id']: r for r in ref}
    jobs = batches()
    models = read_json(ROOT / 'open_models/model_registry.json')
    results = [export_model(spec['repo'].split('/')[-1], jobs, gt) for spec in models]
    side = read_json(RUN / 'LLaVA-OneVision-2-8B-Instruct/experiment2_results_so_far.json')
    full = next(r for r in results if r['model'] == 'LLaVA-OneVision-2-8B-Instruct')
    for setting, metrics in side['metrics'].items():
        assert all(full['metrics'][setting][k] == v for k, v in metrics.items())
    historical = {}
    for model, run in [('Qwen3-VL-30B-A3B', 'qwen30'), ('Qwen3-VL-8B', 'qwen8'), ('InternVL3-38B', 'internvl')]:
        audit = read_json(RUN / 'historical_coordinate_audit' / run / '2d_eval_summary.json')
        historical[model] = {k: audit[k] for k in ['events', *[m + '_f1' for m in METRICS]]}
    summary = {'models': results, 'historical_source_or_canvas_map': historical, 'input_check': read_json(RUN / 'input_preflight.json')}
    (ARCHIVE / 'results.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
