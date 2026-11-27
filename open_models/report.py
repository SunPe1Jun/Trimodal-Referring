"""Aggregate counts across disjoint batches, without averaging batch F1 values."""
import json
from pathlib import Path
from common import ROOT, RUN, SETTINGS, read, write


def report_model(out):
    summaries = [read(p) for p in sorted((out / 'batches').glob('*/summary.json'))]
    result = {'model': out.name, 'completed_batches': len(summaries),
              'completed_requests': sum(s['requests'] for s in summaries),
              'valid_outputs': sum(s['valid_outputs'] for s in summaries), 'metrics': {}}
    for setting in SETTINGS:
        metrics = [s['metrics'][setting] for s in summaries]
        if not metrics:
            continue
        n = sum(s['requests'] // 2 for s in summaries)
        if setting == 'experiment1':
            tp, fp, fn = [sum(m[k] for m in metrics) for k in
                ['set_true_positive_count', 'set_false_positive_count', 'set_false_negative_count']]
            result['metrics'][setting] = {'events': n, 'tp': tp, 'fp': fp, 'fn': fn,
                'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                'exact': sum(m['exact_match_count_overall'] for m in metrics) / n}
        else:
            for label in ('experiment2', 'experiment2_declared_space'):
                values = [s['metrics'][label] for s in summaries]
                record = {'events': n}
                for prefix in ('time', 'point_100', 'point_200', 'joint_100', 'joint_200'):
                    def field(suffix):
                        if prefix == 'time':
                            return 'time_' + suffix
                        kind, threshold = prefix.split('_')
                        return f'{kind}_{suffix}_{threshold}'
                    tp, fp, fn = [sum(m.get(field(k), 0) for m in values) for k in ('tp', 'fp', 'fn')]
                    record[prefix + '_f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0
                result['metrics'][label] = record
    write(out / 'results_so_far.json', result)
    return result


def main():
    records = []
    for spec in read(ROOT / 'open_models/model_registry.json'):
        model = RUN / spec['repo'].split('/')[-1]
        if model.is_dir() and (model / 'protocol.json').exists():
            record = report_model(model)
            record['progress'] = read(model / 'progress.json') if (model / 'progress.json').exists() else {}
            records.append(record)
    write(RUN / 'overview.json', records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
