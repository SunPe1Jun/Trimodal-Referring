"""Back up strict records, reparse saved answers, and reevaluate without generation."""
import argparse
import shutil
from pathlib import Path

from common import RUN, digest, protocol, read, write
from compatibility import VERSION, update_record
from report import report_model


def migrate(model):
    out = RUN / model
    backup = out / 'strict_before_compatibility_v1'
    migration = out / 'compatibility_v1'
    if (migration / 'migration_complete.json').exists():
        return read(migration / 'migration_complete.json')
    m = protocol.modules()
    _, gt = protocol.restore_frozen_gt(migration)
    before_details, after_details = [], []
    counters = {'records': 0, 'strict_valid': 0, 'compatible_valid': 0,
                'recovered': 0, 'still_invalid': 0, 'raw_and_generation_identity_verified': 0}
    for name in ['results_so_far.json', 'parser_protocol.json']:
        source = out / name
        if source.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
    for batch in sorted((out / 'batches').glob('*')):
        paths = sorted((batch / 'experiment1/raw').glob('*/row_*.json'))
        if not paths:
            continue
        for source in paths + [batch / 'summary.json'] + list((batch / 'experiment1/eval').glob('*')):
            if source.is_file():
                target = backup / source.relative_to(out)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
        records = [read(path) for path in paths]
        snapshot_out = backup / 'batches' / batch.name
        strict = protocol.evaluate(records, gt, snapshot_out, m)['experiment1']
        before_details.extend(protocol.read_csv(snapshot_out / 'experiment1/eval/detail.csv'))
        for path, record in zip(paths, records):
            original = read(backup / path.relative_to(out))
            original_fields = {k: record[k] for k in
                               ['model_raw_output', 'request_sha256', 'prompt_sha256', 'images', 'image_audit']}
            strict_ok = original.get('strict_parse_snapshot', {}).get('parse_ok', original['parse_ok'])
            record = update_record(record, record, m)
            assert all(record[k] == value for k, value in original_fields.items())
            write(path, record)
            counters['records'] += 1
            counters['strict_valid'] += int(strict_ok)
            counters['compatible_valid'] += int(record['parse_ok'])
            counters['recovered'] += int(not strict_ok and record['parse_ok'])
            counters['still_invalid'] += int(not record['parse_ok'])
            counters['raw_and_generation_identity_verified'] += 1
        compatible = protocol.evaluate(records, gt, batch, m)['experiment1']
        after_details.extend(protocol.read_csv(batch / 'experiment1/eval/detail.csv'))
        write(migration / 'batches' / (batch.name + '.json'), {
            'events': len(records), 'strict': strict, 'compatible': compatible})
        summary_path = batch / 'summary.json'
        if summary_path.exists():
            summary = read(summary_path)
            summary['metrics']['experiment1'] = compatible
            exp2 = list((batch / 'experiment2/raw').glob('*/row_*.json'))
            summary['valid_outputs'] = sum(record['parse_ok'] for record in records) + sum(read(p)['parse_ok'] for p in exp2)
            summary['parser_version'] = VERSION
            write(summary_path, summary)

    def aggregate(details):
        totals = [sum(len([x for x in r[column].split(',') if x.strip()]) for r in details)
                  for column in ['true_positive_referents', 'false_positive_referents', 'false_negative_referents']]
        tp, fp, fn = totals
        return {'events': len(details), 'tp': tp, 'fp': fp, 'fn': fn,
                'micro_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                'exact': sum(r['exact_match'] == 'True' for r in details) / len(details) if details else 0}
    result = {'model': model, 'parser_version': VERSION, **counters,
              'strict': aggregate(before_details), 'compatible': aggregate(after_details),
              'same_saved_subset': True, 'new_model_calls': 0, 'strict_backup': str(backup)}
    report_model(out)
    write(out / 'parser_protocol.json', {'version': VERSION, 'sha256': digest(Path(__file__).parent / 'compatibility.py'),
          'generation_protocol_unchanged': True, 'scope': 'experiment1'})
    write(migration / 'migration_complete.json', result)
    return result


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    args = parser.parse_args()
    for model in args.models:
        print(json.dumps(migrate(model), ensure_ascii=False, indent=2), flush=True)
