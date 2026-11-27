"""Rescore the three archived open baselines under declared coordinate spaces."""
import csv
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'exam2'))
from run_qwen3vl_2d_point_grounding import map_prediction_coordinates

out = root / 'open_models/outputs/expansion_20261116/historical_coordinate_audit'
runs = {
    'qwen30': 'exam2/outputs_qwen3vl30b_2d_point_hybrid_v10',
    'qwen8': 'qwen8/outputs/exam2_qwen3vl8b_baseline_2d_point_hybrid_v10',
    'internvl': 'internvl/outputs/exam2_internvl3_38b_baseline',
}
for name, path in runs.items():
    src = root / path
    dest = out / name
    dest.mkdir(exist_ok=True)
    archives = [src / 'predictions/json', *sorted((root / 'analysis_outputs/experiment_completion/exp2/raw').glob(name + '_attempt*'))]
    with next((src/'predictions').glob('*.csv')).open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len({(r['scene'], r['row_index']) for r in rows}) == len(rows) == 4000
    for row in rows:
        saved = json.loads(row['parsed_json'] or '{}')
        if not saved.get('referents'):
            continue
        filename = f"{row['scene']}_row_{row['row_index']}.json"
        records = (json.loads((archive/filename).read_text()) for archive in archives if (archive/filename).exists())
        record = next(r for r in records if r['model_raw_output'].strip() == row['model_raw_output'].strip())
        # Use the final merged prediction's stored pre-transform coordinates.
        original = {'referents': []}
        for item in saved['referents']:
            ref = dict(item)
            for field in ('x_norm', 'y_norm', 'coordinate_space'):
                ref[field] = ref.pop('raw_' + field)
            ref.pop('coordinate_transform')
            original['referents'].append(ref)
        meta = record['panel_meta']
        reconstructed = map_prediction_coordinates(original, meta, 'paired_canvas_map')
        assert reconstructed == saved, (name, filename, reconstructed, saved)
        row['parsed_json'] = json.dumps(map_prediction_coordinates(original, meta, 'source_or_canvas_map'))
    predictions = dest/'declared_predictions.csv'
    with predictions.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([sys.executable, str(root/'exam2/evaluate_2d_point_grounding.py'), '--coordinate_mode','panel','--manifest',str(src/'manifest/manifest_all.csv'),'--pred_csv',str(predictions),'--output_dir',str(dest)],check=True,stdout=subprocess.DEVNULL)
    alt = json.loads((dest/'2d_eval_summary.json').read_text())
    old = json.loads((src/'eval/2d_eval_summary.json').read_text())
    assert all(alt[k] == old[k] for k in ('time_tp','time_fp','time_fn','events','predictions','gt_referents'))
    print(name, [(k,old[k],alt[k]) for k in ('point_200_f1','joint_200_f1')],flush=True)
