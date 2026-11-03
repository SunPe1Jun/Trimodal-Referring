#!/usr/bin/env python3
"""Offline smoke diagnostics: coordinate sensitivity, provenance, and same-ID baselines."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from run_responses import ROOT, read_csv, write_csv, write_json, load_module


def prf(tp, fp, fn):
    return {'precision': tp / (tp + fp) if tp + fp else 0,
            'recall': tp / (tp + fn) if tp + fn else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0}


def subset_metrics(setting, rows):
    total = lambda col: sum(int(r[col]) for r in rows)
    if setting == 'experiment1':
        return dict(n=len(rows), **prf(*[total(k) for k in ['tp','fp','fn']]),
                    exact=sum(r['exact'] == 'True' for r in rows) / len(rows))
    if setting == 'experiment2':
        result = {'n':len(rows)}
        for kind, label in [('time','temporal_f1'), ('point','point100_f1'), ('joint','joint100_f1')]:
            cols = [f'{kind}_{k}' if kind == 'time' else f'{kind}_{k}_100' for k in ['tp','fp','fn']]
            result[label] = prf(*[total(k) for k in cols])['f1']
        return result
    return dict(n=len(rows), **prf(*[total(k) for k in ['tp','fp','fn']]),
                exact=sum(r['exact'] == 'True' for r in rows) / len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='closed_models/outputs/taomu_gpt61sol_smoke10')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    summary = json.loads((output / 'smoke_summary.json').read_text())
    tasks = json.loads((output / 'prepared_tasks.json').read_text())
    raw_records = [json.loads((output / t['setting'] / 'raw' / t['scene'] / f"row_{t['row_index']}.json").read_text()) for t in tasks]
    samples = []
    for r in raw_records:
        samples.append({'setting':r['setting'], 'key':r['key'], 'parse_ok':r['parse_ok'],
                        'images':len(r['images']), 'input_tokens':(r.get('usage') or {}).get('input_tokens'),
                        'output_tokens':(r.get('usage') or {}).get('output_tokens'),
                        'reasoning_tokens':(r.get('usage') or {}).get('output_tokens_details', {}).get('reasoning_tokens'),
                        'attempts_in_final_run':len(r.get('attempts', [])),
                        'earlier_failed_runs':len(r.get('request_history', [])),
                        'elapsed_seconds_final_attempts':sum(a.get('elapsed_seconds', 0) for a in r.get('attempts', []))})
    write_csv(output / 'sample_usage.csv', samples)
    subset_records = json.loads((output / 'baseline_subset_records.json').read_text())
    baseline = {setting:{model:subset_metrics(setting,rows) for model,rows in models.items()} for setting,models in subset_records.items()}

    runner2 = load_module('analysis_runner2', 'exam2/run_qwen3vl_2d_point_grounding.py')
    raw2 = [r for r in raw_records if r['setting'] == 'experiment2']
    alt_rows, transform_rows = [], []
    for r in raw2:
        normalized = runner2.extract_json(r['model_raw_output']) or {'referents':[]}
        declared = runner2.map_prediction_coordinates(normalized, r['panel_meta'], 'source_or_canvas_map')
        row = next(x for x in read_csv(output / 'experiment2/predictions.csv') if x['scene'] == r['scene'] and int(x['row_index']) == r['row_index'])
        row['parsed_json'] = json.dumps(declared, ensure_ascii=False)
        alt_rows.append(row)
        for original, frozen, corrected in zip(normalized['referents'], r['parsed_json']['referents'], declared['referents']):
            transform_rows.append({'key':r['key'], 'mention':original['mention'], 'panel_id':original['panel_id'],
                                   'declared_space':original['coordinate_space'], 'raw_x':original['x_norm'], 'raw_y':original['y_norm'],
                                   'frozen_x':frozen['x_norm'], 'frozen_y':frozen['y_norm'],
                                   'declared_x':corrected['x_norm'], 'declared_y':corrected['y_norm'],
                                   'frozen_transform':frozen['coordinate_transform']})
    alt_dir = output / 'experiment2/declared_space_sensitivity'
    write_csv(alt_dir / 'predictions.csv', alt_rows)
    write_csv(alt_dir / 'coordinate_transforms.csv', transform_rows)
    proc = subprocess.run([sys.executable,str(ROOT / 'exam2/evaluate_2d_point_grounding.py'),
                           '--manifest',str(output / 'experiment2/manifest/manifest_all.csv'),
                           '--pred_csv',str(alt_dir / 'predictions.csv'),'--output_dir',str(alt_dir / 'eval'),
                           '--coordinate_mode','panel'], capture_output=True, text=True)
    (alt_dir / 'evaluation.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError('Coordinate diagnostic evaluator failed')
    alt_summary = json.loads((alt_dir / 'eval/2d_eval_summary.json').read_text())

    archived = {(r['scene'],r['row_index'],r['referent_name'],r['panel_id']):r for r in read_csv(ROOT / 'paper_experiment_evidence/manifests/experiment2_projected_gt_manifest.csv')}
    projection_differences = []
    checked = 0
    for r in read_csv(output / 'experiment2/manifest/manifest_all.csv'):
        key = (r['scene'],r['row_index'],r['referent_name'],r['panel_id'])
        old = archived.get(key)
        if old is None:
            projection_differences.append({'key':list(key),'reason':'missing_archived_projection'})
            continue
        checked += 1
        if r['projection_valid'] != old['projection_valid']:
            projection_differences.append({'key':list(key),'reason':'validity_changed'})
        elif r['projection_valid'] == 'True':
            delta = max(abs(float(r['gt_x'])-float(old['gt_x'])),abs(float(r['gt_y'])-float(old['gt_y'])))
            if delta > 0.02:
                projection_differences.append({'key':list(key),'reason':'projected_pixel_changed','max_xy_delta':delta})

    cue = load_module('analysis_cue', 'exam3_point_grounding/run_cue_baselines.py')
    gaze_rows = []
    gaze_comparisons = []
    for t in tasks:
        if t['setting'] != 'experiment3':
            continue
        row = {'scene':t['scene'],'row_index':str(t['row_index']),'event_id':t['event_id'],
               'instruction':t['instruction'],'evidence_json':json.dumps(t['evidence_json'])}
        gaze = cue.dedup(cue.telemetry_points(row,'gaze_hit'),0.5,3)
        gaze_rows.append(cue.make_row(row, Path(''), gaze,'gaze_copy'))
        r = next(r for r in raw_records if r['setting'] == 'experiment3' and r['key'] == t['key'])
        predicted = [x['point'] for x in r['parsed_json'].get('points_3d',[])]
        gaze_comparisons.append({'key':t['key'], 'same_as_direct_gaze':predicted == [list(x) for x in gaze],
                                 'predicted_points':predicted,'direct_gaze_points':[list(x) for x in gaze]})
    gaze_dir = output / 'experiment3/direct_gaze'
    write_csv(gaze_dir / 'predictions.csv', gaze_rows)
    proc = subprocess.run([sys.executable,str(ROOT / 'exam3_point_grounding/evaluate_point_grounding.py'),
                           '--repo_root',str(ROOT),'--gt_manifest',str(output / 'experiment3/gt_manifest_eval.csv'),
                           '--pred_csv',str(gaze_dir / 'predictions.csv'),'--output_dir',str(gaze_dir / 'eval')],capture_output=True,text=True)
    (gaze_dir / 'evaluation.log').write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError('Direct-gaze diagnostic evaluator failed')
    gaze_summary = json.loads((gaze_dir / 'eval/evaluation_summary.json').read_text())['overall']
    analysis = {'samples':samples,'same_id_archived_baseline_metrics':baseline,
                'setting2_frozen':summary['metrics']['experiment2'], 'setting2_declared_space_sensitivity':alt_summary,
                'setting2_coordinate_transforms':transform_rows,
                'setting2_projection_audit':{'checked_rows':checked,'differences':projection_differences,
                    'limitation':'Archived compact GT does not retain evidence_acceptable; temporal equivalence is unverified.'},
                'setting3_direct_gaze':gaze_summary, 'setting3_gaze_comparisons':gaze_comparisons,
                'usage_by_setting':{s:{k:sum((r.get('usage') or {}).get(k,0) for r in raw_records if r['setting']==s) for k in ['input_tokens','output_tokens','total_tokens']} for s in summary['setting_counts']}}
    write_json(output / 'smoke_analysis.json',analysis)
    e1=summary['metrics']['experiment1'];e2=summary['metrics']['experiment2'];e3=summary['metrics']['experiment3']['overall']
    report = f'''**gpt-6.1-sol Responses smoke：结果与诊断**

本轮完成 {summary['requests']} 个真实样本请求：Setting 1 × 4、Setting 2 × 3、Setting 3 × 3。最终有效 JSON {summary['valid_outputs']}/10；三个原 evaluator 均执行成功。调用地址为 `https://www.taomu.xyz/v1/responses`，请求与中转站返回的模型字段均为 `gpt-6.1-sol`。该信息来自中转站，无法独立认证上游模型身份。

本轮只运行 smoke，未启动 4,000 条全量。密钥保存在仓库外、权限 0600 的配置文件中；后续密钥替换不重跑已有成功样本。遇到过本机 DNS 解析失败，已仅补齐失败请求；全部尝试保存在各样本的 `.attempts.json`，跨运行失败记录保存在 `request_history`。

| 设置 | n | 冻结协议结果 |
| --- | ---: | --- |
| Anchor Selection | 4 | Recall {e1['micro_recall']:.4f}；Exact {e1['exact_match_accuracy_overall']:.4f}；Micro-F1 {e1['micro_f1']:.4f} |
| Projected-2D | 3 | T-F1 {e2['time_f1']:.4f}；Point@100 {e2['point_100_f1']:.4f}；Joint@100 {e2['joint_100_f1']:.4f} |
| 3D measured hypotheses | 3 | Anchor-F1 {e3['anchor_set_f1_micro']:.4f}；Margin-F1@1 {e3['margin_f1_at_1_0']:.4f}；Exact {e3['anchor_set_exact_rate']:.4f} |

这些样本是预先固定、覆盖简单/长指令和不同场景的工程验证集，不能据此估计全量性能或排名。旧模型同 ID 的样本记录与重算集合指标位于 `baseline_subset_records.json` 和 `smoke_analysis.json`。所有证据从原视频/遥测重建，尚未证明与缺失的原运行输入逐字节相同。

**Setting 2：坐标冲突已在真实输出上复现**

原 prompt 要求在 `full_panel` / `gaze_crop` 内输出局部坐标，而 `paired_canvas_map` 会按整张 paired canvas 的左右位置再映射。当前输出中，模型声明的 full-panel 局部坐标被再次转换。保留相同模型输出、仅按声明坐标空间转换的离线诊断得到：T-F1 {alt_summary['time_f1']:.4f}、Point@100 {alt_summary['point_100_f1']:.4f}、Joint@100 {alt_summary['joint_100_f1']:.4f}、Joint@200 {alt_summary['joint_200_f1']:.4f}。没有额外调用 API；主结果仍保留原协议。

这是坐标约定的敏感性分析，不能将修正后新模型分数直接与旧协议的旧模型分数比较。正式全量前应先确定统一规则，若更改则给可恢复原始输出的旧模型统一重评。逐 referent 转换记录在 `experiment2/declared_space_sensitivity/coordinate_transforms.csv`。

Setting 2 共检查 {checked} 行重建投影，与归档存在 {len(projection_differences)} 处可观测差异。具体见 `smoke_analysis.json`；归档没有保存 `evidence_acceptable`，所以时间评测条件的一致性仍不能确认。

**Setting 3：必须保留 direct gaze**

在相同重建输入上重新运行 direct gaze，无需 API，得到 Anchor-F1 {gaze_summary['anchor_set_f1_micro']:.4f}、Margin-F1@1 {gaze_summary['margin_f1_at_1_0']:.4f}。{sum(x['same_as_direct_gaze'] for x in gaze_comparisons)}/3 条模型输出与 direct-gaze 点列表完全相同。该观察与最终 prompt 默认复制 gaze hypotheses 的设计一致，不能当作自由 3D 重建能力的结论。

**真实用量与后续执行**

中转站返回总 input tokens {summary['usage']['input_tokens']:,}、output tokens {summary['usage']['output_tokens']:,}、total tokens {summary['usage']['total_tokens']:,}；输出用量包含其报告的 reasoning tokens。每条输入图像数量、源/发送图像哈希、分辨率、请求 hash、完整 usage 均保存。实际金额未获取，不能把 token 用量当作已核实的账单金额。

尽快结束的下一步顺序是：先统一 Setting 2 坐标协议并确认输入恢复条件，再复用当前断点缓存/并发 runner 启动全量；在全量推理进行时用已有结果并行完成自动评测与切片分析，不再按截止日倒排。当前默认三并发、每条最多 4096 输出 tokens、image detail=high、最大边 1920；reasoning effort 未显式指定，使用中转站默认值。若调整这些条件，应使用新的输出目录冻结配置。

离线重算：`bash closed_models/run_smoke.sh --evaluate-only`；重做诊断：`/home/kunyu/miniconda3/envs/trimodal/bin/python closed_models/analyze_smoke.py`。这两条命令不发起模型请求。
'''
    (output / 'SMOKE_REPORT_zh.md').write_text(report)
    print(json.dumps({'valid':summary['valid_outputs'], 'setting1_f1':e1['micro_f1'],
                      'setting2_frozen_joint100':e2['joint_100_f1'], 'setting2_declared_joint100':alt_summary['joint_100_f1'],
                      'setting3_f1':e3['anchor_set_f1_micro'],'direct_gaze_f1':gaze_summary['anchor_set_f1_micro'],
                      'projection_differences':len(projection_differences), 'report':str(output / 'SMOKE_REPORT_zh.md')},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
