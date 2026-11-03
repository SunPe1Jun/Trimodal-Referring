#!/usr/bin/env python3
"""Compare 3D prompt/observation pilots with explicit gaze-copy and cue coverage diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

from run_responses import ROOT, read_csv, write_csv, write_json, load_module

CONDITIONS=['exp3_legacy_pilot20','exp3_evidence_pilot20','exp3_evidence_no_instruction_pilot20',
            'exp3_evidence_no_visual_pilot20','exp3_coverage8_evidence_pilot20',
            'exp3_coverage8_no_instruction_pilot20','exp3_coverage8_no_visual_pilot20']


def analyze_inputs(directory, cue, common):
    tasks=json.loads((directory/'prepared_tasks.json').read_text())
    gt={(r['scene'],int(r['row_index'])):r for r in read_csv(directory/'experiment3/gt_manifest_eval.csv')}
    gaze_rows=[];oracle=[];anchors={};alignment=[]
    for task in tasks:
        scene=task['scene'];index=task['row_index']
        row={'scene':scene,'row_index':index,'event_id':task['event_id'],'instruction':task['instruction'],
             'evidence_json':json.dumps(task['evidence_json'])}
        gaze=cue.dedup(cue.telemetry_points(row,'gaze_hit'),.5,len(task['evidence_json']))
        gaze_rows.append(cue.make_row(row,Path(''),gaze,'gaze_copy'))
        anchors.setdefault(scene,common.load_anchor_table(ROOT,scene))
        ids=set(gt[(scene,index)]['gt_anchor_ids'].split(','))
        nearest={}
        for kind in ['gaze_hit','hand_hit']:
            nearest[kind]={common.nearest_anchor(p,anchors[scene])[0].anchor_id
                           for p in cue.telemetry_points(row,kind)}
        oracle.append({'key':task['key'],'gt_n':len(ids),
                       'gaze_supported_gt':len(ids & nearest['gaze_hit']),
                       'gaze_or_hand_supported_gt':len(ids & (nearest['gaze_hit']|nearest['hand_hit'])),
                       'gaze_candidate_n':len(nearest['gaze_hit']),
                       'gaze_or_hand_candidate_n':len(nearest['gaze_hit']|nearest['hand_hit'])})
        for obs in task['evidence_json']:
            if 'alignment' in obs:alignment.append(obs['alignment'])
    out=directory/'experiment3/direct_gaze'
    write_csv(out/'predictions.csv',gaze_rows)
    proc=subprocess.run([sys.executable,str(ROOT/'exam3_point_grounding/evaluate_point_grounding.py'),
                         '--repo_root',str(ROOT),'--pred_csv',str(out/'predictions.csv'),
                         '--gt_manifest',str(directory/'experiment3/gt_manifest_eval.csv'),'--output_dir',str(out/'eval')],
                        capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr)
    summary=json.loads((out/'eval/evaluation_summary.json').read_text())['overall']
    n=sum(r['gt_n'] for r in oracle)
    result={'direct_gaze':summary,'cue_support_diagnostic':{
        'gt_referents':n,'gaze_recall_ceiling':sum(r['gaze_supported_gt'] for r in oracle)/n,
        'gaze_or_hand_recall_ceiling':sum(r['gaze_or_hand_supported_gt'] for r in oracle)/n,
        'uses_hidden_gt_for_posthoc_analysis_only':True,
        'direct_gaze_policy':'all distinct available gaze points; dedup_distance=0.5; frame-budget-matched max_points',
        'meaning':'Oracle nearest-anchor support of available cue locations, not an inference result or region accuracy'},
        'per_event_cue_support':oracle}
    if alignment:
        result['alignment']={'frames':len(alignment),
            'verified_pts':sum(x['status']=='VERIFIED_CONTAINER_PTS_SEQUENCE' for x in alignment),
            'clock_sources':dict(Counter(x['clock_status'] for x in alignment)),
            'pose_available':sum(x['camera_pose']['status']=='POSE_AVAILABLE' for x in alignment),
            'max_abs_pts_seek_delta_seconds':max(abs(x['pts_minus_requested_seconds']) for x in alignment)}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='closed_models/outputs/exp3_study20')
    args=parser.parse_args()
    output=(ROOT/args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    cue=load_module('study_cues','exam3_point_grounding/run_cue_baselines.py')
    common=load_module('study_common','exam3_point_grounding/point_grounding_common.py')
    results={};rows=[]
    for condition in CONDITIONS:
        directory=ROOT/'closed_models/outputs'/condition
        path=directory/'smoke_summary.json'
        if not path.exists():
            results[condition]={'status':'RUNNING_OR_NOT_COMPLETE'};continue
        run=json.loads(path.read_text());metric=run['metrics']['experiment3']['overall']
        detail=read_csv(directory/'experiment3/eval/evaluation_detail.csv')
        raw=[]
        for r in detail:
            p=directory/'experiment3/raw'/r['scene']/f"row_{r['row_index']}.json"
            raw.append(json.loads(p.read_text()))
        empty=sum(len(r['parsed_json'].get('points_3d',[]))==0 for r in raw)
        sources=Counter()
        for r in raw:
            try:points=json.loads(r['model_raw_output']).get('points_3d',[])
            except (json.JSONDecodeError,AttributeError):points=[]
            for p in points:
                for source in p.get('measurement_sources',[]):sources[source]+=1
        result={'status':'COMPLETE','metrics':metric,'valid':run['valid_outputs'],'empty_predictions':empty,
                'model_claimed_measurement_sources':dict(sources),'usage':run['usage'],
                'qualifier':'source attributions are model reports, not independently established causal evidence'}
        if condition in ['exp3_legacy_pilot20','exp3_coverage8_evidence_pilot20']:
            result['input_diagnostics']=analyze_inputs(directory,cue,common)
        results[condition]=result
        rows.append({'condition':condition,'n':metric['total_samples'],'valid':run['valid_outputs'],
                     'empty_predictions':empty,'precision':metric['anchor_set_precision_micro'],
                     'recall':metric['anchor_set_recall_micro'],'anchor_f1':metric['anchor_set_f1_micro'],
                     'exact':metric['anchor_set_exact_rate'],'margin_f1_1':metric['margin_f1_at_1_0'],
                     'scene_normalized_error':metric['mean_scene_normalized_error']})
    # Explain the known single-target Exact anomaly without altering frozen metrics.
    strict=[]
    for path in (ROOT/'paper_experiment_evidence/predictions/exp3').glob('*.csv'):
        records=read_csv(path)
        for name,multi in [('single',False),('multi',True)]:
            selected=[r for r in records if (len(set(json.loads(r['gt_anchor_ids'])))>1)==multi]
            set_equal=sum(set(json.loads(r['gt_anchor_ids']))==set(json.loads(r['nearest_anchor_predictions'])) for r in selected)
            exact=sum(r['exact']=='True' for r in selected)
            strict.append({'model':path.stem,'partition':name,'n':len(selected),'strict_exact_count':exact,
                           'deduplicated_set_equal_count':set_equal,'duplicate_only_exact_failures':set_equal-exact})
    write_csv(output/'pilot_metrics.csv',rows)
    write_csv(output/'legacy_exact_diagnostic.csv',strict)
    write_json(output/'analysis.json',{'conditions':results,'legacy_exact_diagnostic':strict})
    gaze_base=results['exp3_legacy_pilot20']['input_diagnostics']['direct_gaze']
    gaze_dense=results.get('exp3_coverage8_evidence_pilot20',{}).get('input_diagnostics',{}).get('direct_gaze',{})
    table='\n'.join(f"| {r['condition']} | {r['n']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['anchor_f1']:.4f} | {r['exact']:.4f} | {r['empty_predictions']} |" for r in rows)
    report=f'''**3D setting：gaze shortcut 的实测诊断与替代方案**

只讨论闭源基线与 3D 方法，不包含采集/标注回复。本轮固定 20 条来自 Multiagent 已有清单，选择不根据模型成绩；使用 gpt-6.1-sol、Responses 接口、原 world GT 与原 evaluator。所有数值均为小样本 pilot。

| 条件 | n | Precision | Recall | Anchor-F1 | 旧定义 Exact | 空预测 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

同输入 direct gaze：原稀疏输入 Anchor-F1 {gaze_base['anchor_set_f1_micro']:.4f}；8 帧输入复制所有可用 distinct gaze（上限也为 8，而非仍限 3）Anchor-F1 {gaze_dense.get('anchor_set_f1_micro',0):.4f}、Recall {gaze_dense.get('anchor_set_recall_micro',0):.4f}、Margin-F1@1 {gaze_dense.get('margin_f1_at_1_0',0):.4f}。以上比较不会人为截断 gaze baseline 来制造优势。

**原 setting 为何收敛到 gaze**

最终 v9 prompt 显式要求默认复制全部 distinct gaze 点、最多三条，只有无 gaze 时才使用 hand。这是决策规则的捷径，不能把该协议下强模型仍接近 gaze-copy 当成模型普遍缺乏三模态推理能力的充分证据。必须保留这条原协议作为可复现 baseline，同时另列修订条件。

此外，原代码的 Anchor-F1 会将 nearest anchor 去重，但 Exact 要求去重后集合正确且没有重复 nearest anchor。Qwen-30B 的 930 条单目标中，严格 Exact 仅 2 条；另有 102 条是集合正确但受重复点惩罚。`legacy_exact_diagnostic.csv` 列出三个旧模型完整统计。解释这一区别，不能悄悄改旧表的 Exact。

**本轮已经实现的替代方案**

1. `evidence` prompt：由指令确定不同 referents，结合图像绑定身份，再比较 gaze 与 hand；两种测量不设默认优先级，同一 referent 只输出一个点。不提供 anchor ID、GT 坐标或 GT 目标数。此协议仍是利用测得端点的点监督诊断，不声称从单目图像重建度量 3D。
2. 同输入的 `no_instruction` / `no_visual`：图片/遥测选择冻结，仅去掉相应模型输入，保留空预测与失败。分数变化证明接口输出对输入的敏感性；如果通过拒答使召回下降，不能据此宣称学会了正确融合。
3. `coverage_pts`：参考 Multiagent 的事件覆盖原则，每个时间 bin 保留一张代表帧，共 8 帧；同时保存实际解码 PTS，用相同 PTS 查询插值 camera pose，记录原始 cue 采样时间、时差和超间隔回退。使用已有无 GT/无 gaze 的时钟拟合候选，通过检查则使用，否则回退原始 offset；不读取 referent GT 选择时间或调时钟。相机 pose 不可用时保留图像并明确置空相应遥测，不跨大采样空洞外推。

`coverage_pts` 同时改变了帧预算、覆盖与时钟/姿态因素，因此这一行是组合方案，不把变化归因于某一个因素。实际 PTS 不是渲染曝光时间，现有候选只通过背景旋转一致性验证，不代表绝对像素标定已完成。

**如何判断是否真正改进**

先看相同样本上的 Referent Recall、Anchor-F1、Exact、Margin-F1、空预测和有效覆盖，再看相对同输入 direct gaze 的差值。不要只依赖 Precision 或非空预测上的误差下降。若去掉图像/指令只导致返回空集，需要和正确 referent 数量、反事实换指令下目标集合是否合理变化、模型返回的证据 panel 是否真支持对应 referent 一起检查。

`analysis.json` 给出同输入 direct gaze 及“可用 gaze/hand 端点中有多少 GT anchors 获 nearest-anchor 支持”的推理后诊断。该 oracle 使用隐藏 GT，只定位证据瓶颈，绝不进入 prompt 或选帧。

**若此方案仍不超过 gaze，下一步的具体方案**

将 referent 识别与世界点选择拆开：第一阶段只给指令和对齐的多帧图像，让模型输出每个 referent 的 panel 和 2D point；第二阶段将所有可信 gaze/hand 世界端点用同一参考相机投影回该 panel，以图像射线一致性筛选可用端点，再按 referent 合并多次观测。不能无深度把任意 2D 点直接宣称为准确 3D，也不能用最近 GT anchor 反推输出。没有匹配的深度或端点时保留未定位状态并计入端到端召回。这个两阶段版本在本轮尚未实现或验证，不能把它写成已有性能提升。

首要结论：移除 prompt 的 gaze-copy 默认规则是消除人为 shortcut 的必要步骤，但不保证提高数值；完整 20 条结果如显示召回下降，应如实报告。最终选择全量协议前先完成配对 pilot 的比较，原协议和修订协议的数字分开记录。
'''
    (output/'EXPERIMENT3_SOLUTION_zh.md').write_text(report)
    print(json.dumps({'complete_conditions':len(rows),'metrics':rows,'report':str(output/'EXPERIMENT3_SOLUTION_zh.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
