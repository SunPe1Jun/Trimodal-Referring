#!/usr/bin/env python3
"""Refresh a local overview while the three full API baselines run; no API calls."""
import argparse
import json
from pathlib import Path
import time
from run_responses import ROOT,write_json
from run_full_baseline import aggregate

JOBS={
    'experiment1':'gpt61sol_anchor_full_v1',
    'experiment2':'gpt61sol_projected2d_full_v1',
    'experiment3':'gpt61sol_world3d_legacy_full_v1',
}


def update():
    rows={}
    for setting,name in JOBS.items():
        root=ROOT/'closed_models/outputs'/name
        progress_path=root/'progress.json'
        progress=json.loads(progress_path.read_text()) if progress_path.exists() else {'status':'NOT_STARTED'}
        metrics=aggregate(root,[setting],4000)
        paths=[p for p in root.glob(setting+'/batch_*/'+setting+'/raw/*/row_*.json') if not p.name.endswith('.attempts.json')]
        raw=[json.loads(p.read_text()) for p in paths]
        rows[setting]={'status':progress['status'],'saved_response_records':len(raw),
                       'saved_valid_outputs':sum(r.get('parse_ok',False) for r in raw),
                       'metrics_of_completed_batches':metrics}
    out=ROOT/'closed_models/outputs/full_baseline_overview'
    write_json(out/'overview.json',{'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'settings':rows})
    lines=[]
    for setting,row in rows.items():
        metric=row['metrics_of_completed_batches']['per_setting'].get(setting,{})
        lines.append(f"| {setting} | {row['status']} | {row['saved_response_records']}/4000 | {row['saved_valid_outputs']} | {metric.get('n',0)} |")
    text='''**gpt-6.1-sol：三个 setting 全量运行状态**

每项目标 4,000 条，模型接口为 `/v1/responses`。此页面只汇总本地结果，不请求 API。下表“已评测”按完成的批次统计；当前批次的响应可能已保存但尚未计入该列。

| Setting | 状态 | 已保存响应 | 有效输出 | 已评测 |
| --- | --- | ---: | ---: | ---: |
'''+ '\n'.join(lines)+'''

各批次的原始响应、usage、图片和请求 hash、projection check、evaluator 输出位于对应输出目录。精确的累计指标见同目录 `overview.json`。运行中的场景前缀样本不是随机代表性子集，不能将其阶段性指标用作全量结论。

Setting 2 保留原 `paired_canvas_map` 主结果，另保存同一输出按声明空间转换的敏感性结果；每批在请求 API 前核对归档投影，发现变化则停止，避免带着不一致 GT 继续付费。归档缺失的 `evidence_acceptable` 仍需在论文中说明。

Setting 3 的全量基线保留原 v9 协议以便与旧表比较。证据驱动/8 帧/PTS 对齐是独立 pilot，不将其改善误称为纯模型升级效果。原 v9 的 gaze-copy 规则也不能被当作强模型缺乏融合能力的充分证据。

断点续跑会复用已有成功响应，模型格式错误保留分母，传输错误按固定规则补齐。systemd 用户服务 `trimodal-gpt61sol-anchor`、`trimodal-gpt61sol-projected2d`、`trimodal-gpt61sol-world3d` 负责运行/接续；本地总览服务 `trimodal-gpt61sol-overview` 每 30 秒更新此页。达到失败阈值会停止，没有无限自动重试。
'''
    out.mkdir(parents=True,exist_ok=True)
    (out/'STATUS_zh.md').write_text(text)
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    while True:
        rows=update()
        if not args.watch or all(r['status'] in ['COMPLETE','STOPPED_BATCH_ERROR','STOPPED_FAILURE_THRESHOLD'] for r in rows.values()):
            break
        time.sleep(30)


if __name__=='__main__':main()
