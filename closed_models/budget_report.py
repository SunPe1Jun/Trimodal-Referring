#!/usr/bin/env python3
"""Audit locally recorded responses/usage; no API requests and no invented prices."""
from collections import Counter
import json
from pathlib import Path
import random

from run_responses import ROOT,read_csv,write_json,write_csv


def main():
    root=ROOT/'closed_models/outputs'
    groups={};seen=set()
    for directory in root.iterdir():
        if not directory.is_dir():continue
        responses=[]
        for path in directory.glob('**/raw/*/row_*.json'):
            if path.name.endswith('.attempts.json'):continue
            try:r=json.loads(path.read_text())
            except (OSError,json.JSONDecodeError):continue
            response=r.get('response') or {}
            usage=response.get('usage') or r.get('usage')
            if not usage:continue
            response_id=response.get('id') or str(path)
            if response_id in seen:continue
            seen.add(response_id)
            responses.append(r)
        if responses:
            groups[directory.name]={'responses':len(responses),'valid_outputs':sum(bool(r.get('parse_ok')) for r in responses),
                **{key:sum((r.get('usage') or r['response']['usage']).get(key,0) for r in responses)
                   for key in ['input_tokens','output_tokens','total_tokens']}}
    totals=Counter()
    for v in groups.values():totals.update(v)
    report={'by_run':groups,'total':dict(totals),'monetary_cost':'unknown; relay billing prices and actual ledger not obtained',
            'limitation':'Usage from saved responses only; does not prove actual billing and may omit remote completed responses lost in transit.'}
    out=root/'budget_review';write_json(out/'usage_ledger.json',report)
    write_csv(out/'usage_by_run.csv',[{'run':k,**v} for k,v in groups.items()])

    # Nested random scene quotas: 200/500 shared IDs for three settings, independent of model scores.
    rows=read_csv(ROOT/'paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv')
    grouped={}
    for r in rows:grouped.setdefault(r['scene'],[]).append(r['unified_id'])
    rng=random.Random(20261026)
    for ids in grouped.values():rng.shuffle(ids)
    for n in [200,500]:
        ids=[]
        for scene,pool in grouped.items():
            quota=(40 if len(pool)==800 else 10) if n==200 else (100 if len(pool)==800 else 25)
            ids.extend(pool[:quota])
        write_json(out/f'selection_{n}_shared.json',{s:ids for s in ['experiment1','experiment2','experiment3']})
        selected=set(ids)
        baseline_counts={}
        for exp in ['exp1','exp2','exp3']:
            baseline_counts[exp]={}
            for path in (ROOT/'paper_experiment_evidence/predictions'/exp).glob('*.csv'):
                subset=[r for r in read_csv(path) if r['unified_id'] in selected]
                write_csv(out/f'old_baselines_{n}'/exp/path.name,subset)
                baseline_counts[exp][path.stem]=len(subset)
        write_json(out/f'old_baseline_counts_{n}.json',baseline_counts)
    text=f'''**闭源基线：节约预算后的执行范围**

本轮预算调整已停止提交新的全量批次；只让已启动的当前批次保存和评测。不会追加闭源消融、第二模型或新的 prompt 搜索。已有缓存和原始响应保留。

截至此文件生成，已保存的服务响应共 {totals['responses']} 条，input tokens {totals['input_tokens']:,}、output tokens {totals['output_tokens']:,}。这是去重后的响应 usage，不是实际账单；具体金额未知，需用户提供中转站额度或计费信息。各 run 明细见 `usage_ledger.json`。

建议优先级：

1. 正式补充只跑一个闭源模型，每个 setting 一个固定配置。现有开源消融负责诊断，闭源不再重复完整消融。
2. 以相同、预先固定的场景分层子集先完成三项结果，按预算选择每 setting 200 条（共 600）或 500 条（共 1,500）。已生成共享 ID 清单；500 条包含 200 条，抽样不看模型成绩。
3. 对旧模型在完全相同 ID 上离线重算，避免用闭源子集与旧模型全量直接比较。明确报告 n 与抽样方式。已有全量任务完成的场景前缀不代表完整随机样本，可保留为额外结果或复用严格匹配的请求。
4. 保留 gaze、hand、图像与完整指令，不做未经验证的 aggressive 降分辨率/删帧。先靠减少配置数和评测样本数节省；任何低成本输入变体都须独立标注，不能悄悄替换公平比较条件。
5. 3D 现有 20 条方案已经完成，不再追加探索性调用。8 帧证据驱动版本 F1=0.5263，同输入全 gaze F1=0.4962；它增加了帧预算并采用 PTS/姿态处理，必须与原 3D 协议分开报告。其全量/子集主配置尚需依据最终报告目标确定，不能为省钱把旧 prompt 与新输入的数字混算。
6. token 与单价确认后先设总费用上限。成功请求严格复用缓存；异常输出不以涨分为目的重采样。坐标重映射、同 ID 重评、slice 和统计均离线进行，不增加模型请求。

当前没有恢复付费队列。预算上限待用户提供。
'''
    (out/'BUDGET_PLAN_zh.md').write_text(text)
    print(json.dumps(report['total'],indent=2))


if __name__=='__main__':main()
