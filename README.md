# VR-TriRef / Trimodal-Referring

VR-TriRef 多模态指代定位实验代码。当前论文与 rebuttal 使用以下三个 setting：

1. **Anchor Selection**：指令 + 事件时间窗内 RGB 帧 + gaze/hand/camera 线索 + 场景候选 anchor 表，输出 canonical anchor ID 集合。
2. **Projected-2D Localization**：指令 + 最多 3 个 paired canvas（full panel + gaze-centered crop），输出 panel ID 与归一化 2D 点。
3. **3D Localization**：指令 + 最多 3 张 target-free 证据帧 + 遥测，输出 `points_3d`；评测阶段才做 nearest-anchor / margin / SNE。

`exam3/` 的 camera-centered directional 实验是历史分支，不是论文 Table 2 的第三项。第三项以 `exam3_point_grounding/` 的 v9 measured point-hypothesis 协议为准。

## 当前进度

- [五个新增开源模型的实验一、二全量结果（2026-11-27）](open_models/results/2026-11-27/README.md) · [机器可读汇总](open_models/results/2026-11-27/results.json)

- [实验进度与恢复记录（2026-11-03）](docs/EXPERIMENT_STATUS_20261103.md)
- [gpt-6.1-sol 闭源基线全量记录（2026-11-03）](closed_models/FULL_BASELINE_REPORT_20261103.md)
- [模型与闭源 runner 使用说明](closed_models/README.md)
- [Rebuttal 方案与协议审计（历史）](docs/REBUTTAL_OPENAI_PLAN_20261026.md)
- [三个任务的完整流程说明](full_experiment_flow/README.md)

闭源基线三项均已完成 4,000/4,000 推理与评测；详细指标、异常模型路由和可比性限制见上面的专档。大型结果保存在服务器 `closed_models/outputs/`，不在 Git 中。

## 仓库结构

- `scripts/`：数据准备、关键帧构建和本地 Qwen3-VL grounding。
- `exam2/`：projected-2D manifest、推理和评测。
- `exam3_point_grounding/`：当前第三项 3D measured point-hypothesis 协议、prompt 和评测。
- `exam3/`：历史 camera-centered directional 分支，不作为论文第三项入口。
- `internvl/`、`qwen8/`：旧开源基线运行脚本。
- `ablation/`：模态/输入消融与审计。
- `paper_experiment_evidence/`：论文最终证据、预测、GT 和 evaluator 快照。
- `full_experiment_flow/`：三项任务的流程说明和结果表。
- `closed_models/`：OpenAI-compatible 闭源 runner、prompt、测试和基线报告。
- `analysis_outputs/`：rebuttal 审计、区域/目标分组和指标复核。

## 环境

历史服务器路径：

- repo：`/media/kunyu/16TB/spj/Trimodal-Referring`
- 数据：`/media/kunyu/16TB/spj/V3dMD`
- Qwen3-VL-30B：`/media/kunyu/16TB/spj/Qwen3-VL-30B-A3B-Instruct`
- GroundingDINO：`/media/kunyu/16TB/spj/grounding-dino-base`

旧 shell wrapper 仍含 `/workspace/usr3/...` 默认路径；运行时应显式传入当前路径。`closed_models/run_responses.py` 以仓库根目录动态解析，不依赖固定服务器布局。

```bash
source /home/kunyu/miniconda3/etc/profile.d/conda.sh
conda activate trimodal
```

## 关键入口

```bash
# Experiment 1：closed-set 3D anchor selection
bash scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh

# Experiment 2：projected-2D point diagnostic
bash exam2/run_qwen3vl_30b_2d_full.sh

# Experiment 3：candidate-free measured 3D point hypotheses（论文第三项）
bash exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh

# 历史分支：camera-centered directional（不是论文第三项）
bash exam3/run_qwen3vl_30b_3d_directional.sh

# 闭源 runner：只重解析/评测已保存响应，不调用 API
bash closed_models/run_smoke.sh --evaluate-only
```

`LIMIT=20` 可用于 smoke；完整 wrapper 默认跑全量。闭源基线的正式命令和恢复方式见 [基线专档](closed_models/FULL_BASELINE_REPORT_20261103.md)。

## 大型资产

Git 不包含 V3dMD 数据、视频、模型权重、抽取帧、闭源 raw 响应、生成图片和完整输出。闭源基线约 27 GB，保存在服务器的 `closed_models/outputs/`；论文证据的小型汇总在 `paper_experiment_evidence/`。不要用文件名或旧 README 推断实验已经完成；以 `progress.json`、`run_history.json`、`eval/` 和本仓库进度文档为准。
