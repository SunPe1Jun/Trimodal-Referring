# gpt-6.1-sol 闭源基线全量记录（2026-11-03）

本文记录 `closed_models/` 在 2026-10-26 至 2026-10-27 运行的三项 4,000 条全量结果。原始 Codex 会话已丢失；以下内容由输出目录中的 `progress.json`、`run_history.json`、`batch_*/run_config.json`、`batch_*/smoke_summary.json`、`batch_*/experiment*/eval/` 和 raw 响应复核得到。本次没有新增 API 调用。

## 1. 运行标识

| 项目 | 值 |
| --- | --- |
| 请求模型 | `gpt-6.1-sol` |
| 服务 | `https://www.taomu.xyz/v1`（中转站） |
| Endpoint | `https://www.taomu.xyz/v1/responses` |
| 服务返回模型 | 多数为 `gpt-6.1-sol`；见第 6 节异常 |
| 选择集 | 冻结的 4,000 个唯一 `scene::row_index`，无按性能筛选 |
| 三项运行目录 | `outputs/gpt61sol_anchor_full_v1`、`outputs/gpt61sol_projected2d_full_v1`、`outputs/gpt61sol_world3d_legacy_full_v1` |
| 总览 | `outputs/full_baseline_overview/overview.json`、`STATUS_zh.md` |
| 预算状态 | `outputs/full_budget_state.json` |

这不是 OpenAI 官方 API 结果。`gpt-6.1-sol` 是中转站列出的模型标识；模型身份、usage 和计费均为中转站返回的信息，无法独立认证上游模型。

## 2. 采样与预处理配置

所有 batch 使用同一组模型可见参数（`run_config.json`）：

| 参数 | 值 |
| --- | --- |
| `max_output_tokens` | 4096 |
| `image_detail` | `high` |
| `max_image_edge` | 1920 |
| `timeout_seconds` | 180 |
| `max_attempts` | 2 |
| `store` | false |
| `temperature` / `seed` | 未指定，使用服务默认 |
| `reasoning` | 未指定，使用服务默认 |
| 并发 | Exp1 4；Exp2 前 81 个 batch 为 4、后 44 个为 8；Exp3 4 |
| Batch size | 32 |
| 重试策略 | 408/429/5xx 和传输异常最多 2 次；成功响应和格式失败不自动重跑 |

图片处理：长边超过 1920 时转 RGB、LANCZOS 缩放到 1920、JPEG quality 95；否则直接发送原文件字节。每个请求保存图片路径、源/发送 SHA-256、原/发送尺寸和字节数。

## 3. 三个 Setting 的实际输入

### Experiment 1：Anchor Selection

- 指令来自 `data/<scene>_api_input.csv` 的 `instruction_text`。
- 视频帧由 `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py` 的 `load_video_frames` 按事件 `t_start`/`t_end` 采样，最多 16 帧，保存到 `experiment1/frames/<scene>/row_<index>/frame_*.jpg`。
- 实际发送帧数 6–16，均值 15.247；4,000 条中 2,380 条为 16 帧。
- Prompt 由 `run_qwen3vl_plus_api_single_event_3d.py` 的 `build_3d_object_prompt(..., prompt_strategy='mention_first', max_evidence_segments=0)` 构造；包含完整候选 anchor 表、gaze/hand/camera 结构线索。
- 输出为 canonical anchor ID 集合；解析用 `parse_3d_response_text`。
- 评测用 `scripts/eval/evaluate_local_3d_object_match.py`；GT 来自冻结的 `paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv` 的 `gt_anchor_ids`。

### Experiment 2：Projected-2D

- 复用 `exam2/build_2d_eval_manifest.py` 的 evidence panel 选择、时钟 offset 和 GT 投影；每个事件最多 3 个 panel，`paired_crop`，`gaze_crop_ratio=0.35`，`crop_output_size=768`。
- 图片为 full panel + gaze-centered crop 的 paired canvas；实际每事件 1–3 张，均值 2.959。
- Prompt 由 `exam2/run_qwen3vl_2d_point_grounding.py` 的 `build_prompt(rows, metadata, 'expected_count')` 构造；给定 referent 数量。
- 主结果后处理为 `paired_canvas_map`；同一批 raw 输出另做 `source_or_canvas_map` 的声明空间敏感性评测。
- 评测用 `exam2/evaluate_2d_point_grounding.py --coordinate_mode panel`。
- 证据从原视频和遥测重建；不声称与缺失的历史完整 manifest 逐字节相同。

### Experiment 3：3D measured point hypotheses

- 使用最终 v9 prompt `paper_experiment_evidence/prompts_and_configs/exp3_qwen3vl_point_grounding.md`。
- 证据帧由 `exam3_point_grounding/build_point_grounding_manifest.py` 的 `select_evidence_frames` 选择，最多 3 帧，`candidate_step_seconds=0.5`、`stability_window_seconds=0.3`、`min_sep_seconds=0.85`。
- 实际每事件 1–3 张证据帧，均值 2.750。
- 输入含 camera/gaze/hand 遥测和 broad scene bounds；不含候选 anchor、GT ID 或 GT 坐标。
- 输出 `points_3d`；解析用 `point_parser.parse_points_3d_output`。
- 评测用 `exam3_point_grounding/evaluate_point_grounding.py`；GT 来自 `paper_experiment_evidence/manifests/experiment3_evaluable_manifest.csv`，与 Exp1 GT 集合逐条一致（4,000/4,000，集合比较无差异）。

## 4. 完成状态

| Setting | 状态 | Batch | 保存响应 | 有效输出 | 无效/失败 | 评测行 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Experiment 1 | COMPLETE | 125 | 4000/4000 | 3888 | 112（74 invalid + 38 error） | 4000 |
| Experiment 2 | COMPLETE | 125 | 4000/4000 | 3894 | 106（68 invalid + 38 error） | 4000 |
| Experiment 3 | COMPLETE | 125 | 4000/4000 | 3931 | 69（62 invalid + 7 error） | 4000 |

`progress.json` 的 `completed_requests` 为 4,000；`saved_response_records` 为 4,000；`valid_outputs` 与 `smoke_summary.json` 的逐 batch 汇总一致。Exp1 的 `detail.csv` 和 Exp3 的 `evaluation_detail.csv` 都有 4,000 行；Exp2 的 `events=4000`、`gt_referents=8519`。

## 5. 指标

### 5.1 Experiment 1

| 指标 | 值 |
| --- | ---: |
| Micro Precision | 0.682485 |
| Micro Recall | 0.603157 |
| Micro-F1 | 0.640373 |
| Exact | 0.4115（1646/4000） |
| TP / FP / FN | 5350 / 2489 / 3520 |

独立从 4,000 行 `detail.csv` 重新汇总，得到与 `progress.json`/`overview.json` 完全相同的数值。

### 5.2 Experiment 2

主结果 `paired_canvas_map`：

| 指标 | 值 |
| --- | ---: |
| Temporal F1 | 0.764246 |
| Point@100 F1 | 0.111513 |
| Joint@100 F1 | 0.089444 |
| Point@200 F1 | 0.289117 |
| Joint@200 F1 | 0.243578 |
| GT referents | 8519 |
| Predictions | 8609 |

声明空间敏感性（同一批输出，仅改变坐标映射）：

| 指标 | 值 |
| --- | ---: |
| Temporal F1 | 0.764246 |
| Point@100 F1 | 0.363265 |
| Joint@100 F1 | 0.317258 |
| Point@200 F1 | 0.567258 |
| Joint@200 F1 | 0.495096 |

主结果与敏感性结果不能混为同一协议。主结果保留冻结的 `paired_canvas_map`；敏感性结果用于诊断 prompt 声明空间与后处理映射的冲突。

### 5.3 Experiment 3

| 指标 | 值 |
| --- | ---: |
| Anchor Precision | 0.429637 |
| Anchor Recall | 0.435062 |
| Anchor-F1 | 0.432333 |
| Exact | 0.02275（91/4000） |
| Margin-F1@1 | 0.251672 |
| SNE（加权，7,844 个匹配点） | 0.154122 |

SNE 使用逐行 `mean_scene_normalized_error` 按匹配点数加权；归档 `model_results.csv` 的 Qwen3-VL-30B 为 0.156908、direct gaze 为 0.151876。

## 6. 模型路由异常

对 4,000 条 raw 记录逐条读取 `served_model`：

| Setting | `gpt-6.1-sol` | `gpt-5.6-luna` | `None` | 空字符串 |
| --- | ---: | ---: | ---: | ---: |
| Experiment 1 | 3921 | 1 | 60 | 18 |
| Experiment 2 | 3863 | 0 | 46 | 23 |
| Experiment 3 | 3942 | 9 | 40 | 9 |

- 10 条 `gpt-5.6-luna` 记录都是 HTTP 200、`response.status=completed`、`parse_ok=true`，已计入主结果。
- 113 条有效输出的 `served_model` 缺失（Exp1 40、Exp2 31、Exp3 42）；其中大部分是成功响应但中转站未返回 `model` 字段。无效/失败记录中也常见 `None`。
- 排除 `gpt-5.6-luna` 后的敏感性：Exp1 n=3999、Micro-F1 0.640603、Exact 0.411603；Exp3 n=3991、Anchor-F1 0.432196、Exact 0.022801、SNE 0.153921。
- 仅保留 `served_model=gpt-6.1-sol` 的子集：Exp1 n=3921、Micro-F1 0.643987、Exact 0.416220；Exp3 n=3942、Anchor-F1 0.431994、Exact 0.022324、SNE 0.153570。
- 影响很小，但主结果不能称为“已核实的单一官方模型结果”。若 rebuttal 需要严格模型身份，应改用官方 API 或在报告中明确中转站路由限制。

## 7. 分母、失败与重复审计

- 三项均以 4,000 为分母；无效/失败输出作为空预测保留，不从分母删除。Exp1 的 `detail.csv` 中 `response_status` 为 ok 3888、invalid 74、error 38；Exp3 的 `evaluation_detail.csv` 中 4,000 行包含无效输出。
- 每项 4,000 个 `scene::row_index` 唯一；三项 ID 集合完全相同。没有 full run 内部重复。
- 10 条 smoke 样本包含在 full selection 中，因此被 smoke 和 full 各请求一次；这是预期重复，不是 full run 内部重复。
- 输出目录按 batch 递增，没有发现不同 selection 覆盖同一目录。Exp2 的 `batch_1152` 有两次 `run_history`（中断后续跑），其余 batch 均为一次。
- `run_responses.py` 源码哈希在运行前 30 分钟变化 4 次；早期批次为 Exp1 batch_0000/0032/0064/0096、Exp2 batch_0000/0032、Exp3 batch_0000，共 224 条。最终代码哈希 `c91c9d2cfad74354b7b7e75dc37c49cad4c09fa83544b49583c1f8ec0d07380f` 与当前仓库文件一致。对 224 条早期样本用最终代码重建 request body，`request_sha256` 和 `prompt_sha256` 全部吻合（224/224），未检测到模型可见输入变化；但早期代码版本本身没有留档，这是 provenance 限制。
- 重试：Exp1 101 条、Exp2 103 条、Exp3 24 条有第二次 attempt；Exp1/Exp2 各 38 条、Exp3 7 条为传输失败。失败记录保留在 raw JSON 的 `error_message` 和 `.attempts.json`。

## 8. Setting 2 投影校验异常

`run_full_baseline.py` 的命令没有传 `--validate-projection-archive`；全量中只有 37/125 个 Exp2 batch 有 `projection_archive_check.json`，集中在 `batch_0000` 到 `batch_1152`。

- 36 个 batch 的 `pass=true`，`differences=[]`。
- `batch_1152` 的 `pass=false`：`scene2/362` 的 `chair7 P1` 最大坐标差 634.48 px、`desk5 P1` 的 `projection_valid` 变化、`person1 P1` 最大坐标差 480.54 px。
- 该 batch 仍完成并计入主结果；其评测使用重建的 `manifest_all.csv`，不是归档 `experiment2_projected_gt_manifest.csv`。
- 归档缺少 `evidence_acceptable`，因此不能用归档 manifest 完整重评时间指标；受影响样本很小，但 Setting 2 主结果应标注为“重建 GT + 冻结 `paired_canvas_map`”，不能声称与归档 GT 完全等价。

## 9. 用量与计费

`outputs/full_budget_state.json`（2026-10-27 18:53 本地快照）：

| 字段 | 值 |
| --- | ---: |
| 请求数（含 smoke/重试） | 12,160 |
| Input tokens | 276,238,018 |
| Output tokens | 7,843,018 |
| Total tokens | 330,144,416 |
| Nominal cost | 664.516446 USD |
| `actual_cost` | 66.451645 USD |
| `account_cost` | 339.009379 USD |
| 钱包余额（快照） | 135.590529 USD |
| 观测 debit | 64.920334 USD |

三项 full run 自身的 `progress.json` token 合计：Exp1 223,528,970；Exp2 41,108,097；Exp3 63,645,386；合计 328,282,453。差额来自 smoke、重试和中转站统计口径。计费字段均为中转站报告，未独立核验；token 用量不能替代账单。

## 10. 文件清单与规模

| 目录 | 规模 | 内容 |
| --- | ---: | --- |
| `outputs/gpt61sol_anchor_full_v1` | 16 GB | 125 batch；frames、raw、eval、run_config/history |
| `outputs/gpt61sol_projected2d_full_v1` | 7.6 GB | 125 batch；manifest、model_inputs、raw、predictions、eval、declared_space_sensitivity |
| `outputs/gpt61sol_world3d_legacy_full_v1` | 3.7 GB | 125 batch；evidence_frames、model_manifest、raw、predictions、eval |
| `outputs/taomu_gpt61sol_smoke10` | 25 MB | 10 条 smoke 的 raw、eval、报告和分析 |
| `outputs/full_baseline_overview` | 5 KB | `overview.json`、`STATUS_zh.md` |

每个 batch 的关键文件：

- `run_config.json`、`run_history.json`、`smoke_summary.json`
- `experiment1/raw/<scene>/row_<index>.json`、`experiment1/eval/detail.csv`、`summary.json`
- `experiment2/raw/...`、`predictions.csv`、`manifest/manifest_all.csv`、`eval/2d_eval_summary.json`、`declared_space_sensitivity/`
- `experiment3/raw/...`、`predictions.csv`、`gt_manifest_eval.csv`、`model_manifest.json`、`eval/evaluation_summary.json`、`evaluation_detail.csv`

这些目录已被 `.gitignore` 排除；完整结果留在服务器，不在 Git 中。

## 11. 可复现命令

只做离线重解析/评测，不调用 API：

```bash
cd /path/to/Trimodal-Referring
bash closed_models/run_smoke.sh --evaluate-only
python closed_models/analyze_smoke.py
python -m unittest discover -s closed_models/tests -v
```

正式全量的显式命令（本轮已完成，不要重复付费运行）：

```bash
python closed_models/run_full_baseline.py \
  --output closed_models/outputs/gpt61sol_anchor_full_v1 \
  --settings experiment1 --batch-size 32 --concurrency 4 --wallet-guard
```

另外两个 setting：

```bash
python closed_models/run_full_baseline.py \
  --output closed_models/outputs/gpt61sol_projected2d_full_v1 \
  --settings experiment2 --batch-size 32 --concurrency 4 --wallet-guard

python closed_models/run_full_baseline.py \
  --output closed_models/outputs/gpt61sol_world3d_legacy_full_v1 \
  --settings experiment3 --batch-size 32 --concurrency 4 --wallet-guard
```

`--output` 已存在且 `progress.json` 为 COMPLETE 时，`run_full_baseline.py` 不会重复请求。恢复前先读 `progress.json` 和 `batch_*/smoke_summary.json`。

## 12. 不能声称的内容

- 不能声称这是 OpenAI 官方模型结果，或上游模型身份已核实。
- 不能声称 Setting 2 与归档 GT 完全等价；至少 `batch_1152` 有已知投影差异，且其余 88 个 batch 未做归档校验。
- 不能声称 Setting 2/3 的输入与缺失的历史 manifests 逐字节相同。
- 不能把 Setting 3 的 Anchor-F1 与 direct gaze 的接近解释为模型具备自由 3D 融合能力；v9 prompt 本身偏向复制 gaze。
- 不能把 Exp2 的声明空间敏感性结果与旧模型主表直接比较；后处理不同。
- 不能把 4,000 条中的无效输出删除后重算主指标；当前主结果保留 4,000 分母。
