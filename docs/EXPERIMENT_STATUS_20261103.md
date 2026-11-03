# VR-TriRef / Trimodal-Referring 实验进度与恢复记录（2026-11-03）

本文根据当前服务器副本中的代码、配置、日志、输出文件和 Git 历史整理。原始 Codex 对话已丢失，因此这里只记录能从产物复核的事实；无法从产物确认的内容标为“待确认”。本次没有重新调用收费 API，没有重跑完整推理，也没有启动训练。

## 0. 仓库与证据边界

- 仓库：`D:\spj\Trimodal-Referring`（服务器 Linux 路径在产物中写作 `/media/kunyu/16TB/spj/Trimodal-Referring`）。
- 当前分支：`main`，恢复时 HEAD 为 `4b38f1e61eff5579422f75b8e32aae883dc3d21d`，与 GitHub `origin/main` 一致。
- 本次整理前，工作区只有 6 个跟踪文件有真实内容修改（共 22 行新增），其余约 300 个“修改”是 NTFS/WSL 文件权限位变化，不是实验改动。
- 闭源基线产物位于 `closed_models/outputs/`，约 27 GB，已被 `.gitignore` 排除。本文只记录路径、规模、指标和限制，不把 raw 响应、图片或完整预测提交到 Git。
- 旧的 `docs/REBUTTAL_OPENAI_PLAN_20261026.md` 是方案与协议审计，不是最终结果记录；本轮实际执行状态以本文和 `closed_models/FULL_BASELINE_REPORT_20261103.md` 为准。

## 1. 当前三个 setting 的实际协议

论文 Table 2 的第三项是 `exam3_point_grounding/` 的 measured point-hypothesis v9 协议，不是 `exam3/` 的 camera-centered directional 实验。`exam3/` 仍保留为历史分支，不应作为论文第三项复现入口。

| Setting | 模型可见输入 | 处理流程 | 输出 | 评测 |
| --- | --- | --- | --- | --- |
| 1 Anchor Selection | 指令；事件时间窗内有序 RGB 帧；gaze/hand/camera 结构线索；该场景完整候选 anchor 表 | 复用原视频 loader 的时间窗与最多 16 帧采样规则；`mention_first`/`full` prompt；候选 anchor 是合法输入 | canonical anchor ID 集合 | `scripts/eval/evaluate_local_3d_object_match.py`；Referent Recall / Exact / Macro-F1 / Micro-F1；按集合去重 |
| 2 Projected-2D Localization | 指令；最多 3 个时间 panel，每个为 full-panel + gaze-centered crop 的 paired canvas；给定 GT referent 数量 | 复用原 `build_2d_eval_manifest.py` 选帧、时钟 offset 和 GT 投影；`multi_image / paired_crop`；主结果使用冻结的 `paired_canvas_map` | 每个 referent 的 panel ID 与归一化 2D 点 | `exam2/evaluate_2d_point_grounding.py`；T-F1、Point@50/100/150/200、Joint@50/100/150/200；点误差在原始 panel 像素坐标计算 |
| 3 3D Localization | 指令；最多 3 张 target-free 证据帧；camera/gaze/hand 遥测；broad scene bounds；不给候选 anchor、GT ID 或 GT 坐标 | 复用最终 v9 prompt、证据帧选择和 `point_parser.py`；legacy 模式默认复制不同 gaze hypotheses，无 gaze 才用 hand | `points_3d` 坐标三元组 | `exam3_point_grounding/evaluate_point_grounding.py`；nearest-anchor / margin / SNE；nearest-anchor 只在评测阶段执行 |

三项均覆盖 4,000 个唯一 `scene::row_index`：scene1/scene2/scene3/scene5 各 800 条，scene4 的 4 个 room 各 200 条。scene4 仍属于一个场景，因此论文称 5 个场景。

### 1.1 评测口径

- Setting 1 主表是 Referent Recall / Exact / Macro-F1 / Micro-F1；不要用 any-hit 的 Hit-All 替代 Referent Recall。
- Setting 2 的 P@100、J@100 是相应匹配条件下的 corpus F1，不是 precision 或 interaction accuracy。
- Setting 3 是 measured point-hypothesis 诊断，不是 3D box grounding、3D IoU 或自由物体中心重建；坐标单位按 Unity world unit 描述，除非另行验证为米。
- 归档的 8,519 个可评测 referents 与按 panel 展开的 26,422 行投影 GT CSV 不是同一种计数。

## 2. 相比仓库原有版本的主要变动

### 2.1 新增闭源模型 runner

新增 `closed_models/`，通过 OpenAI-compatible `/v1/responses` 调用中转站模型，并复用论文三个 setting 的 prompt、图片构造和 evaluator：

- `closed_models/run_responses.py`：请求构造、图片编码、重试/缓存/断点续跑、解析、旧 evaluator 调用。
- `closed_models/run_full_baseline.py`：按 32 条一个 batch 增量运行和评测；每批完成后再准备下一批；支持钱包守卫和失败阈值。
- `closed_models/relay_budget.py`：中转站 `/v1/usage` 余额检查；区分 nominal `cost`、`actual_cost` 和 `account_cost`。
- `closed_models/protocols.py`：显式协议变换；不允许把修订 prompt 重标为 legacy。
- `closed_models/analyze_smoke.py`、`budgeted_subset.py`、`budget_report.py`、`watch_full_results.py`、`finish_budget_hold.py`：离线分析、预算和监控工具。
- `closed_models/prompts/exp3_evidence_grounding_v1.md`：仅在 `evidence` 模式下使用的实验 prompt；正式全量使用 `legacy` v9 prompt。
- `closed_models/tests/`：适配器、预算和协议单元测试。

密钥保存在仓库外的 `~/.config/trimodal-referring/taomu.json`，权限 0600；仓库内 `taomu.example.json` 只含公共配置和环境变量名。

### 2.2 数据读取的兼容修复

四个 loader 增加同一个严格 JSON 容器修复：当原始发布文件省略最外层数组、但内容是一个完整对象时，补上 `[` 和 `]`，随后仍按严格 JSON 校验；不修改原始数据，不补造样本内容。改动文件：

- `exam2/build_2d_eval_manifest.py`
- `exam3_point_grounding/point_grounding_common.py`
- `scripts/data_prep/build_keyframe_grounding_input.py`
- `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`
- `scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py`

触发样本是发布的 `scene5/408` 遥测缺少数组外壳。该修复不改变已有正常样本的解析结果。

### 2.3 Rebuttal 审计与分组分析

- `analysis_outputs/rebuttal_openai_20261026/local_audit.json`：论文 PDF、归档、样本、GT 和文件校验。
- `analysis_outputs/rebuttal_openai_20261026/metric_validation.json`：从 9 份 compact 预测独立重算 45 个主指标，全部与归档汇总一致；确认各 setting 内跨模型 GT 一致。
- `analysis_outputs/location_region_audit/`：按已有 37-anchor taxonomy 做非物体/区域目标核查和分组统计；未改 GT、评测规则或模型输出。
- `scripts/data_prep/test_multimodal_json_loading.py`：多模态 JSON 读取回归测试。

### 2.4 README 修正

原 README 把 `exam3/` camera-centered directional 列为第三项，且标题为 `abcde`。本轮改为当前三项协议、闭源基线入口和文档索引；历史 `exam3/` 仍在仓库中，但明确标为历史分支。

## 3. 已完成实验与结果

### 3.1 闭源基线全量（gpt-6.1-sol）

三项均完成 4,000/4,000 请求，每项 125 个 batch，评测分母均为 4,000。详细记录见 `closed_models/FULL_BASELINE_REPORT_20261103.md`。

| Setting | 保存响应 | 有效输出 | 无效/失败 | 关键指标 |
| --- | ---: | ---: | ---: | --- |
| Experiment 1 | 4000/4000 | 3888 | 112 | Micro-F1 0.6404；Recall 0.6032；Exact 0.4115 |
| Experiment 2 | 4000/4000 | 3894 | 106 | T-F1 0.7642；Point@100 0.1115；Joint@100 0.0894；Point@200 0.2891；Joint@200 0.2436（主 `paired_canvas_map`） |
| Experiment 3 | 4000/4000 | 3931 | 69 | Anchor-F1 0.4323；Anchor-P 0.4296；Anchor-R 0.4351；Exact 0.02275；Margin-F1@1 0.2517；SNE 0.1541 |

Experiment 2 的同一批输出另做“按模型声明坐标空间重映射”的敏感性评测：

| 协议 | T-F1 | Point@100 | Joint@100 | Point@200 | Joint@200 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 主结果 `paired_canvas_map` | 0.7642 | 0.1115 | 0.0894 | 0.2891 | 0.2436 |
| 声明空间敏感性 | 0.7642 | 0.3633 | 0.3173 | 0.5673 | 0.4951 |

这两种后处理不能混为同一协议。主结果沿用冻结的 `paired_canvas_map`；敏感性结果用于诊断坐标约定冲突，不能直接与旧模型主表比高低。

### 3.2 与旧开源基线的对照（仅供定位，不直接合并结论）

| Setting | 模型 | 指标 |
| --- | --- | --- |
| 1 | Qwen3-VL-30B-A3B | Micro-F1 0.5399；Recall 0.5169；Exact 0.3287 |
| 1 | gpt-6.1-sol | Micro-F1 0.6404；Recall 0.6032；Exact 0.4115 |
| 2 | Qwen3-VL-30B-A3B | T-F1 0.7348；Point@100 0.2497；Joint@100 0.2067；Point@200 0.4399；Joint@200 0.3674 |
| 2 | gpt-6.1-sol（主） | T-F1 0.7642；Point@100 0.1115；Joint@100 0.0894；Point@200 0.2891；Joint@200 0.2436 |
| 2 | gpt-6.1-sol（声明空间敏感性） | T-F1 0.7642；Point@100 0.3633；Joint@100 0.3173；Point@200 0.5673；Joint@200 0.4951 |
| 3 | Qwen3-VL-30B-A3B | Anchor-F1 0.4326；Exact 0.0185；Margin-F1@1 0.2503；SNE 0.1569 |
| 3 | gpt-6.1-sol | Anchor-F1 0.4323；Exact 0.02275；Margin-F1@1 0.2517；SNE 0.1541 |
| 3 | direct gaze / gaze_copy（归档） | Anchor-F1 0.4325；Exact 0.0435；Margin-F1@1 0.2547；SNE 0.1519 |

Setting 3 上 gpt-6.1-sol 的 Anchor-F1 与 Qwen3-VL-30B 基本持平，Exact 和 Margin 略高，但相对 direct gaze 行在 Exact、Margin 和 SNE 上更差。不能据此宣称模型具备自由 3D 推理能力；v9 prompt 本身默认复制 gaze hypotheses。

### 3.3 Smoke 与分组分析

- 10 条 smoke：Setting 1 为 4 条、Setting 2 为 3 条、Setting 3 为 3 条；10/10 有效 JSON，旧 evaluator 均成功执行。结果在 `closed_models/outputs/taomu_gpt61sol_smoke10/SMOKE_REPORT_zh.md`。smoke 仅用于工程验证，不用于估计全量性能。
- 区域/目标分组：4,000 条 GT 中仅物体 2,628、仅非物体 109、混合 1,263；37 个 non-object anchors。GPT 在 Exp2 声明空间重映射下的非物体组 Point@100/Joint@100 高于开源模型；Exp1 纯非物体组由 Qwen3-VL-30B 略高；Exp3 纯非物体组各模型接近，Exact 多为 0。详见 `analysis_outputs/location_region_audit/REGION_TARGET_MODEL_RESULTS_zh.md`。

## 4. 可比性限制与已知问题

### 4.1 模型身份与服务

- 请求模型标识为 `gpt-6.1-sol`，endpoint 为 `https://www.taomu.xyz/v1/responses`，是中转站，不是 OpenAI 官方 API。模型身份、usage 和计费均为中转站返回的信息，无法独立认证上游模型。
- 全量中有 10 条有效输出由 `gpt-5.6-luna` 返回（Experiment 1 1 条，Experiment 3 9 条）；另有 113 条有效输出的 `served_model` 缺失（Experiment 1 40 条、Experiment 2 31 条、Experiment 3 42 条）。这些记录都保留在主结果分母中。
- 排除 `gpt-5.6-luna` 后：Exp1 Micro-F1 0.6406、Exact 0.4116；Exp3 Anchor-F1 0.4322、Exact 0.02280、SNE 0.1539。仅保留 `served_model=gpt-6.1-sol` 的子集：Exp1 n=3921、Micro-F1 0.6440；Exp3 n=3942、Anchor-F1 0.4320。影响很小，但不能把主结果称为“已核实的单一官方模型结果”。
- 未指定 temperature、seed 或 reasoning effort；使用服务默认值。旧开源模型为 greedy、`do_sample=false`。

### 4.2 输入与输出预算

- Setting 1 使用原视频 loader 的时间窗和最多 16 帧采样规则，以有序多图发送；旧 Qwen/InternVL 使用各自原生 video adapter。帧数分布为 6–16 帧，均值 15.25；两者不是逐字节相同的输入协议。
- Setting 2 使用 3 个 paired canvas（1–3 个，均值 2.96）；Setting 3 使用 1–3 张证据帧（均值 2.75）。旧 Setting 3 的 InternVL 最多 2 帧，Qwen 为 3 帧。
- 新基线 `max_output_tokens=4096`、`image_detail=high`、最长边 1920；旧模型输出上限为 1536/768/512。输出预算和推理默认值不同，不能称为严格等计算量对照。

### 4.3 Setting 2 坐标协议与 GT

- 正式 prompt 要求模型在所选子图（`full_panel` 或 `gaze_crop`）内输出局部归一化坐标；主结果后处理 `paired_canvas_map` 却按整张 paired canvas 的左右区域再映射。两者存在协议冲突；声明空间敏感性结果就是为量化这一冲突而保存的。
- Setting 2 证据从原视频和遥测重建，不能声称与缺失的历史完整 manifest 逐字节相同；归档缺少 `evidence_acceptable`，时间指标条件等同性未确认。
- 全量中只有 37/125 个 Setting 2 batch 保存了 `projection_archive_check.json`（batch_0000 到 batch_1152，连续 37 个）。其中 36 个通过，`batch_1152` 失败：`scene2/362` 的 `chair7 P1` 最大坐标差 634.48 px、`desk5 P1` 有效性变化、`person1 P1` 最大坐标差 480.54 px。该 batch 仍完成并计入主结果，说明“每批请求前都核对归档投影”的旧说法不成立。
- 因此 Setting 2 主结果应标注为“重建 GT + 冻结 `paired_canvas_map`”，而不是“与归档 GT 完全等价的正式全量”。受影响样本量很小，但在论文/ rebuttal 中应披露。

### 4.4 Setting 3 协议

- 全量使用 `legacy` v9 prompt 和 legacy 证据选择（最多 3 帧），不是 `evidence`/`coverage_pts` 新 prompt。
- v9 prompt 默认复制不同的 gaze hypotheses，无 gaze 才用 hand；因此结果更接近“在该 prompt 下的 measured point hypothesis”，不能解释为模型自由进行 3D 融合。
- 归档 direct gaze 行使用旧输入；本轮没有在重建输入上重跑完整 direct gaze 4,000 条，不能把两者当作严格 paired 对照。

### 4.5 分母、失败和重复

- 三个 setting 的分母都是 4,000；无效/失败输出作为空预测保留，不从分母删除。Exp1 的 `detail.csv` 有 4,000 行；Exp3 的 `evaluation_detail.csv` 有 4,000 行；Exp2 的 `events=4000`、`gt_referents=8519`。
- 每个 full run 的 4,000 个 `scene::row_index` 唯一，无重复；三个 setting 的 ID 集合完全相同。10 条 smoke 样本包含在 full selection 中，因此被再次请求（smoke 与 full 各一次），但 full run 内部没有重复。
- 没有发现同一输出目录被不同 selection 覆盖；batch 目录按 `batch_0000` 递增，`run_history.json` 记录每次启动。Exp2 有 1 个 batch 有 2 次 `run_history`（中断后续跑），其余 batch 均为 1 次。
- `run_responses.py` 源码哈希在运行前 30 分钟发生过 4 次变化：Exp1 前 4 个 batch（128 条）、Exp2 前 2 个 batch（64 条）、Exp3 第 1 个 batch（32 条）使用了早期版本；之后 121/123/123 个 batch 使用最终哈希 `c91c9d2c...`。本地对全部 224 条早期样本用最终代码重建 request body，`request_sha256` 和 `prompt_sha256` 全部吻合（224/224），说明可检测到的模型可见输入没有变化；但版本漂移仍应记录为 provenance 限制。
- 重试：Exp1 101 条、Exp2 103 条、Exp3 24 条有一次额外 attempt；Exp1/Exp2 各 38 条、Exp3 7 条为传输失败且无成功响应。失败记录保留。

### 4.6 计费

中转站 usage ledger 记录 12,160 次请求（含 smoke 和重试），总 token 330,144,416；nominal cost 664.52、`actual_cost` 66.45、`account_cost` 339.01 USD；钱包余额从首次记录的 70.87 变为 135.59，观测到的 debit 为 64.92 USD。计费字段均为中转站报告，未独立核验；token 用量不能替代账单。

## 5. 未完成与下一步

已完成：

- 三个 setting 的 4,000 条闭源基线推理与评测。
- 10 条 smoke 与独立指标复算。
- 区域/目标分组分析。
- 源码/请求哈希、分母、重复和异常模型审计。

未完成/待确认：

- 未用 OpenAI 官方 API 核验 `gpt-6.1-sol` 的真实上游身份；未获得官方 model card、价格和配额。
- 未解决 Setting 2 的坐标协议冲突；未在统一规则下重评旧模型全部原始输出。
- 未在重建输入上重跑完整 direct gaze 4,000 条。
- 未重建缺失的历史 full manifests，无法证明 Setting 2/3 输入与历史运行逐字节相同。
- 未运行第二个闭源模型、额外 reasoning 档位或针对新模型的消融。
- 未人工确认全部 GT（尤其 108 个 V3dMD referent 待确认项和 37-anchor taxonomy 的语义边界）。
- 未把 `batch_1152` 的 Setting 2 异常样本从主结果中剔除或重评；当前主结果保留现场。

下一次恢复工作建议按以下顺序阅读：

1. `closed_models/FULL_BASELINE_REPORT_20261103.md`：闭源基线的配置、指标、异常和文件清单。
2. `closed_models/outputs/full_baseline_overview/overview.json`：机器汇总。
3. `closed_models/outputs/gpt61sol_projected2d_full_v1/experiment2/batch_1152/experiment2/projection_archive_check.json`：Setting 2 已知投影异常。
4. `analysis_outputs/location_region_audit/REGION_TARGET_MODEL_RESULTS_zh.md`：分组结果与解释边界。
5. `docs/REBUTTAL_OPENAI_PLAN_20261026.md`：历史方案和协议审计。

## 6. 关键入口与可复现命令

闭源 runner 使用仓库根目录动态解析，不依赖旧的 `/workspace/usr3` 路径。运行前准备 `trimodal` 环境（Pillow、requests、openpyxl、ffmpeg、ffprobe）：

```bash
cd /path/to/Trimodal-Referring
# 仅重新解析/评测已保存响应，不调用 API
bash closed_models/run_smoke.sh --evaluate-only

# 离线分析 smoke
python closed_models/analyze_smoke.py

# 适配器/预算单元测试，无外网
python -m unittest discover -s closed_models/tests -v
```

正式全量的显式命令（**本轮已完成，不要重复付费运行**）：

```bash
python closed_models/run_full_baseline.py \
  --output closed_models/outputs/gpt61sol_anchor_full_v1 \
  --settings experiment1 --batch-size 32 --concurrency 4 \
  --wallet-guard
```

其余两个 setting 分别使用 `gpt61sol_projected2d_full_v1`/`experiment2` 和 `gpt61sol_world3d_legacy_full_v1`/`experiment3`。完整运行由 systemd 用户服务负责，输出目录中的 `progress.json`、`run_history.json`、`batch_*/run.log` 和 `batch_*/experiment*/eval/` 是恢复入口。

## 7. 与 MultiAgent 的依赖关系

`MultiAgent` 复用本仓库的 `data/`、`V3dMD/prepared/`、anchor 表和原 evaluator。其 `MultiAgent/baseline.py` 通过 `SPJ_WORKSPACE_ROOT`、`SPJ_TRIMODAL_ROOT`、`SPJ_MULTIAGENT_ROOT`、`SPJ_V3DMD_ROOT`、`SPJ_PREPARED_DATA_ROOT` 定位这些资源。

- MultiAgent 的结果写入 `D:\spj\Multiagent\runs\`，不属于本仓库。
- 本仓库的闭源基线结果写入 `closed_models/outputs/`，不属于 MultiAgent。
- 两者共享 GT/evaluator，但实验协议和输出目录不同，不能把一方的指标归到另一方。

## 8. 本次整理产生的 Git 变更

- 更新 `README.md`、`closed_models/README.md`、`docs/REBUTTAL_OPENAI_PLAN_20261026.md` 的状态说明。
- 新增本文和 `closed_models/FULL_BASELINE_REPORT_20261103.md`。
- 提交 `closed_models/` 代码、prompt、测试、小型审计 JSON、区域分组文档和文档中的结果表；`baseline_metrics_long.csv` 与 `region_target_model_results.csv` 保留在服务器，路径见上文。
- 不提交 `closed_models/outputs/`、`rebuttal/` 审稿材料、`aaai27_spj.pdf`、原始视频/模型权重/大型运行产物。
