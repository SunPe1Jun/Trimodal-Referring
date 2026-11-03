**OpenAI-compatible Responses runner for VR-TriRef**

当前配置使用桃牧中转站 `https://www.taomu.xyz/v1/responses` 和其列出的 `gpt-6.1-sol`。模型身份与 usage 为中转站返回的信息。实现复用论文三个 setting 的 prompt、图片构造及 evaluator；不会默认启动全量。

三项 4,000 条全量已于 2026-10-27 完成。结果、异常模型路由、Setting 2 投影校验异常和可比性限制见 [`FULL_BASELINE_REPORT_20261103.md`](FULL_BASELINE_REPORT_20261103.md)；项目总进度见 [`../docs/EXPERIMENT_STATUS_20261103.md`](../docs/EXPERIMENT_STATUS_20261103.md)。

真实密钥配置位于仓库外的 `~/.config/trimodal-referring/taomu.json`，权限 0600。仓库内 `taomu.example.json` 只含公共配置和环境变量名。请勿在命令行参数、Git 或共享日志中放入密钥。新密钥替换不会使已完成样本重新计费。

使用已有 `trimodal` 环境（含 Pillow、requests、openpyxl、ffmpeg、ffprobe）：

```bash
# 默认三个 setting 合计 10 条：4 + 3 + 3。
bash closed_models/run_smoke.sh

# 仅准备证据与 prompt，不请求 API。
bash closed_models/run_smoke.sh --prepare-only

# 仅补传输错误；成功结果和模型格式失败不会自动重新采样。
bash closed_models/run_smoke.sh --retry-errors

# 从已保存的响应重新解析、评测，无 API 请求。
bash closed_models/run_smoke.sh --evaluate-only

# 同批输出的坐标敏感性、direct gaze、旧模型同 ID 子集与 usage 分析，无 API 请求。
/home/kunyu/miniconda3/envs/trimodal/bin/python closed_models/analyze_smoke.py

# 适配器测试，无外网或模型调用。
/home/kunyu/miniconda3/envs/trimodal/bin/python -m unittest discover -s closed_models/tests -v
```

若 Python 环境路径不同，设置 `TRIMODAL_PYTHON=/your/env/bin/python`。新环境需将 ffmpeg 和 ffprobe 加入 PATH；runner 会自动将所用 Python 的 bin 目录加入 PATH。

默认样本在 `run_responses.py` 的 `DEFAULT_SELECTION` 中预先固定。`--selection /path/selection.json` 可以指定样本，格式为：

```json
{
  "experiment1": ["scene1::0"],
  "experiment2": ["scene1::1"],
  "experiment3": ["scene2::0"]
}
```

为新模型、不同 prompt/图像 detail、输出预算或样本集指定新的 `--output`。runner 会检查缓存的完整请求 hash 和 endpoint，防止把旧输出误用到新请求；更换密钥可继续同一批结果。准备完成后会冻结 `prepared_tasks.json` 中的实际输入，断点续跑复用它。已收到的成功 HTTP 响应会在解析前落盘。

正式全量的显式命令（本轮已完成，不要重复付费运行）：

```bash
python closed_models/run_full_baseline.py \
  --output closed_models/outputs/gpt61sol_anchor_full_v1 \
  --settings experiment1 --batch-size 32 --concurrency 4 --wallet-guard
```

其余两项分别使用 `gpt61sol_projected2d_full_v1`/`experiment2` 和 `gpt61sol_world3d_legacy_full_v1`/`experiment3`。

需要只运行一个 setting 时加 `--settings experiment1` 等。先确认 smoke 报告中 Setting 2 的坐标规则与输入恢复限制，再确定全量协议。当前版本按冻结的 `paired_canvas_map` 产生主结果，另在 `declared_space_sensitivity/` 进行不增加模型请求的坐标敏感性评测。两者不能混作同一协议，也不能仅给新模型修正后直接对比旧表。

产物默认位于 `closed_models/outputs/taomu_gpt61sol_smoke10/`，已忽略 Git：

- `SMOKE_REPORT_zh.md`、`smoke_summary.json`、`smoke_analysis.json`：结果和分析。
- `experiment*/raw/<scene>/row_<index>.json`：完整响应、模型输出、解析结果、图片路径/尺寸/hash、request hash、usage；不含鉴权头或密钥。
- `*.attempts.json` / `request_history`：当前尝试与跨运行失败记录。网络异常最多两次退避重试，401 不自动重试。
- `experiment*/eval/`：原 evaluator 输出；失败和非法响应保留分母。
- `sample_usage.csv`：每条图片数量、token 用量与请求耗时。
- `run_config.json` / `run_history.json`：脱敏配置和源文件哈希。`max_output_tokens=4096`、并发 4（Exp2 后段为 8）、`detail=high`，最长边 1920；reasoning effort 未指定，使用服务默认设置。

Setting 1 使用原视频 loader 的时间窗、最多 16 帧采样规则，以有序多图发送；它与旧模型的原生 video adapter 有差别。Setting 2/3 证据从原视频和遥测重建，不能声称与缺失的历史完整 manifest 逐字节相同。smoke 的 3 条 Setting 2 样本已核对重建投影与归档一致，但归档缺少 `evidence_acceptable`，无法仅凭 compact 归档证明时间指标条件等同。

2026-10-26 本地 smoke 已完成 10/10 有效输出，详见默认目录中的报告。三项全量随后于 2026-10-27 完成；当前真实金额未取得，token 用量不能替代中转站账单。全量结果不是官方 OpenAI API 结果，且存在少量 `gpt-5.6-luna`/缺失 `served_model` 的路由异常，详见基线专档。

2026-10-26 预算与执行更新：用户确认 300 USD 是可后续追加的余额，正式实验仍要求三个 setting 各 4,000 条。三项全量已完成；不再追加闭源消融、第二模型或 prompt 搜索。共享 `relay_budget.py` 使用 `/v1/usage` 检查真实余额，区分 nominal `cost`、`actual_cost` 和 `account_cost`，不重复乘 0.1。后续任务按 300 USD 运行上限控制，钱包不足时等待充值，已有响应不重复调用。`full_budget_state.json` 保存检查记录，`full_baseline_overview/STATUS_zh.md` 显示实时进度。此前 200/500 条子集建议和 `budgeted_subset.py` 未启动付费运行，不作为当前计划。
