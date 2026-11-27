# 五个新增开源模型的全量基线结果

2026 年 11 月 16 日启动，11 月 27 日核验归档。单张 A100 80 GiB 完成五个模型、两项任务、每项 4,000 个唯一交互，共 **40,000 条模型输出，其中 39,581 条通过解析，419 条解析失败**。所有交互均计入评分。归档时逐条核对了 40,000 个请求哈希、模型版本和对应公共输入。InternVL3.5-8B 在新增模型中取得最高对象选择 Micro-F1（48.58%）和时间 F1（79.23%）；OneVision-2 在原投影协议下取得最高 Joint@200 F1（32.21%）。

## 实验一与实验二主结果

分数单位为 %。实验一的 Exact 要求整个 anchor 集合完全匹配。实验二的时间 F1 评价所选面板，Point 评价空间位置，Joint 要求时间和空间同时匹配。@100 和 @200 是像素距离阈值。表内 F1 均由全语料 TP、FP、FN 计算。实验二主表沿用历史 `paired_canvas_map`。

| 模型 | 来源 | 实验一 Micro-F1 | Exact | 实验二时间 F1 | Point@100 | Joint@100 | Point@200 | Joint@200 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-30B-A3B | 历史 | 53.99 | 32.88 | 73.48 | 24.97 | 20.67 | 43.99 | 36.74 |
| Qwen3-VL-8B | 历史 | 46.79 | 23.78 | 74.61 | 23.50 | 20.13 | 42.34 | 36.69 |
| InternVL3-38B | 历史 | 48.93 | 25.93 | 73.30 | 12.28 | 9.77 | 33.27 | 27.01 |
| InternVL3.5-8B | 新增 | 48.58 | 22.58 | 79.23 | 9.36 | 7.40 | 27.82 | 23.18 |
| MiniCPM-V-4.5 | 新增 | 46.74 | 22.60 | 77.01 | 3.77 | 2.52 | 14.10 | 10.63 |
| Molmo2-8B | 新增 | 40.58 | 17.98 | 67.00 | 11.92 | 10.33 | 26.80 | 22.82 |
| LLaVA-OneVision-2-8B-Instruct | 新增 | 37.48 | 15.03 | 64.66 | 19.87 | 15.92 | 40.10 | 32.21 |
| Qwen3.5-9B | 新增 | 48.28 | 23.75 | 68.10 | 11.59 | 8.98 | 26.36 | 20.61 |

历史结果取自论文最终归档。实验一新增模型使用按时间排序的最多 16 张图像及兼容字段解析，历史 Qwen3-VL-8B 使用最多 8 帧；视觉适配与 token 数也随模型而异。实验二固定 v10 manifest 的 26,422 行投影记录，准备阶段逐行核对投影有效性和 GT 坐标。此前闭源对照及其 GT 重建、服务路由限制见独立报告。[历史主表](../../../paper_experiment_evidence/model_results.csv) · [历史模型协议](../../../paper_experiment_evidence/model_protocol_comparison.csv) · [闭源基线报告](../../../closed_models/FULL_BASELINE_REPORT_20261103.md)

## 解析与运行记录

| 模型 | 实验一有效 / 4,000 | 实验二有效 / 4,000 | 两项解析失败合计 |
|---|---:|---:|---:|
| InternVL3.5-8B | 4,000 | 4,000 | 0 |
| MiniCPM-V-4.5 | 3,973 | 4,000 | 27 |
| Molmo2-8B | 4,000 | 3,881 | 119 |
| LLaVA-OneVision-2-8B-Instruct | 3,995 | 3,758 | 247 |
| Qwen3.5-9B | 3,974 | 4,000 | 26 |

- 所有新增模型使用 BF16、贪心解码。实验一、二分别使用 1,536 和 768 个输出 token 上限。模型版本、推理环境与源码哈希见机器可读结果中的 `generation_protocol`。[模型版本](../../model_registry.json) · [协议与调度记录](../../README.md)
- 实验一统一使用 `anchor_schema_compat_v1`。它识别合法 JSON 中的等价答案字段，并保留显式空集合。MiniCPM 早期 278 条冻结输出中有 95 条通过兼容解析恢复。实验二沿用原字段解析。
- OneVision 先并行完成实验二，完整 worker 随后补齐实验一并复用实验二缓存。归档时已将独立实验二的全部汇总指标与最终结果逐项核对。
- Molmo 首次因 Python 默认 ASCII 解码中断。OneVision 完成后，独立 worker 启用 `PYTHONUTF8=1` 并从缓存续跑。首次调度器日志保留退出错误，最终以 Molmo worker 的 COMPLETE、250 份批次汇总和两项各 4,000 行的评分明细为准。

## 实验二坐标敏感性

提示词允许模型声明完整画面或局部裁剪坐标。主评优先按整张左右拼图的位置解释，`source_or_canvas_map` 优先按声明区域解释。两列使用同一批模型输出和 GT，仅坐标解释不同。历史三款模型使用冻结预测离线重评。坐标解释会改变模型排名，时间 F1 在两种映射下相同。

| 模型 | Joint@200 原映射（%） | Joint@200 声明空间（%） |
|---|---:|---:|
| Qwen3-VL-30B-A3B | 36.74 | 22.17 |
| Qwen3-VL-8B | 36.69 | 41.31 |
| InternVL3-38B | 27.01 | 34.08 |
| InternVL3.5-8B | 23.18 | 44.94 |
| MiniCPM-V-4.5 | 10.63 | 41.60 |
| Molmo2-8B | 22.82 | 22.33 |
| LLaVA-OneVision-2-8B-Instruct | 32.21 | 23.31 |
| Qwen3.5-9B | 20.61 | 27.37 |

## 结果文件与离线复核

[results.json](results.json) 保存五款模型的全精度汇总、两套映射的 Point/Joint@50、100、150、200、检查点与评分文件 SHA-256，同时保存三款历史模型的声明空间复评指标。每款新增模型有一份 CSV，每行对应一个唯一 `scene::row_index`。其中 `valid1/valid2` 是两项解析状态，`exact/tp/fp/fn` 属于实验一，带坐标策略前缀的列为实验二各指标的 TP、FP、FN。汇总一列计数后取 `2TP/(2TP+FP+FN)` 可复核 F1。[InternVL3.5](InternVL3_5-8B.csv) · [MiniCPM](MiniCPM-V-4_5.csv) · [Molmo](Molmo2-8B.csv) · [OneVision](LLaVA-OneVision-2-8B-Instruct.csv) · [Qwen3.5](Qwen3.5-9B.csv)

在存有完整原始输出和影像的实验服务器上，从仓库根目录执行以下命令可从历史预测与新增批次重建归档。普通 Git 检出可直接使用本目录的 CSV 和 JSON。

```bash
export PYTHONUTF8=1
python open_models/replay_historical_exp2.py
python open_models/report.py
python open_models/archive_results.py
```
