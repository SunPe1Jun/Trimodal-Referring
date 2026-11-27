# 开源 baseline 扩充：实验一和二（2026-11-16）

[2026-11-27 全量结果、逐事件评分与复核命令](results/2026-11-27/README.md)

本轮只运行实验一 closed-set anchor selection 与实验二 projected-2D。实验三保持暂停。

## 2026-11-18：OneVision-2 实验二提前并行

`parallel_exp2.py --family onevision` 独立运行已有 OneVision-2 模型的 4000 条实验二；不新增模型、不改变输入、prompt、图像处理、精度、token 预算或评价规则。其原生实验二已有约 18.35 GiB 峰值，因此设置 28% 显存配额，与 MiniCPM 的 65% 配额合计 93%。实验一仍由原队列随后执行。

侧任务直接使用该模型原 `protocol.json` 和相同的请求哈希公式，保存至原 `batches/*/experiment2/raw/`，已完成验证样本直接复用。随后完整 worker 自动读取这些结果，不再次生成。阶段进度在 `experiment2_progress.json`，分批评价在 `experiment2/parallel_summary.json`，单任务汇总在 `experiment2_results_so_far.json`。总览中的两任务 batch summary 待完整 worker 补齐实验一后生成。

调度器读取 `parallel_exp2_reservation.json`，核实进程存活，并将其计入并发数与显存预算；侧任务活跃时不启动同模型完整 worker，避免重复生成和文件写入冲突。侧任务遇到 OOM 会保存错误并释放资源，原队列继续；已保存结果保留。启动、完成或取消实验二侧任务无需终止 MiniCPM worker。

## 2026-11-16：经用户同意启用字段兼容解析

`compatibility.py` 对所有新增模型统一使用 `anchor_schema_compat_v1`。合法 JSON 缺少 `selected_object_names` 时，兼容模型实际输出的 `selected_objectnames` 列表；也允许使用原 evaluator 已支持的 `selected_objects` 或 `selected_object_name`。标准字段存在时优先标准字段，不覆盖显式空列表或格式错误的标准字段。无效 JSON、没有答案字段的输出仍然失败；不从推理文字、GT、候选排序或 mention 猜测最终对象。实验二保持原解析与坐标映射。

模型可见 prompt、源图、模型权重、解码与输出预算不变。已有 raw 记录的 `model_raw_output`、请求哈希、prompt 哈希、图像清单和哈希逐条核验保留。记录新增 `strict_parse_snapshot`、`parser_version`、`compatibility_applied` 与 `compatibility_source`，原调用层解析模块本身未修改。后续所有任务按同一解析政策执行，原输入锁与请求身份保留；解析器版本另存 `parser_protocol.json`。

迁移工具 `reparse_compatibility.py` 在 worker 暂停时先备份，再重解析与调用原 evaluator；不调用模型。修正前记录和评分在各模型的 `strict_before_compatibility_v1/`；独立复评报告在 `compatibility_v1/migration_complete.json`。本次 MiniCPM 实验一冻结比较集 278 条：严格有效 183 条，兼容后有效 278 条，恢复 95 条；同一集合 Micro-F1 0.469925→0.571895，Exact 0.302158→0.460432。数值只表示该已生成子集，不是全量成绩。InternVL 534 条与 OneVision 验证样本 1 条均无需恢复，评分不变。

首批模型：`allenai/Molmo2-8B`、`openbmb/MiniCPM-V-4_5`、`lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct`、`OpenGVLab/InternVL3_5-8B`。检查点提交、下载进程和本机目录在 `/workspace/usr3/models/baseline_expansion_20261116/models.json`；每个模型下载完成才会进入 GPU 队列。

用户随后新增 `Qwen/Qwen3.5-9B`，使用 Transformers 5.9.0 的原生视觉语言模型和 processor，BF16、greedy、官方 `enable_thinking=False`；原输入、prompt、输出预算与 evaluator 不变。模型独立使用 `qwen35_adapter.py`，避免改动已有四模型的 adapter 源码锁。其显存预算为 65%，可与 InternVL 的 30% 配额并行；超出预算时原输入不变，改为独占重试。当前队列共五模型、40000 次计划生成。

Qwen3.5 权重通过官方 ModelScope 镜像下载，模型文件逐个核对 Hugging Face 提交 `c202236235762e1c871ad0ccb60c8ee5ba337b9a` 文件树的大小与 SHA；README/许可证文档使用官方 ModelScope 副本，不把镜像文档版本等同于 HF 文档版本。下载与核验记录在模型目录的 `download_verification.json`。

## 输入和实验合同

- 使用归档 4000 个唯一 `scene::row_index`，不根据难度或结果筛选。每模型两个任务共 8000 次生成，本轮五模型共 40000 次。
- 各 partition 最前 10 条组成固定 80 条 smoke，属于完整 4000 条的一部分，成功生成后不重复推理。所有模型使用相同公共源图、指令和面板元数据。
- 实验一复用原 `mention_first` prompt、场景 anchor 表、事件时间窗和最多 16 帧抽取逻辑，1536 输出 token；通过有序多图输入。原视频适配器与本轮有序多图的区别必须在论文中披露。
- 实验二直接复用本机原完整 v10 manifest：26422 行、4000 个交互。投影有效性和 `gt_x/gt_y` 与论文归档逐行完全一致，所有源帧存在；校验记录为 `outputs/expansion_20261116/input_preflight.json`。
- 实验二沿用最多三个时刻面板、全图与 35% gaze crop 配对、768 局部分辨率、原 expected_count prompt 与 768 输出 token。主结果沿用 `paired_canvas_map`；同一 raw 输出另存 `source_or_canvas_map` 敏感性结果，不把两者混报。
- 两个任务都复用 `closed_models/run_responses.py` 中的解析与原 evaluator，仅替换本地生成端；不会发送收费 API 请求。
- 模型均 BF16、greedy、不量化。保留各模型原生 processor；不同视觉 token/切片与 chat template 不等价于完全相同计算预算。MiniCPM 两项都保留原生 max_slice_nums=9，并显式覆盖 chat 默认 beam-search/repetition penalty 以与原 greedy 解码一致；InternVL 沿用旧 adapter 每图 448×448 单 patch；Molmo 将 system 文本并入 user 以符合其官方模板；不添加新的提示后缀。
- 实验一 GT 使用论文 compact prediction 文件中的冻结 `gt_anchor_ids`，不使用其中预测值。本机两份未提交 Excel 原样保留。

## 调度和恢复

所有输入、输出、进度与日志在 `open_models/outputs/expansion_20261116/`，不会覆盖历史结果。模型检查点在仓库外，不提交 Git。

`prepare.py` 用三个 CPU 进程分批准备公共输入，`supervise.py` 最多同时运行两个 GPU worker，按模型分配显存预算：InternVL 30%、MiniCPM 65%、Molmo 45%、OneVision-2 92%，同时运行任务的预算合计不超过 95%。这些只是进程资源配额，不改变模型可见输入。首次均分 46% 配额使 MiniCPM 在整卡仍有空闲显存时停止，已根据真实样本重新分配；失败日志保留。发生后续 OOM 时保持输入、生成预算与精度，改为独占重试。模型输出非法不自动重新生成；保存为空预测并计入分母，不根据格式失败率或任务成绩剔除模型或样本。用户要求按原方法完成全量；最初的格式失败率停止条件已设为默认关闭。此调度策略修改保留原请求锁，另记 worker_source_history，不改写已生成结果。

不同模型需要不同 Transformers 版本：Molmo/MiniCPM/InternVL 使用模型目录下独立的 `runtime_457`（Transformers 4.57.1、tokenizers 0.22.1、huggingface_hub 0.36.0）；OneVision-2 使用原 trimodal 环境的 Transformers 5.9.0。Molmo 原生 attention mask 还要求 PyTorch >=2.6，因此使用独立 `molmo_env`；原 trimodal 的 PyTorch 2.5.1 保留。没有全局降级或替换既有研究环境。

正式队列启动前，`validate_adapter.py` 对每个新适配器执行实验一、二真实样本校验。其结果直接保存为正式 worker 可复用的 raw 响应，不作为额外重复样本；完整推理仍覆盖固定 4000 条。InternVL 的首个完整 batch 已使用原评价器验证。OneVision-2 自定义 processor 必须先渲染 chat template，再显式传入 images；适配器对视觉张量存在性进行检查，避免误做纯文本推理。

```bash
cd /workspace/usr3/TriModal-Referring
PY=/workspace/usr3/miniconda3/envs/trimodal/bin/python
$PY open_models/status.py
$PY open_models/report.py
nvidia-smi
```

`status.py` 只查看进度；`report.py` 汇总已完成 batch 的 TP/FP/FN 后计算 corpus F1，不平均 batch F1。运行中的部分结果不代表完整 benchmark。

停止后可重新启动 `prepare.py` 与 `supervise.py`，完整输入 batch 和保存的 raw 响应会复用；单例锁防止两个 supervisor 重复推理。源码、模型和输入变更会触发协议检查，应使用独立输出版本，不改写冻结结果。创建 `outputs/expansion_20261116/STOP` 会使调度器终止 worker；恢复前需移除该文件。

## 验证

```bash
$PY -m unittest discover -s open_models -p 'test*.py' -v
```

测试覆盖完整样本唯一性、实验三排除、已生成但未解析响应的恢复、配置漂移拒绝和 corpus 计数聚合。模型真实加载与多图生成的实际验证进度见 worker 日志。
