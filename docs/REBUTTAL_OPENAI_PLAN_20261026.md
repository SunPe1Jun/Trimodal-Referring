**Trimodal-Referring / VR-TriRef：现有实验梳理与 OpenAI rebuttal 补测方案**

整理日期：2026-10-26。截止时间：**2026-10-31 20:00，北京时间（UTC+8）**。初次整理范围为实验审计和补测设计。后续已按用户要求实现 `gpt-6.1-sol` / Responses 适配并完成合计 10 条真实 smoke；三项 4,000 条全量随后于 2026-10-27 完成。**本文是当时的方案与协议审计，不是最终结果记录。** 实际完成情况、指标、异常模型路由和可比性限制见 [`EXPERIMENT_STATUS_20261103.md`](EXPERIMENT_STATUS_20261103.md) 和 [`../closed_models/FULL_BASELINE_REPORT_20261103.md`](../closed_models/FULL_BASELINE_REPORT_20261103.md)。

建议最低交付：**一个经官方和账户核实的 OpenAI 多模态旗舰，在三个 setting 各完成 4,000 条评测，共 12,000 次样本推理**。预算、配额和时间允许后，再加入第二个不同成本档位的 OpenAI 模型。先完成一个模型的三项完整结果，比只补多个模型的零散 setting 更有利于回应 baseline 不够新的问题。

**1. 哪些文件代表论文实验**

已阅读仓库新增的 `aaai27_spj.pdf`，其中 Table 2 与 `paper_experiment_evidence/model_results.csv` 的主要数值一致。最终证据以 `paper_experiment_evidence/` 为准，具体协议以 `run_provenance/run_provenance.csv`、`model_protocol_comparison.csv` 和 `prompts_and_configs/` 为准。PDF 的 SHA256 记录在本轮 `local_audit.json` 中；归档中的 `paper_source_sha256.txt` 是另一份 TeX 源文件的校验值，不能拿来验证当前 PDF。

仓库 README 仍把旧的 `exam3/` camera-centered directional 实验列为第三项。**论文 Table 2 的第三项实际上是 `exam3_point_grounding/` 的最终 v9 measured point-hypothesis 协议**。后续补测不应误跑 `exam3/run_qwen3vl_3d_directional.py`。

| Setting | 模型可见输入与输出 | 必须沿用的口径 |
| --- | --- | --- |
| 1：Anchor Selection | 指令、第一视角视频帧、gaze/hand/camera 结构线索、该场景完整候选 anchor 表；输出可变长度 canonical anchor ID 集合 | mention-first / full prompt；候选 anchor 是合法输入；目标数量由模型推断；按集合去重后评测 |
| 2：Projected-2D Localization | 指令、最多 3 个时间 panel，每个为 full-panel + gaze-centered crop 的 paired canvas；提供 GT referent 数量；输出每个 referent 的 panel 与归一化 2D 点 | `expected_count`；不给 GT 身份和位置；正式设置为 `multi_image / paired_crop / paired_canvas_map`；点误差在原始 panel 像素坐标计算 |
| 3：3D Localization | 指令、最多 3 张 target-free 选取的证据帧、camera/gaze/hand 遥测和 broad scene bounds；输出 `points_3d` | 不给候选 anchor、GT ID 或 GT 坐标；v9 prompt 默认复制不同的 gaze hypotheses，无 gaze 才用 hand；nearest-anchor 只在评测阶段执行 |

三项都包含 4,000 个唯一 `scene::row_index`。8 个分区为 scene1/2/3/5 各 800 条，以及 scene4 的 4 个 room 各 200 条；scene4 仍属于一个场景，因此论文称 5 个场景。

Setting 1 的主表是 Referent Recall / Exact / Macro-F1 / Micro-F1，不能拿 any-hit 的 Hit-All 替代 Referent Recall。Setting 2 的 P@100、J@100 等是相应匹配条件下的 corpus F1，不是 precision 或 interaction accuracy。归档给出的 8,519 个可评测 referents，与按 panel 展开的 26,422 行投影 GT CSV 不是同一种计数。

Setting 3 是 measured point-hypothesis 诊断，不应描述为 3D box grounding、3D IoU 或自由重建物体中心；坐标单位按 Unity world unit 描述，除非另外验证为米。

**2. 已有最终结果：补测应该接在哪些表上**

以下均为 0–1 原始比例，直接取自最终归档。

| Setting 1 模型 | Referent Recall | Exact | Macro-F1 | Micro-F1 | Hit-All（辅助） |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B | 0.5169 | 0.3287 | 0.5657 | 0.5399 | 0.7798 |
| Qwen3-VL-8B | 0.3947 | 0.2377 | 0.4804 | 0.4679 | 0.6733 |
| InternVL3-38B | 0.4479 | 0.2592 | 0.5105 | 0.4893 | 0.7215 |

| Setting 2 模型 | T-F1 | Point@100 F1 | Joint@100 F1 | Point@200 F1 | Joint@200 F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B | 0.7348 | 0.2497 | 0.2067 | 0.4399 | 0.3674 |
| Qwen3-VL-8B | 0.7461 | 0.2350 | 0.2013 | 0.4234 | 0.3669 |
| InternVL3-38B | 0.7330 | 0.1228 | 0.0977 | 0.3327 | 0.2701 |

| Setting 3 模型/方法 | Anchor-P | Anchor-R | Anchor-F1 | Exact | Margin-F1@1 | SNE（越低越好） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B | 0.4256 | 0.4399 | 0.4326 | 0.0185 | 0.2503 | 0.1569 |
| Qwen3-VL-8B | 0.4262 | 0.4166 | 0.4213 | 0.0225 | 0.2440 | 0.1550 |
| InternVL3-38B | 0.4337 | 0.3594 | 0.3931 | 0.0480 | 0.2310 | 0.1615 |
| Direct gaze / gaze_copy | 0.4273 | 0.4379 | 0.4325 | 0.0435 | 0.2547 | 0.1519 |

Setting 3 的 Qwen-30B Anchor-F1 相对 direct gaze 仅高约 **0.01 个百分点**，且 Exact、Margin-F1@1、SNE 都更差。补测必须保留 direct gaze 这一行，不能只比较 VLM 之间的名次。

最终有效输出数：Setting 1 的 Qwen-30B/Qwen-8B/InternVL 分别为 3970/3872/3982；Setting 2 为 3998/4000/4000；Setting 3 均为 4000。无效输出仍保留在 4,000 条分母内，不能只统计 valid subset。

**3. 本地准备状况与真正的阻塞项**

本轮逐条检查 `data/*_api_input.csv` 的引用：4,000 条视频、4,000 条原始遥测 JSON、4,000 条事件 JSON 均存在。本地已有三个模型 × 三项 setting 的 9 份 compact 样本预测文件，每份都有 4,000 个唯一 ID。最终评测 GT、候选 anchor 表和最终配置也已保存。

以下正式运行时目录当前不存在，因此还不能直接运行旧 shell wrapper：

- `data/match_eval_qwen3vl30b_mention_first_v3/`
- `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv`
- `exam3_point_grounding/outputs_full_v9_20260709/manifest.csv`

注意两个归档 manifest 的性质：`experiment2_projected_gt_manifest.csv` 只有简化投影 GT，缺少抽帧路径、图像尺寸、gaze crop、`evidence_acceptable` 等正式推理和评测所需字段；`experiment3_evaluable_manifest.csv` 是隐藏 GT，不是模型输入，缺少 prompt、证据帧和遥测。**不能把这些文件整体直接发给 API。**

建议恢复顺序：

1. 优先找回原始运行的 full manifest、实际输入拼图/帧及其时间戳。找不到时，才根据最终配置重建到新的 rebuttal 目录。
2. 从 frozen compact GT 恢复 builder 所需的最小 `*_match_eval.csv`，明确保留 `scene::row_index`、canonical GT 与行顺序；不要用模型预测当 GT。Setting 1/3 的 GT 集合本轮比对完全相同。
3. Setting 2 逐条对比重建投影与归档 panel ID/GT 像素点，并恢复可接受证据 panel 的标记。投影可比不自动代表时间指标可比；必须同时检查 `evidence_acceptable` 和最终 T/P/J 分母。若无法恢复或验证，不把新数据条件的数值冒充为原协议的直接增补。
4. Setting 3 按最终 v9 prompt 重建 target-free evidence manifest；GT 另存，仅供 evaluator。可以用重建后的 gaze_copy 结果是否匹配归档，额外检查选帧/遥测恢复是否一致。
5. 保存 prompt、候选表、图片 bytes、时间戳、GT 和代码的 hash，并冻结同一组输入给所有新增模型。

`V3dMD/prepared/ready.json` 的 ready 只代表其自身预处理流程完成；该目录的审计仍记录 108 个样本存在未解决 referents，因此不能直接以其新版标签替换论文 frozen GT。部分 anchor 文件 bytes/数字字符串格式不同，本轮逐 anchor 按数值检查未发现坐标差异；哈希不同本身不等于坐标变更。

另有一个分组元数据小问题：`scene4_room1::118` 的 Setting 1/3 GT 集合相同，但 compact 文件的 single/multi 标记不同。因此正文 multiplicity 表若需要加行，应从同一份 canonical GT 集合重新派生分组，不要混合复用两种标记。整体 4,000 条主结果不因此改变。

**4. Setting 2 坐标约定：优先处理的协议问题**

正式 prompt 告诉模型：选 `full_panel` 或 `gaze_crop`，输出所选子图内归一化坐标，“do not use coordinates of the whole paired image canvas”。但正式 `paired_canvas_map` 后处理忽略不可靠的 `coordinate_space`，依照 x 值在整张 paired canvas 的左右区域来转换。

这两处约定存在冲突。如果 OpenAI 更严格遵循 prompt 并输出 full-panel 局部坐标，该转换可能再次缩放它，从而把遵循指令的结果变成错误点。这是根据代码得到的风险判断；本轮尚未调用 OpenAI，不能断言它实际会按哪一种方式回答。

具体执行建议：

- 与论文主表直接比较的主结果，保留冻结的 prompt、图片与后处理，并显式记录这一协议限制。
- 同时保留 raw 坐标、raw `coordinate_space`、crop bbox 和 canvas 几何信息；先用不依赖模型答案的人工几何样例检查映射。
- 如报告修正版，可预先定义按声明的坐标空间转换的规则，并对能恢复 raw 输出的全部比较模型统一重评；修正版单独列为 protocol sensitivity analysis。
- 不按 GT 为每条样本挑选“最好的映射方式”，不只给新增模型更换后处理后与旧分数直接比高低。

**5. OpenAI 模型选择与试跑设计**

本轮已尝试访问 OpenAI 官方 models、latest-model、pricing、vision 和 batch 文档；web 工具未返回可用页面，命令行访问也遇到 DNS 解析失败。因此**当前最新可用 model ID、图像输入限制、reasoning 参数、价格、Batch 支持与账户配额均未核验**。本方案不将任何记忆中的或本地快照中的型号称为“截至 2026-10-26 最新”。

建议先确定两个角色：

| 优先级 | 模型角色 | 工作量 | 用途 |
| --- | --- | ---: | --- |
| P0 | 经官方文档及账户实测支持所需多图输入的 OpenAI 通用多模态旗舰 | 3 × 4,000 | 核心 rebuttal 新闭源 baseline |
| P1 | 同期更低成本的通用多模态模型 | 3 × 4,000 | 增加能力/成本层次；仅在 P0 输入与运行已稳定后追加 |
| P2 | 同一旗舰更高 reasoning 档位 | 预先固定的共同子集，或预算足够时全量 | 分析推理预算影响，标明是同模型的额外配置，不称为独立模型 |

正式选择时要核实精确 API model ID、可固定版本/快照、图像数量与尺寸限制、支持的 endpoint、输出约束与 reasoning 参数。账户能访问才算可执行，不以 ChatGPT 产品名称代替 API ID。

每个 setting 先做 **80 条试跑（每个分区固定随机抽取 10 条，固定种子）**，一个模型共 240 次。这是覆盖分区的工程验证集；因分区大小不同，它不是按总体分布抽取的正式性能估计。补充检查 single/multi、entity/location、长指令、无有效 gaze 等边界例子。若改 prompt 或 parser，留好版本，冻结配置后试跑样本也要按最终配置重跑，不能保留调试阶段更好的答案。

试跑解决：图片确实发送且顺序正确、语言与遥测齐全、输出 schema/坐标合法、没有 GT 泄漏、错误可断点恢复，并记录真实 token 用量、延迟、截断率及花费。图像发送方案作为待验证实现选择：Setting 1 复用旧 runner 的时间窗与最多 16 帧采样规则并绑定时间，Setting 2 复用 3 张 paired canvas，Setting 3 复用最多 3 张证据帧。多图输入与原生视频适配差异需要披露。

预算以试跑实测为准：

`每 setting 预计费用 = 4000 × 该 setting 试跑平均单条费用`

`每模型预计费用 = 三个 setting 预计费用之和 + 预留的固定重试费用`

三个 setting 分开估算，不能用一张图片的价钱乘 12,000；旗舰单轮图像输入上界约为 `(16+3+3)×4000 = 88,000` 张，实际帧数会少于上界。官方图像计费和 reasoning 计费尚未核验，不给无依据的美元总价。

若实测平均每条耗时 L 秒、可持续并发 C，在不受 token/请求限额约束时，12,000 次请求的理想耗时约为 `12000×L/C` 秒，另需加预处理、限流、重试和评测时间。例如 L=20、C=10 是约 6.7 小时，仅是算术示例，不是服务性能承诺。Batch 是否可用、折扣和完成窗口均须官方核验；不把未经验证的 Batch 时限放在最后一天的关键路径上。

**6. 复用什么代码，之后需要实现什么**

当前 `scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py` 是 DashScope 的 Qwen3-VL-Plus 单样本接口，包含特定的 `video_url` payload。文件名里的 `api` 不代表已兼容 OpenAI；只改 base URL 不足以得到三个 setting 的 OpenAI runner。

| Setting | 应复用的逻辑 | 正式 evaluator |
| --- | --- | --- |
| 1 | `run_qwen3vl_local_single_event_3d.py` 的 prompt 构建；`run_qwen3vl_plus_api_single_event_3d.py` 的 mention-first 模板和 anchor/alias 逻辑；本地 runner 的原视频抽帧规则 | `scripts/eval/evaluate_local_3d_object_match.py` |
| 2 | `exam2/run_qwen3vl_2d_point_grounding.py` 的图片构建、`build_prompt`、解析及冻结的坐标转换 | `exam2/evaluate_2d_point_grounding.py`，沿用 `coordinate_mode=panel` |
| 3 | 最终 v9 `prompt_text`、证据帧和 `point_parser.py`；复用 `run_qwen3vl_point_grounding.py` 的输出列 | `exam3_point_grounding/evaluate_point_grounding.py` |

下一阶段的实现建议是一个共享 OpenAI client/限流/重试/缓存层，加三个薄适配器。每个样本保存原请求所需内容的 hash、原始响应、解析结果和标准输出文件；不为三个 setting 各复制一套调度器。先将输出适配成旧 evaluator 接受的 schema，再运行对应 evaluator，不另写一套近似指标。

每次 run 应记录：精确 model ID、服务返回的模型名、推理配置、图像 detail/尺寸/顺序、最大输出预算、prompt hash、图片 hash、样本 hash、GT hash、evaluator hash、原始 usage、截断/拒答/解析错误与重试信息。

原模型的输出上限为 Setting 1/2/3 的 1536/768/512 tokens；这些是协议参考，不能未核验就机械映射为 OpenAI 参数。若新模型输出预算涉及 reasoning，必须先实测是否截断并统一冻结，报告与旧模型的预算差别，不能称计算量严格相等。也不要默认所有 OpenAI 模型都接受 temperature=0 或相同 schema 参数。

为了靠近原协议，先使用同样的文字 JSON 指令和 parser；如采用 API 强制 structured output，作为新的适配条件记录，单独报告 valid-output rate。所有正式模型都采用同一已预定的错误策略：transport/429/5xx 有上限退避，记录全部尝试；格式失败若重试，次数和配置固定，按预定规则选结果，不能按 GT 分数选。最终仍缺失、拒答或非法的记录必须保留为失败样本，不从分母删除。

当前工作区有用户已有的代码修改。本轮仅新增方案和审计产物；后续实现应先记录这些修改的 provenance，避免覆盖，并避免将重建 manifest 与历史正式输入无条件等同。

**7. Rebuttal 的分析重点与文字边界**

补测回答三件事：最新闭源 baseline 在集合 referent 恢复上能提升多少；panel 时间证据选择与空间点定位的差距是否仍存在；在 measured-point 协议下是否超越 direct gaze。

- Setting 1 主报 Recall、Exact、Macro/Micro-F1；关注 multi-target 的 Exact，而不只宣传 any-hit。
- Setting 2 同时报 T-F1、Point@100/200、Joint@100/200；T-F1 高不能证明定位准确；同时披露已知目标数和 gaze crop 条件。
- Setting 3 同时报 Anchor-F1、Exact、Margin-F1@1、SNE 和 direct gaze；接近 direct gaze 的结果只能支持本协议下的结论。由于 prompt 明确偏向复制 gaze，不能据此推出 OpenAI 不具备自由 3D 推理能力。
- 如果 OpenAI 大幅提升，应更新“current VLMs”的范围和论断，写明模型版本与日期；不为保留原结论而换指标或挑选子集。
- 如果只完成子集，则对所有旧 baseline 在同一预定义子集重算，并醒目标注 n。不得把子集新模型与旧模型全量主表当作直接可比结果。
- 先报每分区、single/multi 和 location/region 切片，再挑代表性成功与失败案例。location/region 使用归档可复现的 37 anchors / 1,372 interactions，不用旧的 1,461 说法。
- 若要宣称显著，另做与数据采集依赖结构匹配的成对分析；已有归档没有可用的置信区间/显著性证据，不能继承不存在的 p 值。

与此次新模型补测分开处理旧 ablation 解释：Exp1/2 的多项 ablation 同时涉及 prompt、输入与预处理；strict-hand v1 仍有残留手部和大面积旁损遮挡。此次不将其提升为干净的单模态因果结论，也不把重做所有 ablation 放到新模型主结果的前置路径。

可在结果完成后据实填写的英文骨架：

> We extend the benchmark with [verified OpenAI model/version], evaluated on all 4,000 interactions under each of the three settings. We retain the frozen setting-specific inputs, ground truth, and evaluators, and report the API adapter and decoding differences. The added results achieve [metrics]. [State the observed changes in referent-set recovery and temporal versus spatial localization.] For 3D localization, we additionally retain direct gaze as a reference and interpret the results under the measured point-hypothesis protocol.

若输入最终只能重建而不能确认等同，应修改该骨架的 frozen-input 表述，明确报告恢复验证的范围与无法确认的差异。

**8. 立即执行、尽早结束的顺序**

用户已明确要求尽快启动、尽早完成分析，不按 rebuttal 截止时间倒排。截止日仅作为最后提交边界，不作为等待实验的理由。

已完成：配置桃牧中转站的 `/v1/responses`，核对其 `/v1/models` 列出 `gpt-6.1-sol`；使用原协议构建 10 个真实样本请求（Setting 1/2/3 为 4/3/3），全部得到有效 JSON 并通过旧 evaluator；补齐 DNS 失败请求而不重复调用成功样本；完成坐标敏感性、direct gaze、token 用量和同 ID 旧模型子集的离线分析。密钥保存在仓库外并已按最新用户消息替换。

优先的后续步骤：先统一 Setting 2 坐标约定与输入恢复验证，再冻结全量配置；启动一个模型的三项全量时同步运行增量评测与切片分析；获得完整结果后立即形成表格和 rebuttal 文字；第二模型或额外推理配置不阻塞第一模型结果的交付。当前没有启动全量。

本轮 smoke 的 15 行 Setting 2 投影与归档一致，但原 `evidence_acceptable` 标记未归档，时间评测等同性仍待核验。Setting 2 沿用旧坐标后处理的 Joint@100 为 0，按模型声明坐标空间转换的离线诊断为 0.40；Setting 3 三条输出与同输入 direct gaze 完全相同，Anchor-F1 均为 0.40。以上是工程诊断，不是全量估计。

本轮产物：`analysis_outputs/rebuttal_openai_20261026/local_audit.json`（路径/样本/文件校验及限制）、`baseline_metrics_long.csv`（三项现有指标长表）。另在 `metric_validation.json` 中从 9 份 compact 预测独立重算了 45 个主指标，全部与归档汇总一致，并确认各 setting 内跨模型 GT 完全一致。初次审计未产生新增分数；后续新增 smoke 分数与限制保存在 `closed_models/outputs/taomu_gpt61sol_smoke10/SMOKE_REPORT_zh.md`。
