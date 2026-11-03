# VR-TriRef 非物体位置／区域目标核查

## 结论

本次只读取现有评测使用的 anchor 表、最终 GT 映射、场景 API 输入和已有 taxonomy；未修改原始标注、评分规则或模型结果，也未重新运行模型。

- 实际评测 anchor 表共 **199** 个场景化条目，跨场景同名 ID 未合并；与论文的 199 个 anchors 一致。
- 现有显式 taxonomy 标记 **37** 个 non-object / region-like anchors，与论文中的 37 个一致。
- 4,000 条 GT interaction 中，至少包含一个 taxonomy anchor 的有 **1372** 条；其中纯非物体 **109** 条，混合 **1263** 条。
- 旧的 1,461 条位置／区域交互数没有可追溯的 committed taxonomy；当前显式 taxonomy 下的可复核数字是 1,372 条。已有审计还记录了相同 taxonomy 下从旧证据 1,367 到当前 1,372 的 5 条 GT 修复后新增交互。

## 分类依据与待确认事项

- 37 个 non-object 条目直接来自 `analysis_configs/location_region_anchor_classification.csv`，包括 marked location、named room、large structure、outdoor region 和 outdoor boundary；没有按名称自动推断。
- 其余 162 个 anchor 在本输出中记为 `object`，依据是它们出现在实际 anchor 表中且不在显式 non-object taxonomy 中。这是 taxonomy 补集分类；现有 anchor 表没有单独的 `object_flag`，因此建议在最终 rebuttal 前确认是否接受“对象类 = 显式 non-object taxonomy 的补集”这一表述。
- 现有 GT 中没有找不到场景 anchor 表的引用，也没有空 GT；因此本轮没有生成具体的 `pending_confirmation` anchor 或 interaction。

需要你确认的条目：

1. 是否把 taxonomy 中的 `building*`、`warehouse*`、`fence`、`seawall` 继续统一称为 non-object / region-like targets；当前文件的现有 rationale 已明确这样归类，但其中部分语义上更接近大型结构或边界。
2. 是否接受其余 162 个 anchor 使用 taxonomy 补集作为 object 类；若不接受，需要一个独立的 object/non-object 标注源，当前评测文件中没有找到。

## 按场景汇总

| 场景 | anchors | object | non-object | 仅物体交互 | 仅非物体交互 | 混合交互 | 待确认 |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene1 | 22 | 12 | 10 | 464 | 85 | 251 | 0 |
| scene2 | 42 | 39 | 3 | 694 | 0 | 106 | 0 |
| scene3 | 22 | 14 | 8 | 396 | 21 | 383 | 0 |
| scene4_room1 | 19 | 19 | 0 | 200 | 0 | 0 | 0 |
| scene4_room2 | 20 | 20 | 0 | 200 | 0 | 0 | 0 |
| scene4_room3 | 19 | 16 | 3 | 180 | 3 | 17 | 0 |
| scene4_room4 | 28 | 25 | 3 | 194 | 0 | 6 | 0 |
| scene5 | 27 | 17 | 10 | 300 | 0 | 500 | 0 |

全局交互类别为：仅物体 2,628；仅非物体 109；混合 1,263；待确认 0。

## 标注方式与评测使用

- 实际 anchor 表提供的是每个场景化 anchor 的一个 world-coordinate 代表点，以及部分 aliases；`scene1` 使用 `location_x/location_y/location_z`，其他场景主要使用 `x_world/y_world/z_world`。
- 在本次核查到的评测输入和 manifest 中，没有 per-anchor 区域范围、polygon、mesh、碰撞体、box 或人工边界字段。场景本身存在 Unity 几何或视频画面，不等于已经有对应的人工区域边界标注。
- Exp1 直接以 anchor ID 集合进行选择评测；Exp2 把 anchor 的代表点投影到图像平面；Exp3 以隐藏 anchor 的代表点作为 GT，先将模型预测点映射到最近 anchor，再做集合和距离评测。因此 non-object 的空间监督在当前评测中也是代表性点，不是区域面积或边界。

## 真实交互例子

- `scene1::4`（only_non_object）：Touch the entrance of the main building.；GT=["building7"]；object=[]；non-object=["building7"]
- `scene3::4`（only_non_object）：assess that building.；GT=["building1"]；object=[]；non-object=["building1"]
- `scene4_room3::19`（only_non_object）：Inspect the area in front of room1.；GT=["room1"]；object=[]；non-object=["room1"]
- `scene1::226`（mixed）：Tell that person to check the window of this building0.；GT=["person2","building9"]；object=["person2"]；non-object=["building9"]
- `scene1::227`（mixed）：Ask these workers to check the window of the side building.；GT=["person1","person2","building8"]；object=["person1","person2"]；non-object=["building8"]
- `scene1::228`（mixed）：Instruct these workers to inspect the entrance of the side building.；GT=["person1","person2","building8"]；object=["person1","person2"]；non-object=["building8"]
## 预测结果是否可对齐

- GPT-6.1-Sol：三个 setting 的结果位于 `closed_models/outputs/gpt61sol_anchor_full_v1/`、`gpt61sol_projected2d_full_v1/`、`gpt61sol_world3d_legacy_full_v1/`；每个 batch 的 raw JSON 含 `scene` 和 `row_index`，可通过 `scene::row_index` 对齐本表。
- Qwen：现有 compact 预测位于 `paper_experiment_evidence/predictions/exp1/`、`exp2/`、`exp3/` 下的三份 `qwen3_vl_*.csv`；其中 `unified_id` 直接使用 `scene::row_index`，可以与本表逐条连接。
- 本次没有改变任何评分规则，也没有计算新的模型子集指标；这些 CSV 只提供后续按目标类型分组统计所需的连接键和 GT 分类。

## 来源

- Anchor：`paper_experiment_evidence/anchor_tables/*_anchor_table.tsv`。
- Non-object taxonomy：`analysis_configs/location_region_anchor_classification.csv`。
- GT 与 interaction ID：`paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv`。
- 指令文本：`data/*_api_input.csv`。
- 任务与评测说明：`docs/TASK_DEFINITION_THREE_EXPERIMENTS.md`、`exam3_point_grounding/README.md`。
