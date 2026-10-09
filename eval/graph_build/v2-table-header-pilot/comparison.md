# 表格抽取修复的阶段性结论

后续已完成固定提示词的 13 Chunk 留出验证，发现 3 个 Chunk 失败及多种事实质量问题，
当前实验提示词暂不采用；详见 [留出验证报告](holdout-replay-20260930/README.md)。
以下记录调参阶段，不能代替留出结果。

## 实际抽取结果

2026-09-30，在已授权的同一 4 个新表格数据 Chunk 上运行当前配置的 DeepSeek。
仅发送正文，不发送 Gold；只保存本地响应，不写 PostgreSQL/Neo4j。
Gold 保留原 37 个实体目标、20 个关系目标的名称、类型、方向、条件，重新绑定新 Chunk 和证据偏移。
共有 38 个端点实体记录；额外的跨块辅助“主管”不增加实体 Recall 分母。

| 运行 | Entity Recall | Directed Triple Recall | 两端已抽到时的 Conditional Recall |
| --- | --- | --- | --- |
| 原两份文档的已发布基线 | 16/37（43.24%） | 5/20（25.00%） | 未在此重新计算 |
| 修复列头，当前生产提示词 | 17/37（45.95%） | 2/20（10.00%） | 2/2 |
| 修复列头，表格提示词实验 | 36/37（97.30%） | 19/20（95.00%） | 19/19 |

第一轮受限环境的连接失败保存在 `replay-20260930/`，状态为 API 不可用，Recall 为 null。
这是请求失败，不能算成模型质量降到 0%。联网成功后的两轮目录分别为
`replay-20260930-authorized/`、`replay-20260930-table-guidance/`。

## 损失定位

当前生产提示词运行漏掉 18 条目标关系，全部有至少一个严格 Gold 端点未命中。
这次的具体问题是：

- 两个人事表格 Chunk 用“事项”列的主题名称建 Step，没有抽出另一列明确写出的具体动作。
- `v2-89-01` 首次输出达到 4096 completion token，finish_reason=length；重试未提取保管 Rule。
- `v2-89-02` 恰好提取 25 个实体，最后“临时草稿”行的目标实体未提取。预算与逐行顺序很可能影响覆盖，但这不是受控实验的因果结论。

表格提示词实验明确区分事项主题和执行动作，优先保留明确关系两端，保留条件和否定，
并要求不要将核查触发情形当成对象当前状态。只在实验进程中替换提示词，生产抽取源码没有改动。
本轮四块均一次成功，无输出截断。预测 82 个实体、49 条关系。

严格目标剩下 1 个实体、1 条关系未命中：原 Gold 将“主管与行政”作为一个 Actor；
模型分别抽出“主管”“行政”，并各自连接到“确认配置”。两个真实角色和动作均已抽到，
应归为联合角色建模口径待裁决，不能直接解释成模型不知道负责人。
原 Gold 保持不变，也没有把两个不同角色自动当作别名合并。

## Evidence、Normalization 和最终指标

Codex 对表格实验的 49 条预测逐条初审，查看本 Chunk 列头及原始整行：
47 SUPPORT、0 CONTRADICT、2 INSUFFICIENT。
两条证据不足关系为“岗位资料/资料 GOVERNED_BY 本表不存放个人记录”。
原文约束的是本表的存放内容，不能推出这两种资料对象都受该限制。
这两条没有写图谱，均保留为审核候选。

初审文件：`replay-20260930-table-guidance/codex_evidence_review.jsonl`。
每条记录含模型引文、完整表格行、列头、理由，并绑定抽取结果与目标集哈希。
独立人工裁决仍为 PENDING，47/49 不能作为正式 Evidence Precision。

未命中选择性 Gold 的预测不自动计为 FP。正式 Precision/F1、Unsupported Fact Rate、
Duplicate Entity Rate、Merge Error Rate 均暂不发布。跨 Chunk 的角色重复出现也不能直接
算 Duplicate；联合角色拆分首先涉及实体建模边界，需要业务确定规范。

## 可重跑与下一阶段

```powershell
uv run python -m scripts.evaluate_graph_table_replay verify
uv run python -m scripts.evaluate_graph_table_replay report --output eval/graph_build/v2-table-header-pilot/replay-20260930-authorized
uv run python -m scripts.evaluate_graph_table_replay report --output eval/graph_build/v2-table-header-pilot/replay-20260930-table-guidance
uv run python -m scripts.review_graph_table_guidance
```

迁移、漏抽拆解、条件误配、API 请求失败与质量失败区分的测试，加上响应记录测试：5 passed；Ruff 通过。

下一阶段须用未参与提示词调整的表格运行实际模型抽取，并由人工审查实体边界、49 条候选和证据支持。
本轮不是独立留出评测，只运行每种提示词一次；模型运行时间不同，不能将全部差异归因于某一修改。
调参阶段，6 份留出表格当时只完成本地分块/行内候选检查。随后用户已授权实际模型评测，结果见本文开头的留出验证报告。
满足这些验收条件后再考虑采用实验提示词、重建批次及 PostgreSQL→Neo4j 投影一致性验证。
现有图谱、原冻结 v1 和完整 59 Chunk 的 87/165、40/91 基线仍未改变。
