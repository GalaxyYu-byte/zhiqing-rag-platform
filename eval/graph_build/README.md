# GraphRAG 离线构建评估

完整阶段总结见 [离线测评与修复总结](OFFLINE_EVALUATION_SUMMARY.md)，包含测评范围、问题、修复方法及同口径效果对比。

## 当前目标：Demo 简易验收

验收方式已收敛为构建、查询、完整性和小样本事实检查，见 [Demo 验收结论](DEMO_CHECK.md)。
2026-09-30 实测：构建与直接 Neo4j 查询可用；事实抽查仍有明确错误，质量暂未通过。
历史冻结、Gold 和逐条裁决文件保留作为排查资料，后续无需继续逐批人工确认所有预测。

## 历史评估输入 v1

本目录冻结了图谱构建质量校验的**输入**。`v1/` 包含 2026-09-29 从 PostgreSQL 当前已发布、活动且 READY 的索引批次读取的 10 份文档、59 个完整 Chunk，以及对应的 10 份原始文件。五种格式（PDF、DOCX、Markdown、TXT、XLSX）各两份，覆盖租户 1 的公共、技术、人事、财务、产品五个知识库。选样依据见 `selection-v1.json`；实际冻结时使用的清单副本在 `v1/selection.json`。

## 文件约定

- `v1/documents.jsonl`：每份文档的租户、知识库、修订、活动批次、来源哈希及解析/清洗/分块配置。
- `v1/chunks.jsonl`：Graph Worker 实际使用的逐块正文、定位字段和 `source_refs`，按文档 ID、`chunk_index` 排序。没有向量或先前的图谱抽取缓存。
- `v1/sources/`：冻结的原始文件，按文档 ID 命名；原文件名和路径保存在 `documents.jsonl`。
- `v1/manifest.json`：格式分布、数量、冻结时间及每个文件的 SHA-256。`annotation_status=unlabeled` 表示尚无人工标准答案。

这份快照可以独立验证来源文件、Chunk 正文与文件整体哈希。它固定了图谱构建输入，不代表实体、关系、证据或误抽取已经通过质量评估。后续标注和模型输出应另存版本，不修改 `v1/`。

基于该快照编写的 [Gold 标注草稿](gold-v1/README.md) 位于 `gold-v1/`，包含 59 个原始 Chunk 和 2 个标明来源的模拟 OCR 变体。

## 已验证 Gold 与只读评估

`gold-v1/verification.jsonl` 已将 61 条标注逐条记为 `verified_by_codex`，并通过 `verification_manifest.json` 绑定 Gold 哈希。业务人员的独立复核仍是 `pending`。当前 Gold 只覆盖定向事实，未标出的模型预测不能直接判为假阳性。

`runs/2026-09-29-live/snapshot.json` 是已发布活动批次的 PostgreSQL 与 Neo4j **只读**快照。`runs/2026-09-29-live/evaluation/` 包含 `report.md`、完整 `report.json` 和待裁决的 `review_queue.jsonl`。快照检查文档、Chunk、批次、版本、实体、Fact、Evidence、端点、重复及孤立节点；Gold 对照采用实体类型与规范名称、有向关系三元组、条件及证据原文位置。

进一步拆分的六层评估在 `runs/2026-09-29-live/evaluation-detailed/`。新增 `gold_relation_outcomes.jsonl`（91 条 Gold 关系逐条端点/三元组/条件判定）、`evidence_review.jsonl`（419 个 Fact–Evidence 对）及 `normalization_review.jsonl`（108 个身份对候选）。裁决口径见 [复核指南](REVIEW_GUIDE.md)。原 `evaluation/` 保留为初次报告，细分指标以 `evaluation-detailed/` 为准。

针对实体 87/165 和关系 40/91 的归因结果在 `runs/2026-09-29-live/recall-loss/`：`report.md` 汇总实体类型、零输出 Chunk、名称与类型候选差异，以及关系漏召回的 source/target 端点分解；`entity_outcomes.jsonl`、`relation_loss_outcomes.jsonl` 可逐条回查。可在仓库根目录用以下命令对同一快照重算（使用新的输出目录）：

7 个有正向 Gold 的零输出 Chunk 的历史任务、缓存及当前图谱复查见 [零输出排查记录](runs/2026-09-29-live/zero-chunk-investigation/README.md)。原始抽取记录未留存；向外部模型发送这 7 个 Chunk 的隔离复现尚未获自动审批，故未执行。

```powershell
cd D:\code\zhiqing-rag-platform
uv run python scripts/analyze_graph_recall_loss.py --snapshot eval/graph_build/runs/2026-09-29-live/snapshot.json --output eval/graph_build/runs/2026-09-29-live/recall-loss-recheck
```

在仓库根目录可对同一快照重算评估：

```powershell
cd D:\code\zhiqing-rag-platform
uv run python scripts/evaluate_graph_quality.py --snapshot eval/graph_build/runs/2026-09-29-live/snapshot.json --output eval/graph_build/runs/2026-09-29-live/evaluation-recheck
```

需要重新采集实时状态时指定**新的**文件与输出目录：

```powershell
cd D:\code\zhiqing-rag-platform
uv run python scripts/capture_graph_eval.py --output eval/graph_build/runs/new-run/snapshot.json
uv run python scripts/evaluate_graph_quality.py --snapshot eval/graph_build/runs/new-run/snapshot.json --output eval/graph_build/runs/new-run/evaluation
```

本次实体定向召回为 87/165，关系有向三元组定向召回为 40/91。仅在两个 Gold 端点实体均抽到的条件下，关系类型与方向命中 40/40，条件也一致为 29/40。这不是全量实体或关系质量分数。报告中的否定、条件、时间分层按 Chunk 标签统计，不能解释为对应语义均已正确抽取。所选批次的 PostgreSQL 实体与 Fact 计数和 Neo4j 一致，542 条 Evidence 均可逐字定位；检查还发现 30 个**所选租户和命名空间范围**的孤立实体，不能归因于这 10 份文档。原始抽取缓存为 0/59，故抽取结果到 Neo4j 的逐条无损性尚无法验证。Entity/Relation Precision 与 F1、实体误合并率、语义 Evidence Precision 等数值需先裁决未标注预测及别名身份对。PostgreSQL 与 Neo4j 的读取也不是跨库原子快照。

## 校验与重新冻结

在仓库根目录执行：

```powershell
uv run python scripts/freeze_graph_eval.py verify --output eval/graph_build/v1
```

如需创建新版本，先复制并修改选样清单，然后指定**新**输出目录：

```powershell
uv run python scripts/freeze_graph_eval.py freeze --selection eval/graph_build/selection-v2.json --output eval/graph_build/v2
```

冻结命令仅从数据库读取，要求文档仍已发布、当前批次 READY、修订和文件哈希与选样清单一致，且 Chunk 数量和正文哈希完整。已有输出目录不会被覆盖。重新运行会形成新的快照版本；不要用它替换 v1。
