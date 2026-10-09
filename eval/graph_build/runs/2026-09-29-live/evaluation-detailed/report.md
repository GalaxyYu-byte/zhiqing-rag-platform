# GraphRAG 离线图谱质量评估

快照时间：2026-09-29T03:11:23.433134+00:00。评估范围：10 份文档、59 个原始 Chunk。

## ① Entity Recall

实体命中 87/165，定向召回 0.5273。

## ② Relation Triple Recall

有向三元组命中 40/91，定向召回 0.4396。

## ③ Relation Conditional Recall

仅在 Gold 的 source 和 target 实体都已抽到时计算：关系类型与方向命中 40/40，条件也一致 29/40。
逐条端点、三元组和条件判定见 gold_relation_outcomes.jsonl。

## ④ Evidence Support

419 个 Fact–Evidence 对中，14 个与已验证 Gold 的有向三元组、条件和证据原文完全一致，可标为 SUPPORT；405 个待语义复核。
人工裁决使用 SUPPORT / CONTRADICT / INSUFFICIENT；待复核不等于 INSUFFICIENT。逐条记录见 evidence_review.jsonl。

## ⑤ Normalization

生成 21 对可能应合并未合并、86 对可能误合并、1 对 Gold 中身份不明的别名。
当前没有经裁决的 SAME / DIFFERENT 身份对，Duplicate Entity Rate 和 Merge Error Rate 暂无数值。逐对记录见 normalization_review.jsonl。

## ⑥ 未标注预测裁决与最终指标

1011 个未命中定向 Gold 的实体或关系预测需要人工裁决（实体 632、关系 379），见 review_queue.jsonl。
Gold 仍非穷尽且未获业务签收。只有补齐所有真实事实、裁决预测与身份对后，才计算最终 Precision / Recall / F1；当前只报告上述定向召回。

## 挑战样本分层

按 Chunk 标签的关系定向召回：否定语境 0.2059，条件 0.4483，时间语境 0.2222。这不是各语义属性的正确率；当前 Schema 无独立否定谓词或时间限定字段。

## 投影一致性

- PostgreSQL 原始抽取缓存覆盖 0/59 个 Chunk；原始抽取与 Neo4j 逐条一致性状态：unverified_missing_raw_extractions。
- 检查出 30 条异常记录，按类型计数见 report.json 的 projection_consistency.issue_counts。
- 选中文档的其他批次 Evidence：0。全局孤立 Entity 的检查范围为所选租户和命名空间。
- 所选批次 Evidence 原文位置有效：542/542。这是可定位性检查，不是语义支持率。

## 解释边界

Gold 源文档为虚构企业测试材料。逐字证据与字符位置只能证明可定位，不能替代事实蕴含判断。当前投影由 Neo4j 重建预测；缺少 PostgreSQL 原始抽取缓存，因此无法证明抽取到落图无损。PostgreSQL 和 Neo4j 分别读取，快照之间没有跨库原子事务。
