# GraphRAG 离线图谱质量评估

快照时间：2026-09-29T03:11:23.433134+00:00。评估范围：10 份文档、59 个原始 Chunk。

## 事实质量（定向 Gold）

- 实体命中 87/165，定向召回 0.5273；投影实体预测 719。
- 有向三元组命中 40/91，定向召回 0.4396；投影关系预测 419。
- 匹配关系中条件不一致 11 条；候选方向错误 0 条；负例语境重叠候选 10 条。
- 分层定向召回：否定语境 0.2059，条件 0.4483，时间语境 0.2222。分层按 Chunk 标签统计；当前 Schema 无独立否定谓词或时间限定字段。
- 当前 Gold 非穷尽且业务人员未签收，Entity/Relation Precision 与 F1、Duplicate/Merge Error Rate、Evidence Precision、Unsupported Fact Rate 均不发布数值。未命中 Gold 的预测在 review_queue.jsonl 中待裁决。

## 投影一致性

- PostgreSQL 原始抽取缓存覆盖 0/59 个 Chunk；原始抽取与 Neo4j 逐条一致性状态：unverified_missing_raw_extractions。
- 检查出 30 条异常记录，按类型计数见 report.json 的 projection_consistency.issue_counts。
- 选中文档的其他批次 Evidence：0。全局孤立 Entity 的检查范围为所选租户和命名空间。
- 所选批次 Evidence 原文位置有效：542/542。这是可定位性检查，不是语义支持率。

## 解释边界

Gold 源文档为虚构企业测试材料。逐字证据与字符位置只能证明可定位，不能替代事实蕴含判断。当前投影由 Neo4j 重建预测；缺少 PostgreSQL 原始抽取缓存，因此无法证明抽取到落图无损。PostgreSQL 和 Neo4j 分别读取，快照之间没有跨库原子事务。
