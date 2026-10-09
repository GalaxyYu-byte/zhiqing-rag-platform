# 第一、二批人工裁决记录

第二批用户回复：**第二批 7 条均判 INSUFFICIENT**。

第一批 8 条裁决完整保留；本版仅新增第二批 7 条具体条目的裁决。

| 条目 | Chunk 支持 | 引用支持 | 人工状态 |
| --- | --- | --- | --- |
| holdout-01-02:relation:33 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:34 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:35 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:36 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:37 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:38 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:39 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:40 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:1 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:2 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:3 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:4 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:5 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:6 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:7 | INSUFFICIENT | INSUFFICIENT | REVIEWED |

## 累计进度

- 关系证据：15/202 已人工裁决，187 条待审。
- 已审 Fact：4 SUPPORT、11 INSUFFICIENT；原引用：15 INSUFFICIENT。
- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。
- 异常批次不是随机样本；不将本批比例当作全部预测的 Evidence Precision。
- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。

## 版本

本版 evidence_review.jsonl 保留全部 202 条，仅第二批 7 条新增人工裁决。
human_decisions.json 保留两次用户回复、明确条目范围和原展示清单哈希。
原模型预测、Gold 草稿、第一版及其生成脚本均未修改。

[第二批原文与全部条目](../human-item-adjudication-v1/第二批裁决-状态流转.md)
[第三批 5 条分类责任人关系待裁决](第三批裁决-分类责任人.md)

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m scripts.record_graph_second_adjudication verify
```

本次仅记录本地人工裁决，未调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。
