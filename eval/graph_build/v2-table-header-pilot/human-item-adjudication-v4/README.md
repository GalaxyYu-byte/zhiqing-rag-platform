# 第一至四批人工裁决记录

第四批用户原回复：**全部都判为INSUFFICIENT**。

依据紧接的第四批清单，将“全部”解释为该清单已展示的 14 条。
本版仅新增第四批 14 条裁决，前三批 20 条完整保留。

| 第四批 review_id | Chunk 支持 | 引用支持 | 人工状态 |
| --- | --- | --- | --- |
| holdout-02-01:relation:8 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:9 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:10 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:11 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:12 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:13 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:14 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-01:relation:15 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:7 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:8 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:9 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:10 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:11 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-02-02:relation:12 | INSUFFICIENT | INSUFFICIENT | REVIEWED |

## 累计审核进度

- 关系证据：34/202 已人工裁决，168 条待审。
- 已审 Fact：4 SUPPORT、30 INSUFFICIENT；原引用：34 INSUFFICIENT。
- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。
- 异常批次的比例不作为整批预测的 Evidence Precision。
- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。

## 可复核记录

evidence_review.jsonl 保留全部 202 条；仅指定 14 条新增人工状态。
human_verdicts.jsonl 保存累计 34 条人工裁决。
human_decisions.json 保存四次原回复、批准范围和所展示清单的哈希。
manifest.json 绑定父级文件、草稿原文、生成脚本和所有输出。
原预测、Gold 草稿、前三版裁决和生成脚本保持原样。

[第四批全部条目与原文](../human-item-adjudication-v3/第四批裁决-保留与生成.md)
[第五批 30 条说明字段/步骤关系待裁决](第五批裁决-说明字段与步骤.md)

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m scripts.record_graph_fourth_adjudication verify
```

本次仅记录本地人工裁决，未调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。
