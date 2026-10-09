# 第一批逐条人工裁决：已确认

用户在当前对话明确回复：**第一批按清单裁决**。

本版本记录完整展示的 8 条具体条目，不将批准扩大到其他预测或新增 Gold 口径。

| 条目 | Chunk 对 Fact 的支持 | 模型引用支持 | 人工状态 |
| --- | --- | --- | --- |
| holdout-01-02:relation:33 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:34 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:35 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:36 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:37 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:38 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:39 | SUPPORT | INSUFFICIENT | REVIEWED |
| holdout-01-02:relation:40 | SUPPORT | INSUFFICIENT | REVIEWED |

## 审核进度

- 关系证据逐条人工裁决：8/202；剩余 194 条待审。
- 已审 Fact：4 条 SUPPORT、4 条 INSUFFICIENT；引用：8 条 INSUFFICIENT。
- 266 个预测实体、新增 Gold 类型口径、Normalization 尚待审核。
- 本批是专门选择的异常条目，不将 4/8 当作整批预测的 Evidence Precision。
- 正式 Precision / Recall / F1 和 Evidence Precision 仍未发布。

## 版本与证据

原冻结文档、Gold 草稿、202 条模型预测和父级初审记录均保留。
evidence_review.jsonl 保留全部 202 条，仅对指定 8 条记录人工裁决。
其他条目仍为 PENDING。
human_decision.json 保存原回复、批准范围和所展示清单的哈希。
manifest.json 绑定父级文件、生成脚本和新裁决文件，可重现仅 8 条的状态变化。

[查看原文和第一批全部条目](../six-document-review-draft-v1/第一批裁决-跨行错配.md)
[完整初审清单](../six-document-review-draft-v1/人工裁决清单.md)
[第二批 7 条状态流转待裁决](第二批裁决-状态流转.md)

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m scripts.record_graph_item_adjudication verify
```

本次仅记录本地人工裁决，没有调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。
