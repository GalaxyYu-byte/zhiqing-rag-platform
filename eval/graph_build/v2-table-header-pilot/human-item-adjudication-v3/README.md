# 第一至三批人工裁决记录

第三批用户回复：**第三批 5 条均判 INSUFFICIENT**。

前两批 15 条完整保留；本版本只新增第三批 5 条具体关系的裁决。

| 第三批 review_id | Chunk 支持 | 引用支持 | 人工状态 |
| --- | --- | --- | --- |
| holdout-04-02:relation:1 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-04-02:relation:2 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-04-02:relation:3 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-04-02:relation:4 | INSUFFICIENT | INSUFFICIENT | REVIEWED |
| holdout-04-02:relation:5 | INSUFFICIENT | INSUFFICIENT | REVIEWED |

## 累计审核进度

- 关系证据：20/202 已人工裁决，182 条待审。
- 已审 Fact：4 SUPPORT、16 INSUFFICIENT；原引用：20 INSUFFICIENT。
- 266 个预测实体、新增 Gold 类型口径、Normalization 仍待审核。
- 批次专门选取异常关系，不以已审批次的比例推算整体 Evidence Precision。
- 正式 Precision / Recall / F1、Evidence Precision 仍未发布。

## 版本与证据

evidence_review.jsonl 保留全部 202 条；仅指定 5 条新增人工状态。
human_verdicts.jsonl 保存累计 20 条人工裁决。
human_decisions.json 保存三次原回复和条目范围。
manifest.json 绑定父级文件、草稿原文、脚本和所有输出哈希。
原模型预测、Gold 草稿、前两版裁决及生成脚本保持原样。

[第三批原文及全部条目](../human-item-adjudication-v2/第三批裁决-分类责任人.md)
[第四批 14 条保留与生成关系待裁决](第四批裁决-保留与生成.md)

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m scripts.record_graph_third_adjudication verify
```

本次仅记录本地人工裁决；未调用模型、修改生产抽取或写入 PostgreSQL/Neo4j。
