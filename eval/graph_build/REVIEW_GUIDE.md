# GraphRAG 六层事实质量复核指南

评估固定在 `v1/` 输入、`gold-v1/` 标注和 `runs/2026-09-29-live/snapshot.json` 只读系统快照。详细报告在 `runs/2026-09-29-live/evaluation-detailed/`。对裁决文件进行修改前，先复制到新的人工复核目录，保留原始输出与快照的哈希。

## ①②③ 召回的三个分母

1. **Entity Recall**：165 个原始 Chunk 中的 Gold 实体，按同一 Chunk 的实体类型与规范名称匹配；目前 87/165。
2. **Relation Triple Recall**：91 条 Gold 关系，要求同一 Chunk 的 `source`、`relation_type`、`target` 和方向全匹配；目前 40/91。
3. **Relation Conditional Recall**：91 条 Gold 关系中，先选 source 与 target 的实体类型和规范名称都已抽到的 40 条。其关系类型和方向为 40/40；进一步要求 `condition_text` 相同则为 29/40。详见 `gold_relation_outcomes.jsonl`。它衡量的是“实体已找对后还能否连对关系”，不替代整体关系召回。

两个不同 Gold 实体即使规范名称相同，也要有两个相应的预测实体才算端点均抽到；一条关系引用同一个 Gold 实体作为两个端点时只需一个。

## ④ Evidence Support

`evidence_review.jsonl` 每行对应一个 Fact–Evidence 对。核对原始 Chunk、方向、关系类型、条件、否定和时间语境，再填写 `verdict`：

- `SUPPORT`：该证据足以支持**完整**事实，包括方向与条件。
- `CONTRADICT`：该证据明确否定或推翻该事实。
- `INSUFFICIENT`：证据未推翻事实，但不足以推出该事实；只同时出现两个实体也属于此类。

`verdict=null` 表示**未裁决**，不可计作 `INSUFFICIENT`。目前 14 条与 Codex 验证过的 Gold 关系、条件和原文证据完全一致，自动标记为 `SUPPORT`，其余 405 条待复核。自动标记仍须业务人员签收。原文位置有效只证明可以定位，不证明语义支持。

## ⑤ Entity Normalization

`normalization_review.jsonl` 的 `identity_verdict` 填 `SAME` 或 `DIFFERENT`，判断两处提及在该知识库和文档语境中是否指向同一概念。`system_merged=true` 且裁决为 `DIFFERENT` 是**误合并**；`system_merged=false` 且裁决为 `SAME` 是**漏合并**。若其中一个预测实体缺失，`system_merged=null`，只能记录别名判断，不纳入合并错误率。

这 108 对来自“同一实体键跨 Chunk”“同一规范名称却不同实体键”与 Gold 的显式模糊别名，仅是候选集。即使全部裁决，得到的率也只能称为**候选对错误率**，不能冒充所有实体对的总体错误率。

## ⑥ 未标注预测与最终指标

`review_queue.jsonl` 含 632 个实体、379 条关系的未命中定向 Gold 预测。逐条复核时，可将预测裁决为 `CORRECT`、`INCORRECT` 或 `DUPLICATE`，并说明对应 Chunk 与证据。`CORRECT` 预测要补入下一版 Gold；同时逐块检查**系统未预测但真实存在**的实体和关系。只有 59 个原始 Chunk 的 Gold 穷尽性得到业务签收，且未标注预测全部裁决，才计算最终 Precision、Recall、F1。当前报告将这些最终指标设为 `null`，以免把定向 Gold 的未标注内容误算成假阳性。

两个模拟 OCR 样本没有对应的实时系统预测；若要评估 OCR 鲁棒性，须单独运行抽取并记录模型版本与输出。
