# GraphRAG 图谱构建 Gold 草稿 v1

本数据集基于已冻结的 `../v1/` 输入：59 个原始 Chunk 全部纳入，并增加 2 个**明确标记为模拟 OCR 噪声**的变体，共 61 条样本。原始文档是仓库中用于测试的虚构企业材料，不是真实企业生产文档；模拟噪声也不是实际 OCR 输出。

`cases.jsonl` 每行一个样本，包含完整 `content`、`entities`、`relations`、`evidence`、`hard_negatives` 和 `coverage_tags`。目前有 180 个实体、104 条关系和 21 条显式负例。每条证据的 `start` 包含起始字符，`end` 不包含结束字符，均针对该样本的 `content`。OCR 变体保留 `parent_case_id` 和逐字替换记录，可以与原始 Chunk 对照。

## 标注口径

- 正向实体和关系只使用当前 `graph-extractor-v1` 的类型与端点约束。实体名称、关系证据和非空条件必须逐字出现在**同一个 Chunk** 中。跨句关系只在相邻句或相邻步骤共同提供证据时标注。
- 标注集中关注各文档自身的流程、责任、状态、材料字段和规则。重复的通用套话原则上不纳入正向召回目标；四个标为 `boilerplate_consistency` 的 Chunk 保留同一类经办责任，用于检查跨文档一致性。因此**未列出的事实不能直接计作模型误抽取**。这套 Gold 草稿适合先测目标事实召回、类型/端点、证据有效性和明确负例的误抽取，不适合直接计算全量事实精确率。
- `hard_negatives` 是原文明确否定或不支持的推论，不应反向生成为肯定关系。当前 Schema 没有否定、时间或 `USES` 专用关系；`negation`、`time` 标签及 `condition_text` 用于分层检查，不能被误读为新增谓词。
- `similar_entity_alias` 中的候选称呼带 `status=ambiguous`，不强制合并实体。PDF 的换行伪影标作 `layout_noise`，与两个 `synthetic_ocr_noise` 变体分开统计。
- `review_status=draft_awaiting_human_review`：这是一版可检查的标注草稿，尚未由业务人员双人复核，不能宣称为最终人工 Gold。复核修订应另建版本，不改动本版文件。

## 验证状态

61 条样本已在 `verification.jsonl` 中逐条标记为 `verified_by_codex`。验证覆盖冻结 Chunk 归属、原文或已声明 OCR 扰动、实体/关系 Schema、证据原文及字符位置、关系端点与方向、条件和否定句；验证结果通过 `verification_manifest.json` 的哈希绑定到本版 Gold。**业务人员人工复核仍为 `pending`**，因此原有 `review_status=draft_awaiting_human_review` 保持不变。这两个状态分别表示技术与 Codex 复核、以及独立业务签收。

## 文件与验证

- `annotation_plan.py`：逐块编写的事实、负例和难例标签，便于审阅标注选择。
- `cases.jsonl`：可直接交给离线评测器读取的结构化样本。
- `manifest.json`：冻结输入、标注方案、JSONL 的 SHA-256 与样本统计。
- `verification.jsonl`、`verification_manifest.json`：逐样本验证状态和校验范围，不冒充业务人员签收。

在仓库根目录执行：

```powershell
cd D:\code\zhiqing-rag-platform
uv run python scripts/build_graph_gold.py verify
uv run python scripts/verify_graph_gold_review.py verify
```

验证会重新编译所有样本，检查冻结输入版本、逐字证据、字符位置、关系端点、条件和文件哈希；不调用模型或数据库。后续正式评估应记录模型、提示词、抽取规则版本与本数据集 ID，并把预测结果写在另一个目录。
