# 六份留出表格冻结输入

2026-09-30，在用户授权后固定 6 份测试 xlsx、13 个此前已检查的 Chunk。
来源文件副本、正文、来源与正文 SHA256、固定提示词、选择性标注和字符偏移均已保存。
提示词逐字来自上一轮 `replay-20260930-table-guidance/system_prompt.txt`，不是本轮结果出来后调整的版本。

`cases.jsonl` 中只有项目交付物与采购节点的 4 个 Chunk 有预先选定的 Gold 草稿：57 实体、30 关系。
其余 9 个 Chunk 标为 `unannotated_not_negative`，不能按负样本计算 Precision。
这是 Codex 草稿，独立人工审核仍待完成；表格中的某些内容描述是否应作为 Material 需业务确认。
六份文档属于同仓库的虚构企业测试材料，模板相近，不能代表真实生产文档或任意表格的泛化质量。

冻结来源不会覆盖原 v1。脚本拒绝覆盖已有输入/模型输出。
模型仅收到正文与固定提示词，未收到 Gold；运行脚本不连接 PostgreSQL 或 Neo4j。

校验：`uv run python -m scripts.replay_graph_table_holdout verify`。
结果及问题明细见 [实际验证报告](../holdout-replay-20260930/README.md)。
