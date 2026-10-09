# 修复后四个表格 Chunk 的实际模型抽取

评估状态：INCOMPLETE_API_UNAVAILABLE。API 无返回 4/4；若有 API 不可用，下列为已收到结果的计数，不代表完整 Recall。

Entity Recall：0/37；Relation Triple Recall：0/20。

两端已抽到的 Conditional Recall：0/0；条件逐字严格匹配：0/20。

关系损失：端点漏抽 0，两端已抽到但关系未命中 0。失败 Chunk 4/4。

对照旧两份文档为实体 16/37、关系 5/20。目标名称、类型、方向不变；Chunk 边界与模型运行时间改变，不能将差异全部归因于列头修复。这四块不是独立留出集，也不是完整 59 Chunk 复测。

Gold 是选择性迁移草稿。所有预测写入 review_queue.jsonl 等待独立人工裁决；证据逐字出现不代表语义支持。Precision、F1、Evidence Precision、Normalization 错误率暂不计算。无 PostgreSQL/Neo4j 写入。extractor 仍使用当前生产提示词，不包含离线列抽取规则。
