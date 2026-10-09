# 零图谱输出 Chunk 排查

检查对象是冻结集 `v1` 中 9 个在 Neo4j 没有任何 Evidence 或 Entity 提及的原始 Chunk；其中 7 个有正向 Gold，共 23 个实体和 16 条关系。检查均未修改 PostgreSQL 或 Neo4j。

| Gold Case | 文档 ID | Gold 实体 | Gold 关系 | 当前正文哈希 |
| --- | ---: | ---: | ---: | --- |
| gb-35-00 | 35 | 8 | 7 | 与冻结集一致 |
| gb-35-01 | 35 | 2 | 1 | 与冻结集一致 |
| gb-45-00 | 45 | 5 | 4 | 与冻结集一致 |
| gb-56-00 | 56 | 1 | 0 | 与冻结集一致 |
| gb-71-07 | 71 | 1 | 0 | 与冻结集一致 |
| gb-73-03 | 73 | 1 | 0 | 与冻结集一致 |
| gb-76-01 | 76 | 5 | 4 | 与冻结集一致 |

另有 `gb-63-00`、`gb-71-00` 没有正向 Gold，零输出本身不构成已知漏召回。

## 已确认的事实

- `history-with-keys.json` 是 PostgreSQL **只读**任务历史：上述 7 份文档的 Graph 任务最终均为 `SUCCEEDED`，对应投影均为 `READY`。文档 73 的任务先后有两次 `GRAPH_PROCESSING_FAILED`，第三次成功；失败记录没有 Chunk ID 或具体异常。任务只记录文档级实体和关系总数。
- 7 个任务的 `payload_keys` 均只有 `pipeline`。当前 `GraphQueue.checkpoint()` 每处理一块会写 `next_chunk_index`、`entity_keys`、`relation_keys`；当前 `save_extraction()` 会写 `_graph_extraction_cache`。现存 59 个 Chunk 的该缓存覆盖率为 **0/59**。所以**历史记录与当前代码的留痕行为不一致**，不能拿当前代码假定历史每块的原始响应。
- 任务就绪时间为 2026-09-28 08:24–08:45 UTC；当前 `graph.py`、`graph_queue.py` 文件的修改时间约为同日 12:57–12:58 UTC。这是代码后来变化的线索，文件时间本身不能证明旧版实现细节。
- `current-snapshot.json` 再次只读采集后，Neo4j 的 Evidence 键集合与原评估快照完全相同，419 个 Assertion、542 个 Evidence 计数也相同；缓存仍为 0/59。9 个 Chunk 目前依然没有 Evidence。
- 涉及零输出的 7 份文档投影总数均为正，且同文档的其他 Chunk 有 Evidence，说明不是整篇文档未构建。当前代码的完成路径允许合法的空抽取结果；本地模拟空模型响应的测试证实它会被接受。但历史原始模型响应、逐 Chunk 校验结果和写入前后的记录均不存在。因此**无法从现有记录断定**这 7 个有正向 Gold 的 Chunk 当时是模型返回空、旧 Worker 未写入、还是后续投影丢失。文档级 PostgreSQL/Neo4j 总数一致不能替代逐 Chunk 对照。

## 隔离复现状态

用户明确授权后，已于 2026-09-30 运行 `scripts/replay_graph_zero_chunks.py`：它只读取冻结正文，用项目当前 `deepseek-flash` 配置运行同一 `extract_chunk()`，逐次保存原始模型响应和验证后的结果；不调用图写入或 PostgreSQL 更新。结果及逐次诊断见 [replay-20260930/README.md](replay-20260930/README.md)。

即使得到授权并成功复现，那也是**当前模型与当前代码**的行为，不能直接还原 2026-09-28 的原始运行。下一次正式图谱构建应保留每块模型响应、验证结果、写入前后计数、模型与提示词哈希及 Chunk ID，才可把同类问题定位到具体阶段。
