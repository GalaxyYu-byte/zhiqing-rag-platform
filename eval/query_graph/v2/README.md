# 在线 GraphRAG 第二版开发组件集

五份虚构文本与五题在实现前冻结，字节 SHA-256 为
`76a3e94c34de01350496d20e2c8bf02aa4d26fb6402c552898ea3012f1470ecc`。
三道正题涵盖步骤→材料→字段、明示别名和必要条件；两道负题检查缺少字段来源和同名材料属于不同流程。
这是开发组件集，不是独立测试集或生产准确率样本。

脚本使用真实 Neo4j，每份文档写入本次随机隔离 namespace，并在结束时定向清理所有节点与 fence、检查残留为零。
不创建正式 PG 知识库或文档，不重建正式索引。授权仓储、查询实体和关系抽取采用固定组件适配器；
词面通道为固定本地字对匹配，向量通道为当前向量的本地余弦排序，因此不能替代真实 HTTP/PG 的端到端验收。

```powershell
# 真实图数据库，不外发模型
.\.venv\Scripts\python.exe scripts\evaluate_online_graph_v2.py --output eval/query_graph/runs/NEW_GRAPH_RUN

# 同一冻结来源使用真实 DashScope 向量化、重排、回答与复核
.\.venv\Scripts\python.exe scripts\evaluate_online_graph_v2.py --models --output eval/query_graph/runs/NEW_MODEL_RUN
```

使用新运行目录，历史失败记录不会被覆盖。字节 SHA 不一致时在模型调用和图写入前拒绝。
同一组问题、来源、向量、候选预算 30 和 Top-K 2 比较 `hybrid`、`one_hop`、`two_hop` 共 15 组。
两跳不会扩大候选或最终证据预算。

结果分别记录 `path_passed`、`answer_evidence_checks_passed` 和未经人工判定的 `semantic_correct=null`。
路径检查要求实际路由、完整两条业务关系、跨文档/namespace 出处、原文 SHA/连续偏移、桥接和完整条件。
回答检查要求最终发布、自己的连续引用、必要文档、字段、条件和明示别名依据；正题的拒绝不能计作通过。
脚本退出成功要求全部模式的路径检查、两跳五题的回答检查和清理完成。基线回答结果如实保留，只作为对照。

本地模型诊断记录草稿、修复选择及复核 JSON，并保存实际提示词 SHA；这些只用于获准的虚构验收，不是已发布回答。
最终事实结论仍须独立逐条对照原文，模型的 `supported` 或引用可定位不能代替该检查。
运行结果、实现哈希和限制见 [验收](ACCEPTANCE.md)。
