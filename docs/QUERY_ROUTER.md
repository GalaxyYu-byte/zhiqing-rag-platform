# 在线 Query Router

`QueryRouterService` 按确定性规则生成执行计划，不额外调用 LLM。
`RagAnswerService` 已接入该计划，`POST /chat/answer` 依次执行：

```text
KB / 文档权限校验 → 会话归属与 KB 范围校验 → 近期用户问题
    → DeepSeek Analyzer → Query Router
        → hybrid：Dense + BM25 → 融合 / 按需重排 → 基于引用生成回答
        → hybrid_graph：实体与活动图谱证据检查 → 三路召回 → 融合 / 重排 → 生成
        → clarify：返回澄清问题
        → structured：返回暂不支持精确统计 / 全量查询的说明
        → none：跳过检索，调用现有问答模型回应纯闲聊
```

澄清、能力说明及闲聊响应都会沿用现有会话持久化流程，sources 为空，reranked 为 false。
KB/文档权限检查仍在所有路由之前执行；本次未新增不选择知识库的通用聊天入口。

## 六种情况

| Analyzer 情况 | 当前行为 |
| --- | --- |
| hybrid | 使用原问题执行 Dense + BM25，普通查询权重 0.5/0.5，精确查询 0.3/0.7 |
| hybrid_graph | 对全部原文实体候选做唯一名称/别名匹配，并检查授权当前文档版本内是否存在活动关系证据；匹配完整时增加 Graph 分支，权重 0.4/0.3/0.3 |
| clarify | 返回 Analyzer 的澄清问题，不调用召回、重排或回答生成模型 |
| structured | 当前没有统计/全量查询执行器，明确返回能力限制，不给出推测总数、总额或完整清单 |
| none | 不调用召回或重排；使用现有 DashScope 问答模型进行无来源引用的简短闲聊 |
| status=degraded | 无历史时保留原问题走基础 Hybrid，忽略模型抽取和路径；有历史时返回澄清，避免错误指代 |

Router 会重算分析特征对应的路径和固定权重，不直接采用模型建议中的权重。
必要澄清优先于统计、闲聊、图谱和普通召回；Query Analyzer 的原始建议路径仍保存在
`routing.requested_path`，实际执行路径返回为 `routing.path`。

## 图谱检查与降级

图谱检查要求调用方提供权限服务已经求交集的具体 `doc_ids`。
API 在 Analyzer 之前完成授权检查，模型不能新增或扩大知识库和文档范围。
内部 CLI 若只提供 KB 而未提供具体授权文档范围，则不会启动 Graph，退回基础 Hybrid。

覆盖检查先查询 PostgreSQL 当前有效文档版本，再查询 Neo4j：

- 使用规范化名称或别名精确匹配，不使用子串猜测节点 ID。
- 限定 seed、claim、chunk、doc 属于目标 KB，文档版本与 PostgreSQL 当前版本一致。
- 仅使用 `graph_status=ACTIVE` 且与 `active_graph_version` 相符的关系证据。
- 每个名称最多取两个不同 UID，多个匹配返回同名实体澄清；缺少任何候选匹配则降级。
- Graph 检索使用确认后的 `seed_entity_uids`，不会再次用问题子串扩大种子范围。
- 图谱结果映射回 PostgreSQL 时再次限定 KB、doc_ids、当前版本和未删除/索引完成状态。

覆盖检查只证明授权文档中有相关实体及活动关系证据，不证明已经覆盖问题要求的关系链。
最终生成依旧依据检索到的原文，并保留来源引用；若 Graph 返回空结果，也会降级。

```dotenv
QUERY_ROUTER_GRAPH_TIMEOUT_SECONDS=2
```

此配置分别限制覆盖检查和实际 Graph 召回，每阶段默认 2 秒。
图谱连接异常、超时、实体缺失、当前版本没有证据都会退回 Dense + BM25。
图谱同名歧义返回澄清。取消传播给调用方；若取消发生在 PostgreSQL 图分支读取期间，
先回滚只读事务以恢复 Session，再处理超时或取消，不并发使用同一 AsyncSession。

## 上下文与待接入能力

现有会话检查通过之后，API 按时间顺序加载近期最多 6 条用户问题供 Analyzer 分析。
历史助手回答不会传给模型，避免把过往生成内容当作事实，或重用已失去权限的文档证据。

Query Rewrite、Multi Query 和 HyDE 执行器尚未接入。Router 在 `deferred_features` 中
说明改写/多查询建议未执行；无上下文依赖时按原问题进行一次检索。
需要语义补全的指代问题即使有历史实体候选，也先要求用户写出完整对象和条件，
返回 `context_rewrite_required`；不会把实体列表附加到问题上伪装成已经可靠改写。
HyDE 继续关闭，元数据继续保留原文软约束，不直接生成 SQL 或硬过滤。

## 响应新增字段

`/chat/answer` 的已有字段保留，新增 `routing`，并扩展 timing：

```json
{
  "routing": {
    "requested_path": "hybrid_graph",
    "path": "hybrid",
    "action": "retrieve",
    "reason": "graph_not_covered",
    "analyzer_status": "ok",
    "graph_degraded": true,
    "warnings": ["实体未完整匹配到当前授权文档的活动图谱事实，使用基础混合检索。"],
    "deferred_features": [],
    "router_version": "query-router-v1"
  },
  "timing": {
    "analysis_ms": 1800,
    "routing_ms": 15,
    "retrieval_ms": 220,
    "generation_ms": 500,
    "total_ms": 2535
  }
}
```

action 为 retrieve / clarify / unsupported / chat。
`graph_degraded=false` 也可能表示没有请求 Graph，不能将该字段单独解读为图谱执行成功。
模型名仍表示配置的回答生成模型；澄清和能力说明分支没有执行回答生成，token_count=0。
total_ms 包含 Analyzer、Router、检索和生成，未包含 API 外层权限校验及消息落库耗时。

主要 reason：

| reason | 含义 |
| --- | --- |
| hybrid_selected / graph_covered / no_retrieval_required | 普通检索 / 图增强检索 / 无检索 |
| clarification_required / context_rewrite_required | 主体或约束不明确 / 需要未接入的上下文改写 |
| structured_not_supported | 当前统计/全量查询能力未实现 |
| analysis_degraded / analysis_degraded_with_history | 分析失败走基础检索 / 带历史时请求完整问题 |
| graph_scope_required / graph_not_covered / graph_entity_ambiguous | 缺授权文档范围 / 缺图谱证据 / 同名歧义 |
| graph_check_timeout / graph_check_unavailable | 图谱预检查超时或不可用 |
| graph_execution_empty / graph_execution_timeout / graph_execution_unavailable | 实际 Graph 召回为空、超时或不可用 |

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B -m scripts.check_query_router_graph
```

自动测试覆盖各路由是否真的跳过不需要的服务、授权范围传递、同名与缺失实体、
超时和取消、Graph 后续执行失败，以及会话历史加载前的归属和权限检查。
LLM 和检索测试使用注入的客户端与服务，不产生网络调用。

第二条命令需要已配置且可连接的 Neo4j，只对两条 Cypher 执行 EXPLAIN，验证编译，
不执行图检索或修改数据。真实关系覆盖与最终回答质量仍需使用授权业务语料验证。

2026-09-17 验证：项目全量 246 项测试通过；两条 Cypher 在已配置的 Neo4j 上
通过只读 EXPLAIN 编译检查。本轮未运行真实数据库检索与回答生成的端到端评测。
