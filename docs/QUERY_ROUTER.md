# 在线 Query Router

`QueryRouterService` 按确定性规则生成执行计划，不额外调用 LLM。
`RagAnswerService` 已接入该计划，`POST /chat/answer` 依次执行：

```text
KB / 文档权限校验 → 会话归属与 KB 范围校验 → 近期用户问题
    → DeepSeek Analyzer → 澄清或独立问题 → 按需更新分析 → 最终 Query Router（只执行一次）
        → hybrid：按需 Multi Query / HyDE → Dense + BM25 → 融合 / 按需重排 → 基于引用生成回答
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
| hybrid | 使用最终问题执行 Dense + BM25，HyDE 时仅 Dense 向量输入使用假想文档；普通查询权重 0.5/0.5，精确查询 0.3/0.7 |
| hybrid_graph | 对全部原文实体候选做唯一名称/别名匹配，并检查授权当前文档版本内是否存在活动关系证据；匹配完整时增加 Graph 分支，权重 0.4/0.3/0.3 |
| clarify | 返回 Analyzer 的澄清问题，不调用召回、重排或回答生成模型 |
| structured | 当前没有统计/全量查询执行器，明确返回能力限制，不给出推测总数、总额或完整清单 |
| none | 不调用召回或重排；使用现有 DashScope 问答模型进行无来源引用的简短闲聊 |
| status=degraded | 仅按当前问题的明显指代或追问信号决定是否澄清；完整问题保留原文走基础 Hybrid，忽略会话历史和故障模型的抽取、路径与预处理标记 |

Router 会重算分析特征对应的路径和固定权重，不直接采用模型建议中的权重。
权重统一维护在 `query_analysis/policy.py`，Router 使用同一策略确定主分支和 Hybrid
回退权重；Executor 只使用计划中的权重，Graph 失败时采用计划中的回退值。
必要澄清优先于统计、闲聊、图谱和普通召回；Query Analyzer 的原始建议路径仍保存在
`routing.requested_path`，实际执行路径返回为 `routing.path`。

## 模块边界与误路由诊断

| 模块 | 负责内容 |
| --- | --- |
| Analyzer | 意图、实体、原文约束、上下文依赖和澄清需求；保留模型策略建议 |
| Query Rewrite | 根据用户原文及可信上下文补全问题，校验否定、时间、编号等条件是否保留 |
| Router | 根据分析特征、已授权范围及运行时能力生成不可变执行计划 |
| RetrievalExecutor | 执行计划中的召回、融合及重排，记录实际降级与最终参数 |

诊断字段分别记录不同阶段，执行降级时保留 Router 的原始决策：

- `routing.model_suggestion`：原始模型建议；分析降级时为 null。
- `routing.requested_path`：原始建议路径；分析降级时仅为基础回退标记。
- `routing.policy_path`：统一规则依据分析特征选择的路径，尚未检查运行时覆盖。
- `routing.rule_path`、`rule_reason`、`rule_weights`：Router 最终批准的计划。
- `routing.path`、`reason`：保留现有接口含义，反映实际执行路径与原因。
- `execution`：最终查询、授权 KB/doc 范围、候选和返回预算、最终三路权重、
  实际 Graph 尝试参数、重排状态、Dense 输入类型、已应用扩展和降级原因。

例如 Graph 覆盖检查通过但执行超时：`rule_path=hybrid_graph`、
`rule_weights=[0.4,0.3,0.3]` 保持不变，实际 `execution.path=hybrid`，
普通查询 `execution.weights=[0.5,0.5,0]`，降级原因是 `graph_execution_timeout`。
这些字段与 Rewrite、Multi-Query 摘要一起存入助手消息的 `retrieval_trace` JSONB。
已有数据库需先应用 `zq_rag_app/schemas/migrations/007_chat_retrieval_trace.sql`。
独立 `/query/analyze` 的策略输出不再包含权重，需读取 Router 规则权重或实际执行权重。

## 图谱检查与降级

图谱检查要求调用方提供权限服务已经求交集的具体 `doc_ids`。
API 在 Analyzer 之前完成授权检查，模型不能新增或扩大知识库和文档范围。
内部 CLI 若只提供 KB 而未提供具体授权文档范围，则不会启动 Graph，退回基础 Hybrid。

覆盖检查先查询 PostgreSQL 当前有效文档版本，再查询 Neo4j：

- 使用规范化名称或别名精确匹配，不使用子串猜测节点 ID。
- 限定 seed、claim、chunk、doc 属于目标 KB，文档版本与 PostgreSQL 当前版本一致。
- 仅使用 `graph_status=ACTIVE` 且与 `active_graph_version` 相符的关系证据。
- Analyzer 在原有一次调用中区分 `graph_role=subject/qualifier/auxiliary`，并保存原文身份限定 `qualifiers`。
- 每个候选最多取两个不同 UID；必需主体歧义才澄清，必需主体缺失才降级。
- 辅助实体缺失或歧义时忽略该候选，保留已确认种子，返回 `graph_partial_coverage` 和部分证据警告。
- 使用图谱支持的实体类型消歧；PROJECT/CONTRACT/IDENTIFIER 不臆造类型映射，只按名称和限定匹配。
- 身份限定必须逐字包含在主体证据中，并有当前授权版本内的正向 PART_OF/OWNED_BY/LOCATED_IN 关系支持。
  一般关联、负向关系、旧版本或未授权证据不能用来确认限定身份；不支持的限定不会被悄悄忽略。
- Graph 检索使用确认后的 `seed_entity_uids`，不会再次用问题子串扩大种子范围。
- 图谱结果映射回 PostgreSQL 时再次限定 KB、doc_ids、当前版本和未删除/索引完成状态。

覆盖检查只证明授权文档中有相关实体及活动关系证据，不证明已经覆盖问题要求的关系链。
最终生成依旧依据检索到的原文，并保留来源引用；若 Graph 返回空结果，也会降级。
Graph 执行超时、异常或返回空结果时，恢复 Router 预先确定的 Hybrid 权重：
普通查询 Dense/BM25 为 0.5/0.5，精确查询（含 exact 关键词）为 0.3/0.7，Graph 权重归零。
后续 Multi-Query 扩展候选再次融合时沿用恢复后的权重。

```dotenv
QUERY_ROUTER_GRAPH_TIMEOUT_SECONDS=2
```

此配置分别限制覆盖检查和实际 Graph 召回，每阶段默认 2 秒。
图谱连接异常、超时、必需主体缺失、当前版本没有证据都会退回 Dense + BM25。
必需主体同名歧义返回澄清。旧分析结果未提供角色时，有命名业务主体则将附带技术/概念作为辅助；
没有业务主体时保守地将已有候选作为必需主体，避免关闭以技术本身为主体的关系检索。
取消传播给调用方；若取消发生在 PostgreSQL 图分支读取期间，
先回滚只读事务以恢复 Session，再处理超时或取消，不并发使用同一 AsyncSession。

## 上下文与待接入能力

现有会话检查通过之后，API 按时间顺序加载近期最多 6 条用户问题供 Analyzer 分析。
历史助手回答不会传给模型，避免把过往生成内容当作事实，或重用已失去权限的文档证据。

Query Rewrite 在 Router 之前接入，HyDE 在 Router 确定检索计划后执行，见 [Query Rewrite 文档](QUERY_REWRITE.md)。
普通改写后按需更新分析，使用独立问题及其最终特征执行 Router；不提前检查图谱或运行旧路由。
已审核的表达整理若抽取依据及决策输入仍可复用，不再次调用 Analyzer。
已抽取的唯一主体仅作代词替换时，可直接更新指代标记与抽取来源。
改写后的分析若仍建议改写，不再次执行，避免循环。
Multi Query 已在最终 Router 后按需接入，默认关闭；未开启或独立调用 Router 时仍返回 deferred_features。
开启后使用同一最终路由和授权范围生成等价搜索问题，见 [Multi Query 文档](MULTI_QUERY.md)。
上下文补全或重新分析失败时返回澄清。独立调用 Router 未补全指代时返回 context_rewrite_required。
元数据继续保留原文软约束，不直接生成 SQL 或硬过滤。

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
    "rewrite_ms": 0,
    "retrieval_ms": 220,
    "generation_ms": 500,
    "total_ms": 2535
  }
}
```

action 为 retrieve / clarify / unsupported / chat。
`graph_degraded=false` 也可能表示没有请求 Graph，不能将该字段单独解读为图谱执行成功。
模型名仍表示配置的回答生成模型；澄清和能力说明分支没有执行回答生成，token_count=0。
total_ms 包含 Analyzer（含重新分析）、Rewrite、Router、检索和生成，未包含 API 外层权限校验及消息落库耗时。
响应同时提供 rewrite 和 multi_query 摘要，以及 timing.expansion_ms；假想文档不在 Chat 响应或会话消息中。

主要 reason：

| reason | 含义 |
| --- | --- |
| hybrid_selected / graph_covered / no_retrieval_required | 普通检索 / 图增强检索 / 无检索 |
| graph_partial_coverage / graph_subject_required | 保留已确认图谱种子但只提供部分证据 / 缺少必需主体信号 |
| clarification_required / context_rewrite_required | 主体或约束不明确 / 指代尚未可靠补全 |
| structured_not_supported | 当前统计/全量查询能力未实现 |
| analysis_degraded / analysis_degraded_context_required | 分析失败时完整问题走基础检索 / 明显依赖上文时请求完整问题 |
| graph_scope_required / graph_not_covered / graph_entity_ambiguous | 缺授权文档范围 / 缺图谱证据 / 同名歧义 |
| graph_check_timeout / graph_check_unavailable | 图谱预检查超时或不可用 |
| graph_execution_empty / graph_execution_timeout / graph_execution_unavailable | 实际 Graph 召回为空、超时或不可用 |

## 会话澄清恢复

Chat 在 `kb_chat_session.pending_clarification` 保存待澄清任务，包括 `original_query`、
`missing_slots`、澄清问题、原授权文档范围、候选文档对象、近期用户指代上下文、补充和带时区的
`expires_at`（30 分钟）。候选文档仅保存 ID，不代表已验证的业务实体或事实。
状态与本轮消息在同一事务提交；已有会话加行锁，防止并发请求覆盖待澄清任务。

收到补充后，先检查会话归属和当前 KB/文档权限，将原文档范围及候选与当前范围求交集。
有效任务先做语义补全并由独立语义审核确认，再分析完整问题并路由；恢复任务不能扩大原范围。
短回复不能直接作为新任务检索，补全失败继续澄清并保留原任务、补充和原过期时间。
完成任务或明确切换新问题后清除状态。过期、无效或授权交集为空的状态不参与恢复。
历史助手回答不参与补全，也不作为回答证据；最终答案仍只依据本轮授权检索结果。

Router 的未解析上下文检查位于 `structured` 和 `none` 返回之前，因此统计和闲聊路径同样
必须先解决指代；不能绕过澄清保护直接返回能力说明或生成闲聊。

已有数据库上线前需要执行 `zq_rag_app/schemas/migrations/006_chat_clarification.sql`。
新建数据库的 `schema.sql` 已包含该字段。本地测试不自动执行数据库迁移。

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
