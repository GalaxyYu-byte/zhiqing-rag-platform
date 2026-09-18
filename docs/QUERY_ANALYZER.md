# 在线 Query Analyzer

Analyzer 使用 DeepSeek 分析意图、查询类型、实体、关键词、原文约束、上下文依赖及澄清需求。
`retrieval_strategy` 保留模型的策略建议，不生成分支权重或执行计划；`model_suggestion`
保存同一原始建议，避免预处理更新策略后丢失诊断依据。
Analyzer 服务本身不回答问题、不访问数据库、不执行检索。`/chat/answer` 已通过
`RagAnswerService` 接入 Analyzer → 澄清或独立问题 → 按需更新分析 → 最终 Query Router → 按需 Multi Query / HyDE，现有独立召回接口保持直接检索。
实际路由行为见 [Query Router 文档](QUERY_ROUTER.md)。

## 配置

在进程环境变量或仓库 `.env` 中配置密钥，不将密钥写入代码或文档：

```dotenv
DEEPSEEK_API_KEY=<your-key>
DEEPSEEK_BASE_URL=https://api.deepseek.com
QUERY_ANALYZER_MODEL=deepseek-flash
QUERY_ANALYZER_TIMEOUT_SECONDS=12
QUERY_ANALYZER_RETRY_ATTEMPTS=2
QUERY_ANALYZER_MAX_TOKENS=3072
```

模型使用当前 DeepSeek 官方提供的 `deepseek-flash`，可通过配置更换。
调用 OpenAI 兼容的 Chat Completions，设置 `response_format={"type":"json_object"}`
和 `thinking={"type":"disabled"}`。配置独立于现有 DashScope 问答生成、Embedding 和图抽取。
修改配置后重启服务。

官方依据：[模型说明](https://api-docs.deepseek.com/quick_start/pricing/)、
[JSON 模式](https://api-docs.deepseek.com/guides/json_mode/)、
[思考模式开关](https://api-docs.deepseek.com/guides/thinking_mode/)。
JSON 模式约束 JSON 语法；字段结构和一致性由本地 Pydantic 协议校验，
不能保证意图和实体在语义上始终识别正确。

## 接口

```http
POST /query/analyze
Content-Type: application/json
```

```json
{
  "query": "他是否也负责天河项目？",
  "history": [
    {"role": "user", "content": "张三是星河项目的负责人。"}
  ],
  "reference_date": "2026-09-17"
}
```

- `query`：去除首尾空白后 1–2000 字符。
- `history`：可省略，默认空；按时间顺序，最多 6 条，每条 1–2000 字符。
  只接受 `user` / `assistant`，不允许客户端注入 `system` 消息。
- `reference_date`：可省略，默认中国标准时间当天。为相对时间解析和复测提供基准；
  当前返回原始时间描述，不实际转换日期边界。

接口沿用项目 `CurrentUser` 依赖；未绑定当前用户返回 401。
当前应用仍使用项目已有的固定管理员中间件，此接口未替换认证系统。
接口只分析调用方传入的文字，不接收 KB/文档 ID，也不读取会话。
后续在线编排从数据库加载历史时，必须先校验会话所有权及知识库权限。

输出示例（编号查询）：

```json
{
  "intent": {"primary": "fact", "secondary": [], "confidence": 0.95},
  "query_type": {
    "primary": "exact",
    "confidence": 0.95,
    "has_context_reference": false,
    "relation_required": false,
    "potential_multi_hop": false,
    "requires_exhaustive": false,
    "needs_clarification": false,
    "clarification_question": null
  },
  "entities": [{
    "source": "query",
    "history_index": null,
    "evidence_quote": "HT-2025-001",
    "confidence": 0.99,
    "entity_type": "CONTRACT",
    "name": "HT-2025-001"
  }],
  "keywords": [{
    "source": "query",
    "history_index": null,
    "evidence_quote": "金额",
    "confidence": 0.95,
    "text": "金额",
    "kind": "domain"
  }],
  "metadata": [],
  "retrieval_strategy": {
    "path": "hybrid",
    "use_query_rewrite": false,
    "use_multi_query": false,
    "use_hyde": false,
    "rewrite_method": "none",
    "rewrite_operations": [],
    "reason": "合同编号精确检索与正文语义召回结合。"
  },
  "original_query": "HT-2025-001 合同的金额是多少？",
  "reference_date": "2026-09-17",
  "model": "deepseek-flash",
  "analyzer_version": "query-analyzer-v2",
  "status": "ok",
  "degradation_reason": null,
  "attempts": 1,
  "latency_ms": 1952,
  "warnings": []
}
```

## 六项分析协议

实体新增 `graph_role`（subject / qualifier / auxiliary）及 `qualifiers`（原文身份限定对象）。
角色在现有 Analyzer 调用内产生，不增加模型调用。限定对象必须逐字出现在该实体的证据中；
旧结果允许不提供这两项，由 Router 采用保守类型规则。技术或概念本身为关系查询主体时仍应标为 subject。

| 字段 | 含义与允许值 |
| --- | --- |
| `intent` | 用户目标；primary 为 fact / explanation / procedure / comparison / summary / relationship / statistics / recommendation / chitchat / other；secondary 最多 3 个且去重 |
| `query_type` | 检索形态；primary 为 exact / semantic / relational / multi_hop / aggregate / compound / conversational / ambiguous；附带指代、关系、多跳、完整性和澄清标记 |
| `entities` | 最多 20 个候选，保留类型、原文名称、来源、证据和自评置信度；不是已匹配的图谱节点或业务 ID |
| `keywords` | 最多 20 个原文词组；kind 为 entity / domain / exact / relation / action / qualifier；不在这里生成同义词扩展 |
| `metadata` | 最多 15 项原文约束；保留字段、比较操作、原始值和证据，不生成数据库过滤表达式 |
| `retrieval_strategy` | 原始模型建议，包含路径及预处理建议，不含权重、权限或已确认的运行时能力；不直接控制执行 |

抽取项的 `evidence_quote` 必须是对应输入的连续原文片段，且 name/text/value 必须
逐字出现在证据中。`source=query` 时 `history_index=null`；`source=history` 时指定
输入数组的有效索引，只允许引用用户消息，且当前问题必须标记上下文依赖。
历史助手回复只帮助理解对话，不作为事实依据。即使证据匹配成功，语义分类仍可能出错。

协议禁止未知字段、未知枚举、字符串伪装布尔/置信度、非有限数值、重复 JSON 字段、
不一致的澄清/多跳标记，以及模型生成的 SQL、Cypher、权限或检索权重字段。
顶层六项及嵌套必填字段不能省略；空抽取用 `[]`。同类同名抽取项保留第一项。

## 元数据边界

允许字段：time / event_time / uploaded_at / effective_time / publication_time / location /
document_name / document_code / file_type / department / version / business_status / amount。
允许操作：eq / neq / contains / lt / lte / gt / gte / range / latest / relative。

例如“2025 年之前金额大于100 万元的未验收项目”应保留：

| field | operator | value |
| --- | --- | --- |
| time | lt | 2025 年之前 |
| amount | gt | 100 万元 |
| business_status | eq | 未验收 |

所有元数据当前均为 `soft` 模式，不能直接套到 ORM 字段或 SQL 上。
业务“未验收”不等于索引状态，用户表达的制度版本不等于索引版本；部门名称不等于部门 ID。
事件时间也不等于上传日期。执行硬过滤之前，需要确认目标字段支持、单位/日期解析、
实体或部门映射及授权范围。相对日期、最新版本、空结果回退尚未实现。

## 下游规则策略优先级

服务端按以下顺序选择路径，覆盖不适用的模型路径建议：

1. 需要澄清，或模型识别出上下文指代但没有历史：`clarify`。
2. 统计意图、aggregate 查询或要求完整数据：`structured`。
   这是能力需求标记，当前没有结构化统计/全量检索执行器；不能静默当作 Top-K 问答。
3. 没有实体、元数据和关系需求的纯闲聊：`none`。
4. 具有关系需求且有原文实体候选：`hybrid_graph`，先匹配实体并检查图谱覆盖。
5. 其他情况：`hybrid`。多个实体或跨文档比较本身不触发 Graph。

权重是初始规则值，尚未通过真实召回数据调优：exact 查询或 exact 关键词采用
Dense/BM25 = 0.3/0.7，普通查询 0.5/0.5，图增强查询 Dense/BM25/Graph = 0.4/0.3/0.3。
clarify / structured / none 分支权重为零，调用方应先分支处理。
上述权重由统一 Policy 为下游计划确定，不属于 Analyzer 返回协议。
图谱不匹配时 Router 调整计划；计划执行中图谱不可用时 Executor 使用计划的 Hybrid 回退权重。

Analyzer 通过 `rewrite_method=none/query_rewrite/hyde` 建议是否及如何预处理；
`rewrite_operations` 指定上下文补全、检索标准化、关键词优化、约束显式化中的实际需要项。
`use_query_rewrite` / `use_hyde` 与 method 严格一致且互斥。`/query/analyze` 只返回建议，
`/chat/answer` 执行改写和审核，再进行一次最终 Router。实现和降级见 [Query Rewrite 文档](QUERY_REWRITE.md)。
HyDE 仅允许无精确约束、业务实体或关系需求的语义解释/方法/建议问题；上下文补全优先普通改写。
`use_multi_query` 建议对同一完整问题生成等价搜索表达，不表示拆分多个任务。
开启 `MULTI_QUERY_ENABLED` 后，`/chat/answer` 在最终 Router 后执行，首次零结果也可以补救触发；
见 [Multi Query 文档](MULTI_QUERY.md)。`potential_multi_hop` 表示可能需要关系链，
不保证知识库内存在对应关系或确定的跳数。

## 超时和降级

12 秒整体预算覆盖全部 LLM 尝试和短退避；SDK 自动重试关闭，默认最多 2 次尝试。
空内容、非 JSON、截断输出、格式/证据校验失败会带固定修正提示再次请求。
连接异常、429 和服务端错误可重试；401/403 等配置错误不重试；任务取消向上传播。
日志只记录尝试次数和错误分类，不输出问题正文、模型内容、密钥或异常响应体。

请求格式不合法返回 422。分析不可用则仍返回结构完整的 200 响应：
`status=degraded`，`degradation_reason` 为 missing_api_key / timeout / provider_error /
invalid_output，置信度为 0、抽取为空、策略为基础 hybrid，并保留原问题。
调用方必须检查 status 和 warnings；空抽取不代表问题确实没有约束。
降级时不会解析或继承会话历史；Router 用有限的本地文本规则检查当前问题的明显指代和追问信号。
这些问题需要补充对象名称和条件；完整问题直接使用原文执行基础 Hybrid，不因存在历史而持续澄清。
本地规则只提供故障保护，不替代正常 Analyzer 的语义识别。

## 验证与复测

### 调用量、降级率与端到端 P95

保持现有调用结构：未触发预处理的普通问答通常为 Analyzer 一次、回答生成一次；
Router 不调用 LLM，澄清及统计能力说明通常仅调用 Analyzer。改写、澄清补全、重新分析和重试
可能增加调用次数，不能将逻辑问答请求数直接视为模型调用量。本轮未增加简单请求快速路径。

`/metrics` 提供以下 Prometheus 指标，不使用问题、实体、用户、KB/doc ID 或模型名作为标签：

| 指标 | 统计口径 |
| --- | --- |
| `zq_query_analyzer_requests_total{status,reason}` | 每次逻辑分析（包括重新分析）；成功、降级、取消或异常 |
| `zq_query_analyzer_calls_total` | 实际发起的模型调用尝试，包括失败及重试；无密钥降级不计调用 |
| `zq_query_analyzer_retries_total{reason}` | 实际发起的第二次及后续调用，按前次失败原因分类 |
| `zq_query_analyzer_duration_seconds` | 分析耗时，包含全部调用和退避，失败及取消也记录 |
| `zq_chat_http_duration_seconds{outcome}` | `/chat/answer` HTTP 耗时，含权限、状态恢复、模型处理、序列化及落库；含错误及取消 |

过去 5 分钟 Analyzer 模型调用速率、重试速率、降级率及成功问答端到端 P95 可查询：

```promql
rate(zq_query_analyzer_calls_total[5m])
sum(rate(zq_query_analyzer_retries_total[5m]))
sum(rate(zq_query_analyzer_requests_total{status="degraded"}[5m]))
  / sum(rate(zq_query_analyzer_requests_total{status=~"ok|degraded"}[5m]))
histogram_quantile(0.95,
  sum by (le) (rate(zq_chat_http_duration_seconds_bucket{outcome="success"}[5m])))
```

降级率分母为已返回成功或降级结果的分析，取消和异常可单独查看。P95 为服务端 HTTP 分桶估计，
不包含客户端网络耗时；现有 `zq_rag_answer_stage_duration_seconds{stage="total"}` 只覆盖 RAG 服务内部。
需先积累真实流量再评估快速路径，本地替身测试不会给出实际生产调用成本或 P95。

无需网络的自动测试：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_query_analyzer_service.py tests/test_query_analysis_api.py -q
```

故障会话回归见 `tests/test_analyzer_failure_context.py` 和 `tests/test_rag_routing_pipeline.py`：
覆盖缺少密钥、超时、供应商错误和无效输出，以及同一会话中先澄清、补充完整问题后恢复检索、再切换主题。
2026-09-18 完整测试集通过：377 项；这些回归使用本地替身，不调用模型或数据库。

使用配置密钥对 10 个合成问题进行真实调用（产生 API 用量，不访问数据库）：

```powershell
.\.venv\Scripts\python.exe -m scripts.eval_query_analyzer
```

报告默认写入 `eval-design/query-analyzer-smoke.json`，支持 `--output` 指定位置。
脚本校验输出状态、意图、路径、关键实体和元数据，并对话题切换检查历史条件泄漏。
配置或连通性错误时提前停止，失败退出码为 1。

2026-09-17 验证：40 项 Analyzer/接口自动测试通过；接入 Router 后项目全量 246 项测试通过。
真实 DeepSeek 合成样例 10/10 通过，单题延迟 1227–1952 ms，均一次尝试完成。
这些样例属于冒烟检查，不能作为业务识别准确率或生产延迟 SLA。
后续应使用真实业务标注集检查意图混淆、实体歧义、时间/否定条件保持，
并比较接入前后的 Recall@K、MRR、Graph 误触发率与 P95 延迟。
