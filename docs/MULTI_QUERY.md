# 按需多查询扩展

`/chat/answer` 在最终 Router 之后接入 MultiQueryService，沿用 DeepSeek API 和已有授权范围。
默认关闭，开启后有两处触发：

| 触发 | 判断位置 | 行为 |
| --- | --- | --- |
| analyzer | 最终 Analyzer 的 use_multi_query=true，意图明确但检索表达不足 | 检索前生成并审核等价查询，原问题仍必定检索 |
| empty_retrieval | 首次 Dense/BM25/按需 Graph 融合后结果为空 | 生成并审核补救查询，保留已执行的原问题，不重复基础检索 |

每个请求最多运行一次扩展流程；Analyzer 触发后即使生成失败或检索零结果，也不再次补救。
扩展流程内部有有限重试，默认最多两次生成尝试。当前不判断非空结果是否足以回答，
不根据低分或低置信度直接触发补救；“零结果”也不表示知识库中事实不存在。

## 配置

使用已有 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL，无需新密钥。
进程环境变量或 `.env` 中可配置，修改后重启：

```dotenv
MULTI_QUERY_ENABLED=true
MULTI_QUERY_EMPTY_RETRIEVAL_ENABLED=true
MULTI_QUERY_MODEL=deepseek-flash
MULTI_QUERY_TIMEOUT_SECONDS=20
MULTI_QUERY_RETRY_ATTEMPTS=2
MULTI_QUERY_MAX_TOKENS=2048
MULTI_QUERY_MAX_EXPANSIONS=2
MULTI_QUERY_RETRIEVAL_TIMEOUT_SECONDS=8
```

MULTI_QUERY_ENABLED 默认 false；其他值如上。关闭 EMPTY_RETRIEVAL_ENABLED 仅关闭零结果补救，
保留 Analyzer 建议触发。MAX_EXPANSIONS 支持 1 或 2，含原问题最多三条检索问题。
生成/审核的 20 秒整体预算覆盖重试和退避；追加查询召回共享 8 秒预算。
SDK 自动重试关闭。取消向上传播，日志不包含问题、输出、密钥或供应商异常响应体。

## 等价与路由约束

扩展的是同一完整问题的检索表达，不拆分子任务，不补充相关但不同的问题，不生成假想答案。
例如“星河项目的验收审批流程是怎样的？”可以调整为同一审批流程的步骤提问；
不能扩展成“星河项目为什么未验收？”或假设具体失败原因。

严格 JSON 协议拒绝未知/缺失字段、重复 JSON 键、截断、超量或空字符串。
去重会忽略空白、全半角、英文大小写及常见提问标点，丢弃与原问题相同的表达。
本地校验共享普通改写的原文条件保护，生成时明确传入 protected_phrases：
命名主体、元数据原文、精确/限定关键词、编号、数字、常见数量单位、否定、范围及时间词。
不能引入无来源的数字或标识符，不能猜日期边界或实体别名。

候选通过本地检查后，DeepSeek 单独逐项审核：完整问题等价、意图保留、条件保留、无无据扩展、
不会改变实体/关系/多跳/精确条件/统计/澄清需求及检索范围。所有查询都需审核，漏项或重复拒绝。
仅经审核的候选参与检索，历史不传给扩展模型；未解析的指代必须先补全或澄清。
模型审核不能保证识别全部语义偏移，真实业务标注集评估仍然必要。

分析降级、意图不明确、澄清、闲聊、全量统计及精确编号/精确关键词查询不扩展。
本次选了 HyDE 也不叠加扩展，包括 HyDE 降级的请求。
Router 只运行一次，扩展不重新路由、不新增实体种子、不修改 KB/doc IDs。

## 召回与融合

原问题先正常检索；每个新增问题仅追加 Dense 和 BM25，串行复用同一 AsyncSession。
追加读取在独立 SAVEPOINT 内执行，失败/超时回滚该可选读取，保留外层事务与已校验对象。
服务继续执行已有 KB/doc/version/未删除过滤；追加结果还检查 doc_ids，越权候选整组丢弃。
两通道成功且 SAVEPOINT 正常退出后才接纳该查询，单个查询失败不影响已经成功的查询。

先分别对 Dense 榜单和 BM25 榜单做加权 RRF，原问题权重 0.6，成功扩展平分其余 0.4，k=60。
同一榜单中的同一 chunk ID 只投一次票，每个通道保留 candidate_k，再进入既有通道融合。
该初始权重尚未通过业务评测调优；跨查询后的 score 为排序融合值，不是相关性置信度。
Graph 复用最终实体 UID 确认的原问题结果，不为等价表达重复执行或投票。
融合后去重，candidate_k 和 top_k 保持原预算，统一重排一次。
重排和回答生成使用最终完整问题，回答仅依据实际授权文档证据，原用户问题仍保留在响应和历史中。

扩展生成/审核失败时保留原检索结果；部分召回失败或超时时保留原问题及已经完成的扩展结果。
总召回为空时沿用原有“当前授权知识库证据不足”的回答。当前服务不提供多查询独立 HTTP 接口，
`/query/analyze` 仍只返回建议，其他独立召回接口仍直接检索。

## 响应

`/chat/answer` 新增 multi_query 和 timing.expansion_ms：

```json
{
  "multi_query": {
    "status": "expanded",
    "queries": ["星河项目的验收审批流程是怎样的？", "星河项目的验收审批需要哪些步骤？"],
    "trigger": "analyzer",
    "applied_queries": ["星河项目的验收审批流程是怎样的？", "星河项目的验收审批需要哪些步骤？"],
    "failed_queries": [],
    "attempts": 1,
    "reason": "equivalent_queries_generated",
    "degradation_reason": null,
    "warnings": []
  },
  "timing": {"expansion_ms": 2100}
}
```

status 为 disabled/skipped/expanded/unchanged/degraded；trigger 为 analyzer/empty_retrieval/null。
queries 是原问题加已审核扩展，applied_queries 是原问题加成功完成两通道追加召回的查询，
不表示这些查询必然命中结果；failed_queries 包括失败或因整体预算耗尽未执行完的追加查询。
reason 说明触发/跳过原因；degradation_reason 为 missing_api_key/timeout/provider_error/invalid_output/
retrieval_error/retrieval_timeout。部分失败时 status=degraded 但成功结果仍可参与融合。
expansion_ms 仅计生成和审核，追加检索计入 retrieval_ms，补救生成耗时不重复计入 retrieval_ms。
master 关闭时 Analyzer 的 multi_query 建议仍可在 routing.deferred_features 中出现，开启时由摘要说明执行状态。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_multi_query_service.py tests/test_multi_query_pipeline.py -q
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B -m scripts.eval_multi_query
```

前两条无需网络，覆盖触发/跳过、条件保持、逐项审核、去重、固定预算、Graph 单次执行、
范围保护、部分失败、SAVEPOINT 退出、超时和取消。最后一条产生真实 DeepSeek 用量，
只使用合成问题，模拟首次零结果，不连接数据库；报告写入 eval-design/multi-query-smoke.json。
若 Analyzer 未自主建议，专项触发样例可覆盖 use_multi_query 并明确记录 method_source=test_override，
这验证执行器分支，不能据此计算 Analyzer 建议准确率。没有验证生产数据库或最终召回质量。
上线启用前应比较 Recall@K、最终回答正确率、P95 延迟、调用成本和实体/否定/时间保持情况。

2026-09-18 验证：新增 47 项测试通过，项目全量 338 项通过；真实 DeepSeek 合成冒烟 6/6 通过。
Analyzer 触发专项使用显式建议覆盖，报告已标记；零结果补救仅模拟触发条件。
本地及语义检查确认“去年”“未验收”等原文条件保持，不把未执行验收改成未通过验收。
