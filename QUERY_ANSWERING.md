# 带引用的单轮证据回答

`POST /api/query/answer` 需要登录，请求体与 `/api/query/search` 相同：

```json
{"query":"工单关闭前需要满足什么确认条件？","knowledge_base_ids":["1"]}
```

知识库和文档 ID 使用十进制字符串。范围可省略，此时查询当前用户可读资源。
客户端只能提交问题和资源选择，不能提交证据、答案、历史、执行计划或模型参数。

接口返回 `analysis`、`plan`、`scope`、`retrieval` 和 `answer`。回答由若干 `statements`
组成，每条包含 `text` 和 `citations`；引用包含本次响应的 `citation_id` 和逐字 `quote`。
来源文件、页码、章节和原文保留在 `retrieval.evidence`。页面引用可跳到对应原文卡片。

## 发布过程

1. 当前鉴权和范围检查后执行 Analyzer、Router、向量/BM25 召回、融合与重排。
2. 重新读取证据权限、活动批次、原文及哈希，再向生成模型发送当前问题与已授权原文。
3. 缓冲完整 JSON 草稿，拒绝不完整输出、重复字段、额外字段、未知引用和改写引文。
4. 模型省略 PDF 中的 CR/LF 时，仅在去除 CR/LF 后能唯一定位同一连续片段的情况下，
   恢复精确原文换行。不修改空格、标点、词语、数字、否定或实体名；有歧义则拒绝。
5. 再次复核权限，调用独立模型请求检查每条陈述的证据支持以及问题是否被完整回答。
6. 发布前再复核权限和活动批次。任一输入证据失效时丢弃整份回答，并移除失效原文。

旧索引中部分续表仅重复稀疏标题行，缺少真正列名。服务端逐张识别 Markdown 表格，
只认可直接列头，或同表内与切块器一致的稀疏标题、说明、真实列名和数据结构。
无列头的表格碎片、跨表借列名和仅凭首块序号的豁免均不被接受。上下文不完整时不能
作为回答依据，原文仍可查看。历史数据可通过 [重建工具](TABLE_REINDEX.md) 修复；
本轮已重建八份测试 XLSX。此判断只识别明确的缺列名情况，不证明所有表格语义正确。

显式“A 和 B 分别／各自……”问题在同一授权候选池中增加子问题重排，最多三个子问题。
原句与各子问题重排第一名优先保留，剩余位置按原融合秩与分面秩的等权 RRF 交替补足。
向量化、数据库召回、候选预算和 Top-K 不增加；重排请求可从一次增至最多四次，
`retrieval.trace.rerank_query_count` 记录实际数量。返回分数始终来自原句，等待后重核权限。

模型复核是错误拦截措施，不能视为事实正确的保证；仍需固定问题回归和人工抽查。
未通过检查的草稿不会进入 HTTP 响应。回答、引用和证据响应均禁止缓存。

| answer.status | 行为 |
| --- | --- |
| answered | 所有陈述有可定位引用，模型复核通过，发布前授权复查通过 |
| insufficient | 没有证据、证据不能完整回答，或复核认为不足；无草稿 |
| rejected | 引用或输出格式不合法，或处理期间资料/权限变化；无草稿 |
| unavailable | 模型关闭、未配置、超时或失败；可保留仍获授权的原文证据 |
| skipped | 需澄清或无需文档检索，提示补充问题 |
| unsupported | 当前计划能力未接入，不将片段检索包装成统计或筛选答案 |

JSON 接口认证失效返回 401；无法确认权限返回脱敏 503。模型等待时不占用数据库连接。
客户端取消会停止展示并忽略晚到结果；SSE 断开会取消后台任务并等待上游清理，
但不能保证供应商尚未处理已经发出的模型请求。

## SSE 流式回答

`POST /api/query/answer/stream` 接受同样的请求体、Bearer 身份和资源范围。
页面“生成回答”通过带身份的 `fetch` 读取流；令牌不会写入 URL。
“仅检索证据”继续使用 `/api/query/search`，原 `POST /api/query/answer` JSON 接口保留。

| SSE 事件 | 数据与行为 |
| --- | --- |
| progress | `{ "stage": "analysis" / "retrieval" / "generation" / "review" }`；在对应工作开始时实时发送，跳过的生成或复核阶段不发送 |
| answer | `QueryAnswerResponse`；每次追加一条已经完整核验的陈述，携带累计陈述、引用和当前授权原文 |
| done | 完整 `QueryAnswerResponse`；唯一成功终点，包括证据不足、拒绝、不可用等结果 |
| error | `{ "status": 401/403/503, "code": "...", "message": "..." }`；发送 HTTP 头后的错误，随后关闭流，无 done |

服务端仍缓冲完整模型草稿，并执行逐字引用、完整性、语义和发布前授权复核。
所有检查通过后按完整陈述发送，不发送未经复核的模型 token，也不人为延迟制造打字效果。
只有一条陈述时只发送一次 answer；SSE 改善等待阶段的反馈，不缩短生成与复核耗时。
证据不足、拒绝、不可用等结果不发送 answer，直接由 done 返回明确状态。

身份、资源范围和请求参数在发送 HTTP 头前校验，失败沿用 JSON HTTP 错误，
凭据过期可按原策略刷新一次。流开始后的 error 不自动重连或重复生成；401 清除登录状态。
前端在每次显示前检查引用和复核状态，要求后续陈述只追加、来源和最终结果保持一致。
中断、缺少 done、无效帧或错误会清除未完成结果；修改问题、范围、权限变化和取消也会清除。

响应为 `text/event-stream`，设置 `Cache-Control: no-store`、`X-Accel-Buffering: no`，
每 10 秒发送注释心跳，单次网络发送超时为 15 秒。生产反向代理还需关闭此路径的响应缓冲，
并允许至少 120 秒的读取等待；客户端总等待上限仍为 2 分钟。
队列有界，慢客户端会产生背压；断开时取消分析、检索、生成或复核的在途任务。
当前不提供断线续传或自动重连。

## 配置

沿用 `DASHSCOPE_API_KEY`、`OPENAI_BASE_URL`、`CHAT_MODEL` 和 `CHAT_TEMPERATURE`。
生成和复核使用相同配置模型，分别请求；复核温度为 0，供应商调用不自动重试。

```dotenv
QUERY_ANSWER_ENABLED=true
QUERY_ANSWER_TIMEOUT_SECONDS=30
QUERY_ANSWER_MAX_OUTPUT_TOKENS=2048
QUERY_ANSWER_CONTEXT_TOKENS=6000
```

超时与输出上限按每次模型调用计算。上下文预算统计问题和完整证据 payload 的 token，
不截断证据正文；超限时建议缩小问题。使用量只累计已获得有效响应的调用，不能用于核算
超时请求是否计费。`answer.trace` 记录生成、复核、授权、总耗时与 token 数量。

当前接口按每次独立问题生成回答，不保存问题、回答和会话历史。
长会话摘要及在线图谱召回后续接入。

## 本地验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_query_answering.py tests/test_query_answer_api.py
cd frontend
npm test
npm run build
```

测试使用本地替身；真实模型验收脚本和固定数据见 [检索验收](eval/query_retrieval/README.md)。
本轮结论及已知限制见 [验收记录](eval/query_retrieval/ACCEPTANCE.md)。
