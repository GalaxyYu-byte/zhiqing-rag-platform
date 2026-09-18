# 在线 Query Rewrite / HyDE

`/chat/answer` 的执行顺序是权限及会话校验 → Analyzer → 澄清或生成独立问题 → 按需更新分析 → 最终 Router → 按需 Multi Query / HyDE → 检索 → 回答。
改写之前不调用 Router，不提前检查图谱。`/query/analyze` 仍只分析，独立召回接口仍直接检索。

## 方式选择

Analyzer v2 的 `retrieval_strategy` 新增必填 `rewrite_method` 和 `rewrite_operations`：

| method | 使用条件 | operations |
| --- | --- | --- |
| none | 问题完整、已有清晰检索词，或需要澄清/统计/闲聊 | [] |
| query_rewrite | 确实需要补全、整理搜索表达或显式化条件 | 从下列四项选择，不能空 |
| hyde | 缺少合适检索表达的抽象 semantic 原因/方法/建议问题 | [] |

四项普通改写操作：`context_completion` 上下文补全、`retrieval_normalization` 检索标准化、
`keyword_optimization` 搜索词优化、`constraint_explicitization` 约束显式化。
`use_query_rewrite` 和 `use_hyde` 与 method 一致且互斥；无需转换不强行改写。

服务端规则优先：上下文指代只走普通改写；必要澄清、统计、闲聊不走改写。
HyDE 仅对 hybrid/semantic 的 explanation/procedure/recommendation 启用，
业务命名实体、精确编号/数字、元数据、关系需求及否定/范围/相对时间等条件会关闭 HyDE。
此规则偏保守，可用标注集评估后调整；Multi Query 已按需接入，默认关闭，见 [Multi Query 文档](MULTI_QUERY.md)。

## 普通改写的保护

DeepSeek 生成结构化候选（rewritten / unchanged / clarify），本地检查：

- 拒绝缺字段、未知字段、重复 JSON 键、截断输出、不一致状态或方式、未选择的操作。
- 保留当前问题的实体名称、元数据原文、精确/限定关键词及编号、数字、常见数量单位、否定和范围表达。
- 不新增无来源数字，不计算或猜测相对日期边界；元数据仍是软约束。
- 上下文补全必须引用相关 user 历史的连续原文片段；禁止 assistant 事实和新话题继承旧条件。

候选通过本地检查后，再调用一次 DeepSeek 独立审核语义、用户意图、约束、无依据扩展及指代唯一性。
当前明确条件优先于旧条件。有歧义时返回具体澄清问题。
审核是模型判断，不能保证所有语义偏移都被发现；保守拒绝和真实业务评测仍有必要。

不默认再次调用 Analyzer。经过本地检查和独立语义审核的 `retrieval_normalization`，
若无上下文依赖、抽取证据仍逐字存在于独立问题中且规则决策输入未变，则复用分析特征并更新问题标签。
若 Analyzer 已抽取出唯一的历史主体，经过审核的补全仅把一个“他/她/它”替换为该主体，
可直接更新指代标记和抽取来源；抽取值须在新问题中再次通过原文依据检查。
其他上下文补全、关键词优化、条件显式化或抽取依据失效时，保守地重新运行 Analyzer。
独立问题的历史置空，保留同一 reference_date。
比较前后路径、固定权重、意图、查询类型、关系/多跳/完整性、实体、精确关键词和元数据等决策输入，
返回 `route_changed`。它表示路由特征发生变化，不表示提前执行过旧 Router 或必然改变最终路径。
最终 Router 使用更新后的分析和问题，只运行一次，并按最终实体检查授权图谱覆盖。
最多改写一次、重新分析一次；重新分析再次建议改写时不循环。

普通无上下文改写失败：丢弃候选，使用原问题和原分析。
上下文补全失败：要求完整对象及条件，不检索未解析指代。
重新分析失败：丢弃候选；有指代时澄清，否则恢复原问题。
Analyzer 不可用时跳过改写，仅检查当前文本的明显指代或追问信号：依赖上文时澄清，完整问题保留原文走基础 Hybrid，即使会话已有历史也可以继续检索。

## HyDE 隔离

最终 Router 确定 Hybrid 检索计划后，DeepSeek 为独立问题生成简短假想文档，再通过结构检查和主题/范围审核。
澄清、闲聊及暂不支持的路由不生成假想文档。
问题必须逐字保留，不能同时普通改写或引入具体人名、业务项目、日期、数字、出处等假设事实。
假想文档只传给 `search_by_cosine(..., embedding_text=...)` 生成向量，召回真实且授权的文档分块。
Dense 结果的 query 标签仍是用户问题，BM25、图谱、重排、Router 和回答生成均使用问题。
假想文档不放进回答证据、来源、Chat 响应或历史消息。
HyDE 生成/审核失败退回原问题；假想文档的 embedding 调用失败则用原问题生成向量，
返回 `hyde_embedding_failed`。数据库错误不重复检索，任务取消向上传播。

方法依据：[HyDE 原始论文](https://aclanthology.org/2023.acl-long.99/)。
该论文将包含可能幻觉的假想文档嵌入，用于定位真实文档；此实现不把假想文档当作答案证据。

## 配置和响应

沿用 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`，无需新密钥：

```dotenv
QUERY_REWRITE_MODEL=deepseek-flash
QUERY_REWRITE_TIMEOUT_SECONDS=20
QUERY_REWRITE_RETRY_ATTEMPTS=2
QUERY_REWRITE_MAX_TOKENS=2048
```

20 秒整体预算覆盖生成、审核、有限重试和短退避；SDK 自动重试关闭。
失败日志只记录尝试次数和错误类别，不记录问题、输出、密钥或供应商响应体。
初次及重新 Analyzer 预算分别受 QUERY_ANALYZER_TIMEOUT_SECONDS 控制，额外调用会增加在线延迟。

`/chat/answer` 新增：

```json
{
  "rewrite": {
    "method": "query_rewrite",
    "status": "rewritten",
    "retrieval_query": "张三是否也负责天河项目？",
    "operations": ["context_completion"],
    "reanalyzed": true,
    "route_changed": true,
    "attempts": 1,
    "degradation_reason": null,
    "warnings": []
  },
  "timing": {"analysis_ms": 3200, "rewrite_ms": 2600, "routing_ms": 15}
}
```

原 query 和会话用户消息保留用户输入。回答生成同时接收原问题和经过校验的独立问题；
恢复澄清任务时，原问题取自待澄清状态，独立问题包含已审核的补充条件。
两种问题都不是事实证据，回答仍须由授权检索证据支持。
`analysis_ms` 包含实际执行的所有分析，`rewrite_ms` 含普通改写及 HyDE 的生成和审核。
status 可为 skipped / rewritten / unchanged / clarify / degraded。
degradation_reason 为 missing_api_key / timeout / provider_error / invalid_output /
reanalysis_failed / hyde_embedding_failed；超时、空输出、语义审核拒绝都不使用生成内容。
响应字段为增量；Analyzer 协议版本由 query-analyzer-v1 升为 query-analyzer-v2。

## 验证

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B -m scripts.eval_query_rewrite
.\.venv\Scripts\python.exe -B -m scripts.eval_query_analyzer --output eval-design/query-analyzer-v2-smoke.json
```

前一条无需网络。后两条产生 DeepSeek 用量，使用公开合成问题，不访问数据库或修改知识库。
改写报告默认 `eval-design/query-rewrite-smoke.json`；包含上下文、当前条件覆盖、歧义、精确条件、
新话题及 HyDE 隔离。HyDE 专项在真实 Analyzer 后显式选择 HyDE，并经过相同服务端安全规则，
报告 method_source=test_override；它验证执行器，不证明 Analyzer 自主选择的准确率。
合成样例仅是冒烟检查，不代表生产召回质量或语义保持准确率。

2026-09-17 验证：项目全量 291 项测试通过；真实改写/HyDE 6/6 和 Analyzer v2 10/10 通过。
两个普通上下文改写样例一次生成及审核完成，重新分析后使用新问题；
生成和审核耗时 1740–3181 ms，额外分析耗时计入 analysis_ms。此延迟来自少量样例，不是 SLA。
