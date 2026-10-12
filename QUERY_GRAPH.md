# 在线 GraphRAG（第二版）

在线图检索通过 `hybrid_graph` 路由扩展现有混合检索。图谱在文档发布后由 Worker 异步构建；查询阶段只读已有图谱，定位关系出处，再从 PostgreSQL 回读当前授权原文。

## 查询链路

`/api/query/plan`、`/search`、`/answer`、`/answer/stream` 共用服务器规划检查；多轮聊天调用同一回答链。客户端仍只提交问题与资源选择，不能指定图种子、投影、Cypher 或跳数。

1. Analyzer 判断需要关系检索，并提供逐字有据的问题实体。
2. PostgreSQL 刷新会话身份，取初始授权范围与当前 READ 权限的交集；只批准已发布文档的活动 READY 索引和 READY 图投影。
3. 使用批准投影的实际 namespace 匹配问题实体。别名必须有当前原文的明示声明；相同规范名只用于寻找候选。存在有效一跳证据时生成 `GraphPlan`，预检找到完整两跳来源才批准 `max_hops=2`；缺来源或身份有歧义时使用一跳或 `hybrid`。
4. 向量、BM25 与图关系召回并行。固定 Cypher 模板限制关系白名单、数量和超时；第二版只批准下文列出的两跳业务路径。
5. 图 Evidence 回查真实 chunk，核验 tenant/kb/document/revision/generation、内容 SHA-256、连续原句及其偏移。关系条件必须保留在对应原句中。
6. 三路按 RRF 融合、去重，再使用现有重排。候选和 Top-K 选择以完整路径来源组为单位，不扩大原预算；放不下时只保留独立原文，删除路径元数据。模型请求前复核原文权限；输出检索结果前再次确认图投影仍 READY，剥离失效图出处。
7. 回答生成及引用、语义、权限复核继续读取原文。使用路径的正文须保留其全部原文明示条件，条件陈述用自己的引用支持对应 hop。多个完整路径有一条满足即可，不合并备选条件；条件和字段可以分句。图三元组不会作为额外事实传给回答模型。

一跳指一条业务断言：`Entity ← SUBJECT — Assertion — OBJECT → Entity`。这里有两条 Neo4j 技术边，不能将技术边数量当作业务跳数。

## 两跳与身份依据

当前允许两种路径：

- `Step — REQUIRES_MATERIAL → Material — HAS_FIELD → Field`，定位步骤所需材料及其信息项，支持同知识库跨文档、跨 namespace。
- `Process — HAS_STEP → Step ← RESPONSIBLE_FOR / REVIEWS / CONFIRMS — Actor`，定位流程步骤的职责；中间 Step 仍要求同文档、同投影身份。

两跳业务关系必须逐条由当前原文支持；明确的动作否定不能组成正向路径，条件必须完整保留。
跨文档的 Material、Process、Actor 对应关系使用当前出处中的共同流程上下文、以对象命名的文档或明示别名作为依据。
明确属于不同流程、不同对象或禁止混用的来源不能合并。Step、State、Field、Rule 不按同名跨文档合并。

映射记录包含版本化 `identity_id` 和各自的原文出处。每份来源仍保留自己的 document/revision/generation/projection/namespace；
不会改写历史实体 UID 或把两个 namespace 的节点永久合并。旧图的 MENTIONED_IN 缺少新属性时，从已核验原文保守读取明示依据。
同文档旧共享 UID 也不能覆盖出处中的明确上下文冲突。

## 权限与投影

种子实体的 `MENTIONED_IN`、关系的 `SUPPORTED_BY`、路径两端的可见出处均限定在批准的 namespace/tenant/kb/document/generation 范围内。共享 Entity 的租户和知识库属性不能单独授予访问权限。

投影 namespace 取自当前 PostgreSQL 记录，支持现有 `:rebuild:<UUID>` 投影；旧版本、STALE、BUILDING、FAILED 投影不参与查询。执行前和回读时再次检查当前投影与计划投影的精确交集。

每个 hop 与映射来源都独立参与原文权限及投影 READY 复查。只要一个来源被撤权，整条路径失效并移除撤权正文；
投影退休但原文仍可读时，剥离路径并保留原文事实。回答等待期间也重复这两类检查。
发布前若原路径退休，仅剩被完整引用但条件不符合正文的备选路径，也剥离路径元数据。
Neo4j 驱动仅在一次图操作内复用，退出即关闭；不同请求不共享授权检查结果。

## 配置

| 配置 | 默认值 | 作用 |
| --- | --- | --- |
| `NEO4J_ENABLED` | `false` | 图服务总开关；使用已有 Neo4j 连接配置 |
| `QUERY_GRAPH_ENABLED` | `true` | 在线图检索开关 |
| `QUERY_GRAPH_TIMEOUT_SECONDS` | `3` | 规划预检和在线图分支各自的总预算；Neo4j 查询也设置服务端超时 |
| `QUERY_GRAPH_SEED_LIMIT` | `20` | 问题种子预算 |
| `QUERY_GRAPH_RELATION_LIMIT` | `100` | 每轮关系候选预算 |
| `QUERY_GRAPH_MAX_HOPS` | `2` | 最多业务跳数，可设为 `1` 保留一跳模式 |
| `QUERY_GRAPH_PATH_LIMIT` | `10` | 最多完整路径数 |
| `QUERY_GRAPH_MENTION_LIMIT` | `300` | 明示实体出处候选预算 |

预算按实际查询检查。关闭图服务或证据不完整时仍使用可用的混合文本检索。

## 结果与降级

原有 evidence、回答和 SSE 契约保持兼容。证据增加 `scores.graph_rank/graph_score` 与 `graph_relations`，记录断言 ID、端点 ID、关系类型、条件、原句及偏移；原文仍是回答引用的来源。

`retrieval.graph_paths` 返回有序 hops、映射 bridges 和完整 `required_chunk_ids`。各来源包含内容哈希和原句偏移。
前端展示业务关系、条件与可跳转的原文引用；JSON、SSE 和恢复历史使用同一完整性检查。

检索 trace 增加 `graph_status/count/seed_count/assertion_count/hops/ms`、融合前路径候选数 `graph_path_candidate_count` 与最终完整路径数 `graph_path_count`。
图分数用于排序，不表示事实置信度。前端检索详情显示图执行情况。

- 规划 `GRAPH_NOT_READY`：图关闭、无 READY 投影、实体缺失或同名歧义，使用基础混合检索。
- 规划或检索 `GRAPH_UNAVAILABLE`：图服务异常或超时，使用仍可用的文本召回。
- 检索 `GRAPH_NO_EVIDENCE`：计划后图范围变化或无有效关系证据，使用文本召回。
- 检索 `GRAPH_PATH_INCOMPLETE`：两跳来源组不完整或超出原预算，只保留通过检查的独立证据。
- 检索 `GRAPH_PROVENANCE_CHANGED/UNAVAILABLE`：最终图出处失效或无法复核，剥离关系元数据；仍通过权限复核的原文可保留。
- 会话失效、身份不符及越权结果始终传播授权错误，不作为可选图分支故障吞掉。

## 验收与边界

第二版支持上述有限两跳模板及有原文依据的对象映射。社区摘要、任意深度推理、全库图聚合和跨知识库共指尚未实现。
中文否定与身份检查采用保守规则，不能声称覆盖任意自然语言表达；当前历史图的抽取质量仍需要独立验收。

测试覆盖图权限范围、独立命名空间、撤权、旧批次、同名歧义、哈希和引用位置、条件、超时、三路融合、最终投影变化以及 JSON/SSE 接线。真实验收还须检查路由和图 trace，并人工复核答案语义；单纯 Top5 原文命中不能证明图分支已执行。

第一版真实 HTTP 结果见 [一跳验收](eval/query_graph/ACCEPTANCE.md)，第二版隔离组件对照和回答回归见 [第二版验收](eval/query_graph/v2/ACCEPTANCE.md)。
隔离脚本 `scripts/evaluate_online_graph_v2.py` 使用冻结虚构语料、真实 Neo4j 和可选真实 DashScope，使用固定授权仓储、确定性抽取及实体分析；
它验证跨 namespace 的路径组件，不替代真实 PG/HTTP 权限验收，也不证明生产准确率提升。
