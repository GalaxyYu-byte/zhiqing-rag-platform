# RAG 企业知识库系统整体架构

## 1. 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                                   用户层                                     │
│                       Vue3（文档管理 + 智能对话）                              │
└────────────────────────┬────────────────────────────┬────────────────────────┘
                         │ 文档上传                    │ 提问（HTTP / SSE）
                         │                            │
┌────────────────────────▼────────────────────────────▼────────────────────────┐
│                            接入层（Controller）                               │
│  DocumentController     ChatController     KnowledgeBaseController          │
│  HealthController                                                          │
│  ├── Sa-Token / JWT 认证（Token → UserContext）                              │
│  └── KB / Document 权限校验                                                  │
└────────────────────────┬────────────────────────────┬────────────────────────┘
                         │                            │
        ┌────────────────▼────────────────┐   ┌───────▼─────────────────────────┐
        │           离线索引管道            │   │           在线查询管道            │
        │                                 │   │                                 │
        │ 1. 文档上传 → MinIO              │   │ 1. Query Analyzer              │
        │ 2. 创建 IndexTask               │   │ 2. Query Router                │
        │ 3. 异步执行：                    │   │    ├─ 普通事实型 → RAG 检索      │
        │    ├─ 格式解析                   │   │    └─ 多跳关系型 → GraphRAG      │
        │    ├─ 文档清洗                   │   │                                 │
        │    ├─ Chunk 分块                 │   │ 3A. 普通 RAG 路径               │
        │    ├─ Embedding                 │   │    ├─ 自适应 Rewrite / HyDE     │
        │    ├─ 建立 HNSW 向量索引          │   │    ├─ HNSW + BM25 混合召回       │
        │    ├─ 建立 BM25 / 全文索引        │   │    ├─ Metadata Filter ACL       │
        │    ├─ 实体/关系抽取               │   │    ├─ RRF 融合                  │
        │    └─ 写入 Neo4j                 │   │    └─ Cross-Encoder 重排        │
        │ 4. 更新任务状态                   │   │                                 │
        └─────────────────────────────────┘   │ 3B. GraphRAG 路径              │
                                              │    ├─ 实体识别 / 关系查询         │
                                              │    ├─ Neo4j 多跳路径检索          │
                                              │    ├─ Metadata / ACL 过滤        │
                                              │    └─ 图谱结果 + 文档证据融合      │
                                              │                                 │
                                              │ 4. Context Budget Manager       │
                                              │    ├─ System Prompt 配额         │
                                              │    ├─ History 动态裁剪            │
                                              │    ├─ 历史摘要压缩                │
                                              │    └─ RAG Context 动态配额        │
                                              │ 5. 引用溯源组装                   │
                                              │ 6. LLM 流式生成（SSE）            │
                                              └─────────────────────────────────┘
                         │                            │
┌────────────────────────▼────────────────────────────▼────────────────────────┐
│                               Service 层                                     │
│  IndexService        ChunkService          EmbeddingService                 │
│  QueryRewriteService QueryRouterService    HybridSearchService              │
│  RrfFusionService    RerankerService       GraphRetrievalService            │
│  PermissionService   ContextBudgetService  RagQueryService                  │
│  RagEvaluationService（RAGAS / MRR / Recall@K / HitRate）                   │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────────┐
│                               数据访问层                                     │
│  KbDocumentRepository    DocChunkRepository    ChatSessionRepository        │
│  IndexTaskRepository     EvaluationRepository   GraphRepository             │
└───────────────┬──────────────────────────────┬──────────────────────┬────────┘
                │                              │                      │
┌───────────────▼──────────────┐  ┌────────────▼────────────┐  ┌────▼─────────────┐
│ PostgreSQL + PGVector        │  │ Redis 7                 │  │ Neo4j            │
│ ├── 业务表                   │  │ ├── Embedding 缓存       │  │ ├── 企业实体       │
│ ├── Chunk / Metadata         │  │ ├── Query Rewrite 缓存   │  │ ├── 实体关系       │
│ ├── HNSW 向量索引             │  │ ├── Query Router 缓存    │  │ └── 多跳关系路径    │
│ ├── BM25 / 全文索引           │  │ ├── 查询结果缓存          │  └──────────────────┘
│ └── ACL Metadata             │  │ └── IndexTask 进度推送   │
└───────────────┬──────────────┘  └─────────────────────────┘
                │
┌───────────────▼──────────────┐
│ MinIO                        │
│ 原始文档 / 附件存储            │
└──────────────────────────────┘

                                 离线评测闭环
┌──────────────────────────────────────────────────────────────────────────────┐
│  Golden Dataset → RAG Pipeline → RAGAS / MRR / Recall@K → 参数对比 → 调优   │
│  示例：MRR 0.62 → 0.78                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 架构分层说明

### 2.1 用户层

前端采用 **Vue3**，主要包含两类核心功能：

- **文档管理**
  - 创建/管理知识库
  - 上传、删除、查看文档
  - 查看文档解析和索引进度

- **智能对话**
  - 用户输入问题
  - 通过 HTTP / SSE 与后端通信
  - 实时展示大模型流式回答
  - 展示引用来源和原始文档信息

---

### 2.2 接入层（Controller）

负责接收前端请求，并完成认证、权限校验以及参数转换。

主要 Controller：

- `DocumentController`
  - 文档上传
  - 文档删除
  - 文档查询
  - 索引任务触发

- `ChatController`
  - RAG 问答
  - SSE 流式输出
  - 会话管理

- `KnowledgeBaseController`
  - 知识库创建
  - 知识库修改
  - 知识库删除
  - 成员和权限管理

- `HealthController`
  - 服务健康检查

安全机制：

1. **Sa-Token 认证拦截器**
   - 解析 JWT Token
   - 获取当前用户信息
   - 构建 `UserContext`

2. **知识库权限拦截器**
   - 校验当前用户是否拥有目标知识库访问权限
   - 防止跨知识库越权访问

---

## 3. 离线索引管道

离线索引管道负责将用户上传的原始文档转换为可以进行语义检索的向量数据。

整体流程：

```text
文档上传
   ↓
MinIO 保存原始文件
   ↓
创建 IndexTask
   ↓
异步文档解析
   ↓
文档分块 Chunk
   ↓
Embedding 向量化
   ↓
写入 PostgreSQL + PGVector
   ↓
更新 IndexTask 状态
```

### 3.1 文档上传

用户上传文档后：

- 原始文件存入 **MinIO**
- PostgreSQL 保存文档元数据
- 创建对应 `IndexTask`

### 3.2 文档解析

根据不同文件格式进行解析，例如：

- PDF
- Word
- Markdown
- TXT
- Excel

解析后统一转换为标准文本结构。

### 3.3 文档分块

通过 `ChunkService` 对文本进行 Chunk 切分。

每个 Chunk 通常保存：

```text
chunk_id
document_id
knowledge_base_id
content
chunk_index
metadata
token_count
```

### 3.4 Embedding

由 `EmbeddingService` 调用 Embedding 模型，将文本转换为向量。

```text
文本 Chunk
   ↓
Embedding Model
   ↓
Vector
```

生成后的向量写入 PostgreSQL 的 PGVector 字段中。

### 3.5 索引任务状态

`IndexTask` 用于记录索引流程状态，例如：

```text
PENDING
PARSING
CHUNKING
EMBEDDING
INDEXING
SUCCESS
FAILED
```

任务进度可以通过 Redis 推送给前端。

---

## 4. 在线查询管道

在线查询采用 **Query Router 智能分流 + Hybrid RAG + GraphRAG** 的双路径架构。

普通事实性问题优先走低延迟的传统 RAG 检索链路；涉及跨文档、多实体、多跳关系的问题才动态路由到 Neo4j 图谱增强检索，从而在保证复杂问答能力的同时，避免所有请求都进入图谱检索所带来的额外延迟和 Token 开销。

整体流程：

```text
用户问题
   ↓
JWT → UserContext
   ↓
Query Analyzer
   ↓
Query Router
   ├──────────────────────────────────────────┐
   │                                          │
   ▼                                          ▼
普通事实型 Query                         多跳 / 关系型 Query
   │                                          │
Rewrite / HyDE（按需）                   Entity / Relation Detect
   │                                          │
HNSW + BM25                               Neo4j Multi-hop Search
   │                                          │
Metadata Filter ACL                      Metadata / ACL Filter
   │                                          │
RRF Fusion                                Graph Evidence
   │                                          │
Cross-Encoder                             Document Evidence
   └───────────────────┬──────────────────────┘
                       ↓
              Context Budget Manager
                       ↓
                 Citation Builder
                       ↓
                     LLM
                       ↓
                  SSE Stream
```

---

### 4.1 Query Analyzer

`Query Analyzer` 首先判断问题特征：

- 是否为简单事实型查询
- 是否包含多个实体
- 是否存在实体之间的关系询问
- 是否涉及跨文档追溯
- 是否需要多跳推理
- 是否存在上下文指代
- 是否需要 Query Rewrite / HyDE

---

### 4.2 Query Router 智能分流

通过 `QueryRouterService` 将问题路由到不同检索路径。

```text
简单事实型问题
    ↓
Hybrid RAG

多实体 / 多跳 / 跨文档关系问题
    ↓
GraphRAG
```

#### 普通 RAG 路径

适合：

- 制度条款查询
- 文档内容问答
- 单文档事实查询
- 精确关键词查询
- 普通语义检索

优势：

- 延迟低
- Token 开销低
- 检索链路简单

#### GraphRAG 路径

适合：

- 某项目负责人还负责哪些项目？
- 某客户与哪些合同、产品、部门存在关联？
- 某问题从制度、项目到责任人的完整关系是什么？
- 跨多个文档才能得到答案的问题

GraphRAG 仅在必要时启用，从而避免每次查询都访问 Neo4j。

---

### 4.3 自适应 Query Rewrite / HyDE

对于简单 Query，直接进入检索。

对于复杂、模糊、表达不完整的问题，根据 Query Analyzer 结果按需启用：

- Query Rewrite
- Multi Query
- HyDE

```text
Simple Query
   ↓
Direct Retrieval

Complex Query
   ↓
Rewrite / HyDE
   ↓
Retrieval
```

这种自适应策略可以减少额外的大模型调用。

---

### 4.4 普通 RAG：HNSW + BM25 混合召回

第一阶段进行双路召回：

```text
Query
 ├── Embedding → HNSW Vector Search
 └── BM25 / Full Text Search
```

HNSW 解决语义相似问题，BM25 解决专有名词、错误码、型号、缩写、关键词精确匹配问题。

---

### 4.5 Metadata Filter 文档级动态授权

用户身份经过 JWT 认证后生成 `UserContext`。

```text
JWT Token
   ↓
UserContext
   ├── userId
   ├── tenantId
   ├── departmentId
   ├── roles
   └── accessibleDocumentIds / scopes
```

检索时不采用“先检索、后过滤”的方式，而是将 ACL 条件直接下推到检索层：

```text
Vector Search
   +
Metadata Filter
   ↓
Only Authorized Chunks
```

示例过滤字段：

```text
tenant_id
knowledge_base_id
document_id
department_id
security_level
allowed_role
allowed_user
```

这样可以保证未经授权的 Chunk 根本不会进入候选集。

权限链路：

```text
JWT
 ↓
UserContext
 ↓
PermissionService
 ↓
Build Metadata Filter
 ↓
HNSW / BM25 Search
 ↓
仅返回用户有权限的数据
```

通过 **JWT + Metadata Filter + KB/Document ACL**，在向量检索和索引层实现文档级动态授权，从底层降低越权检索风险。

---

### 4.6 RRF 融合

HNSW 与 BM25 的评分体系不同，因此采用 RRF 进行结果融合：

```text
HNSW Result
     +
BM25 Result
     ↓
RRF Fusion
     ↓
Candidate Top-N
```

公式：

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

---

### 4.7 Cross-Encoder 重排

RRF 得到的候选结果进入 Cross-Encoder：

```text
RRF Top 30 ~ 50
      ↓
Cross-Encoder
      ↓
Top 5 ~ 10
```

Cross-Encoder 联合编码 Query 和 Chunk，可以获得比单纯向量相似度更精确的相关性评分。

如果 Reranker 服务异常，则直接降级使用 RRF 结果。

---

## 4.8 Neo4j 企业实体关系图谱

为了处理跨文档、多实体和多跳关系问题，在离线索引阶段增加实体关系抽取。

```text
Document
   ↓
Parser
   ↓
Entity / Relation Extraction
   ↓
Entity Normalization
   ↓
Neo4j
```

图谱中可保存：

```text
(:Person)
(:Department)
(:Project)
(:Customer)
(:Contract)
(:Product)
(:Document)
```

关系示例：

```text
(:Person)-[:RESPONSIBLE_FOR]->(:Project)
(:Project)-[:BELONGS_TO]->(:Department)
(:Project)-[:RELATED_TO]->(:Contract)
(:Contract)-[:SIGNED_WITH]->(:Customer)
(:Document)-[:DESCRIBES]->(:Project)
```

---

### 4.9 GraphRAG 多跳检索

当 Query Router 判断为关系型或多跳问题时：

```text
User Query
   ↓
Entity Recognition
   ↓
Relation Intent
   ↓
Neo4j Cypher / Graph Search
   ↓
Multi-hop Path
   ↓
关联 Document / Chunk
   ↓
Evidence Fusion
```

GraphRAG 不只返回实体关系，还需要关联回原始文档 Chunk，避免只依赖图结构生成答案。

最终上下文可以由两部分组成：

```text
Graph Evidence
+
Document Evidence
```

这样既能够完成多跳推理，又保留原始文档引用和可追溯性。

---

## 4.10 精细化 Context Budget

最终送入 LLM 的上下文由以下部分组成：

```text
System Prompt
History
RAG / Graph Context
User Query
Reserved Output Tokens
```

由 `ContextBudgetService` 动态分配 Token。

例如：

```text
Model Context Window
        ↓
┌────────────────────────────┐
│ System Prompt Budget       │
│ Conversation Budget        │
│ RAG Context Budget         │
│ Graph Context Budget       │
│ Output Reserved Budget     │
└────────────────────────────┘
```

根据当前查询动态调整各部分占比，而不是固定拼接。

---

### 4.11 历史摘要压缩

对于长期会话，不持续携带全部历史消息。

采用：

```text
最近 N 轮原始消息
        +
历史摘要 Summary
```

当历史 Token 超过阈值时：

```text
Long History
   ↓
Summary Compression
   ↓
Conversation Summary
```

例如：

```text
优化前长期会话上下文：约 12K Token
优化后有效历史上下文：约 3K Token
```

核心策略：

- 最近若干轮保留原文
- 较早历史压缩为 Summary
- 与当前问题无关的历史直接裁剪
- RAG Context 根据相关性排序动态截断
- 为回答输出预留固定 Token

目标是降低：

- Prompt Token 成本
- 首 Token 延迟
- 无关历史干扰
- 长会话上下文膨胀

---

### 4.12 引用溯源与 SSE 流式生成

每条检索结果保留：

```text
knowledge_base_id
document_id
document_name
chunk_id
page_number
source_url
```

GraphRAG 结果同时保留关系路径和对应原始文档证据。

最终：

```text
Context
   ↓
Citation Builder
   ↓
LLM
   ↓
SSE
   ↓
Vue3
```

---

## 4.13 RAG 离线评测与数据驱动调优

使用 Golden Dataset 评估：

- MRR
- Recall@K
- HitRate@K
- RAGAS Context Precision
- RAGAS Context Recall
- RAGAS Faithfulness
- RAGAS Answer Relevancy

评测链路：

```text
Golden Dataset
      ↓
RAG Pipeline
      ↓
RAGAS / MRR / Recall@K
      ↓
参数对比
      ↓
调优：
├── HNSW Top-K
├── BM25 Top-K
├── RRF k
├── Reranker Candidate Size
├── Final Top-K
├── Rewrite / HyDE 触发阈值
└── Query Router 路由阈值
      ↓
重新评测
```

示例：

```text
优化前 MRR：0.62
优化后 MRR：0.78
```

## 5. Service 层

| Service | 主要职责 |
|---|---|
| `IndexService` | 文档解析、分块、向量化、索引和图谱构建编排 |
| `ChunkService` | 文档 Chunk 切分与 Metadata 维护 |
| `EmbeddingService` | Query / Chunk Embedding |
| `QueryRewriteService` | Query Rewrite、Multi Query、HyDE |
| `QueryRouterService` | 普通 RAG / GraphRAG 智能分流 |
| `HybridSearchService` | HNSW + BM25 混合召回 |
| `RrfFusionService` | 多路结果 RRF 融合 |
| `RerankerService` | Cross-Encoder 精排与降级 |
| `GraphRetrievalService` | Neo4j 实体关系和多跳路径检索 |
| `PermissionService` | JWT 用户上下文与文档级 ACL 构建 |
| `ContextBudgetService` | System / History / RAG Context 动态 Token 配额 |
| `RagQueryService` | 在线 RAG / GraphRAG 总流程编排 |
| `RagEvaluationService` | RAGAS、MRR、Recall@K 等离线评测 |

核心编排：

```text
RagQueryService
    ↓
QueryRouterService
    ├── HybridSearchService
    │      ↓
    │   RRF → Reranker
    │
    └── GraphRetrievalService
           ↓
        Neo4j Multi-hop

两路结果
    ↓
ContextBudgetService
    ↓
LLM
```

---

## 6. 数据访问层

Repository 层负责数据库访问。

主要 Repository：

| Repository | 说明 |
|---|---|
| `KbDocumentRepository` | 知识库文档 |
| `DocChunkRepository` | 文档 Chunk 与向量 |
| `ChatSessionRepository` | 对话会话 |
| `IndexTaskRepository` | 文档索引任务 |

---

## 7. 数据存储

### 7.1 PostgreSQL + PGVector

PostgreSQL 负责：

- 业务数据
- 知识库和文档元数据
- Chunk 内容
- Embedding Vector
- HNSW 向量索引
- BM25 / 全文检索
- ACL Metadata

Chunk Metadata 可包含：

```text
tenant_id
knowledge_base_id
document_id
department_id
security_level
allowed_role
allowed_user
```

检索时直接基于这些 Metadata 进行授权过滤。

---

### 7.2 Neo4j

Neo4j 用于存储企业知识实体与关系。

典型实体：

```text
Person
Department
Project
Customer
Contract
Product
Document
```

主要用于：

- 多实体关系查询
- 跨文档追溯
- 多跳路径检索
- GraphRAG

---

### 7.3 Redis 7

Redis 主要用于：

- Embedding 缓存
- Query Rewrite / HyDE 缓存
- Query Router 结果缓存
- 热点查询缓存
- IndexTask 任务进度推送
- 会话 Summary 缓存

---

### 7.4 MinIO

MinIO 保存用户上传的原始文档和附件。

数据库与图数据库只保存结构化数据、Metadata、Chunk、向量和图谱信息。

---

## 8. 核心业务链路

### 文档索引链路

```text
Vue3
 ↓
DocumentController
 ↓
PermissionService
 ↓
MinIO
 ↓
IndexTask
 ↓
Parser
 ↓
ChunkService
 ├───────────────┐
 ↓               ↓
Embedding      Entity / Relation Extraction
 ↓               ↓
PGVector        Neo4j
```

### 普通 RAG 查询链路

```text
Vue3
 ↓
ChatController
 ↓
JWT / UserContext
 ↓
Query Router
 ↓
Hybrid RAG
 ↓
Rewrite / HyDE（按需）
 ↓
HNSW + BM25
 ↓
Metadata Filter ACL
 ↓
RRF
 ↓
Cross-Encoder
 ↓
Context Budget
 ↓
LLM
 ↓
SSE
```

### GraphRAG 查询链路

```text
Vue3
 ↓
ChatController
 ↓
JWT / UserContext
 ↓
Query Router
 ↓
GraphRAG
 ↓
Entity / Relation Detection
 ↓
Neo4j Multi-hop Search
 ↓
ACL Filter
 ↓
Graph Evidence + Document Evidence
 ↓
Context Budget
 ↓
LLM
 ↓
SSE
```

### 长会话上下文压缩链路

```text
Conversation History
 ↓
Token Threshold Check
 ↓
Recent Messages + Historical Summary
 ↓
Context Budget Manager
 ↓
约 12K Token → 约 3K Token
```

### RAG 评测调优链路

```text
Golden Dataset
 ↓
RAG Pipeline
 ↓
RAGAS / MRR / Recall@K
 ↓
参数对比
 ↓
HNSW / BM25 / RRF / Reranker / Router 参数调优
 ↓
重新评测
```

---

## 9. 架构特点

- **HNSW + BM25 混合召回，兼顾语义匹配与关键词精确检索**
- **RRF 融合 + Cross-Encoder 重排形成三阶段检索链路**
- **复杂 Query 按需启用 Query Rewrite / HyDE，降低无效模型调用**
- **Query Router 自动区分普通 RAG 与 GraphRAG**
- **Neo4j 支持跨文档、多实体、多跳关系追溯**
- **仅复杂关系问题进入图谱检索，避免额外延迟和 Token 消耗**
- **JWT → UserContext → Metadata Filter，将 ACL 下推到检索层**
- **未授权 Chunk 在召回阶段即被过滤，降低越权检索风险**
- **System Prompt / History / RAG Context 按 Token Budget 动态分配**
- **历史摘要压缩使长期会话上下文可由约 12K Token 压缩至约 3K Token**
- **Reranker 异常自动降级至 RRF 结果**
- **RAGAS + MRR + Recall@K 构成离线评测闭环**
- **通过 Golden Dataset 驱动 Top-K、RRF、Reranker、Router 等参数调优**
- **示例检索指标 MRR：0.62 → 0.78**
- **完整引用溯源，GraphRAG 结果同时绑定原始文档证据**
- **Redis 缓存降低重复 Embedding、改写和路由计算成本**
- **MinIO、PostgreSQL/PGVector、Neo4j 分别承担对象、检索和图谱存储职责**
- **SSE 提供大模型流式响应**
