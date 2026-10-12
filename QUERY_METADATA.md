# 统计与元数据过滤

当前实现文档目录精确计数、分页清单和共享元数据筛选。统计不使用 Top-K 原文片段推算总量。
统计与 Dense、BM25、图投影、原文回读及返回前授权复核使用同一套参数化过滤条件。

## 查询范围

默认口径为当前账号可读、用户选择范围内、已发布且活动索引批次为 READY 的文档。
每份文档计数一次；修订历史和多个 Chunk 不增加文档数量。撤回、删除、跨租户和无权限
文档不能进入统计。无符合条件的文档返回成功的 0；数据库故障不包装成 0。

计数和清单依据同一次目录查询快照计算。清单按当前发布修订上传时间倒序、文档 ID
倒序排序，分页不改变 total。不同分页请求会重新查询，executed_at 标明每次查询时间。
首期最多支持 10000 份符合条件的文档，以保存并复核完整统计来源；超出返回
`422 / STRUCTURED_SCOPE_LIMIT`，要求缩小范围，不截断后返回不完整总量。

## 接口

`POST /api/query/plan`、`/search`、`/answer`、`/answer/stream` 和聊天回答接口均接受
`filters`；目录分页接受 `offset`（0～100000）、`limit`（1～100，默认 20）。
字段白名单和类型校验由服务端执行，客户端不能提交 SQL、租户或执行权限。
资源 ID 和标准化过滤中的部门 ID 均使用十进制字符串，避免浏览器大整数精度损失。

```json
{
  "query": "文档有多少份？",
  "knowledge_base_ids": ["1"],
  "filters": [
    {"field": "file_type", "operator": "eq", "value": "PDF"},
    {"field": "business_status", "operator": "eq", "value": "EFFECTIVE"},
    {"field": "uploaded_at", "operator": "range", "value": "去年"}
  ]
}
```

将问题改为“列出所有文档”，并提交 `offset`、`limit` 即可分页。响应的 `request`
保存此次实际请求，前端翻页重用它，不读取已经修改的问题草稿或筛选控件。

| 条件字段 | 数据来源 | 支持的操作 |
| --- | --- | --- |
| department | 当前租户部门名称或代码 → document.department_id | eq、neq；重名或未知部门要求澄清 |
| document_name | 活动索引关联修订的 original_filename | eq、neq、contains |
| document_code | document.document_code | eq、neq、contains |
| file_type | 活动修订 mime_type | eq、neq；PDF、DOCX、XLSX、XLS、Excel、Word、TXT、Markdown |
| business_status | document.business_status | eq、neq；UNKNOWN、DRAFT、EFFECTIVE、ARCHIVED、REPEALED |
| uploaded_at | 活动修订 created_at | eq、gt、gte、lt、lte、range；精确带时区时间戳还支持 neq |
| event_time | document.event_date | eq、gt、gte、lt、lte、range |

多个条件使用 AND。显式筛选和自然语言抽取条件共同生效，不能互相覆盖。contains 是
字面子串匹配，百分号和下划线不作为通配符。可空字段的 neq 使用 SQL 不等于语义，
不包含 NULL；UNKNOWN 业务状态是独立枚举，不等同于生效。

日期按请求执行时间和 Asia/Shanghai 解析，支持明确年、月、日、今年、去年、本月、
上月、本周、上周、今天和昨天；范围统一左闭右开。日期的 lte 包含当天，精确时间戳
保留瞬时比较。uploaded_at 表示当前发布版本上传时间，文档首次创建时间不是该字段。
event_time 表示业务生效或事件日期，不使用上传时间代替。

“业务状态为生效的文档有多少份”等受支持句式可由 Analyzer 提取条件后执行。
模糊时间、未知枚举、无明确比较方式、条件冲突均返回澄清。分组、任意排序、OR、正文
主题统计、金额汇总和复合任务尚未开放；不能删除这些要求后执行简化统计。
Analyzer 故障时自然语言统计仍明确受阻；显式过滤在普通检索降级时继续保留。

## 结果和会话

`structured` 与 `retrieval` 独立。目录结果包含 operation、total、documents、offset、
limit、applied_filters、executed_at 和 scope。目录执行时原文检索标为 skipped。
回答使用 `answer.status=structured`，确定性文本在 message 中，statements 为空；
原文回答的逐字引用规则保持不变。SSE 发送完整统计 answer 与 done，不调用回答模型。

服务端私有 provenance 保存全部参与统计的文档版本和元数据指纹，以及知识库授权
锚点和所选文档范围，不在 HTTP 响应中暴露。聊天提交、恢复历史和再次发布时复核来源、
会话身份、权限和文档属性；零结果也复核知识库与所选文档权限。
依赖旧统计的追问同时保存并复核其来源和选择范围。
条件和分页参数进入请求幂等摘要，重试不能改变条件后复用原请求 ID。

## 文档元数据维护

`GET /api/documents/{document_id}/metadata` 返回标题、业务编号、业务状态、业务日期、
归属部门、updated_at 和 can_write。`PATCH` 同路径要求 WRITE 权限，并提交：

```json
{
  "expected_updated_at": "2026-10-11T01:00:00+00:00",
  "document_code": "制度-2026-01",
  "business_status": "EFFECTIVE",
  "event_date": "2026-01-01"
}
```

只允许 title、document_code、business_status、event_date；编号和日期可用 null 清空。
updated_at 过期返回 409 METADATA_CONFLICT；同知识库编号重复返回 409
DOCUMENT_CODE_CONFLICT。修改记录审计，不能通过此接口修改部门、租户或授权字段。
新上传支持这些业务属性；修订上传保留原文档属性。已有文档可在文档页“属性”中补录。

## 代码与验证

- query_metadata：原始和标准化条件、日期/部门解析、固定 SQL 谓词。
- query_structured：目录查询、结果和完整来源复核。
- query_execution：共享查询执行分派，search、answer、chat 共用。

测试覆盖条件边界、权限复核、活动修订、计数与分页、零结果、元数据并发修改、统计
历史来源和 JSON/SSE 客户端协议。可选真实 PostgreSQL 测试只使用本地库的隔离 schema。
