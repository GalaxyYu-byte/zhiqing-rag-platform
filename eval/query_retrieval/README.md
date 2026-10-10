# 检索与回答的固定问题验收

本轮结果、逐条核对及模型服务限制见 [验收记录](ACCEPTANCE.md)。

问题来自现有虚构企业文档。首次模型调用前固定 11 道正向题及逐字答案依据，覆盖 PDF、
DOCX、Markdown、TXT、XLSX 各两题和一题跨文档；另有澄清、统计能力限制和无答案控制题。

`v1/cases.json` 是首次基线。`v2/cases.json` 保持全部正向题和答案依据不变，增加一道人为
设定的无答案题，区分未接入元数据过滤的情况和有相似检索结果但原文不能回答的情况。
引用的冻结输入为 `eval/graph_build/v1/chunks.jsonl`。每次运行都会复核当前账号权限、
活动批次、原文和哈希；来源变化时停止评估，不沿用过期答案。

## 运行

在项目根目录执行。脚本加载当前 FastAPI 应用并调用真实 ASGI HTTP 接口，不需启动后端。
鉴权、PostgreSQL、Redis、分析、嵌入和重排均使用现有配置；`--answer` 还调用生成和复核模型。

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_query_retrieval.py --cases eval/query_retrieval/v2/cases.json --output eval/query_retrieval/runs/new-run --email admin@zhiqing.test --credentials-file data/dev-seed/credentials.json
```

凭据文件沿用本机被忽略的私有测试数据。也可通过 `QUERY_EVAL_PASSWORD` 环境变量提供密码，
不将密码放在命令行。脚本不打印或保存 Token、Cookie、密码。评估结束后只撤销自己新建的
认证会话及对应 Redis Refresh 摘要，不退出其他设备，不修改文档。

`--answer` 执行回答链；`--only Q01` 可按题号缩小范围；`--capture-model-output` 仅用于
回答排查，保存当次模型草稿和复核结果。草稿包含未验收输出，不能当作已通过的回答。
真实模型调用需网络访问并可能产生用量；本地替身测试不会调用模型。

## 结果口径

- 正向题完整命中：Top-5 中存在指定文档的逐字答案依据；跨文档题需要全部依据命中。
- 允许同一文档的其他块含有相同依据；不是要求固定的 Chunk 排名。
- 逐字引文可定位只证明引用来自原文，不证明陈述受到原文支持。
- 无答案题仍可能召回相似块，须在回答环节拒答，不能要求 Top-K 必须为空。
- 模型复核通过不能代替独立事实验收。固定集属于开发回归，不是独立留出集。

各次运行独立保存结果、耗时及提示词。首次基线及修复后结果均保留在 `runs/`，避免覆盖历史。

同一原文件修订完成表格索引重建后，可加 `--allow-reindexed-sources`。脚本仍要求原文件
SHA-256、租户、知识库、文档和修订匹配冻结来源，并逐字确认每条原答案依据存在于活动
分块；问题和答案依据保持不变。报告的 `source_transitions` 记录新旧批次与分块哈希，
默认模式仍要求原冻结分块完全一致。`v3` 在原十五题之外增加三道预先固定的并列问题，
属于扩展开发回归，不是外部独立验收。
