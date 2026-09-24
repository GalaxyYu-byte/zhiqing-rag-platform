# 知擎 RAG 平台

新项目位于 `D:\code\zhiqing-rag-platform`，Python 固定为 **3.12**，使用 uv 管理环境和依赖。
当前已完成开发环境、健康检查入口、独立数据库和 27 张业务表的设计与初始化；身份与知识库调试数据已录入。部分业务 ORM 已实现，业务 API 与 Alembic 将后续完善。

实体与技术方案文档单独维护；各业务模块将分阶段实现。

## 开始开发

在 PowerShell 中执行：

```powershell
cd D:\code\zhiqing-rag-platform
uv sync --locked
uv run python -m zhiqing_rag.main
```

启动后访问 `http://127.0.0.1:8000/docs`。
`GET /health/live` 检查进程；`GET /health/ready` 只读检查数据库、Redis、MinIO 和已启用的 Neo4j。
服务启动不连接中间件；就绪接口会在依赖不可用时返回 503。

## 配置

本机 `.env` 已迁移旧项目的中间件及模型配置，并生成新的 JWT 密钥。
新机器先将 `.env.example` 复制为 `.env`，填写实际凭据并生成独立的 JWT 密钥。
系统环境变量优先于 `.env`。本地缓存保存在 `.uv-cache`。

```powershell
uv run python -m zhiqing_rag.checks
uv run ruff check .
uv run ruff format --check .
```

环境检查仅验证中间件连接、pgvector 扩展和存储桶存在性；不会建表、建桶或写入队列。
当前连接独立数据库 `zhiqing_rag_platform`，项目 schema 为 `zhiqing_rag`。环境检查中的 `project_schema: present` 只表示 schema 存在，不代表业务表已初始化。
模型项显示凭据是否配置，不代表已完成模型 API 调用验证。
DeepSeek 为后续查询分析预留，可暂不配置。

## 技术组成

| 用途 | 组件 |
| --- | --- |
| HTTP 服务 / 流式输出 | FastAPI、Uvicorn、SSE |
| 配置 | Pydantic Settings、dotenv |
| 关系与向量数据 | PostgreSQL、pgvector、SQLAlchemy 2、psycopg 3 |
| 关键词检索 | PostgreSQL pg_search（BM25，已确认扩展存在） |
| 数据迁移 | Alembic（已安装，待表设计后配置迁移） |
| 缓存与异步任务 | Redis、ARQ |
| 文件存储 | MinIO |
| 图谱存储 | Neo4j |
| 模型接口 | OpenAI SDK、LangChain、DashScope；预留 DeepSeek |
| 文档解析 | PDF、DOCX、XLS/XLSX、Markdown |
| 开发工具 | pytest、pytest-asyncio、Ruff |

使用 `src/zhiqing_rag` 布局，uv 以 editable 模式安装本项目。
依赖沿用旧项目主技术栈，通过 `uv.lock` 固定实际版本；评估工具将在评估方案确定后单独配置。
新依赖使用 `uv add 包名`，开发工具使用 `uv add --dev 包名`。
修改依赖后提交 `pyproject.toml` 和 `uv.lock`；不要提交 `.env`、`.venv` 和缓存目录。

## 后续数据隔离约定

现阶段沿用已有服务实例与连接凭据。PostgreSQL 已新建独立数据库 `zhiqing_rag_platform`，其中 schema 为 `zhiqing_rag`；
Redis/ARQ 使用 `zhiqing-rag:` 前缀；MinIO 对象使用 `zhiqing-rag-platform/` 前缀；
Neo4j 预留 `GRAPH_NAMESPACE`。这些配置需要后续存储层、任务层和迁移代码落实。
数据库、schema 与 27 张业务表已创建；身份与知识库调试数据已录入，文档类数据待后续离线导入。
