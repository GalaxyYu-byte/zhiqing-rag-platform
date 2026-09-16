from functools import lru_cache

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ============================================================
    # Application
    # ============================================================

    app_name: str = "zq-rag-py"
    app_env: str = "dev"

    # ============================================================
    # PostgreSQL
    # ============================================================

    db_host: str = "103.236.92.173"
    db_port: int = 5432
    db_name: str = "ragkb"
    db_username: str = "ragkb"
    db_password: str = "ragkb"

    db_pool_size: int = 20
    db_max_overflow: int = 10
    db_pool_timeout: int = 30
    db_echo: bool = False

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://"
            f"{self.db_username}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ============================================================
    # Redis
    # ============================================================

    redis_host: str = "103.236.92.173"
    redis_port: int = 6379
    redis_password: str = "redis6379"
    redis_db: int = 0

    redis_max_connections: int = 16

    @computed_field
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return (
                f"redis://:{self.redis_password}"
                f"@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            )

        return (
            f"redis://"
            f"{self.redis_host}:{self.redis_port}/{self.redis_db}"
        )

    # ============================================================
    # Async Task
    # ============================================================

    task_core_workers: int = 4
    task_max_workers: int = 8
    task_queue_capacity: int = 100

    # ============================================================
    # File Upload
    # ============================================================

    max_file_size_mb: int = 50
    max_request_size_mb: int = 100

    # ============================================================
    # DashScope / LLM
    # ============================================================

    dashscope_api_key: str

    openai_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    chat_model: str = "qwen-plus"
    chat_temperature: float = 0.1
    chat_max_tokens: int = 2048

    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    # DashScope text-embedding-v3 单次最多接收 10 条文本。
    embedding_batch_size: int = 10
    embedding_concurrency: int = 4
    embedding_request_timeout_seconds: float = 30.0
    embedding_retry_attempts: int = 5
    embedding_retry_min_wait_seconds: float = 0.5
    embedding_retry_max_wait_seconds: float = 30.0

    # ============================================================
    # MinIO
    # ============================================================

    minio_endpoint: str = "http://103.236.92.173:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minio9000"
    minio_bucket: str = "rag-documents"

    # ============================================================
    # Neo4j 图数据库
    # ============================================================

    # 密码为空时仅表示尚未启用图谱；驱动采用懒加载，不会在导入配置时连接服务器。
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50
    neo4j_connection_timeout_seconds: float = 15.0

    # 图抽取独立于向量索引，默认不自动执行。qwen-plus 使用 JSON Object 后再由
    # Pydantic 严格校验；切换到支持 JSON Schema 的模型后可升级调用方式。
    graph_extraction_enabled: bool = False
    graph_extraction_model: str = "qwen-plus"
    graph_extraction_temperature: float = 0.0
    graph_extraction_max_tokens: int = 4096
    graph_extraction_timeout_seconds: float = 60.0
    graph_extraction_retry_attempts: int = 3
    graph_extractor_version: str = "graph-extractor-v1"
    graph_task_queue_name: str = "arq:graph"
    graph_worker_max_jobs: int = 2
    graph_task_max_retry: int = 3
    graph_task_retry_base_seconds: float = 5.0
    graph_heartbeat_interval_seconds: int = 15
    graph_lease_seconds: int = 90

    # ============================================================
    # Reranker
    # ============================================================

    reranker_endpoint: str
    reranker_model: str = "gte-rerank-v2"
    reranker_timeout_ms: int = 800
    reranker_top_n: int = 5

    # ============================================================
    # RAG
    # ============================================================

    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64

    rag_vector_top_k: int = 20
    rag_fulltext_top_k: int = 20
    rag_return_top_n: int = 5
    rag_min_score: float = 0.5

    rag_context_max_tokens: int = 3000

    # ============================================================
    # Cache
    # ============================================================

    embedding_cache_ttl: int = 7 * 24 * 60 * 60
    embedding_cache_version: str = "v1"
    embedding_local_cache_size: int = 2_000
    embedding_local_cache_ttl: int = 30 * 60
    embedding_lock_ttl: int = 180
    embedding_lock_wait_seconds: float = 30.0
    embedding_lock_poll_seconds: float = 0.2
    query_cache_ttl: int = 10 * 60

    # ============================================================
    # Document indexing
    # ============================================================

    index_upsert_batch_size: int = 300
    index_db_retry_attempts: int = 5
    index_heartbeat_interval_seconds: int = 15
    index_lease_seconds: int = 60
    index_task_max_retry: int = 3
    index_task_retry_base_seconds: float = 2.0

    # ============================================================
    # JWT
    # ============================================================

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"

    token_expire_seconds: int = 86400
    token_prefix: str = "Bearer"
    token_header: str = "Authorization"

    # ============================================================
    # Monitoring
    # ============================================================

    metrics_enabled: bool = True

    # ============================================================
    # Logging
    # ============================================================

    log_level: str = "INFO"

    # ============================================================
    # Pydantic Settings
    # ============================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
