"""从项目 .env 或系统环境变量读取配置，密钥不进入代码和日志。"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import URL

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    app_name: str = "zhiqing-rag-platform"
    app_env: Literal["dev", "test", "production"] = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"

    db_host: str = "127.0.0.1"
    db_port: int = Field(default=5432, ge=1, le=65535)
    db_name: str = "zhiqing_rag_platform"
    db_username: str = "ragkb"
    db_password: SecretStr = Field(default=SecretStr(""), repr=False)
    db_schema: str = Field(default="zhiqing_rag", pattern=r"^[a-z][a-z0-9_]*$")
    db_pool_size: int = Field(default=10, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    db_pool_timeout: int = Field(default=30, ge=1)
    db_echo: bool = False

    redis_host: str = "127.0.0.1"
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_password: SecretStr = Field(default=SecretStr(""), repr=False)
    redis_db: int = Field(default=0, ge=0)
    redis_max_connections: int = Field(default=16, ge=1)
    redis_key_prefix: str = "zhiqing-rag:"
    arq_queue_name: str = "zhiqing-rag:arq:default"

    minio_endpoint: str = "http://127.0.0.1:9000"
    minio_access_key: SecretStr = Field(default=SecretStr(""), repr=False)
    minio_secret_key: SecretStr = Field(default=SecretStr(""), repr=False)
    minio_bucket: str = "rag-documents"
    minio_object_prefix: str = "zhiqing-rag-platform/"

    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = Field(default=SecretStr(""), repr=False)
    neo4j_database: str = "neo4j"
    graph_namespace: str = "zhiqing-rag-platform"

    dashscope_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    openai_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-plus"
    chat_temperature: float = Field(default=0.1, ge=0, le=2)
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = Field(default=1024, gt=0)
    reranker_endpoint: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    reranker_model: str = "gte-rerank-v2"
    deepseek_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    query_analyzer_model: str = "deepseek-flash"

    jwt_secret_key: SecretStr = Field(default=SecretStr(""), repr=False)
    jwt_issuer: str = "zhiqing-rag-platform"
    jwt_audience: str = "zhiqing-rag-web"
    refresh_cookie_name: str = "zhiqing_refresh_token"
    refresh_cookie_secure: bool = False
    service_timeout_seconds: float = Field(default=5, gt=0, le=30)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        hide_input_in_errors=True,
    )

    @property
    def database_url(self) -> URL:
        # URL.create 会正确处理密码中的 @、冒号等特殊字符。
        return URL.create(
            "postgresql+psycopg",
            username=self.db_username,
            password=self.db_password.get_secret_value(),
            host=self.db_host,
            port=self.db_port,
            database=self.db_name,
        )

    @model_validator(mode="after")
    def validate_production(self):
        if self.app_env == "production":
            if len(self.jwt_secret_key.get_secret_value().encode()) < 32:
                raise ValueError("生产环境 JWT_SECRET_KEY 至少需要 32 字节")
            if not self.refresh_cookie_secure:
                raise ValueError("生产环境必须启用 REFRESH_COOKIE_SECURE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
