"""Redis 向量缓存，不保存正文；缓存故障按未命中处理。"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from collections.abc import Mapping, Sequence

from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff
from redis.exceptions import RedisError

from zhiqing_rag.core.config import Settings

from .embedding import EMBEDDING_DIMENSIONS, validate_embedding

logger = logging.getLogger(__name__)
VECTOR_STRUCT = struct.Struct(f"<{EMBEDDING_DIMENSIONS}f")


def create_cache_client(settings: Settings) -> Redis:
    return Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        password=settings.redis_password.get_secret_value() or None,
        db=settings.redis_db,
        max_connections=settings.redis_max_connections,
        socket_connect_timeout=settings.embedding_cache_timeout_seconds,
        socket_timeout=settings.embedding_cache_timeout_seconds,
        retry=Retry(NoBackoff(), 0),
    )


class EmbeddingCache:
    def __init__(self, client: Redis, settings: Settings) -> None:
        self.client = client
        self.ttl = settings.embedding_cache_ttl_seconds
        spec = json.dumps(
            {
                "provider": "dashscope",
                "base_url": settings.openai_base_url.rstrip("/"),
                "model": settings.embedding_model,
                "dimensions": settings.embedding_dimensions,
                "revision": settings.embedding_cache_revision,
                "encoding_format": "float",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        spec_hash = hashlib.sha256(spec.encode()).hexdigest()
        self.prefix = f"{settings.redis_key_prefix}embedding:v2:float32:{spec_hash}:"

    def key(self, text: str) -> str:
        return self.prefix + hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get_many(self, texts: Sequence[str]) -> dict[str, tuple[float, ...]]:
        if not texts:
            return {}
        try:
            values = await self.client.mget([self.key(text) for text in texts])
        except RedisError as exc:
            logger.warning("向量缓存读取失败，回退模型调用: %s", type(exc).__name__)
            return {}
        result = {}
        for text, value in zip(texts, values, strict=True):
            if value is None:
                continue
            try:
                if not isinstance(value, bytes) or len(value) != VECTOR_STRUCT.size:
                    continue
                vector = VECTOR_STRUCT.unpack(value)
                validate_embedding(vector)
                result[text] = vector
            except (ValueError, TypeError, struct.error):
                # 损坏、过期格式或不合法数值按未命中处理，成功生成后会覆盖。
                continue
        return result

    async def set_many(self, vectors: Mapping[str, tuple[float, ...]]) -> None:
        if not vectors:
            return
        for vector in vectors.values():
            validate_embedding(vector)
        try:
            async with self.client.pipeline(transaction=False) as pipeline:
                for text, vector in vectors.items():
                    pipeline.set(self.key(text), VECTOR_STRUCT.pack(*vector), ex=self.ttl)
                await pipeline.execute()
        except RedisError as exc:
            logger.warning("向量缓存写入失败，继续返回已校验向量: %s", type(exc).__name__)
