"""分块批量向量化：按响应索引匹配，校验后才返回完整结果。"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from redis.asyncio import Redis
from redis.exceptions import RedisError

from zhiqing_rag.core.config import Settings, get_settings

from .chunking import DocumentChunkDraft

if TYPE_CHECKING:
    from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
EMBEDDING_DIMENSIONS = 1024


class EmbeddingValidationError(ValueError):
    """模型返回的向量或索引不满足完整性要求。"""


def validate_embedding(vector: Sequence[float]) -> None:
    """保证向量可以保存为 1024 维 pgvector 并用于余弦检索。"""
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise EmbeddingValidationError(f"向量维度应为 1024，实际为 {len(vector)}")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
        raise EmbeddingValidationError("向量必须只包含数值")
    if any(not math.isfinite(value) for value in vector):
        raise EmbeddingValidationError("向量包含 NaN 或无穷大")
    # pgvector 使用 float32；不能让有限的 Python float 在入库时溢出。
    if any(abs(value) > 3.4028234663852886e38 for value in vector):
        raise EmbeddingValidationError("向量数值超出 float32 范围")
    if not any(abs(value) >= 1.401298464324817e-45 for value in vector):
        raise EmbeddingValidationError("向量不能为全零向量")


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """保留原分块及生成它的模型名称，向量不允许被原地修改。"""

    chunk: DocumentChunkDraft
    embedding: tuple[float, ...]
    model_name: str

    def __post_init__(self) -> None:
        validate_embedding(self.embedding)


@dataclass(slots=True)
class EmbeddingStats:
    cache_hits: int = 0
    model_chunks: int = 0
    model_batches: int = 0


async def embed_chunks(
    chunks: Sequence[DocumentChunkDraft],
    *,
    settings: Settings | None = None,
    client: AsyncOpenAI | None = None,
    redis_client: Redis | None = None,
    stats: EmbeddingStats | None = None,
) -> tuple[EmbeddedChunk, ...]:
    """每批至多 10 块，全部成功才返回；传入的客户端由调用方管理。"""
    settings = settings or get_settings()
    stats = stats if stats is not None else EmbeddingStats()
    stats.cache_hits = stats.model_chunks = stats.model_batches = 0
    if settings.embedding_dimensions != EMBEDDING_DIMENSIONS:
        raise ValueError("当前数据库只支持 1024 维向量")
    chunks = tuple(chunks)
    if len({chunk.chunk_index for chunk in chunks}) != len(chunks):
        raise ValueError("chunk_index 不能重复")
    if any(chunk.chunk_index < 0 or not chunk.embedding_content.strip() for chunk in chunks):
        raise ValueError("分块序号必须非负，embedding_content 不能为空")
    if not chunks:
        return ()
    # 局部导入，缓存解码复用本模块的向量校验，避免导入循环。
    from .embedding_cache import EmbeddingCache, create_cache_client

    owned_redis = None
    cache = None
    if settings.embedding_cache_enabled:
        if redis_client is None:
            owned_redis = create_cache_client(settings)
            redis_client = owned_redis
        cache = EmbeddingCache(redis_client, settings)
    try:
        return await _embed_with_cache(chunks, settings, client, cache, stats)
    finally:
        if owned_redis is not None:
            try:
                await owned_redis.aclose()
            except RedisError as exc:
                logger.warning("向量缓存连接关闭失败: %s", type(exc).__name__)


async def _embed_with_cache(
    chunks: tuple[DocumentChunkDraft, ...],
    settings: Settings,
    client: AsyncOpenAI | None,
    cache: EmbeddingCache | None,
    stats: EmbeddingStats,
) -> tuple[EmbeddedChunk, ...]:
    # 相同实际输入在同一调用中只计算一次，再映射回每个原始分块。
    unique = {chunk.embedding_content: chunk for chunk in chunks}
    vectors = await cache.get_many(tuple(unique)) if cache is not None else {}
    stats.cache_hits = sum(chunk.embedding_content in vectors for chunk in chunks)
    missing = tuple(chunk for text, chunk in unique.items() if text not in vectors)
    stats.model_chunks = len(missing)
    if missing:
        generated = await _generate_chunks(missing, settings, client, stats)
        new_vectors = {item.chunk.embedding_content: item.embedding for item in generated}
        if cache is not None:
            await cache.set_many(new_vectors)
        vectors.update(new_vectors)
    return tuple(
        EmbeddedChunk(chunk, vectors[chunk.embedding_content], settings.embedding_model)
        for chunk in chunks
    )


async def _generate_chunks(
    chunks: tuple[DocumentChunkDraft, ...],
    settings: Settings,
    client: AsyncOpenAI | None,
    stats: EmbeddingStats,
) -> tuple[EmbeddedChunk, ...]:
    if client is not None:
        return await _embed_batches(chunks, settings, client, stats)
    key = settings.dashscope_api_key.get_secret_value()
    if not key:
        raise ValueError("未配置 DASHSCOPE_API_KEY")
    async with AsyncOpenAI(
        api_key=key,
        base_url=settings.openai_base_url,
        timeout=30.0,
        max_retries=2,
    ) as owned_client:
        return await _embed_batches(chunks, settings, owned_client, stats)


async def _embed_batches(
    chunks: tuple[DocumentChunkDraft, ...],
    settings: Settings,
    client: AsyncOpenAI,
    stats: EmbeddingStats,
) -> tuple[EmbeddedChunk, ...]:
    result: list[EmbeddedChunk] = []
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        stats.model_batches += 1
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=[chunk.embedding_content for chunk in batch],
            dimensions=EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        if len(response.data) != len(batch):
            raise EmbeddingValidationError(f"第 {start // BATCH_SIZE + 1} 批返回数量不匹配")
        by_index: dict[int, tuple[float, ...]] = {}
        for item in response.data:
            if item.index not in range(len(batch)) or item.index in by_index:
                raise EmbeddingValidationError("响应 index 重复或越界")
            vector = tuple(item.embedding)
            validate_embedding(vector)
            by_index[item.index] = vector
        result.extend(
            EmbeddedChunk(chunk, by_index[index], settings.embedding_model)
            for index, chunk in enumerate(batch)
        )
    return tuple(result)
