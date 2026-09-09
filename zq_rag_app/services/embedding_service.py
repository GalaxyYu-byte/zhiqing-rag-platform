"""带两级缓存、批处理、分布式锁和网络重试的 Embedding 服务。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import secrets
import struct
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from redis.asyncio import Redis
from redis.exceptions import RedisError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_random_exponential

from ..core.config import settings
from ..core.redis import redis_binary_client


logger = logging.getLogger(__name__)

_LOCK_RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""

_CACHE_IO_ERRORS = (RedisError, OSError, TimeoutError, ConnectionError)


def build_embedding_cache_key(
    text: str,
    *,
    model: str,
    dimensions: int,
    version: str = "v1",
) -> str:
    """用真正发送给模型的文本生成稳定、可升级的缓存键。"""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"embedding:{version}:sha256:{model}:{dimensions}:{digest}"


class Float32VectorCodec:
    """在 Python 浮点列表和固定小端序 float32 bytes 之间转换。"""

    @staticmethod
    def encode(vector: Sequence[float], dimensions: int) -> bytes:
        if len(vector) != dimensions:
            raise ValueError(
                f"Embedding 维度不匹配: {len(vector)} != {dimensions}"
            )
        return struct.pack(f"<{dimensions}f", *vector)

    @staticmethod
    def decode(data: bytes, dimensions: int) -> list[float]:
        expected_size = dimensions * 4
        if len(data) != expected_size:
            raise ValueError(
                f"缓存向量字节数不匹配: {len(data)} != {expected_size}"
            )
        return list(struct.unpack(f"<{dimensions}f", data))


class LocalVectorCache:
    """进程内有界 LRU + TTL 缓存；多节点共享由 Redis L2 负责。"""

    def __init__(
        self,
        *,
        max_size: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_size < 0:
            raise ValueError("本地缓存容量不能小于 0")
        if ttl_seconds < 0:
            raise ValueError("本地缓存 TTL 不能小于 0")
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._values: OrderedDict[str, tuple[float, tuple[float, ...]]] = (
            OrderedDict()
        )
        self._lock = RLock()

    def get(self, key: str) -> list[float] | None:
        if self._max_size == 0:
            return None
        now = self._clock()
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires_at, vector = item
            if expires_at <= now:
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return list(vector)

    def set(self, key: str, vector: Sequence[float]) -> None:
        if self._max_size == 0 or self._ttl_seconds == 0:
            return
        with self._lock:
            self._values[key] = (
                self._clock() + self._ttl_seconds,
                tuple(float(value) for value in vector),
            )
            self._values.move_to_end(key)
            while len(self._values) > self._max_size:
                self._values.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()


@dataclass(slots=True, frozen=True)
class EmbeddingProgress:
    total: int
    completed: int
    local_cache_hits: int
    redis_cache_hits: int
    generated: int

    @property
    def cache_hits(self) -> int:
        return self.local_cache_hits + self.redis_cache_hits


@dataclass(slots=True, frozen=True)
class EmbeddingBatchResult:
    vectors: list[list[float]]
    local_cache_hits: int
    redis_cache_hits: int
    generated: int

    @property
    def cache_hits(self) -> int:
        return self.local_cache_hits + self.redis_cache_hits


EmbeddingProgressCallback = Callable[
    [EmbeddingProgress], Awaitable[None] | None
]


def is_retryable_embedding_exception(exc: BaseException) -> bool:
    """仅将网络、超时、限流和服务端错误视为可重试异常。"""

    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            TimeoutError,
            ConnectionError,
            OSError,
        ),
    ):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429} or exc.status_code >= 500
    return False


class EmbeddingService:
    """批量解析 L1/L2 缓存未命中项，并仅向模型发送必要文本。"""

    def __init__(
        self,
        *,
        client: Any | None = None,
        redis: Redis | None = redis_binary_client,
        local_cache: LocalVectorCache | None = None,
        model: str = settings.embedding_model,
        dimensions: int = settings.embedding_dimensions,
        batch_size: int = settings.embedding_batch_size,
        concurrency: int = settings.embedding_concurrency,
        cache_ttl: int = settings.embedding_cache_ttl,
        cache_version: str = settings.embedding_cache_version,
        lock_ttl: int = settings.embedding_lock_ttl,
        lock_wait_seconds: float = settings.embedding_lock_wait_seconds,
        lock_poll_seconds: float = settings.embedding_lock_poll_seconds,
        retry_attempts: int = settings.embedding_retry_attempts,
        retry_min_wait_seconds: float = settings.embedding_retry_min_wait_seconds,
        retry_max_wait_seconds: float = settings.embedding_retry_max_wait_seconds,
    ) -> None:
        if (
            dimensions <= 0
            or batch_size <= 0
            or concurrency <= 0
            or retry_attempts <= 0
        ):
            raise ValueError(
                "Embedding 维度、批次大小、并发数和重试次数必须大于 0"
            )
        self.client = client or AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.embedding_request_timeout_seconds,
            # 重试由本服务统一控制，避免 SDK 重试与任务重试成倍叠加。
            max_retries=0,
        )
        self.redis = redis
        self.local_cache = local_cache or LocalVectorCache(
            max_size=settings.embedding_local_cache_size,
            ttl_seconds=settings.embedding_local_cache_ttl,
        )
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.cache_ttl = cache_ttl
        self.cache_version = cache_version
        self.lock_ttl = lock_ttl
        self.lock_wait_seconds = lock_wait_seconds
        self.lock_poll_seconds = lock_poll_seconds
        self.retry_attempts = retry_attempts
        self.retry_min_wait_seconds = retry_min_wait_seconds
        self.retry_max_wait_seconds = retry_max_wait_seconds
        self._api_semaphore = asyncio.Semaphore(concurrency)

    async def embed_many(
        self,
        texts: Sequence[str],
        *,
        progress_callback: EmbeddingProgressCallback | None = None,
    ) -> EmbeddingBatchResult:
        """保持输入顺序返回向量；重复文本在同一批中只生成一次。"""

        if not texts:
            return EmbeddingBatchResult([], 0, 0, 0)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("待向量化文本不能为空")

        keys = [
            build_embedding_cache_key(
                text,
                model=self.model,
                dimensions=self.dimensions,
                version=self.cache_version,
            )
            for text in texts
        ]
        unique_texts: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            existing = unique_texts.setdefault(key, text)
            if existing != text:
                # 概率极低，但不能在检测到 Hash 碰撞时复用错误向量。
                raise RuntimeError("检测到 Embedding 缓存键碰撞")
        unique_entries = list(unique_texts.items())
        vectors: dict[str, list[float]] = {}
        sources: dict[str, str] = {}

        for key, _ in unique_entries:
            vector = self.local_cache.get(key)
            if vector is not None:
                vectors[key] = vector
                sources[key] = "local"

        redis_missing = [
            (key, text) for key, text in unique_entries if key not in vectors
        ]
        if redis_missing:
            cached, _ = await self._read_redis(
                [key for key, _ in redis_missing]
            )
            for key, vector in cached.items():
                vectors[key] = vector
                sources[key] = "redis"
                self.local_cache.set(key, vector)

        await self._notify_progress(
            keys, sources, progress_callback
        )

        unresolved = [
            (key, text) for key, text in unique_entries if key not in vectors
        ]
        tasks = [
            asyncio.create_task(self._resolve_miss_batch(batch))
            for batch in self._batches(unresolved, self.batch_size)
        ]
        try:
            for completed_task in asyncio.as_completed(tasks):
                resolved = await completed_task
                for key, (vector, source) in resolved.items():
                    vectors[key] = vector
                    sources[key] = source
                await self._notify_progress(
                    keys, sources, progress_callback
                )
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        ordered_vectors = [list(vectors[key]) for key in keys]
        return EmbeddingBatchResult(
            vectors=ordered_vectors,
            local_cache_hits=sum(sources[key] == "local" for key in keys),
            redis_cache_hits=sum(sources[key] == "redis" for key in keys),
            generated=sum(sources[key] == "generated" for key in keys),
        )

    async def _resolve_miss_batch(
        self,
        entries: list[tuple[str, str]],
    ) -> dict[str, tuple[list[float], str]]:
        if not entries:
            return {}
        if self.redis is None:
            return await self._generate_entries(entries)

        acquired: list[tuple[str, str, bytes]] = []
        waiting: list[tuple[str, str]] = []
        redis_available = True
        for key, text in entries:
            token = secrets.token_hex(16).encode("ascii")
            try:
                locked = await self.redis.set(
                    self._lock_key(key),
                    token,
                    nx=True,
                    ex=self.lock_ttl,
                )
            except _CACHE_IO_ERRORS:
                logger.warning("Redis 分布式锁不可用，降级为直接向量化", exc_info=True)
                redis_available = False
                break
            if locked:
                acquired.append((key, text, token))
            else:
                waiting.append((key, text))

        if not redis_available:
            for key, _, token in acquired:
                await self._release_lock(key, token)
            return await self._generate_entries(entries)

        resolved: dict[str, tuple[list[float], str]] = {}
        try:
            generated = await self._generate_entries(
                [(key, text) for key, text, _ in acquired]
            )
            resolved.update(generated)
        finally:
            await asyncio.gather(
                *(
                    self._release_lock(key, token)
                    for key, _, token in acquired
                ),
                return_exceptions=True,
            )

        if waiting:
            cached = await self._wait_for_redis(
                [key for key, _ in waiting]
            )
            for key, vector in cached.items():
                self.local_cache.set(key, vector)
                resolved[key] = (vector, "redis")

            # 锁持有者可能崩溃或 Redis 缓存写入失败。等待超时后直接生成，
            # 依靠确定性缓存键和数据库 Upsert 保证最终业务幂等。
            remaining = [
                (key, text) for key, text in waiting if key not in resolved
            ]
            resolved.update(await self._generate_entries(remaining))
        return resolved

    async def _generate_entries(
        self,
        entries: list[tuple[str, str]],
    ) -> dict[str, tuple[list[float], str]]:
        if not entries:
            return {}
        response_vectors = await self._request_embeddings(
            [text for _, text in entries]
        )
        result: dict[str, tuple[list[float], str]] = {}
        for (key, _), vector in zip(entries, response_vectors, strict=True):
            # API 返回通常是 Python float（双精度对象）；统一为 float32 后再返回，
            # 确保首次结果、Redis 命中结果和 pgvector 入库结果完全一致。
            payload = Float32VectorCodec.encode(vector, self.dimensions)
            result[key] = (
                Float32VectorCodec.decode(payload, self.dimensions),
                "generated",
            )
        for key, (vector, _) in result.items():
            self.local_cache.set(key, vector)
        await self._write_redis(
            {key: vector for key, (vector, _) in result.items()}
        )
        return result

    async def _request_embeddings(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(is_retryable_embedding_exception),
            wait=wait_random_exponential(
                multiplier=self.retry_min_wait_seconds,
                max=self.retry_max_wait_seconds,
            ),
            stop=stop_after_attempt(self.retry_attempts),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                async with self._api_semaphore:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                    )

        items = sorted(response.data, key=lambda item: item.index)
        if len(items) != len(texts):
            raise ValueError(
                f"Embedding 返回数量不匹配: {len(items)} != {len(texts)}"
            )
        vectors = [list(item.embedding) for item in items]
        for vector in vectors:
            # 编码同时执行统一的维度校验。
            Float32VectorCodec.encode(vector, self.dimensions)
        return vectors

    async def _read_redis(
        self,
        keys: list[str],
    ) -> tuple[dict[str, list[float]], bool]:
        if not keys or self.redis is None:
            return {}, self.redis is not None
        try:
            values = await self.redis.mget(keys)
        except _CACHE_IO_ERRORS:
            logger.warning("Redis Embedding 缓存读取失败，降级回源", exc_info=True)
            return {}, False

        result: dict[str, list[float]] = {}
        corrupt_keys: list[str] = []
        for key, value in zip(keys, values, strict=True):
            if value is None:
                continue
            try:
                result[key] = Float32VectorCodec.decode(
                    bytes(value), self.dimensions
                )
            except (TypeError, ValueError):
                corrupt_keys.append(key)
                logger.warning("忽略损坏的 Embedding 缓存: %s", key)
        if corrupt_keys:
            try:
                await self.redis.delete(*corrupt_keys)
            except _CACHE_IO_ERRORS:
                logger.warning("删除损坏的 Embedding 缓存失败", exc_info=True)
        return result, True

    async def _write_redis(self, values: dict[str, list[float]]) -> None:
        if not values or self.redis is None:
            return
        try:
            async with self.redis.pipeline(transaction=False) as pipe:
                for key, vector in values.items():
                    pipe.set(
                        key,
                        Float32VectorCodec.encode(vector, self.dimensions),
                        ex=self.cache_ttl,
                    )
                await pipe.execute()
        except _CACHE_IO_ERRORS:
            # Redis 是缓存，写失败不能让文档索引失败。
            logger.warning("Redis Embedding 缓存写入失败，继续持久化", exc_info=True)

    async def _wait_for_redis(self, keys: list[str]) -> dict[str, list[float]]:
        deadline = time.monotonic() + self.lock_wait_seconds
        remaining = list(keys)
        result: dict[str, list[float]] = {}
        while remaining and time.monotonic() < deadline:
            cached, available = await self._read_redis(remaining)
            result.update(cached)
            remaining = [key for key in remaining if key not in result]
            if not available or not remaining:
                break
            await asyncio.sleep(self.lock_poll_seconds)
        return result

    async def _release_lock(self, key: str, token: bytes) -> None:
        if self.redis is None:
            return
        try:
            await self.redis.eval(
                _LOCK_RELEASE_SCRIPT,
                1,
                self._lock_key(key),
                token,
            )
        except _CACHE_IO_ERRORS:
            logger.warning("释放 Embedding 分布式锁失败", exc_info=True)

    @staticmethod
    def _lock_key(cache_key: str) -> str:
        return f"lock:{cache_key}"

    @staticmethod
    def _batches(
        values: list[tuple[str, str]],
        batch_size: int,
    ) -> list[list[tuple[str, str]]]:
        return [
            values[start : start + batch_size]
            for start in range(0, len(values), batch_size)
        ]

    @staticmethod
    async def _notify_progress(
        keys: list[str],
        sources: dict[str, str],
        callback: EmbeddingProgressCallback | None,
    ) -> None:
        if callback is None:
            return
        progress = EmbeddingProgress(
            total=len(keys),
            completed=sum(key in sources for key in keys),
            local_cache_hits=sum(sources.get(key) == "local" for key in keys),
            redis_cache_hits=sum(sources.get(key) == "redis" for key in keys),
            generated=sum(sources.get(key) == "generated" for key in keys),
        )
        callback_result = callback(progress)
        if inspect.isawaitable(callback_result):
            await callback_result


_default_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """返回进程级服务实例，使 L1 缓存在当前 Worker 内持续复用。"""

    global _default_embedding_service
    if _default_embedding_service is None:
        _default_embedding_service = EmbeddingService()
    return _default_embedding_service
