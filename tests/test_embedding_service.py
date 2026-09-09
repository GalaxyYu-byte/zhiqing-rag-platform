from types import SimpleNamespace

import pytest
from redis.exceptions import RedisError

from zq_rag_app.services.embedding_service import (
    EmbeddingService,
    Float32VectorCodec,
    LocalVectorCache,
    build_embedding_cache_key,
)


class _FakePipeline:
    def __init__(self, redis: "_FakeRedis") -> None:
        self.redis = redis
        self.commands: list[tuple[str, bytes, int | None]] = []

    async def __aenter__(self) -> "_FakePipeline":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def set(self, key: str, value: bytes, *, ex: int | None = None):
        self.commands.append((key, value, ex))
        return self

    async def execute(self) -> list[bool]:
        for key, value, _ in self.commands:
            self.redis.values[key] = value
        return [True] * len(self.commands)


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.locks: dict[str, bytes] = {}

    async def mget(self, keys: list[str]) -> list[bytes | None]:
        return [self.values.get(key) for key in keys]

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        del ex
        target = self.locks if key.startswith("lock:") else self.values
        if nx and key in target:
            return False
        target[key] = value
        return True

    async def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            removed += int(self.values.pop(key, None) is not None)
        return removed

    async def eval(
        self,
        _: str,
        __: int,
        key: str,
        token: bytes,
    ) -> int:
        if self.locks.get(key) == token:
            del self.locks[key]
            return 1
        return 0

    def pipeline(self, *, transaction: bool = False) -> _FakePipeline:
        del transaction
        return _FakePipeline(self)


class _BrokenRedis(_FakeRedis):
    async def mget(self, keys: list[str]) -> list[bytes | None]:
        del keys
        raise RedisError("redis unavailable")

    async def set(
        self,
        key: str,
        value: bytes,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        del key, value, nx, ex
        raise RedisError("redis unavailable")


class _FakeEmbeddings:
    def __init__(self, dimensions: int, fail_times: int = 0) -> None:
        self.dimensions = dimensions
        self.fail_times = fail_times
        self.calls: list[list[str]] = []

    async def create(self, *, model: str, input: list[str]):
        del model
        self.calls.append(list(input))
        if self.fail_times:
            self.fail_times -= 1
            raise OSError("temporary network failure")
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=index,
                    embedding=[float(index + 1)] * self.dimensions,
                )
                for index, _ in enumerate(input)
            ]
        )


class _FakeClient:
    def __init__(self, dimensions: int, fail_times: int = 0) -> None:
        self.embeddings = _FakeEmbeddings(dimensions, fail_times)


def test_cache_key_is_stable_and_namespaced() -> None:
    key = build_embedding_cache_key(
        "章节：部署\n\n安装服务",
        model="text-embedding-v3",
        dimensions=1024,
    )

    assert key.startswith(
        "embedding:v1:sha256:text-embedding-v3:1024:"
    )
    assert key == build_embedding_cache_key(
        "章节：部署\n\n安装服务",
        model="text-embedding-v3",
        dimensions=1024,
    )
    assert key != build_embedding_cache_key(
        "章节：部署\n\n安装服务。",
        model="text-embedding-v3",
        dimensions=1024,
    )


def test_float32_codec_round_trip_and_dimension_validation() -> None:
    encoded = Float32VectorCodec.encode([0.25, -0.5, 1.0], 3)

    assert len(encoded) == 12
    assert Float32VectorCodec.decode(encoded, 3) == pytest.approx(
        [0.25, -0.5, 1.0]
    )
    with pytest.raises(ValueError, match="维度不匹配"):
        Float32VectorCodec.encode([1.0], 2)


def test_local_cache_uses_ttl_and_lru() -> None:
    now = [10.0]
    cache = LocalVectorCache(
        max_size=1,
        ttl_seconds=5,
        clock=lambda: now[0],
    )
    cache.set("a", [1.0])
    assert cache.get("a") == [1.0]

    cache.set("b", [2.0])
    assert cache.get("a") is None
    now[0] = 16.0
    assert cache.get("b") is None


@pytest.mark.asyncio
async def test_embed_many_deduplicates_and_refills_l1_from_redis() -> None:
    redis = _FakeRedis()
    first_client = _FakeClient(dimensions=3)
    first = EmbeddingService(
        client=first_client,
        redis=redis,  # type: ignore[arg-type]
        local_cache=LocalVectorCache(max_size=10, ttl_seconds=60),
        model="test-model",
        dimensions=3,
        batch_size=16,
        concurrency=2,
        retry_attempts=2,
        retry_min_wait_seconds=0.001,
        retry_max_wait_seconds=0.001,
    )

    result = await first.embed_many(["相同内容", "不同内容", "相同内容"])

    assert len(first_client.embeddings.calls) == 1
    assert first_client.embeddings.calls[0] == ["相同内容", "不同内容"]
    assert result.generated == 3
    assert result.vectors[0] == result.vectors[2]

    second_client = _FakeClient(dimensions=3)
    second = EmbeddingService(
        client=second_client,
        redis=redis,  # type: ignore[arg-type]
        local_cache=LocalVectorCache(max_size=10, ttl_seconds=60),
        model="test-model",
        dimensions=3,
        batch_size=16,
        concurrency=2,
        retry_attempts=2,
        retry_min_wait_seconds=0.001,
        retry_max_wait_seconds=0.001,
    )

    cached = await second.embed_many(["相同内容", "不同内容", "相同内容"])

    assert second_client.embeddings.calls == []
    assert cached.redis_cache_hits == 3
    assert cached.vectors == result.vectors


@pytest.mark.asyncio
async def test_embedding_network_error_is_retried() -> None:
    client = _FakeClient(dimensions=2, fail_times=1)
    service = EmbeddingService(
        client=client,
        redis=None,
        local_cache=LocalVectorCache(max_size=0, ttl_seconds=0),
        model="test-model",
        dimensions=2,
        batch_size=4,
        concurrency=1,
        retry_attempts=2,
        retry_min_wait_seconds=0.001,
        retry_max_wait_seconds=0.001,
    )

    result = await service.embed_many(["网络重试"])

    assert len(client.embeddings.calls) == 2
    assert result.generated == 1


@pytest.mark.asyncio
async def test_redis_failure_degrades_to_embedding_api() -> None:
    client = _FakeClient(dimensions=2)
    service = EmbeddingService(
        client=client,
        redis=_BrokenRedis(),  # type: ignore[arg-type]
        local_cache=LocalVectorCache(max_size=10, ttl_seconds=60),
        model="test-model",
        dimensions=2,
        batch_size=4,
        concurrency=1,
        retry_attempts=1,
    )

    result = await service.embed_many(["Redis 故障仍需继续"])

    assert len(client.embeddings.calls) == 1
    assert result.generated == 1


@pytest.mark.asyncio
async def test_progress_callback_tracks_all_chunks_including_duplicates() -> None:
    progress = []
    service = EmbeddingService(
        client=_FakeClient(dimensions=2),
        redis=None,
        local_cache=LocalVectorCache(max_size=0, ttl_seconds=0),
        model="test-model",
        dimensions=2,
        batch_size=1,
        concurrency=2,
        retry_attempts=1,
    )

    await service.embed_many(
        ["a", "b", "a"],
        progress_callback=progress.append,
    )

    assert progress[-1].completed == 3
    assert progress[-1].generated == 3
