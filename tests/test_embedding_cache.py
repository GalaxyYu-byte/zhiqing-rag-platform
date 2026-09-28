"""验证缓存命中、部分命中、输入去重、缓存隔离和 Redis 故障回退。"""

import json
import struct
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError
from redis.exceptions import ConnectionError
from test_embedding import make_chunks, response_data

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing import EmbeddingStats, embed_chunks
from zhiqing_rag.document_processing.embedding_cache import EmbeddingCache


class MemoryRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}
        self.read_failure = False
        self.write_failure = False
        self.reads = 0

    async def mget(self, keys):
        self.reads += 1
        if self.read_failure:
            raise ConnectionError("unavailable")
        return [self.data.get(key) for key in keys]

    def pipeline(self, transaction=False):
        return MemoryPipeline(self)


class MemoryPipeline:
    def __init__(self, redis):
        self.redis = redis
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def set(self, key, value, ex):
        self.commands.append((key, value, ex))

    async def execute(self):
        if self.redis.write_failure:
            raise ConnectionError("unavailable")
        for key, value, ttl in self.commands:
            self.redis.data[key] = value
            self.redis.ttls[key] = ttl


def cache_settings(**kwargs):
    return Settings(_env_file=None, dashscope_api_key="test-key", **kwargs)


def mock_client(requests):
    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        response = response_data(len(body["input"]))
        for item, text in zip(response["data"], body["input"], strict=True):
            item["embedding"] = [float(int(text.rsplit(" ", 1)[-1]) + 1)] * 1024
        response["data"].reverse()
        return httpx.Response(200, json=response)

    return AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def test_20_cold_chunks_use_two_batches_and_warm_chunks_use_no_model():
    redis = MemoryRedis()
    chunks = make_chunks()
    settings = cache_settings()
    stats = EmbeddingStats()
    requests = []
    async with mock_client(requests) as client:
        cold = await embed_chunks(
            chunks, settings=settings, client=client, redis_client=redis, stats=stats
        )
        assert stats == EmbeddingStats(cache_hits=0, model_chunks=20, model_batches=2)
        # 全部命中时不需要模型凭据，也不需要建立模型连接。
        warm = await embed_chunks(
            chunks,
            settings=settings.model_copy(update={"dashscope_api_key": ""}),
            redis_client=redis,
            stats=stats,
        )
    assert stats == EmbeddingStats(cache_hits=20, model_chunks=0, model_batches=0)
    assert cold == warm
    assert [len(body["input"]) for body in requests] == [10, 10]
    assert len(redis.data) == 20
    assert set(redis.ttls.values()) == {604800}
    assert all(isinstance(value, bytes) and len(value) == 4096 for value in redis.data.values())
    for index, item in enumerate(warm):
        assert item.chunk is chunks[index]
        assert item.embedding == (float(index + 1),) * 1024


async def test_partial_hits_rebatch_only_misses_and_preserve_all_original_chunks():
    redis = MemoryRedis()
    settings = cache_settings()
    cache = EmbeddingCache(redis, settings)
    chunks = make_chunks()
    await cache.set_many(
        {
            chunk.embedding_content: (float(i + 1),) * 1024
            for i, chunk in enumerate(chunks)
            if i % 2 == 0
        }
    )
    requests = []
    stats = EmbeddingStats()
    async with mock_client(requests) as client:
        result = await embed_chunks(
            chunks, settings=settings, client=client, redis_client=redis, stats=stats
        )
    assert stats == EmbeddingStats(cache_hits=10, model_chunks=10, model_batches=1)
    assert requests[0]["input"] == [chunk.embedding_content for chunk in chunks[1::2]]
    assert all(
        item.chunk is chunks[i] and item.embedding[0] == i + 1 for i, item in enumerate(result)
    )


async def test_float32_cache_round_trip_preserves_storage_precision():
    redis = MemoryRedis()
    cache = EmbeddingCache(redis, cache_settings())
    vector = (0.1, -0.2, 1.23456789, 1e-40) * 256
    await cache.set_many({"正文": vector})
    expected = struct.pack("<1024f", *vector)
    assert redis.data[cache.key("正文")] == expected
    result = await cache.get_many(["正文"])
    assert result["正文"] == struct.unpack("<1024f", expected)
    assert result["正文"] != vector


async def test_legacy_json_cache_is_isolated_from_float32_cache():
    redis = MemoryRedis()
    cache = EmbeddingCache(redis, cache_settings())
    key = cache.key("正文")
    legacy_key = key.replace("embedding:v2:float32:", "embedding:v1:", 1)
    redis.data[legacy_key] = json.dumps([1.0] * 1024)
    assert await cache.get_many(["正文"]) == {}
    await cache.set_many({"正文": (2.0,) * 1024})
    assert json.loads(redis.data[legacy_key]) == [1.0] * 1024
    assert (await cache.get_many(["正文"]))["正文"] == (2.0,) * 1024


async def test_heading_change_misses_cache_even_when_content_hash_is_unchanged():
    redis = MemoryRedis()
    settings = cache_settings()
    requests = []
    chunks = make_chunks(2)
    async with mock_client(requests) as client:
        await embed_chunks(chunks, settings=settings, client=client, redis_client=redis)
        changed = replace(chunks[0], embedding_content="新标题\n正文 0")
        stats = EmbeddingStats()
        result = await embed_chunks(
            (changed, chunks[1]), settings=settings, client=client, redis_client=redis, stats=stats
        )
    assert changed.content_sha256 == chunks[0].content_sha256
    assert stats == EmbeddingStats(cache_hits=1, model_chunks=1, model_batches=1)
    assert requests[1]["input"] == [changed.embedding_content]
    assert result[0].chunk is changed


async def test_duplicate_inputs_use_one_vector_but_keep_two_original_chunks():
    chunks = make_chunks(2)
    second = replace(chunks[1], embedding_content=chunks[0].embedding_content)
    requests = []
    async with mock_client(requests) as client:
        result = await embed_chunks(
            (chunks[0], second),
            settings=cache_settings(),
            client=client,
            redis_client=MemoryRedis(),
        )
    assert len(requests[0]["input"]) == 1
    assert len(result) == 2
    assert result[0].chunk is chunks[0] and result[1].chunk is second
    assert result[0].embedding == result[1].embedding


@pytest.mark.parametrize(
    "value",
    [
        b"not-json",
        b"\xff",
        "null",
        "{}",
        "[true]",
        "[1]",
        json.dumps([0] * 1024),
        json.dumps([float("nan")] * 1024),
        json.dumps([1e39] * 1024),
        struct.pack("<1023f", *([1.0] * 1023)),
        struct.pack("<1025f", *([1.0] * 1025)),
        struct.pack("<1024f", *([0.0] * 1024)),
        struct.pack("<1024f", *([float("nan")] * 1024)),
        struct.pack("<1024f", *([float("inf")] * 1024)),
    ],
)
async def test_corrupt_cache_entry_is_recomputed_and_replaced(value):
    redis = MemoryRedis()
    settings = cache_settings()
    chunk = make_chunks(1)[0]
    key = EmbeddingCache(redis, settings).key(chunk.embedding_content)
    redis.data[key] = value
    requests = []
    async with mock_client(requests) as client:
        result = await embed_chunks((chunk,), settings=settings, client=client, redis_client=redis)
    assert len(requests) == 1 and result[0].embedding == (1.0,) * 1024
    assert redis.data[key] == struct.pack("<1024f", *([1.0] * 1024))


@pytest.mark.parametrize("failure", ["read_failure", "write_failure"])
async def test_redis_failure_does_not_fail_vector_generation(failure):
    redis = MemoryRedis()
    setattr(redis, failure, True)
    requests = []
    async with mock_client(requests) as client:
        result = await embed_chunks(
            make_chunks(), settings=cache_settings(), client=client, redis_client=redis
        )
    assert len(requests) == 2 and len(result) == 20


async def test_disabled_cache_does_not_access_redis():
    redis = MemoryRedis()
    redis.read_failure = redis.write_failure = True
    requests = []
    async with mock_client(requests) as client:
        await embed_chunks(
            make_chunks(1),
            settings=cache_settings(embedding_cache_enabled=False),
            client=client,
            redis_client=redis,
        )
    assert redis.reads == 0 and not redis.data


@pytest.mark.parametrize(
    "change",
    [
        {"embedding_model": "different-model"},
        {"embedding_dimensions": 512},
        {"embedding_cache_revision": "v2"},
        {"openai_base_url": "https://example.test/v1"},
        {"redis_key_prefix": "other-project:"},
    ],
)
def test_cache_key_isolates_model_configuration(change):
    settings = cache_settings()
    original = EmbeddingCache(MemoryRedis(), settings).key("标题\n正文")
    changed = EmbeddingCache(MemoryRedis(), settings.model_copy(update=change)).key("标题\n正文")
    assert original != changed
    assert original.startswith("zhiqing-rag:embedding:v2:float32:")
    assert "正文" not in original and "test-key" not in original


async def test_second_batch_failure_does_not_cache_partial_vectors():
    requests = []
    redis = MemoryRedis()

    def handler(request):
        requests.append(request)
        if len(requests) == 2:
            return httpx.Response(400, json={"error": {"message": "invalid"}})
        return httpx.Response(200, json=response_data(10))

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        with pytest.raises(BadRequestError):
            await embed_chunks(
                make_chunks(), settings=cache_settings(), client=client, redis_client=redis
            )
    assert len(requests) == 2 and not redis.data
