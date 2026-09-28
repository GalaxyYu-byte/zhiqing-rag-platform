"""验证批量请求、响应错序映射及不完整向量拒绝行为，无外部 API 调用。"""

import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from openai import AsyncOpenAI, BadRequestError

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing import DocumentChunkDraft, embed_chunks
from zhiqing_rag.document_processing.embedding import (
    EmbeddedChunk,
    EmbeddingValidationError,
    validate_embedding,
)


def make_chunks(count: int = 20) -> tuple[DocumentChunkDraft, ...]:
    return tuple(
        DocumentChunkDraft(
            chunk_index=index,
            content=f"正文 {index}",
            embedding_content=f"标题 {index}\n正文 {index}",
            content_sha256=hashlib.sha256(f"正文 {index}".encode()).hexdigest(),
            token_count=12,
            page_number=index + 1,
            section_title=f"标题 {index}",
            source_refs=(f"page:{index + 1}/block:1",),
            metadata={"heading_path": [f"标题 {index}"]},
        )
        for index in range(count)
    )


def settings() -> Settings:
    return Settings(_env_file=None, dashscope_api_key="test-key", embedding_cache_enabled=False)


def response_data(count: int) -> dict:
    return {
        "object": "list",
        "model": "text-embedding-v3",
        "data": [
            {"object": "embedding", "index": i, "embedding": [float(i + 1)] * 1024}
            for i in range(count)
        ],
        "usage": {"prompt_tokens": count * 12, "total_tokens": count * 12},
    }


@pytest.mark.parametrize("count,expected_batches", [(1, [1]), (20, [10, 10]), (23, [10, 10, 3])])
async def test_batches_and_out_of_order_response_match_original_chunks(count, expected_batches):
    chunks = make_chunks(count)
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        response = response_data(len(body["input"]))
        for item, text in zip(response["data"], body["input"], strict=True):
            item["embedding"] = [float(int(text.split("\n")[0].split()[-1]) + 1)] * 1024
        response["data"].reverse()
        return httpx.Response(200, json=response)

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        result = await embed_chunks(chunks, settings=settings(), client=client)

    assert [len(body["input"]) for body in requests] == expected_batches
    assert all(body["model"] == "text-embedding-v3" for body in requests)
    assert all(body["dimensions"] == 1024 for body in requests)
    assert all(body["encoding_format"] == "float" for body in requests)
    assert [text for body in requests for text in body["input"]] == [
        chunk.embedding_content for chunk in chunks
    ]
    for index, item in enumerate(result):
        assert isinstance(item, EmbeddedChunk)
        assert item.chunk is chunks[index]
        assert item.embedding == (float(index + 1),) * 1024
        assert item.model_name == "text-embedding-v3"


@pytest.mark.parametrize(
    "invalid", ["missing", "duplicate", "negative", "out_of_range", "dimension"]
)
async def test_rejects_invalid_batch(invalid):
    body = response_data(2)
    if invalid == "missing":
        body["data"].pop()
    elif invalid == "duplicate":
        body["data"][1]["index"] = 0
    elif invalid == "negative":
        body["data"][1]["index"] = -1
    elif invalid == "out_of_range":
        body["data"][1]["index"] = 2
    else:
        body["data"][1]["embedding"].pop()
    async with AsyncOpenAI(
        api_key="test",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
        ),
    ) as client:
        with pytest.raises(EmbeddingValidationError):
            await embed_chunks(make_chunks(2), settings=settings(), client=client)


@pytest.mark.parametrize(
    "vector",
    [
        (0.0,) * 1024,
        (float("nan"),) * 1024,
        (float("inf"),) * 1024,
        (1e39,) * 1024,
        (1e-50,) * 1024,
        (True,) * 1024,
        ("1",) * 1024,
        (1.0,) * 1023,
    ],
)
def test_rejects_invalid_vectors(vector):
    with pytest.raises(EmbeddingValidationError):
        validate_embedding(vector)


async def test_empty_input_and_invalid_input_do_not_send_requests():
    assert await embed_chunks((), settings=settings()) == ()
    chunk = make_chunks(1)[0]
    with pytest.raises(ValueError, match="不能重复"):
        await embed_chunks((chunk, chunk), settings=settings())
    with pytest.raises(ValueError, match="不能为空"):
        await embed_chunks((replace(chunk, embedding_content=" "),), settings=settings())
    with pytest.raises(ValueError, match="1024"):
        await embed_chunks((chunk,), settings=Settings(_env_file=None, embedding_dimensions=512))


async def test_second_batch_failure_does_not_return_partial_results():
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 2:
            return httpx.Response(400, json={"error": {"message": "invalid input"}})
        return httpx.Response(200, json=response_data(10))

    async with AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    ) as client:
        with pytest.raises(BadRequestError):
            await embed_chunks(make_chunks(), settings=settings(), client=client)
    assert len(calls) == 2
