"""对同一份文档执行两次向量化，验证真实 Redis 缓存及第二次零模型请求。"""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from zhiqing_rag.core.config import get_settings
from zhiqing_rag.document_processing import (
    ChunkConfig,
    EmbeddingStats,
    embed_chunks,
    process_document,
)
from zhiqing_rag.document_processing.embedding_cache import (
    VECTOR_STRUCT,
    EmbeddingCache,
    create_cache_client,
)


async def verify(path: Path) -> dict:
    settings = get_settings()
    if not settings.embedding_cache_enabled:
        raise ValueError("缓存开关未启用")
    processed = process_document(path, chunk_config=ChunkConfig(min_chunk_tokens=0))
    if not processed.chunks or not processed.cleaned.is_complete:
        raise ValueError("文档为空或解析不完整")
    first_stats, second_stats = EmbeddingStats(), EmbeddingStats()
    first = await embed_chunks(processed.chunks, settings=settings, stats=first_stats)
    second = await embed_chunks(processed.chunks, settings=settings, stats=second_stats)
    # 首次模型结果为 Python float，缓存命中结果为 float32；按存储精度比较。
    same_results = len(first) == len(second) and all(
        left.chunk == right.chunk
        and left.model_name == right.model_name
        and VECTOR_STRUCT.pack(*left.embedding) == VECTOR_STRUCT.pack(*right.embedding)
        for left, right in zip(first, second, strict=True)
    )
    if not same_results or second_stats.model_batches != 0:
        raise ValueError("缓存复用验证失败")
    if second_stats.cache_hits != len(processed.chunks):
        raise ValueError("第二次调用未全部命中缓存")
    async with create_cache_client(settings) as redis:
        cache = EmbeddingCache(redis, settings)
        values = await redis.mget(
            [cache.key(chunk.embedding_content) for chunk in processed.chunks]
        )
        if any(
            not isinstance(value, bytes) or len(value) != VECTOR_STRUCT.size for value in values
        ):
            raise ValueError("缓存 float32 二进制格式验证失败")
        async with redis.pipeline(transaction=False) as pipeline:
            for chunk in processed.chunks:
                pipeline.ttl(cache.key(chunk.embedding_content))
            ttls = await pipeline.execute()
    if any(ttl <= 0 or ttl > settings.embedding_cache_ttl_seconds for ttl in ttls):
        raise ValueError("缓存 TTL 验证失败")
    return {
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
        "cache_format": "float32-le",
        "bytes_per_vector": VECTOR_STRUCT.size,
        "chunks": len(processed.chunks),
        "first_call": asdict(first_stats),
        "second_call": asdict(second_stats),
        "ttl_min_seconds": min(ttls),
        "ttl_max_seconds": max(ttls),
        "identical_vectors_and_chunk_mapping": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = asyncio.run(verify(args.file.resolve()))
        output = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output, encoding="utf-8")
        print(output)
    except Exception as exc:
        print(f"缓存验证失败: {type(exc).__name__}")
        raise SystemExit(1) from None
