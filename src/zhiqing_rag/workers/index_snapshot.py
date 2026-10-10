"""重建计划与实际索引输入共享的无凭据内容指纹。"""

import hashlib
import json


def chunk_record(chunk) -> dict:
    metadata = dict(getattr(chunk, "metadata_", getattr(chunk, "metadata", {})))
    if hasattr(chunk, "source_refs"):
        metadata["source_refs"] = list(chunk.source_refs)
    # 图抽取缓存可由独立 Worker 补齐，不属于解析/向量输入。
    metadata = {key: value for key, value in metadata.items() if not key.startswith("_graph")}
    return {
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
        "embedding_content": chunk.embedding_content,
        "content_sha256": chunk.content_sha256,
        "token_count": chunk.token_count,
        "page_number": chunk.page_number,
        "section_title": chunk.section_title,
        "metadata": metadata,
    }


def chunk_fingerprint(chunks) -> str:
    records = [chunk_record(chunk) for chunk in chunks]
    if not records or any(
        record["chunk_index"] != index
        or record["content_sha256"] != hashlib.sha256(record["content"].encode()).hexdigest()
        for index, record in enumerate(records)
    ):
        raise ValueError("CHUNK_INTEGRITY_INVALID")
    value = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()
