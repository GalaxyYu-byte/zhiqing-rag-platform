"""连续写入固定 JSON，并查询 Neo4j 验证幂等性。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.graph.models import GraphImportSample
from zq_rag_app.graph.repository import GraphRepository


DEFAULT_INPUT = (
    Path(__file__).parents[1]
    / "zq_rag_app"
    / "schemas"
    / "graph_extraction_sample.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="用固定 JSON 连续写入 Neo4j 并校验结果不增生",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    if args.repeat < 2:
        parser.error("--repeat 至少为 2，才能验证幂等性")
    return args


async def write_and_verify(
    input_path: Path,
    repeat: int,
    *,
    repository: GraphRepository | None = None,
) -> dict:
    sample = GraphImportSample.model_validate_json(
        input_path.read_text(encoding="utf-8")
    )
    sample.extraction.validate_evidence(sample.context.content)

    repository = repository or GraphRepository()
    await repository.ensure_schema()

    summaries = []
    for attempt in range(1, repeat + 1):
        write_result = await repository.upsert_extraction(
            context=sample.context,
            extraction=sample.extraction,
            extractor_version=sample.extractor_version,
        )
        summary = await repository.get_chunk_summary(write_result.chunk_uid)
        if summary is None:
            raise RuntimeError("写入后没有查到目标 Chunk")
        summaries.append(summary)
        print(
            json.dumps(
                {"attempt": attempt, **asdict(summary)},
                ensure_ascii=False,
            )
        )

    payload = GraphRepository.prepare_payload(
        context=sample.context,
        extraction=sample.extraction,
        extractor_version=sample.extractor_version,
    )
    expected_entities = len(
        {
            entity["uid"]
            for entity in payload.entities
        }
    )
    expected_claims = len(
        {
            claim["uid"]
            for claim in payload.claims
        }
    )
    final = summaries[-1]
    expected_counts = (
        expected_entities,
        expected_entities,
        expected_claims,
        expected_claims,
    )
    actual_counts = (
        final.entity_count,
        final.mention_count,
        final.claim_count,
        final.evidence_count,
    )
    if any(summary != summaries[0] for summary in summaries[1:]):
        raise RuntimeError("重复写入后图计数发生变化，幂等校验失败")
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"写入计数与样例不一致: actual={actual_counts}, "
            f"expected={expected_counts}"
        )

    return {
        "status": "idempotent",
        "repeat": repeat,
        **asdict(final),
    }


async def main() -> None:
    args = parse_args()
    try:
        result = await write_and_verify(args.input.resolve(), args.repeat)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        await close_neo4j()


if __name__ == "__main__":
    asyncio.run(main())
