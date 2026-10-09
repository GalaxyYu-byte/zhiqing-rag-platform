"""Replay seven frozen zero-output chunks through the current extractor without graph writes."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from openai import AsyncOpenAI

from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import (
    EXTRACTOR_VERSION,
    SYSTEM_PROMPT,
    GraphExtractionError,
    extract_chunk,
    normalize_name,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class RecordingCompletions:
    def __init__(self, client: AsyncOpenAI, trace: list[dict]):
        self.client = client
        self.trace = trace

    async def create(self, **kwargs):
        started = monotonic()
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except Exception as error:
            self.trace.append(
                {
                    "attempt": len(self.trace) + 1,
                    "elapsed_seconds": round(monotonic() - started, 3),
                    "error_type": type(error).__name__,
                }
            )
            raise
        choice = response.choices[0]
        self.trace.append(
            {
                "attempt": len(self.trace) + 1,
                "elapsed_seconds": round(monotonic() - started, 3),
                "response_model": response.model,
                "finish_reason": choice.finish_reason,
                "raw_output": choice.message.content,
                "usage": response.usage.model_dump(mode="json") if response.usage else None,
            }
        )
        return response


class RecordingClient:
    def __init__(self, client: AsyncOpenAI, trace: list[dict]):
        class Chat:
            pass

        self.chat = Chat()
        self.chat.completions = RecordingCompletions(client, trace)


def compare_to_gold(extraction, case: dict) -> dict:
    gold_entities = Counter(
        (entity["type"], normalize_name(entity["name"])) for entity in case["entities"]
    )
    predicted_entities = Counter(
        (entity.type, normalize_name(entity.canonical_name)) for entity in extraction.entities
    )
    by_id = {entity.id: entity for entity in extraction.entities}
    gold_by_id = {entity["id"]: entity for entity in case["entities"]}
    gold_relations = Counter(
        (
            gold_by_id[relation["source_id"]]["type"],
            normalize_name(gold_by_id[relation["source_id"]]["name"]),
            relation["type"],
            gold_by_id[relation["target_id"]]["type"],
            normalize_name(gold_by_id[relation["target_id"]]["name"]),
        )
        for relation in case["relations"]
    )
    predicted_relations = Counter(
        (
            by_id[relation.source_id].type,
            normalize_name(by_id[relation.source_id].canonical_name),
            relation.type,
            by_id[relation.target_id].type,
            normalize_name(by_id[relation.target_id].canonical_name),
        )
        for relation in extraction.relations
    )
    return {
        "gold_entities": len(case["entities"]),
        "predicted_entities": len(extraction.entities),
        "exact_entity_matches": sum((gold_entities & predicted_entities).values()),
        "gold_relations": len(case["relations"]),
        "predicted_relations": len(extraction.relations),
        "exact_directed_triple_matches": sum((gold_relations & predicted_relations).values()),
    }


async def replay(snapshot_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    frozen_hash = hashlib.sha256((FROZEN / "manifest.json").read_bytes()).hexdigest()
    gold_hash = hashlib.sha256((GOLD / "manifest.json").read_bytes()).hexdigest()
    if snapshot["frozen_manifest_sha256"] != frozen_hash:
        raise ValueError("快照与冻结输入不一致")
    if snapshot["gold_manifest_sha256"] != gold_hash:
        raise ValueError("快照与 Gold 不一致")
    cases = [
        case for case in read_jsonl(GOLD / "cases.jsonl") if case["source_variant"] == "frozen"
    ]
    evidence_chunks = {row["chunk_id"] for row in snapshot["neo4j"]["evidence"]}
    targets = [
        case for case in cases if case["entities"] and case["chunk_id"] not in evidence_chunks
    ]
    if len(targets) != 7:
        raise ValueError(f"预期 7 个正向零输出 Chunk，实际 {len(targets)}")
    settings = Settings()
    if not settings.deepseek_api_key.get_secret_value():
        raise ValueError("DEEPSEEK_API_KEY 未配置")
    frozen_chunks = {row["id"]: row for row in read_jsonl(FROZEN / "chunks.jsonl")}
    for case in targets:
        frozen = frozen_chunks[case["chunk_id"]]
        if (
            case["content"] != frozen["content"]
            or case["content_sha256"] != frozen["content_sha256"]
        ):
            raise ValueError(f"Gold 与冻结 Chunk 不一致: {case['case_id']}")
    metadata = {
        "schema_version": "graph-zero-chunk-replay-v1",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "frozen_manifest_sha256": frozen_hash,
        "gold_manifest_sha256": gold_hash,
        "model": settings.graph_extraction_model,
        "extractor_version": EXTRACTOR_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "mode": "current_model_replay_no_postgres_or_neo4j_writes",
        "target_case_ids": [case["case_id"] for case in targets],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    api_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
        timeout=settings.graph_llm_timeout_seconds,
        max_retries=0,
    )
    try:
        with (output_dir / "replay.jsonl").open("w", encoding="utf-8") as output:
            for case in targets:
                trace = []
                row = {
                    "case_id": case["case_id"],
                    "chunk_id": case["chunk_id"],
                    "document_id": case["document_id"],
                    "content_sha256": case["content_sha256"],
                    "started_at_utc": datetime.now(UTC).isoformat(),
                    "attempts": trace,
                }
                try:
                    extraction = await extract_chunk(
                        settings, case["content"], client=RecordingClient(api_client, trace)
                    )
                    row["status"] = "VALID_EXTRACTION"
                    row["extraction"] = extraction.model_dump(mode="json")
                    row["gold_comparison"] = compare_to_gold(extraction, case)
                except Exception as error:
                    row["status"] = "ERROR"
                    row["error_type"] = type(error).__name__
                    if isinstance(error, GraphExtractionError):
                        row["error_code"] = error.code
                row["finished_at_utc"] = datetime.now(UTC).isoformat()
                output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                output.flush()
                print(
                    f"{case['case_id']}: {row['status']} ({len(trace)} API attempt(s))", flush=True
                )
    finally:
        await api_client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(replay(args.snapshot, args.output))


if __name__ == "__main__":
    main()
