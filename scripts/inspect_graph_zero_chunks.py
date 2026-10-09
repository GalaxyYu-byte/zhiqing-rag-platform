"""Read-only inspection of graph tasks and extraction caches for frozen zero-output chunks."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def inspect(snapshot_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise FileExistsError(f"输出已存在: {output_path}")
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
    chunks_with_evidence = {row["chunk_id"] for row in snapshot["neo4j"]["evidence"]}
    zero_cases = [case for case in cases if case["chunk_id"] not in chunks_with_evidence]
    if len(zero_cases) != 9:
        raise ValueError(f"预期 9 个零 Evidence Chunk，实际 {len(zero_cases)}")
    generation_ids = sorted({case["generation_id"] for case in zero_cases})
    chunk_ids = sorted(case["chunk_id"] for case in zero_cases)
    settings = Settings()
    schema = settings.db_schema
    with database_connection(settings) as connection:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            cursor.execute(
                f"""SELECT id, tenant_id, document_id, generation_id, status, stage,
                           attempt_count, error_code, created_at, finished_at,
                           payload,
                           payload ->> 'next_chunk_index' AS next_chunk_index,
                           jsonb_array_length(COALESCE(payload -> 'entity_keys', '[]'::jsonb))
                             AS checkpoint_entity_keys,
                           jsonb_array_length(COALESCE(payload -> 'relation_keys', '[]'::jsonb))
                             AS checkpoint_relation_keys
                    FROM {schema}.background_task
                    WHERE task_type='GRAPH' AND generation_id = ANY(%s)
                    ORDER BY generation_id, id""",
                (generation_ids,),
            )
            tasks = cursor.fetchall()
            for task in tasks:
                task["payload_keys"] = sorted(task.pop("payload"))
            task_ids = [row["id"] for row in tasks]
            cursor.execute(
                f"""SELECT task_id, attempt_no, status, stage, started_at, finished_at,
                           error_code, error_detail, metrics
                    FROM {schema}.task_attempt
                    WHERE task_id = ANY(%s) ORDER BY task_id, attempt_no""",
                (task_ids,),
            )
            attempts = cursor.fetchall()
            cursor.execute(
                f"""SELECT id, document_id, generation_id, chunk_index, content_sha256,
                           length(content) AS character_count, created_at,
                           metadata ? '_graph_extraction_cache' AS cache_present,
                           metadata -> '_graph_extraction_cache' ->> 'model' AS cached_model,
                           metadata -> '_graph_extraction_cache' ->> 'extractor_version'
                             AS cached_extractor_version
                    FROM {schema}.document_chunk WHERE id = ANY(%s)
                    ORDER BY document_id, chunk_index""",
                (chunk_ids,),
            )
            current_chunks = cursor.fetchall()
            cursor.execute(
                f"""SELECT document_id, generation_id, status, entity_count, relation_count,
                           model_name, extractor_version, ready_at, created_at, updated_at
                    FROM {schema}.graph_projection
                    WHERE generation_id = ANY(%s) ORDER BY generation_id""",
                (generation_ids,),
            )
            projections = cursor.fetchall()
    frozen_chunks = {row["id"]: row for row in read_jsonl(FROZEN / "chunks.jsonl")}
    current_by_id = {row["id"]: row for row in current_chunks}
    for case in zero_cases:
        current = current_by_id.get(case["chunk_id"])
        case["current_chunk_present"] = current is not None
        case["content_hash_still_matches_frozen"] = (
            current is not None
            and current["content_sha256"] == frozen_chunks[case["chunk_id"]]["content_sha256"]
        )
    report = {
        "schema_version": "zero-graph-chunk-history-v1",
        "inspected_at_utc": datetime.now(UTC).isoformat(),
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "zero_cases": [
            {
                "case_id": case["case_id"],
                "chunk_id": case["chunk_id"],
                "document_id": case["document_id"],
                "generation_id": case["generation_id"],
                "gold_entity_count": len(case["entities"]),
                "gold_relation_count": len(case["relations"]),
                "current_chunk_present": case["current_chunk_present"],
                "content_hash_still_matches_frozen": case["content_hash_still_matches_frozen"],
            }
            for case in zero_cases
        ],
        "tasks": tasks,
        "attempts": attempts,
        "current_chunks": current_chunks,
        "projections": projections,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        f"零输出 {len(zero_cases)} Chunk；含正向 Gold "
        f"{sum(bool(case['entities']) for case in zero_cases)}；"
        f"当前缓存 {sum(row['cache_present'] for row in current_chunks)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.snapshot, args.output)


if __name__ == "__main__":
    main()
