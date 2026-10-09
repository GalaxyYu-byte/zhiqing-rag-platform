"""用现有授权的 13 个冻结 Chunk 验证预算/类型提示修改；仅写本地结果。"""

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from scripts.evaluate_graph_table_replay import PILOT, ROOT, read_jsonl, sha, write_json
from scripts.replay_graph_table_holdout import verify
from scripts.replay_graph_zero_chunks import RecordingClient, compare_to_gold
from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import (
    EXTRACTOR_VERSION, SYSTEM_PROMPT, Extraction, GraphExtractionError,
    extract_chunk, normalize_name, validate_evidence,
)

GOLD = PILOT / "six-document-review-draft-v1/cases.jsonl"
BASELINE = PILOT / "name-feedback-regression-20260930/replay.jsonl"
GOLD_SHA = "d663bfbc7068ce2edcf71c9b9c3ba2a85f71bf7612e28c6f9ad20985fe2648dd"
BASELINE_SHA = "0fd226554fc70360bcad98fcd66b024127cd092ae432fc75e71310cec4b4dde3"


def metrics(rows, cases):
    counts = Counter()
    documents = {}
    for row in rows:
        case = cases[row["case_id"]]
        if row["content_sha256"] != case["content_sha256"]:
            raise ValueError("结果与冻结正文不一致")
        extraction = Extraction.model_validate(row.get("extraction", {"entities": [], "relations": []}))
        validate_evidence(extraction, case["content"])
        comparison = compare_to_gold(extraction, case)
        counts.update(comparison)
        documents.setdefault(case["source"], Counter()).update(comparison)
        counts["failed_chunks"] += row["status"] != "VALID_EXTRACTION"
        counts["api_attempts"] += len(row["attempts"])
        counts["truncated_attempts"] += sum(a.get("finish_reason") == "length" for a in row["attempts"])
        typed = {(e.type, normalize_name(e.canonical_name)) for e in extraction.entities}
        names = {name for _, name in typed}
        for entity in case["entities"]:
            name = normalize_name(entity["name"])
            outcome = ("entity_hit" if (entity["type"], name) in typed else
                       "entity_wrong_type" if name in names else "entity_missing_name")
            counts[outcome] += 1
        by_id = {e["id"]: e for e in case["entities"]}
        predicted = {e.id: e for e in extraction.entities}
        triples = Counter((predicted[r.source_id].type, normalize_name(predicted[r.source_id].canonical_name),
                           r.type, predicted[r.target_id].type,
                           normalize_name(predicted[r.target_id].canonical_name)) for r in extraction.relations)
        for relation in case["relations"]:
            source, target = by_id[relation["source_id"]], by_id[relation["target_id"]]
            skey = (source["type"], normalize_name(source["name"]))
            tkey = (target["type"], normalize_name(target["name"]))
            eligible = skey in typed and tkey in typed
            key = (*skey, relation["type"], *tkey)
            hit = triples[key] > 0
            if hit:
                triples[key] -= 1
            counts["conditional_eligible"] += eligible
            counts["conditional_hits"] += eligible and hit
            if not hit:
                counts["relation_missing_with_endpoints" if eligible else "relation_missing_endpoint"] += 1
    result = {"counts": dict(counts), "documents": {key: dict(value) for key, value in documents.items()}}
    for kind, hit_key, pred_key, gold_key in (
        ("entity", "exact_entity_matches", "predicted_entities", "gold_entities"),
        ("relation", "exact_directed_triple_matches", "predicted_relations", "gold_relations"),
    ):
        hit, pred, gold = counts[hit_key], counts[pred_key], counts[gold_key]
        result[kind] = {"matches": hit, "predictions": pred, "gold_targets": gold,
                        "precision_conservative_reference": hit / pred if pred else None,
                        "targeted_recall": hit / gold if gold else None,
                        "f1_conservative_reference": 2 * hit / (pred + gold) if pred + gold else None}
    return result


async def replay(output):
    frozen, manifest = verify()
    if sha(GOLD) != GOLD_SHA or sha(BASELINE) != BASELINE_SHA:
        raise ValueError("Gold 或基线发生变化，不能作同口径比较")
    cases = {case["case_id"]: case for case in read_jsonl(GOLD)}
    if len(cases) != 13 or set(cases) != {case["case_id"] for case in frozen}:
        raise ValueError("输入超出既有授权范围")
    for case in frozen:
        if cases[case["case_id"]]["content"] != case["content"]:
            raise ValueError("Gold 正文与冻结输入不一致")
    settings = Settings()
    previous_model = json.loads((PILOT / "name-feedback-regression-20260930/metadata.json").read_text(encoding="utf-8"))["model"]
    if settings.graph_extraction_model != previous_model:
        raise ValueError("模型变化，不能作同模型回归")
    output.mkdir(parents=True, exist_ok=False)
    (output / "system_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    (output / "extractor_source.py").write_bytes((ROOT / "src/zhiqing_rag/workers/graph_extraction.py").read_bytes())
    write_json(output / "metadata.json", {
        "started_at_utc": datetime.now(UTC).isoformat(), "model": settings.graph_extraction_model,
        "extractor_version": EXTRACTOR_VERSION, "max_output_tokens": settings.graph_llm_max_output_tokens,
        "timeout_seconds": settings.graph_llm_timeout_seconds,
        "gold_sha256": GOLD_SHA, "baseline_sha256": BASELINE_SHA,
        "frozen_cases_sha256": manifest["cases_sha256"],
        "system_prompt_sha256": sha(output / "system_prompt.txt"),
        "extractor_source_sha256": sha(output / "extractor_source.py"),
        "authorized_case_ids": list(cases), "mode": "isolated_no_database_writes",
    })
    semaphore = asyncio.Semaphore(2)
    async with AsyncOpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                           base_url=settings.deepseek_base_url,
                           timeout=settings.graph_llm_timeout_seconds, max_retries=0) as client:
        with (output / "replay.jsonl").open("w", encoding="utf-8") as stream:
            async def run(case):
                async with semaphore:
                    trace = []
                    row = {"case_id": case["case_id"], "content_sha256": case["content_sha256"], "attempts": trace}
                    try:
                        extraction = await extract_chunk(settings, case["content"], client=RecordingClient(client, trace))
                        row.update(status="VALID_EXTRACTION", extraction=extraction.model_dump(mode="json"))
                    except Exception as error:
                        row.update(status="ERROR", error_type=type(error).__name__)
                        if isinstance(error, GraphExtractionError):
                            row["error_code"] = error.code
                    row["finished_at_utc"] = datetime.now(UTC).isoformat()
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(f"{case['case_id']}: {row['status']} ({len(trace)} requests)", flush=True)
                    return row
            rows = await asyncio.gather(*(run(case) for case in cases.values()))
    write_json(output / "summary.json", {
        "baseline": metrics(read_jsonl(BASELINE), cases), "current": metrics(rows, cases),
        "gold_unchanged": True, "independent_holdout": False,
        "scope": "Post-prediction diagnostic Gold; unmatched predictions are unjudged. Structural matches do not establish evidence support or conditional effects.",
        "finished_at_utc": datetime.now(UTC).isoformat(),
    })
    print("Comparison saved to summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(replay(parser.parse_args().output))
