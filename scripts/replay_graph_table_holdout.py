"""Freeze and replay exactly thirteen authorized holdout chunks with the fixed pilot prompt."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from scripts.evaluate_graph_table_replay import (
    PILOT,
    ROOT,
    read_jsonl,
    sha,
    write_json,
    write_jsonl,
)
from scripts.replay_graph_zero_chunks import RecordingClient, compare_to_gold
from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing import ChunkConfig, process_document
from zhiqing_rag.workers import graph_extraction
from zhiqing_rag.workers.graph_extraction import (
    Extraction,
    GraphExtractionError,
    extract_chunk,
    normalize_name,
    validate_evidence,
)

LOCAL = PILOT / "unseen-validation"
FROZEN = PILOT / "holdout-frozen-v1"
PROMPT_RUN = PILOT / "replay-20260930-table-guidance"
CONFIG = ChunkConfig(max_tokens=512, overlap_tokens=48)


def logical_prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def selected_gold(chunk, facts):
    entities = []
    relations = []
    keys = {}
    evidence = []

    def entity(name, entity_type, quote):
        key = (entity_type, normalize_name(name))
        if key not in keys:
            entity_id = f"e{len(entities) + 1}"
            keys[key] = entity_id
            entities.append({"id": entity_id, "name": name, "canonical_name": name,
                             "type": entity_type, "evidence_quote": quote})
        return keys[key]

    for index, fact in enumerate(facts, 1):
        quote = fact["evidence_quote"]
        if quote not in chunk["content"]:
            raise ValueError("预先选定的行证据不在冻结 Chunk 中")
        if fact["source_entity"] not in quote or fact["target_entity"] not in quote:
            raise ValueError("选择性目标两端不在同一行证据中")
        if fact["relation_type"] not in ("RESPONSIBLE_FOR", "GOVERNED_BY"):
            raise ValueError("选择性目标关系类型变化")
        responsible = fact["relation_type"] == "RESPONSIBLE_FOR"
        source_id = entity(fact["source_entity"], "Actor" if responsible else "Material", quote)
        target_id = entity(fact["target_entity"], "Step" if responsible else "Rule", quote)
        relations.append({"id": f"r{index}", "source_id": source_id, "target_id": target_id,
                          "type": fact["relation_type"], "evidence_quote": quote,
                          "condition_text": None})
    for kind, records in (("entity", entities), ("relation", relations)):
        for record in records:
            start = chunk["content"].index(record["evidence_quote"])
            evidence.append({"target_kind": kind, "target_id": record["id"], "start": start,
                             "end": start + len(record["evidence_quote"]),
                             "quote": record["evidence_quote"]})
    return {**chunk, "entities": entities, "relations": relations, "evidence": evidence,
            "gold_scope": "preexisting_selected_row_targets_not_exhaustive" if facts
            else "unannotated_not_negative",
            "review_status": "codex_draft_independent_human_review_pending"}


def prepare():
    if FROZEN.exists():
        raise FileExistsError("留出输入已冻结，不覆盖")
    checks = read_jsonl(LOCAL / "chunk_checks.jsonl")
    facts = read_jsonl(LOCAL / "fact_candidates.jsonl")
    if len(checks) != 13 or len(facts) != 30:
        raise ValueError("预先留出输入计数变化")
    prompt = (PROMPT_RUN / "system_prompt.txt").read_text(encoding="utf-8")
    prompt_meta = json.loads((PROMPT_RUN / "metadata.json").read_text(encoding="utf-8"))
    if logical_prompt_hash(prompt) != prompt_meta["system_prompt_sha256"]:
        raise ValueError("调参结束时的提示词哈希不一致")
    source_paths = list(dict.fromkeys(check["source"] for check in checks))
    if len(source_paths) != 6:
        raise ValueError("授权范围必须是既有的六份文档")
    chunks = []
    sources = []
    for document_index, source_name in enumerate(source_paths, 1):
        source = ROOT / source_name
        source_checks = [check for check in checks if check["source"] == source_name]
        if any(sha(source) != check["source_sha256"] for check in source_checks):
            raise ValueError("原留出来源哈希变化")
        processed = process_document(source, chunk_config=CONFIG)
        by_index = {draft.chunk_index: draft for draft in processed.chunks}
        for check in source_checks:
            draft = by_index[check["chunk_index"]]
            if draft.content_sha256 != check["content_sha256"]:
                raise ValueError("原留出 Chunk 无法复现")
            chunk = {"case_id": f"holdout-{document_index:02d}-{draft.chunk_index:02d}",
                     "source": source_name, "chunk_index": draft.chunk_index,
                     "content": draft.content, "content_sha256": draft.content_sha256,
                     "token_count": draft.token_count, "metadata": draft.metadata}
            chunk_facts = [fact for fact in facts if fact["source"] == source_name
                           and fact["chunk_index"] == draft.chunk_index]
            chunks.append(selected_gold(chunk, chunk_facts))
        sources.append({"source": source_name, "source_sha256": sha(source),
                        "frozen_filename": f"sources/{document_index:02d}.xlsx"})
    FROZEN.mkdir(exist_ok=False)
    (FROZEN / "sources").mkdir()
    for source in sources:
        (FROZEN / source["frozen_filename"]).write_bytes((ROOT / source["source"]).read_bytes())
    (FROZEN / "system_prompt.txt").write_bytes(prompt.encode("utf-8"))
    write_jsonl(FROZEN / "cases.jsonl", chunks)
    write_json(FROZEN / "manifest.json", {
        "schema_version": "graph-table-holdout-frozen-v1",
        "created_at_utc": datetime.now(UTC).isoformat(), "sources": sources,
        "case_count": len(chunks), "selected_gold_case_count": sum(bool(c["relations"])
                                                                    for c in chunks),
        "selected_gold_entity_count": sum(len(c["entities"]) for c in chunks),
        "selected_gold_relation_count": sum(len(c["relations"]) for c in chunks),
        "cases_sha256": sha(FROZEN / "cases.jsonl"),
        "prompt_sha256": sha(FROZEN / "system_prompt.txt"),
        "prompt_run_metadata_sha256": sha(PROMPT_RUN / "metadata.json"),
        "preexisting_fact_candidates_sha256": sha(LOCAL / "fact_candidates.jsonl"),
        "preexisting_chunk_checks_sha256": sha(LOCAL / "chunk_checks.jsonl"),
        "extractor_source_sha256": sha(ROOT / "src/zhiqing_rag/workers/graph_extraction.py"),
        "review_status": "codex_draft_independent_human_review_pending",
    })
    verify()


def verify():
    manifest = json.loads((FROZEN / "manifest.json").read_text(encoding="utf-8"))
    for path, key in ((FROZEN / "cases.jsonl", "cases_sha256"),
                      (FROZEN / "system_prompt.txt", "prompt_sha256"),
                      (LOCAL / "fact_candidates.jsonl", "preexisting_fact_candidates_sha256"),
                      (LOCAL / "chunk_checks.jsonl", "preexisting_chunk_checks_sha256"),
                      (PROMPT_RUN / "metadata.json", "prompt_run_metadata_sha256")):
        if sha(path) != manifest[key]:
            raise ValueError(f"冻结输入或前序记录哈希变化: {path.name}")
    cases = read_jsonl(FROZEN / "cases.jsonl")
    if len(cases) != 13 or len({c["case_id"] for c in cases}) != 13:
        raise ValueError("输入不符合授权的十三块")
    checks = read_jsonl(LOCAL / "chunk_checks.jsonl")
    if {(c["source"], c["chunk_index"], c["content_sha256"]) for c in cases} != {
        (c["source"], c["chunk_index"], c["content_sha256"]) for c in checks
    }:
        raise ValueError("授权范围与原留出 Chunk 清单不一致")
    if len(manifest["sources"]) != 6 or {s["source"] for s in manifest["sources"]} != {
        c["source"] for c in checks
    }:
        raise ValueError("授权范围与原六份留出文档不一致")
    facts = read_jsonl(LOCAL / "fact_candidates.jsonl")
    for source in manifest["sources"]:
        frozen_source = FROZEN / source["frozen_filename"]
        if sha(frozen_source) != source["source_sha256"]:
            raise ValueError("冻结来源文件损坏")
        current = {draft.chunk_index: draft for draft in
                   process_document(frozen_source, chunk_config=CONFIG).chunks}
        for case in (c for c in cases if c["source"] == source["source"]):
            if current[case["chunk_index"]].content != case["content"]:
                raise ValueError("冻结 Chunk 重分块无法复现")
            if logical_prompt_hash(case["content"]) != case["content_sha256"]:
                raise ValueError("正文哈希不一致")
            if case != selected_gold(
                {key: case[key] for key in ("case_id", "source", "chunk_index", "content",
                                            "content_sha256", "token_count", "metadata")},
                [fact for fact in facts if fact["source"] == case["source"]
                 and fact["chunk_index"] == case["chunk_index"]],
            ):
                raise ValueError("选择性目标无法从预先记录复现")
    if sum(len(c["relations"]) for c in cases) != 30:
        raise ValueError("目标关系数量变化")
    print(f"留出输入校验通过：6 文档、13 Chunk、{manifest['selected_gold_entity_count']} "
          "实体目标、30 关系目标（后两项只覆盖 2 份文档）", flush=True)
    return cases, manifest


async def replay(output, *, name_feedback_regression=False):
    cases, manifest = verify()
    if output.exists():
        raise FileExistsError("结果目录已存在，不覆盖")
    extractor_hash = sha(ROOT / "src/zhiqing_rag/workers/graph_extraction.py")
    if name_feedback_regression and graph_extraction.EXTRACTOR_VERSION != "graph-extractor-v2":
        raise ValueError("名称反馈回归要求明确的 v2 抽取器版本")
    if extractor_hash != manifest["extractor_source_sha256"] and not name_feedback_regression:
        raise ValueError("抽取器已变化，不能作为固定版本留出验证")
    settings = Settings()
    if not settings.deepseek_api_key.get_secret_value():
        raise ValueError("DeepSeek 未配置")
    graph_extraction.SYSTEM_PROMPT = (FROZEN / "system_prompt.txt").read_text(encoding="utf-8")
    previous_model = json.loads((PROMPT_RUN / "metadata.json").read_text(encoding="utf-8"))["model"]
    if settings.graph_extraction_model != previous_model:
        raise ValueError("模型与调参轮次不一致")
    output.mkdir(parents=True, exist_ok=False)
    (output / "extractor_source.py").write_bytes(
        (ROOT / "src/zhiqing_rag/workers/graph_extraction.py").read_bytes()
    )
    (output / "system_prompt.txt").write_bytes((FROZEN / "system_prompt.txt").read_bytes())
    write_json(output / "metadata.json", {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "frozen_manifest_sha256": sha(FROZEN / "manifest.json"),
        "system_prompt_sha256": manifest["prompt_sha256"],
        "extractor_source_sha256": extractor_hash,
        "baseline_extractor_source_sha256": manifest["extractor_source_sha256"],
        "extractor_version": graph_extraction.EXTRACTOR_VERSION,
        "name_feedback_regression": name_feedback_regression,
        "initial_prompt_mode": "unchanged_frozen_table_prompt",
        "model": settings.graph_extraction_model,
        "authorized_case_ids": [case["case_id"] for case in cases],
        "mode": ("name_feedback_regression_no_database_writes" if name_feedback_regression
                 else "isolated_fixed_prompt_holdout_no_database_writes"),
    })
    async with AsyncOpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                           base_url=settings.deepseek_base_url,
                           timeout=settings.graph_llm_timeout_seconds, max_retries=0) as client:
        with (output / "replay.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                trace = []
                row = {"case_id": case["case_id"], "content_sha256": case["content_sha256"],
                       "attempts": trace}
                try:
                    extraction = await extract_chunk(settings, case["content"],
                                                     client=RecordingClient(client, trace))
                    row.update(status="VALID_EXTRACTION",
                               extraction=extraction.model_dump(mode="json"))
                except Exception as error:
                    row.update(status="ERROR", error_type=type(error).__name__)
                    if isinstance(error, GraphExtractionError):
                        row["error_code"] = error.code
                row["finished_at_utc"] = datetime.now(UTC).isoformat()
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                print(f"{case['case_id']}: {row['status']} ({len(trace)} 次请求)", flush=True)
    report(output)


def report(output):
    case_rows, _ = verify()
    cases = {c["case_id"]: c for c in case_rows}
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if meta["frozen_manifest_sha256"] != sha(FROZEN / "manifest.json"):
        raise ValueError("结果绑定的冻结版本不一致")
    rows = read_jsonl(output / "replay.jsonl")
    if len(rows) != 13 or {r["case_id"] for r in rows} != set(cases):
        raise ValueError("结果不完整或超出授权范围")
    counts = Counter()
    document_counts = {}
    diagnostics = []
    queue = []
    for row in rows:
        case = cases[row["case_id"]]
        if row["content_sha256"] != case["content_sha256"]:
            raise ValueError("结果正文哈希不符")
        unavailable = row["status"] == "ERROR" and not any(
            attempt.get("raw_output") is not None for attempt in row["attempts"])
        counts["unavailable_chunks"] += unavailable
        counts["failed_chunks"] += row["status"] == "ERROR"
        counts["api_attempts"] += len(row["attempts"])
        counts["truncated_attempts"] += sum(a.get("finish_reason") == "length"
                                             for a in row["attempts"])
        extraction = Extraction.model_validate(row.get("extraction", {"entities": [],
                                                                     "relations": []}))
        validate_evidence(extraction, case["content"])
        counts["all_predicted_entities"] += len(extraction.entities)
        counts["all_predicted_relations"] += len(extraction.relations)
        doc = document_counts.setdefault(case["source"], Counter())
        doc["chunks"] += 1
        doc["failed_chunks"] += row["status"] == "ERROR"
        doc["all_predicted_entities"] += len(extraction.entities)
        doc["all_predicted_relations"] += len(extraction.relations)
        predicted = {e.id: e for e in extraction.entities}
        for index, relation in enumerate(extraction.relations, 1):
            queue.append({"case_id": case["case_id"], "relation_index": index,
                          "source": predicted[relation.source_id].model_dump(),
                          "relation": relation.model_dump(),
                          "target": predicted[relation.target_id].model_dump(),
                          "evidence_verdict": "PENDING", "independent_human_review": "PENDING"})
        for entity in extraction.entities:
            queue.append({"case_id": case["case_id"], "entity": entity.model_dump(),
                          "entity_verdict": "PENDING", "independent_human_review": "PENDING"})
        if not case["relations"]:
            continue
        comparison = compare_to_gold(extraction, case)
        counts.update(comparison)
        doc.update(comparison)
        counts["selected_unavailable_chunks"] += unavailable
        entity_keys = {(e.type, normalize_name(e.canonical_name)) for e in extraction.entities}
        triples = Counter((predicted[r.source_id].type,
                           normalize_name(predicted[r.source_id].canonical_name), r.type,
                           predicted[r.target_id].type,
                           normalize_name(predicted[r.target_id].canonical_name))
                          for r in extraction.relations)
        gold = {e["id"]: e for e in case["entities"]}
        for relation in case["relations"]:
            source, target = gold[relation["source_id"]], gold[relation["target_id"]]
            skey = (source["type"], normalize_name(source["name"]))
            tkey = (target["type"], normalize_name(target["name"]))
            key = (*skey, relation["type"], *tkey)
            eligible = skey in entity_keys and tkey in entity_keys
            hit = triples[key] > 0
            if hit:
                triples[key] -= 1
            loss = ("EXTRACTION_UNAVAILABLE" if unavailable else "HIT" if hit else
                    "RELATION_WITH_ENDPOINTS_PRESENT" if eligible else "MISSING_ENTITY_ENDPOINT")
            counts[loss] += 1
            counts["conditional_eligible"] += eligible
            counts["conditional_hits"] += eligible and hit
            diagnostics.append({"case_id": case["case_id"], "source": source["name"],
                                "target": target["name"], "type": relation["type"],
                                "loss": loss, "source_found": skey in entity_keys,
                                "target_found": tkey in entity_keys})
    summary = {
        "counts": dict(counts), "documents": {k: dict(v) for k, v in document_counts.items()},
        "selected_entity_recall": (None if counts["selected_unavailable_chunks"] else
                                   counts["exact_entity_matches"] / counts["gold_entities"]),
        "selected_triple_recall": (
            None if counts["selected_unavailable_chunks"] else
            counts["exact_directed_triple_matches"] / counts["gold_relations"]),
        "conditional_recall": (counts["conditional_hits"] / counts["conditional_eligible"]
                               if counts["conditional_eligible"] else None),
        "full_six_document_recall": None, "final_precision": None, "final_f1": None,
        "evidence_precision": None, "normalization_error_rates": None,
        "human_adjudication": "PENDING", "production_promotion": "NOT_ACCEPTED",
        "scope": "13 chunks replayed; selected draft targets cover only two of six documents",
    }
    write_json(output / "summary.json", summary)
    write_jsonl(output / "relation_diagnostics.jsonl", diagnostics)
    write_jsonl(output / "review_queue.jsonl", queue)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "replay", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--name-feedback-regression", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    elif args.output is None:
        parser.error("replay/report 需要 --output")
    elif args.command == "replay":
        asyncio.run(replay(args.output, name_feedback_regression=args.name_feedback_regression))
    else:
        report(args.output)


if __name__ == "__main__":
    main()
