"""Compare a captured Neo4j graph with targeted Gold and audit projection integrity."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from zhiqing_rag.workers.graph_extraction import ENDPOINTS, normalize_name, stable_key

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"
CONTEXT_TYPES = {"Step", "State", "Field", "Rule"}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def entity_signature(entity: dict) -> tuple[str, str]:
    return entity["type"], normalize_name(entity["name"])


def gold_relation_signature(relation: dict, entities: dict[str, dict]) -> tuple:
    source = entities[relation["source_id"]]
    target = entities[relation["target_id"]]
    return (*entity_signature(source), relation["type"], *entity_signature(target))


def predicted_relation_signature(row: dict) -> tuple | None:
    if len(row["subjects"]) != 1 or len(row["objects"]) != 1:
        return None
    source = row["subjects"][0]
    target = row["objects"][0]
    if source.get("type") is None or target.get("type") is None:
        return None
    return (*entity_signature(source), row["assertion"]["predicate"], *entity_signature(target))


def issue(items: list[dict], code: str, **details) -> None:
    items.append({"code": code, **details})


def audit_projection(snapshot: dict, frozen_chunks: list[dict]) -> dict:
    neo = snapshot["neo4j"]
    pg = snapshot["postgres"]
    namespace = snapshot["graph_namespace"]
    chunks = {row["id"]: row for row in frozen_chunks}
    evidence_by_key = {row["key"]: row for row in neo["evidence"]}
    problems = []
    entity_keys_by_gen: dict[int, set[str]] = defaultdict(set)
    assertion_keys_by_gen: dict[int, set[str]] = defaultdict(set)
    evidence_keys_by_gen: dict[int, set[str]] = defaultdict(set)
    linked_evidence = set()
    natural_fact_keys: dict[tuple, list[str]] = defaultdict(list)

    for evidence in neo["evidence"]:
        key = evidence["key"]
        generation = evidence["generation_id"]
        evidence_keys_by_gen[generation].add(key)
        chunk = chunks.get(evidence.get("chunk_id"))
        if chunk is None:
            issue(problems, "EVIDENCE_UNKNOWN_CHUNK", evidence_key=key)
            continue
        for field in (
            "tenant_id",
            "kb_id",
            "document_id",
            "revision_id",
            "generation_id",
            "chunk_index",
        ):
            if evidence.get(field) != chunk[field]:
                issue(problems, "EVIDENCE_SCOPE_MISMATCH", evidence_key=key, field=field)
        if evidence.get("content_sha256") != chunk["content_sha256"]:
            issue(problems, "EVIDENCE_CONTENT_HASH_MISMATCH", evidence_key=key)
        quote = evidence.get("quote")
        start, end = evidence.get("quote_start"), evidence.get("quote_end")
        if (
            not isinstance(quote, str)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or chunk["content"][start:end] != quote
        ):
            issue(problems, "EVIDENCE_QUOTE_OFFSET_INVALID", evidence_key=key)
        else:
            expected_key = stable_key(
                namespace, evidence["tenant_id"], generation, evidence["chunk_id"], start, quote
            )
            if key != expected_key:
                issue(problems, "EVIDENCE_KEY_MISMATCH", evidence_key=key)
        try:
            source_refs = json.loads(evidence.get("source_refs_json", "[]"))
        except (TypeError, ValueError):
            source_refs = None
        if source_refs != chunk["source_refs"]:
            issue(problems, "EVIDENCE_SOURCE_REFS_MISMATCH", evidence_key=key)

    for mention in neo["mentions"]:
        entity, evidence_key = mention["entity"], mention["evidence_key"]
        evidence = evidence_by_key.get(evidence_key)
        if evidence is None:
            issue(problems, "MENTION_WITHOUT_EVIDENCE", entity_key=entity["key"])
            continue
        linked_evidence.add(evidence_key)
        entity_keys_by_gen[evidence["generation_id"]].add(entity["key"])
        if any(
            entity.get(field) != evidence.get(field)
            for field in ("tenant_id", "kb_id", "namespace")
        ):
            issue(problems, "ENTITY_SCOPE_MISMATCH", entity_key=entity["key"])
        canonical = normalize_name(entity.get("name") or "")
        if not canonical or entity.get("normalized_name") != canonical:
            issue(problems, "ENTITY_NORMALIZATION_MISMATCH", entity_key=entity["key"])
        else:
            context = (
                (evidence["document_id"], normalize_name(evidence.get("section_title") or ""))
                if entity["type"] in CONTEXT_TYPES
                else None
            )
            expected_key = stable_key(
                namespace, entity["tenant_id"], entity["kb_id"], entity["type"], canonical, context
            )
            if entity["key"] != expected_key:
                issue(problems, "ENTITY_KEY_MISMATCH", entity_key=entity["key"])

    for row in neo["assertions"]:
        assertion = row["assertion"]
        key = assertion["key"]
        generation = assertion["generation_id"]
        assertion_keys_by_gen[generation].add(key)
        subjects, objects, evidences = row["subjects"], row["objects"], row["evidences"]
        if len(subjects) != 1 or len(objects) != 1:
            issue(
                problems,
                "ORPHAN_OR_AMBIGUOUS_FACT_ENDPOINT",
                assertion_key=key,
                subject_count=len(subjects),
                object_count=len(objects),
            )
        if not evidences:
            issue(problems, "FACT_WITHOUT_EVIDENCE", assertion_key=key)
        if len(evidences) != 1:
            issue(problems, "FACT_EVIDENCE_CARDINALITY", assertion_key=key, count=len(evidences))
        predicate = assertion.get("predicate")
        if len(subjects) == 1 and len(objects) == 1:
            source, target = subjects[0], objects[0]
            allowed = ENDPOINTS.get(predicate)
            if (
                allowed is None
                or source.get("type") not in allowed[0]
                or target.get("type") not in allowed[1]
            ):
                issue(problems, "INVALID_RELATION_ENDPOINT", assertion_key=key)
            if any(
                endpoint.get(field) != assertion.get(field)
                for endpoint in (source, target)
                for field in ("tenant_id", "kb_id", "namespace")
            ):
                issue(problems, "FACT_ENTITY_SCOPE_MISMATCH", assertion_key=key)
        for evidence in evidences:
            evidence_key = evidence["key"]
            linked_evidence.add(evidence_key)
            if evidence_key not in evidence_by_key:
                issue(
                    problems,
                    "FACT_EVIDENCE_OUTSIDE_SCOPE",
                    assertion_key=key,
                    evidence_key=evidence_key,
                )
            if any(
                assertion.get(field) != evidence.get(field)
                for field in ("tenant_id", "kb_id", "document_id", "generation_id", "namespace")
            ):
                issue(problems, "FACT_EVIDENCE_SCOPE_MISMATCH", assertion_key=key)
            if len(subjects) == 1 and len(objects) == 1:
                source_key, target_key = subjects[0]["key"], objects[0]["key"]
                natural = (
                    evidence_key,
                    source_key,
                    predicate,
                    target_key,
                    assertion.get("condition_text"),
                )
                natural_fact_keys[natural].append(key)
                expected_key = stable_key(*natural)
                if key != expected_key:
                    issue(problems, "FACT_KEY_MISMATCH", assertion_key=key)

    for natural, keys in natural_fact_keys.items():
        if len(set(keys)) > 1:
            issue(
                problems,
                "DUPLICATE_FACT",
                assertion_keys=sorted(set(keys)),
                evidence_key=natural[0],
            )
    for key in sorted(set(evidence_by_key) - linked_evidence):
        issue(problems, "ORPHAN_EVIDENCE", evidence_key=key)
    for key in neo["orphan_entity_keys"]:
        issue(problems, "ORPHAN_ENTITY_GLOBAL", entity_key=key)
    for row in neo["duplicate_entity_keys"]:
        issue(problems, "DUPLICATE_ENTITY_KEY_GLOBAL", **row)

    pg_documents = {row["active_generation_id"]: row for row in pg["documents"]}
    per_generation = []
    for generation in sorted(pg_documents):
        row = pg_documents[generation]
        actual_entities = len(entity_keys_by_gen[generation])
        actual_facts = len(assertion_keys_by_gen[generation])
        actual_evidence = len(evidence_keys_by_gen[generation])
        expected_entities, expected_facts = row["entity_count"], row["relation_count"]
        if row["projection_status"] != "READY" or row["generation_status"] != "READY":
            issue(problems, "PROJECTION_NOT_READY", generation_id=generation)
        if row["desired_revision_id"] != row["generation_revision_id"]:
            issue(problems, "ACTIVE_REVISION_MISMATCH", generation_id=generation)
        if row["graph_namespace"] != namespace:
            issue(problems, "NAMESPACE_MISMATCH", generation_id=generation)
        if actual_entities != expected_entities:
            issue(
                problems,
                "ENTITY_COUNT_MISMATCH",
                generation_id=generation,
                postgres=expected_entities,
                neo4j=actual_entities,
            )
        if actual_facts != expected_facts:
            issue(
                problems,
                "FACT_COUNT_MISMATCH",
                generation_id=generation,
                postgres=expected_facts,
                neo4j=actual_facts,
            )
        per_generation.append(
            {
                "generation_id": generation,
                "document_id": row["document_id"],
                "entity_count_postgres": expected_entities,
                "entity_count_neo4j": actual_entities,
                "fact_count_postgres": expected_facts,
                "fact_count_neo4j": actual_facts,
                "evidence_count_neo4j": actual_evidence,
                "active": row["status"] == "PUBLISHED" and row["projection_status"] == "READY",
            }
        )
    raw_count = sum(bool(row["extraction_cache"]) for row in pg["chunks"])
    return {
        "per_generation": per_generation,
        "selected_generation_counts": {
            "entities_by_generation_sum": sum(len(keys) for keys in entity_keys_by_gen.values()),
            "assertions": len(neo["assertions"]),
            "evidence": len(neo["evidence"]),
            "evidence_with_valid_quote_offset": len(neo["evidence"])
            - sum(row["code"] == "EVIDENCE_QUOTE_OFFSET_INVALID" for row in problems),
        },
        "issue_counts": dict(sorted(Counter(row["code"] for row in problems).items())),
        "issues": problems,
        "raw_extraction_cache_chunks": raw_count,
        "raw_to_neo4j_status": (
            "unverified_missing_raw_extractions"
            if raw_count < len(pg["chunks"])
            else "unverified_raw_comparison_not_run"
        ),
        "inactive_generation_evidence_count": len(neo["other_generation_evidence"]),
        "global_orphan_entity_scope": (
            "selected tenants and graph namespace, not only the 10 documents"
        ),
    }


def compare_gold(snapshot: dict, cases: list[dict]) -> tuple[dict, list[dict]]:
    neo = snapshot["neo4j"]
    evidences = {row["key"]: row for row in neo["evidence"]}
    mentions_by_chunk: dict[int, dict[str, dict]] = defaultdict(dict)
    assertions_by_chunk: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for mention in neo["mentions"]:
        evidence = evidences.get(mention["evidence_key"])
        if evidence:
            mentions_by_chunk[evidence["chunk_id"]][mention["entity"]["key"]] = mention["entity"]
    for row in neo["assertions"]:
        for evidence in row["evidences"]:
            assertions_by_chunk[evidence["chunk_id"]].append((row, evidence))
    counts = Counter()
    strata: dict[str, Counter] = defaultdict(Counter)
    review = []
    for case in cases:
        if case["source_variant"] != "frozen":
            continue
        case_id = case["case_id"]
        chunk_id = case["chunk_id"]
        gold_entities = case["entities"]
        predicted_entities = list(mentions_by_chunk[chunk_id].values())
        gold_by_sig: dict[tuple, list[dict]] = defaultdict(list)
        predicted_by_sig: dict[tuple, list[dict]] = defaultdict(list)
        for row in gold_entities:
            gold_by_sig[entity_signature(row)].append(row)
        for row in predicted_entities:
            predicted_by_sig[entity_signature(row)].append(row)
        counts["gold_entities"] += len(gold_entities)
        counts["predicted_entities"] += len(predicted_entities)
        gold_names = {name for _, name in gold_by_sig}
        for signature in sorted(set(gold_by_sig) | set(predicted_by_sig)):
            gold_rows = gold_by_sig.get(signature, [])
            predicted_rows = predicted_by_sig.get(signature, [])
            matched = min(len(gold_rows), len(predicted_rows))
            counts["matched_entities"] += matched
            for row in predicted_rows[matched:]:
                review.append(
                    {
                        "review_id": f"{case_id}:entity:{row['key']}",
                        "case_id": case_id,
                        "kind": "entity",
                        "verdict": None,
                        "entity_key": row["key"],
                        "type": row["type"],
                        "name": row["name"],
                        "reason": "not_in_targeted_gold",
                    }
                )
                if normalize_name(row["name"]) in gold_names:
                    counts["entity_type_mismatch_candidates"] += 1

        by_id = {row["id"]: row for row in gold_entities}
        gold_rel_by_sig: dict[tuple, list[dict]] = defaultdict(list)
        predicted_rel_by_sig: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
        for relation in case["relations"]:
            gold_rel_by_sig[gold_relation_signature(relation, by_id)].append(relation)
        for row, evidence in assertions_by_chunk[chunk_id]:
            signature = predicted_relation_signature(row)
            if signature is not None:
                predicted_rel_by_sig[signature].append((row, evidence))
        counts["gold_relations"] += len(case["relations"])
        counts["predicted_relations"] += sum(map(len, predicted_rel_by_sig.values()))
        case_matches = 0
        for signature in sorted(set(gold_rel_by_sig) | set(predicted_rel_by_sig)):
            predicted = predicted_rel_by_sig.get(signature, [])
            gold = gold_rel_by_sig.get(signature, [])
            matched = min(len(gold), len(predicted))
            counts["matched_relations"] += matched
            case_matches += matched
            for relation, (row, evidence) in zip(gold[:matched], predicted[:matched], strict=True):
                if (relation.get("condition_text") or None) != (
                    row["assertion"].get("condition_text") or None
                ):
                    counts["condition_mismatch_on_matched_relation"] += 1
                else:
                    counts["matched_relation_with_condition"] += 1
                if evidence.get("quote") == relation["evidence_quote"]:
                    counts["exact_gold_evidence_quote_matches"] += 1
            for row, evidence in predicted[matched:]:
                assertion = row["assertion"]
                reasons = ["not_in_targeted_gold"]
                inverse = (signature[3], signature[4], signature[2], signature[0], signature[1])
                if inverse in gold_rel_by_sig:
                    reasons.append("possible_direction_error")
                    counts["direction_error_candidates"] += 1
                start, end = evidence.get("quote_start"), evidence.get("quote_end")
                if (
                    isinstance(start, int)
                    and isinstance(end, int)
                    and any(
                        start < item["end"] and end > item["start"]
                        for item in case["hard_negatives"]
                    )
                ):
                    reasons.append("negative_context_overlap")
                    counts["negative_context_candidates"] += 1
                review.append(
                    {
                        "review_id": f"{case_id}:relation:{assertion['key']}:{evidence['key']}",
                        "case_id": case_id,
                        "kind": "relation",
                        "verdict": None,
                        "assertion_key": assertion["key"],
                        "evidence_key": evidence["key"],
                        "source": row["subjects"][0]["name"],
                        "predicate": assertion["predicate"],
                        "target": row["objects"][0]["name"],
                        "condition_text": assertion.get("condition_text"),
                        "evidence_quote": evidence.get("quote"),
                        "reasons": reasons,
                    }
                )
        for tag in case["coverage_tags"]:
            strata[tag]["gold_relations"] += len(case["relations"])
            strata[tag]["matched_relations"] += case_matches
    entity_recall = ratio(counts["matched_entities"], counts["gold_entities"])
    relation_recall = ratio(counts["matched_relations"], counts["gold_relations"])
    return (
        {
            "population": "59 frozen chunks; two synthetic OCR cases have no live predictions",
            "entity": {
                "gold_count": counts["gold_entities"],
                "prediction_count": counts["predicted_entities"],
                "exact_type_name_match_count": counts["matched_entities"],
                "targeted_recall": entity_recall,
                "precision": None,
                "f1": None,
                "unjudged_prediction_count": counts["predicted_entities"]
                - counts["matched_entities"],
                "type_mismatch_candidates": counts["entity_type_mismatch_candidates"],
            },
            "relation": {
                "gold_count": counts["gold_relations"],
                "prediction_count": counts["predicted_relations"],
                "exact_directed_triple_match_count": counts["matched_relations"],
                "targeted_recall": relation_recall,
                "directed_triple_and_condition_match_count": counts[
                    "matched_relation_with_condition"
                ],
                "precision": None,
                "f1": None,
                "unjudged_prediction_count": counts["predicted_relations"]
                - counts["matched_relations"],
                "direction_error_candidates": counts["direction_error_candidates"],
                "condition_mismatch_on_matched_relation": counts[
                    "condition_mismatch_on_matched_relation"
                ],
                "negative_context_candidates": counts["negative_context_candidates"],
            },
            "normalization": {
                "duplicate_entity_rate": None,
                "merge_error_rate": None,
                "reason": (
                    "Gold only marks ambiguous aliases; it has no adjudicated "
                    "same/different identity pairs."
                ),
            },
            "evidence": {
                "exact_gold_quote_matches_on_matched_relations": counts[
                    "exact_gold_evidence_quote_matches"
                ],
                "evidence_precision": None,
                "unsupported_fact_rate": None,
                "reason": (
                    "Verbatim occurrence and target-Gold matching do not establish "
                    "entailment for unjudged predictions."
                ),
            },
            "relation_strata": {
                tag: {
                    "gold_count": values["gold_relations"],
                    "matched_count": values["matched_relations"],
                    "targeted_recall": ratio(values["matched_relations"], values["gold_relations"]),
                }
                for tag, values in sorted(strata.items())
            },
            "schema_capabilities": {
                "explicit_negation_predicate": False,
                "explicit_temporal_qualifier": False,
                "condition_text": True,
            },
            "metric_gate": (
                "Precision, F1, duplicate/merge rates and semantic evidence rates "
                "require exhaustive or adjudicated Gold."
            ),
        },
        review,
    )


def conditional_relation_review(snapshot: dict, cases: list[dict]) -> tuple[dict, list[dict]]:
    """Measure triple extraction only where both Gold endpoint entities were found."""
    neo = snapshot["neo4j"]
    evidence_by_key = {row["key"]: row for row in neo["evidence"]}
    entities_by_chunk: dict[int, dict[str, dict]] = defaultdict(dict)
    assertions_by_chunk: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for mention in neo["mentions"]:
        evidence = evidence_by_key.get(mention["evidence_key"])
        if evidence is not None:
            entities_by_chunk[evidence["chunk_id"]][mention["entity"]["key"]] = mention["entity"]
    for row in neo["assertions"]:
        for evidence in row["evidences"]:
            assertions_by_chunk[evidence["chunk_id"]].append((row, evidence))

    outcomes = []
    counts = Counter()
    for case in cases:
        if case["source_variant"] != "frozen":
            continue
        chunk_id = case["chunk_id"]
        gold_entities = {entity["id"]: entity for entity in case["entities"]}
        predicted_entities = Counter(
            entity_signature(entity) for entity in entities_by_chunk[chunk_id].values()
        )
        predictions = assertions_by_chunk[chunk_id]
        used = set()
        local = []
        for index, relation in enumerate(case["relations"], start=1):
            source = entity_signature(gold_entities[relation["source_id"]])
            target = entity_signature(gold_entities[relation["target_id"]])
            required = Counter([source, target])
            if relation["source_id"] == relation["target_id"]:
                required[source] = 1
            endpoints_found = all(predicted_entities[sig] >= n for sig, n in required.items())
            local.append(
                {
                    "gold_relation_id": f"{case['case_id']}:r{index}",
                    "case_id": case["case_id"],
                    "chunk_id": chunk_id,
                    "source": gold_entities[relation["source_id"]]["name"],
                    "predicate": relation["type"],
                    "target": gold_entities[relation["target_id"]]["name"],
                    "gold_condition_text": relation.get("condition_text"),
                    "gold_evidence_quote": relation["evidence_quote"],
                    "signature": gold_relation_signature(relation, gold_entities),
                    "source_signature": source,
                    "target_signature": target,
                    "endpoints_found": endpoints_found,
                    "prediction_index": None,
                }
            )

        # Prefer exact condition and evidence before accepting a triple-only match.
        for level in ("condition_and_quote", "condition", "triple"):
            for outcome in local:
                if not outcome["endpoints_found"] or outcome["prediction_index"] is not None:
                    continue
                for index, (row, evidence) in enumerate(predictions):
                    if index in used or predicted_relation_signature(row) != outcome["signature"]:
                        continue
                    condition_match = (row["assertion"].get("condition_text") or None) == (
                        outcome["gold_condition_text"] or None
                    )
                    quote_match = evidence.get("quote") == outcome["gold_evidence_quote"]
                    if level == "condition_and_quote" and not (condition_match and quote_match):
                        continue
                    if level == "condition" and not condition_match:
                        continue
                    outcome["prediction_index"] = index
                    used.add(index)
                    break

        for outcome in local:
            counts["gold_relations"] += 1
            if outcome["endpoints_found"]:
                counts["eligible_relations"] += 1
            index = outcome.pop("prediction_index")
            source, target = outcome.pop("source_signature"), outcome.pop("target_signature")
            signature = outcome.pop("signature")
            if index is not None:
                row, evidence = predictions[index]
                outcome["triple_status"] = "MATCH"
                outcome["condition_status"] = (
                    "MATCH"
                    if (row["assertion"].get("condition_text") or None)
                    == (outcome["gold_condition_text"] or None)
                    else "MISMATCH"
                )
                outcome["assertion_key"] = row["assertion"]["key"]
                outcome["evidence_key"] = evidence["key"]
                counts["triple_matches_given_endpoints"] += 1
                if outcome["condition_status"] == "MATCH":
                    counts["condition_matches_given_endpoints"] += 1
            else:
                outcome["triple_status"] = (
                    "NO_MATCH" if outcome["endpoints_found"] else "INELIGIBLE"
                )
                outcome["condition_status"] = "NOT_EVALUATED"
                if outcome["endpoints_found"]:
                    signatures = [predicted_relation_signature(row) for row, _ in predictions]
                    if any(sig and sig[:2] == source and sig[3:] == target for sig in signatures):
                        outcome["diagnostic_candidate"] = "WRONG_PREDICATE"
                    elif any(
                        sig and sig[:2] == target and sig[2] == signature[2] and sig[3:] == source
                        for sig in signatures
                    ):
                        outcome["diagnostic_candidate"] = "REVERSED_DIRECTION"
                    else:
                        outcome["diagnostic_candidate"] = "MISSING_RELATION"
            outcomes.append(outcome)

    denominator = counts["eligible_relations"]
    return (
        {
            "gold_relation_count": counts["gold_relations"],
            "both_gold_endpoints_found_count": denominator,
            "triple_match_count_given_endpoints": counts["triple_matches_given_endpoints"],
            "relation_type_and_direction_recall_given_endpoints": ratio(
                counts["triple_matches_given_endpoints"], denominator
            ),
            "triple_and_condition_match_count_given_endpoints": counts[
                "condition_matches_given_endpoints"
            ],
            "triple_and_condition_recall_given_endpoints": ratio(
                counts["condition_matches_given_endpoints"], denominator
            ),
            "denominator_definition": (
                "Gold relation whose source and target type+normalized-name entities "
                "are both present in the same frozen chunk."
            ),
        },
        outcomes,
    )


def evidence_support_review(snapshot: dict, cases: list[dict]) -> tuple[dict, list[dict]]:
    """Carry only exact verified Gold facts into SUPPORT; queue all other facts."""
    gold: dict[tuple, list[str]] = defaultdict(list)
    case_by_chunk = {}
    negative_by_chunk = {}
    for case in cases:
        if case["source_variant"] != "frozen":
            continue
        chunk_id = case["chunk_id"]
        case_by_chunk[chunk_id] = case["case_id"]
        negative_by_chunk[chunk_id] = case["hard_negatives"]
        entities = {entity["id"]: entity for entity in case["entities"]}
        for index, relation in enumerate(case["relations"], start=1):
            key = (
                chunk_id,
                gold_relation_signature(relation, entities),
                relation.get("condition_text") or None,
                relation["evidence_quote"],
            )
            gold[key].append(f"{case['case_id']}:r{index}")

    reviews = []
    for row in snapshot["neo4j"]["assertions"]:
        assertion = row["assertion"]
        signature = predicted_relation_signature(row)
        for evidence in row["evidences"]:
            chunk_id = evidence["chunk_id"]
            key = (
                chunk_id,
                signature,
                assertion.get("condition_text") or None,
                evidence.get("quote"),
            )
            gold_refs = gold.get(key, [])
            start, end = evidence.get("quote_start"), evidence.get("quote_end")
            negative_overlap = (
                isinstance(start, int)
                and isinstance(end, int)
                and any(
                    start < item["end"] and end > item["start"]
                    for item in negative_by_chunk.get(chunk_id, [])
                )
            )
            reviews.append(
                {
                    "review_id": f"evidence:{assertion['key']}:{evidence['key']}",
                    "case_id": case_by_chunk.get(chunk_id),
                    "chunk_id": chunk_id,
                    "assertion_key": assertion["key"],
                    "evidence_key": evidence["key"],
                    "source": row["subjects"][0]["name"] if len(row["subjects"]) == 1 else None,
                    "predicate": assertion.get("predicate"),
                    "target": row["objects"][0]["name"] if len(row["objects"]) == 1 else None,
                    "condition_text": assertion.get("condition_text"),
                    "evidence_quote": evidence.get("quote"),
                    "negative_context_overlap": negative_overlap,
                    "gold_relation_refs": gold_refs,
                    "verdict": "SUPPORT" if gold_refs else None,
                    "verdict_basis": "exact_verified_gold_fact_and_quote" if gold_refs else None,
                }
            )
    reviews.sort(key=lambda row: (row["chunk_id"], row["assertion_key"], row["evidence_key"]))
    support = sum(row["verdict"] == "SUPPORT" for row in reviews)
    return (
        {
            "total_fact_evidence_pairs": len(reviews),
            "support_from_exact_verified_gold": support,
            "human_judged_fact_evidence_pairs": 0,
            "human_confirmed_contradict": 0,
            "human_confirmed_insufficient": 0,
            "pending_semantic_review": len(reviews) - support,
            "evidence_precision": None,
            "unsupported_fact_rate": None,
            "verdicts": ["SUPPORT", "CONTRADICT", "INSUFFICIENT"],
        },
        reviews,
    )


def normalization_review(snapshot: dict, cases: list[dict]) -> tuple[dict, list[dict]]:
    """Build identity-pair queues for duplicate and wrong-merge adjudication."""
    evidence_by_key = {row["key"]: row for row in snapshot["neo4j"]["evidence"]}
    by_entity: dict[str, dict[int, dict]] = defaultdict(dict)
    entities = {}
    for mention in snapshot["neo4j"]["mentions"]:
        evidence = evidence_by_key.get(mention["evidence_key"])
        if evidence is None:
            continue
        entity = mention["entity"]
        entities[entity["key"]] = entity
        by_entity[entity["key"]].setdefault(evidence["chunk_id"], evidence)

    reviews = []
    for entity_key, chunks in sorted(by_entity.items()):
        for first, second in combinations(sorted(chunks), 2):
            reviews.append(
                {
                    "review_id": f"normalization:merged:{stable_key(entity_key, first, second)}",
                    "candidate_kind": "possible_wrong_merge",
                    "system_merged": True,
                    "entity_keys": [entity_key, entity_key],
                    "entity_type": entities[entity_key]["type"],
                    "names": [entities[entity_key]["name"]] * 2,
                    "chunk_ids": [first, second],
                    "evidence_quotes": [chunks[first]["quote"], chunks[second]["quote"]],
                    "identity_verdict": None,
                }
            )
    by_canonical: dict[tuple, set[str]] = defaultdict(set)
    for key, entity in entities.items():
        scope = (
            entity["tenant_id"],
            entity["kb_id"],
            entity["type"],
            entity["normalized_name"],
        )
        by_canonical[scope].add(key)
    for keys in by_canonical.values():
        for first, second in combinations(sorted(keys), 2):
            first_chunk = min(by_entity[first])
            second_chunk = min(by_entity[second])
            reviews.append(
                {
                    "review_id": f"normalization:split:{stable_key(first, second)}",
                    "candidate_kind": "possible_duplicate",
                    "system_merged": False,
                    "entity_keys": [first, second],
                    "entity_type": entities[first]["type"],
                    "names": [entities[first]["name"], entities[second]["name"]],
                    "chunk_ids": [first_chunk, second_chunk],
                    "evidence_quotes": [
                        by_entity[first][first_chunk]["quote"],
                        by_entity[second][second_chunk]["quote"],
                    ],
                    "identity_verdict": None,
                }
            )
    for case in cases:
        if case["source_variant"] != "frozen":
            continue
        chunk_entities = [
            entities[key] for key, chunks in by_entity.items() if case["chunk_id"] in chunks
        ]
        for alias in case["alias_candidates"]:
            first = [
                entity["key"]
                for entity in chunk_entities
                if normalize_name(entity["name"]) == normalize_name(alias["first"])
            ]
            second = [
                entity["key"]
                for entity in chunk_entities
                if normalize_name(entity["name"]) == normalize_name(alias["second"])
            ]
            first_key = first[0] if first else None
            second_key = second[0] if second else None
            reviews.append(
                {
                    "review_id": f"normalization:alias:{case['case_id']}:1",
                    "candidate_kind": "gold_ambiguous_alias",
                    "system_merged": first_key == second_key if first_key and second_key else None,
                    "entity_keys": [first_key, second_key],
                    "entity_type": None,
                    "names": [alias["first"], alias["second"]],
                    "chunk_ids": [case["chunk_id"], case["chunk_id"]],
                    "evidence_quotes": [None, None],
                    "gold_note": alias["note"],
                    "identity_verdict": None,
                }
            )
    reviews.sort(key=lambda row: row["review_id"])
    counts = Counter(row["candidate_kind"] for row in reviews)
    return (
        {
            "possible_duplicate_pairs": counts["possible_duplicate"],
            "possible_wrong_merge_pairs": counts["possible_wrong_merge"],
            "ambiguous_gold_alias_pairs": counts["gold_ambiguous_alias"],
            "identity_pairs_judged": 0,
            "duplicate_entity_rate": None,
            "merge_error_rate": None,
            "identity_verdicts": ["SAME", "DIFFERENT"],
            "scope": "candidate pairs from this snapshot; not a complete identity-pair census",
        },
        reviews,
    )


def render_markdown(report: dict) -> str:
    facts = report["fact_quality"]
    audit = report["projection_consistency"]
    conditional = facts["relation_conditional"]
    evidence = facts["evidence_support"]
    normalization = facts["normalization_pairs"]
    lines = [
        "# GraphRAG 离线图谱质量评估",
        "",
        f"快照时间：{report['captured_at_utc']}。评估范围：10 份文档、59 个原始 Chunk。",
        "",
        "## ① Entity Recall",
        "",
        (
            f"实体命中 {facts['entity']['exact_type_name_match_count']}/"
            f"{facts['entity']['gold_count']}，定向召回 {facts['entity']['targeted_recall']}。"
        ),
        "",
        "## ② Relation Triple Recall",
        "",
        (
            f"有向三元组命中 {facts['relation']['exact_directed_triple_match_count']}/"
            f"{facts['relation']['gold_count']}，定向召回 {facts['relation']['targeted_recall']}。"
        ),
        "",
        "## ③ Relation Conditional Recall",
        "",
        (
            "仅在 Gold 的 source 和 target 实体都已抽到时计算：关系类型与方向命中 "
            f"{conditional['triple_match_count_given_endpoints']}/"
            f"{conditional['both_gold_endpoints_found_count']}，"
            f"条件也一致 {conditional['triple_and_condition_match_count_given_endpoints']}/"
            f"{conditional['both_gold_endpoints_found_count']}。"
        ),
        "逐条端点、三元组和条件判定见 gold_relation_outcomes.jsonl。",
        "",
        "## ④ Evidence Support",
        "",
        (
            f"{evidence['total_fact_evidence_pairs']} 个 Fact–Evidence 对中，"
            f"{evidence['support_from_exact_verified_gold']} 个与已验证 Gold 的"
            "有向三元组、条件和证据原文完全一致，可标为 SUPPORT；"
            f"{evidence['pending_semantic_review']} 个待语义复核。"
        ),
        (
            "人工裁决使用 SUPPORT / CONTRADICT / INSUFFICIENT；待复核不等于 "
            "INSUFFICIENT。逐条记录见 evidence_review.jsonl。"
        ),
        "",
        "## ⑤ Normalization",
        "",
        (
            f"生成 {normalization['possible_duplicate_pairs']} 对可能应合并未合并、"
            f"{normalization['possible_wrong_merge_pairs']} 对可能误合并、"
            f"{normalization['ambiguous_gold_alias_pairs']} 对 Gold 中身份不明的别名。"
        ),
        (
            "当前没有经裁决的 SAME / DIFFERENT 身份对，Duplicate Entity Rate 和 "
            "Merge Error Rate 暂无数值。逐对记录见 normalization_review.jsonl。"
        ),
        "",
        "## ⑥ 未标注预测裁决与最终指标",
        "",
        (
            f"{report['review_queue_count']} 个未命中定向 Gold 的实体或关系预测"
            f"需要人工裁决（实体 {facts['entity']['unjudged_prediction_count']}、"
            f"关系 {facts['relation']['unjudged_prediction_count']}），"
            "见 review_queue.jsonl。"
        ),
        (
            "Gold 仍非穷尽且未获业务签收。只有补齐所有真实事实、裁决预测与身份对后，"
            "才计算最终 Precision / Recall / F1；当前只报告上述定向召回。"
        ),
        "",
        "## 挑战样本分层",
        "",
        (
            "按 Chunk 标签的关系定向召回：否定语境 "
            f"{facts['relation_strata']['negation']['targeted_recall']}，"
            f"条件 {facts['relation_strata']['condition']['targeted_recall']}，"
            f"时间语境 {facts['relation_strata']['time']['targeted_recall']}。"
            "这不是各语义属性的正确率；当前 Schema 无独立否定谓词或时间限定字段。"
        ),
        "",
        "## 投影一致性",
        "",
        (
            f"- PostgreSQL 原始抽取缓存覆盖 {audit['raw_extraction_cache_chunks']}/59 "
            "个 Chunk；原始抽取与 Neo4j 逐条一致性状态："
            f"{audit['raw_to_neo4j_status']}。"
        ),
        (
            f"- 检查出 {len(audit['issues'])} 条异常记录，按类型计数见 report.json "
            "的 projection_consistency.issue_counts。"
        ),
        (
            "- 选中文档的其他批次 Evidence："
            f"{audit['inactive_generation_evidence_count']}。"
            "全局孤立 Entity 的检查范围为所选租户和命名空间。"
        ),
        (
            "- 所选批次 Evidence 原文位置有效："
            f"{audit['selected_generation_counts']['evidence_with_valid_quote_offset']}/"
            f"{audit['selected_generation_counts']['evidence']}。"
            "这是可定位性检查，不是语义支持率。"
        ),
        "",
        "## 解释边界",
        "",
        (
            "Gold 源文档为虚构企业测试材料。逐字证据与字符位置只能证明可定位，"
            "不能替代事实蕴含判断。当前投影由 Neo4j 重建预测；缺少 PostgreSQL "
            "原始抽取缓存，因此无法证明抽取到落图无损。"
            "PostgreSQL 和 Neo4j 分别读取，快照之间没有跨库原子事务。"
        ),
        "",
    ]
    return "\n".join(lines)


def evaluate(snapshot_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在: {output_dir}")
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot["frozen_manifest_sha256"] != file_hash(FROZEN / "manifest.json"):
        raise ValueError("快照与冻结集版本不一致")
    if snapshot["gold_manifest_sha256"] != file_hash(GOLD / "manifest.json"):
        raise ValueError("快照与 Gold 版本不一致")
    frozen_chunks = read_jsonl(FROZEN / "chunks.jsonl")
    cases = read_jsonl(GOLD / "cases.jsonl")
    projection = audit_projection(snapshot, frozen_chunks)
    quality, review = compare_gold(snapshot, cases)
    conditional, relation_outcomes = conditional_relation_review(snapshot, cases)
    evidence_support, evidence_reviews = evidence_support_review(snapshot, cases)
    normalization, normalization_reviews = normalization_review(snapshot, cases)
    quality["relation_conditional"] = conditional
    quality["evidence_support"] = evidence_support
    quality["normalization_pairs"] = normalization
    quality["final_metrics"] = {
        "entity_precision": None,
        "entity_recall": None,
        "entity_f1": None,
        "relation_precision": None,
        "relation_recall": None,
        "relation_f1": None,
        "status": "awaiting_exhaustive_human_gold_and_prediction_adjudication",
    }
    report = {
        "schema_version": "graph-quality-report-v2",
        "captured_at_utc": snapshot["captured_at_utc"],
        "snapshot_sha256": file_hash(snapshot_path),
        "gold_manifest_sha256": snapshot["gold_manifest_sha256"],
        "fact_quality": quality,
        "projection_consistency": projection,
        "review_queue_count": len(review),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "review_queue.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review),
        encoding="utf-8",
    )
    for filename, rows in (
        ("gold_relation_outcomes.jsonl", relation_outcomes),
        ("evidence_review.jsonl", evidence_reviews),
        ("normalization_review.jsonl", normalization_reviews),
    ):
        (output_dir / filename).write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        f"完成：实体定向召回 {quality['entity']['targeted_recall']}、"
        f"关系定向召回 {quality['relation']['targeted_recall']}、"
        f"投影异常 {len(projection['issues'])} 条"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.snapshot, args.output)


if __name__ == "__main__":
    main()
