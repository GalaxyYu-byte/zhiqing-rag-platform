"""Compile and verify the hand-authored GraphRAG gold draft against frozen input."""

import argparse
import copy
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from zhiqing_rag.workers.graph_extraction import Extraction, validate_evidence

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"
PLAN = GOLD / "annotation_plan.py"
CASES = GOLD / "cases.jsonl"
MANIFEST = GOLD / "manifest.json"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def as_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def load_plan():
    spec = importlib.util.spec_from_file_location("graph_gold_annotation_plan", PLAN)
    if spec is None or spec.loader is None:
        raise ValueError("无法读取标注方案")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quote_span(content: str, quote: str) -> dict:
    start = content.find(quote)
    if not quote or start < 0:
        raise ValueError(f"标注引句未逐字出现在 Chunk 中: {quote[:60]}")
    return {"quote": quote, "start": start, "end": start + len(quote)}


def make_case(
    chunk: dict,
    facts: list[tuple],
    standalone: list[tuple],
    negatives: list[str],
    aliases: list[tuple],
    tags: list[str],
):
    content = chunk["content"]
    entities = []
    entity_ids = {}
    relations = []
    evidence = []

    def add_entity(entity_type: str, name: str, support: str) -> str:
        if name not in support:
            raise ValueError(f"实体名称不在证据中: {name}")
        key = (entity_type, name)
        if key in entity_ids:
            return entity_ids[key]
        entity_id = f"e{len(entities) + 1}"
        entity_ids[key] = entity_id
        entities.append(
            {
                "id": entity_id,
                "type": entity_type,
                "name": name,
                "canonical_name": name,
                "evidence_quote": support,
            }
        )
        evidence.append(
            {"target_kind": "entity", "target_id": entity_id, **quote_span(content, support)}
        )
        return entity_id

    for fact in facts:
        if len(fact) not in (6, 7):
            raise ValueError("关系标注必须有 6 或 7 个字段")
        source_type, source_name, predicate, target_type, target_name, support = fact[:6]
        condition = fact[6] if len(fact) == 7 else None
        if condition is not None and condition not in support:
            raise ValueError(f"条件不在关系证据中: {condition}")
        source_id = add_entity(source_type, source_name, support)
        target_id = add_entity(target_type, target_name, support)
        relations.append(
            {
                "type": predicate,
                "source_id": source_id,
                "target_id": target_id,
                "evidence_quote": support,
                "condition_text": condition,
            }
        )
        evidence.append(
            {
                "target_kind": "relation",
                "target_id": f"r{len(relations)}",
                **quote_span(content, support),
            }
        )
    for entity_type, name in standalone:
        add_entity(entity_type, name, name)
    extraction = Extraction.model_validate({"entities": entities, "relations": relations})
    validate_evidence(extraction, content)
    negative_rows = [
        {**quote_span(content, quote), "reason": "explicit_negation_or_non_entailment"}
        for quote in negatives
    ]
    alias_rows = []
    for first, second, note in aliases:
        if first not in content or second not in content:
            raise ValueError("别名候选不在 Chunk 中")
        alias_rows.append({"first": first, "second": second, "note": note, "status": "ambiguous"})
    if (
        not entities
        and not negative_rows
        and (chunk["document_id"], chunk["chunk_index"]) not in {(63, 0), (71, 0)}
    ):
        raise ValueError("空标注仅允许元数据标题块")
    return {
        "case_id": f"gb-{chunk['document_id']}-{chunk['chunk_index']:02d}",
        "source_variant": "frozen",
        "tenant_id": chunk["tenant_id"],
        "kb_id": chunk["kb_id"],
        "document_id": chunk["document_id"],
        "revision_id": chunk["revision_id"],
        "generation_id": chunk["generation_id"],
        "chunk_id": chunk["id"],
        "chunk_index": chunk["chunk_index"],
        "content_sha256": sha256(content.encode("utf-8")),
        "content": content,
        "coverage_tags": sorted(set(tags)),
        "entities": entities,
        "relations": relations,
        "evidence": evidence,
        "hard_negatives": negative_rows,
        "alias_candidates": alias_rows,
        "review_status": "draft_awaiting_human_review",
    }


def compile_cases() -> list[dict]:
    plan = load_plan()
    frozen_chunks = [
        json.loads(line)
        for line in (FROZEN / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    keys = {(chunk["document_id"], chunk["chunk_index"]) for chunk in frozen_chunks}
    if keys != set(plan.FACTS):
        raise ValueError("标注方案必须覆盖冻结集每个 Chunk，且不能包含多余 Chunk")
    for group in (
        plan.STANDALONE_ENTITIES,
        plan.HARD_NEGATIVES,
        plan.ALIAS_CANDIDATES,
        plan.TAGS,
    ):
        if set(group) - keys:
            raise ValueError("标注方案引用了不存在的 Chunk")
    cases = []
    for chunk in frozen_chunks:
        key = (chunk["document_id"], chunk["chunk_index"])
        tags = plan.TAGS.get(key, ["ordinary_fact"] if plan.FACTS[key] else [])
        if plan.HARD_NEGATIVES.get(key):
            tags = [*tags, "negation"]
        if not plan.FACTS[key] and not plan.STANDALONE_ENTITIES.get(key):
            tags = [*tags, "no_positive_graph_fact"]
        cases.append(
            make_case(
                chunk,
                plan.FACTS[key],
                plan.STANDALONE_ENTITIES.get(key, []),
                plan.HARD_NEGATIVES.get(key, []),
                plan.ALIAS_CANDIDATES.get(key, []),
                tags,
            )
        )
    # Controlled OCR-like corruption is explicit augmentation, never presented as source text.
    variants = (
        (35, 0, "第一步", "第—步"),
        (45, 1, "第二步", "第ニ步"),
    )
    for document_id, chunk_index, original, replacement in variants:
        source = next(
            case
            for case in cases
            if case["document_id"] == document_id and case["chunk_index"] == chunk_index
        )
        if source["content"].count(original) != 1:
            raise ValueError("OCR 扰动锚点必须恰好出现一次")
        changed = copy.deepcopy(source)
        changed["content"] = source["content"].replace(original, replacement, 1)
        changed["case_id"] = source["case_id"] + "-ocr1"
        changed["source_variant"] = "synthetic_ocr_noise"
        changed["parent_case_id"] = source["case_id"]
        changed["parent_content_sha256"] = source["content_sha256"]
        changed["perturbation"] = {"from": original, "to": replacement}
        changed["content_sha256"] = sha256(changed["content"].encode("utf-8"))
        changed["coverage_tags"] = sorted(set([*source["coverage_tags"], "synthetic_ocr_noise"]))
        for entity in changed["entities"]:
            entity["evidence_quote"] = entity["evidence_quote"].replace(original, replacement)
        for relation in changed["relations"]:
            relation["evidence_quote"] = relation["evidence_quote"].replace(original, replacement)
        # Recompute offsets because the replacement can shift subsequent evidence.
        changed["evidence"] = [
            {
                **row,
                **quote_span(changed["content"], row["quote"].replace(original, replacement)),
            }
            for row in source["evidence"]
        ]
        changed["hard_negatives"] = [
            {**row, **quote_span(changed["content"], row["quote"])}
            for row in source["hard_negatives"]
        ]
        extraction = Extraction.model_validate(
            {"entities": changed["entities"], "relations": changed["relations"]}
        )
        validate_evidence(extraction, changed["content"])
        cases.append(changed)
    return cases


def summary(cases: list[dict]) -> dict:
    return {
        "case_count": len(cases),
        "frozen_case_count": sum(case["source_variant"] == "frozen" for case in cases),
        "synthetic_ocr_case_count": sum(
            case["source_variant"] == "synthetic_ocr_noise" for case in cases
        ),
        "entity_count": sum(len(case["entities"]) for case in cases),
        "relation_count": sum(len(case["relations"]) for case in cases),
        "hard_negative_count": sum(len(case["hard_negatives"]) for case in cases),
        "tag_counts": dict(
            sorted(Counter(tag for case in cases for tag in case["coverage_tags"]).items())
        ),
    }


def build() -> None:
    if CASES.exists() or MANIFEST.exists():
        raise FileExistsError("Gold v1 已存在，禁止覆盖；修改标注请创建新版本")
    cases = compile_cases()
    cases_data = b"".join(as_bytes(case) for case in cases)
    manifest = {
        "schema_version": "graph-build-gold-v1",
        "dataset_id": "graph-build-gold-v1",
        "parent_dataset_id": "graph-build-v1",
        "frozen_manifest_sha256": sha256((FROZEN / "manifest.json").read_bytes()),
        "annotation_plan_sha256": sha256(PLAN.read_bytes()),
        "cases_sha256": sha256(cases_data),
        "annotation_scope": "explicit_document_specific_facts_under_graph_extractor_v1",
        "review_status": "draft_awaiting_human_review",
        **summary(cases),
    }
    CASES.write_bytes(cases_data)
    MANIFEST.write_bytes(as_bytes(manifest))
    verify()
    print(f"生成 {len(cases)} 条 Gold 草稿，{manifest['relation_count']} 条关系")


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "graph-build-gold-v1":
        raise ValueError("Gold 版本不匹配")
    if sha256((FROZEN / "manifest.json").read_bytes()) != manifest["frozen_manifest_sha256"]:
        raise ValueError("冻结输入清单发生变化")
    if sha256(PLAN.read_bytes()) != manifest["annotation_plan_sha256"]:
        raise ValueError("人工标注方案发生变化")
    data = CASES.read_bytes()
    if sha256(data) != manifest["cases_sha256"]:
        raise ValueError("Gold JSONL 发生变化")
    actual = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    expected = compile_cases()
    if actual != expected or summary(actual) != {key: manifest[key] for key in summary(actual)}:
        raise ValueError("Gold 内容、证据或数量校验失败")
    print(f"Gold 校验通过：{len(actual)} 条样本")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args()
    if args.command == "build":
        build()
    else:
        verify()


if __name__ == "__main__":
    main()
