"""Independently audit frozen GraphRAG labels and record Codex verification status."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from zhiqing_rag.workers.graph_extraction import Extraction, validate_evidence

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "eval" / "graph_build" / "v1"
GOLD = ROOT / "eval" / "graph_build" / "gold-v1"
LEDGER = GOLD / "verification.jsonl"
ATTESTATION = GOLD / "verification_manifest.json"
CHECKS = [
    "frozen_chunk_identity",
    "source_or_declared_ocr_variant",
    "entity_and_endpoint_schema",
    "verbatim_entity_and_relation_evidence",
    "evidence_character_offsets",
    "condition_inside_relation_evidence",
    "explicit_negative_quote_presence",
    "relation_direction_and_entailment_codex_review",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_cases() -> list[dict]:
    frozen = {row["id"]: row for row in load_lines(INPUT / "chunks.jsonl")}
    cases = load_lines(GOLD / "cases.jsonl")
    manifest = json.loads((GOLD / "manifest.json").read_text(encoding="utf-8"))
    if sha((GOLD / "cases.jsonl").read_bytes()) != manifest["cases_sha256"]:
        raise ValueError("Gold 内容哈希不匹配")
    if sha((INPUT / "manifest.json").read_bytes()) != manifest["frozen_manifest_sha256"]:
        raise ValueError("冻结输入哈希不匹配")
    if len(cases) != manifest["case_count"] or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("样本数量或 ID 不正确")
    original_cases = [row for row in cases if row["source_variant"] == "frozen"]
    if {row["chunk_id"] for row in original_cases} != set(frozen) or len(original_cases) != len(
        frozen
    ):
        raise ValueError("原始 Chunk 未完整覆盖")
    by_id = {row["case_id"]: row for row in cases}
    results = []
    for case in cases:
        source = frozen.get(case["chunk_id"])
        if source is None:
            raise ValueError(f"来源 Chunk 不存在: {case['case_id']}")
        for field in (
            "tenant_id",
            "kb_id",
            "document_id",
            "revision_id",
            "generation_id",
            "chunk_index",
        ):
            if case[field] != source[field]:
                raise ValueError(f"来源范围不匹配: {case['case_id']} {field}")
        if case["source_variant"] == "frozen":
            if case["content"] != source["content"]:
                raise ValueError(f"原始正文不匹配: {case['case_id']}")
        elif case["source_variant"] == "synthetic_ocr_noise":
            parent = by_id.get(case.get("parent_case_id"))
            change = case.get("perturbation", {})
            if (
                parent is None
                or parent["source_variant"] != "frozen"
                or parent["chunk_id"] != case["chunk_id"]
                or parent["content_sha256"] != case.get("parent_content_sha256")
                or parent["content"].count(change.get("from", "")) != 1
                or case["content"] != parent["content"].replace(change["from"], change["to"], 1)
            ):
                raise ValueError(f"模拟 OCR 来源不正确: {case['case_id']}")
        else:
            raise ValueError(f"未知来源变体: {case['case_id']}")
        if sha(case["content"].encode("utf-8")) != case["content_sha256"]:
            raise ValueError(f"正文哈希不正确: {case['case_id']}")
        extraction = Extraction.model_validate(
            {"entities": case["entities"], "relations": case["relations"]}
        )
        validate_evidence(extraction, case["content"])
        entity_ids = {entity["id"] for entity in case["entities"]}
        expected_refs = {
            *(("entity", entity_id) for entity_id in entity_ids),
            *(("relation", f"r{index}") for index in range(1, len(case["relations"]) + 1)),
        }
        actual_refs = {(row["target_kind"], row["target_id"]) for row in case["evidence"]}
        if actual_refs != expected_refs or len(case["evidence"]) != len(expected_refs):
            raise ValueError(f"证据未逐条对应事实: {case['case_id']}")
        for row in [*case["evidence"], *case["hard_negatives"]]:
            if case["content"][row["start"] : row["end"]] != row["quote"]:
                raise ValueError(f"证据位置不正确: {case['case_id']}")
        by_entity = {entity["id"]: entity for entity in case["entities"]}
        for relation in case["relations"]:
            quote = relation["evidence_quote"]
            if (
                by_entity[relation["source_id"]]["name"] not in quote
                or by_entity[relation["target_id"]]["name"] not in quote
                or (relation["condition_text"] and relation["condition_text"] not in quote)
            ):
                raise ValueError(f"关系证据缺少端点或条件: {case['case_id']}")
        for item in case.get("alias_candidates", []):
            if item["first"] not in case["content"] or item["second"] not in case["content"]:
                raise ValueError(f"别名候选缺少原文: {case['case_id']}")
        results.append(
            {
                "case_id": case["case_id"],
                "content_sha256": case["content_sha256"],
                "status": "verified_by_codex",
                "checks": CHECKS,
                "human_review_status": "pending",
            }
        )
    return results


def mark() -> None:
    if LEDGER.exists() or ATTESTATION.exists():
        raise FileExistsError("验证记录已存在；标注变化应创建新的版本和验证记录")
    results = audit_cases()
    data = b"".join(json_bytes(row) for row in results)
    attestation = {
        "status": "verified_by_codex",
        "human_review_status": "pending",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "case_count": len(results),
        "gold_manifest_sha256": sha((GOLD / "manifest.json").read_bytes()),
        "verification_jsonl_sha256": sha(data),
        "checks": CHECKS,
        "scope_note": "Codex source review; business-domain human sign-off is pending.",
    }
    LEDGER.write_bytes(data)
    ATTESTATION.write_bytes(json_bytes(attestation))
    verify()
    print(f"已标记 {len(results)} 条样本为 verified_by_codex；人工业务复核待完成")


def verify() -> None:
    attestation = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    if sha((GOLD / "manifest.json").read_bytes()) != attestation["gold_manifest_sha256"]:
        raise ValueError("Gold 标注在验证后发生变化")
    if sha(LEDGER.read_bytes()) != attestation["verification_jsonl_sha256"]:
        raise ValueError("验证记录哈希不匹配")
    expected = audit_cases()
    actual = load_lines(LEDGER)
    if actual != expected or len(actual) != attestation["case_count"]:
        raise ValueError("验证记录与标注内容不一致")
    print(f"验证记录有效：{len(actual)} 条样本")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("mark", "verify"))
    args = parser.parse_args()
    if args.command == "mark":
        mark()
    else:
        verify()


if __name__ == "__main__":
    main()
