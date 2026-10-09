"""Gold draft stays tied to frozen inputs and has auditable challenge coverage."""

import json
from pathlib import Path

from scripts.build_graph_gold import verify
from scripts.verify_graph_gold_review import verify as verify_review

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "eval" / "graph_build" / "gold-v1" / "cases.jsonl"


def test_graph_gold_integrity_and_strata():
    verify()
    verify_review()
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()]
    assert len(cases) == 61
    assert sum(case["source_variant"] == "frozen" for case in cases) == 59
    tags = {tag for case in cases for tag in case["coverage_tags"]}
    assert {
        "ordinary_fact",
        "multi_entity",
        "multi_relation",
        "cross_sentence",
        "negation",
        "condition",
        "time",
        "table",
        "similar_entity_alias",
        "synthetic_ocr_noise",
    } <= tags
    for case in cases:
        for item in [*case["evidence"], *case["hard_negatives"]]:
            assert case["content"][item["start"] : item["end"]] == item["quote"]
