"""索引重建后的验收仍绑定原文件、修订、租户与冻结逐字依据。"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
from test_query_retrieval_service import chunk

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_query_retrieval.py"
SPEC = importlib.util.spec_from_file_location("query_evaluation_sources", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

FILE_HASH = "a" * 64
REFERENCE = {"quotes": ["主管确认", "管理员开通"], "source_file_sha256": FILE_HASH}
FROZEN = {"tenant_id": 1, "kb_id": 3, "document_id": 4, "revision_id": 11}


def test_same_revision_new_generation_preserves_original_answer_references():
    record = chunk(content="| 条件 | 执行 |\n| 主管确认 | 管理员开通 |", generation_id=20)
    assert MODULE.match_reindexed_reference(REFERENCE, FROZEN, (record,), FILE_HASH) == (record,)


@pytest.mark.parametrize(
    "change",
    [
        {"tenant_id": 2},
        {"knowledge_base_id": 8},
        {"document_id": 6},
        {"revision_id": 12},
        {"content_sha256": "b" * 64},
    ],
)
def test_reindexed_evaluation_rejects_changed_identity_revision_or_content_hash(change):
    record = replace(chunk(content="主管确认后由管理员开通。", generation_id=20), **change)
    with pytest.raises(ValueError, match="活动索引缺少"):
        MODULE.match_reindexed_reference(REFERENCE, FROZEN, (record,), FILE_HASH)


def test_changed_original_file_cannot_be_counted_as_same_frozen_evaluation():
    with pytest.raises(ValueError, match="原文件与冻结来源不一致"):
        MODULE.match_reindexed_reference(
            REFERENCE, FROZEN, (chunk(content="主管确认管理员开通"),), "b" * 64
        )


def test_rechunking_does_not_allow_partial_gold_reference():
    with pytest.raises(ValueError, match="活动索引缺少"):
        MODULE.match_reindexed_reference(REFERENCE, FROZEN, (chunk(content="主管确认"),), FILE_HASH)
