"""Holdout checks for column-aware table evidence, without model or graph writes."""

from pathlib import Path

from openpyxl import Workbook

from scripts.experiment_graph_table_extraction import extract_tables
from zhiqing_rag.document_processing import ChunkConfig, process_document
from zhiqing_rag.workers.graph_extraction import validate_evidence


def test_unseen_table_rows_support_only_their_own_governed_by_facts(tmp_path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "运维资料"
    sheet.append(["运维资料保管台账", None, None, None, None])
    sheet.append(["说明：每类资料按本行要求保管", None, None, None, None])
    sheet.append(["资料类型", "主要读者", "典型内容", "保管关注点", "重新核查时"])
    expected = []
    for index in range(1, 13):
        material = f"巡检记录{index}"
        rule = f"仅向值班岗位开放第{index}类记录"
        expected.append((material, rule))
        sheet.append([material, "值班岗位", f"设备状态{index}", rule, "轮班调整"])
    source = tmp_path / "holdout-operations.xlsx"
    workbook.save(source)

    result = process_document(
        source,
        chunk_config=ChunkConfig(max_tokens=115, overlap_tokens=8),
    )
    actual = []
    data_chunks = [
        chunk for chunk in result.chunks if not chunk.metadata.get("table_preamble")
    ]
    assert len(data_chunks) > 1
    for chunk in data_chunks:
        extraction, _, header_in_chunk = extract_tables(
            {"content": chunk.content}, None
        )
        assert header_in_chunk
        validate_evidence(extraction, chunk.content)
        by_id = {entity.id: entity for entity in extraction.entities}
        for relation in extraction.relations:
            source_entity = by_id[relation.source_id]
            target_entity = by_id[relation.target_id]
            assert relation.type == "GOVERNED_BY"
            assert source_entity.name in relation.evidence_quote
            assert target_entity.name in relation.evidence_quote
            actual.append((source_entity.name, target_entity.name))
    assert actual == expected
