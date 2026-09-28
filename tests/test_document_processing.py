"""清洗、分块和完整离线入口的行为测试。"""

from pathlib import Path

from pypdf import PdfWriter

from zhiqing_rag.document_processing import ChunkConfig, chunk_document, clean_document
from zhiqing_rag.document_processing.format_validation import DocumentFormat
from zhiqing_rag.document_processing.parsers.models import (
    ElementKind,
    ParsedDocument,
    ParsedElement,
)
from zhiqing_rag.document_processing.processing import process_document


def _document(*elements: ParsedElement) -> ParsedDocument:
    return ParsedDocument(
        format=DocumentFormat.PDF,
        original_filename="sample.pdf",
        file_sha256="0" * 64,
        elements=elements,
    )


def test_cleaning_removes_page_artifacts_and_keeps_source_location() -> None:
    document = _document(
        ParsedElement(ElementKind.PARAGRAPH, "页眉", 0, page_number=1, source_type="header"),
        ParsedElement(
            ElementKind.PARAGRAPH,
            "  第一段\u00a0 内容  ",
            1,
            heading_path=("  章节  ",),
            page_number=1,
            source_ref="page:1/block:2",
        ),
        ParsedElement(
            ElementKind.TABLE,
            "<table><tr><td>甲</td></tr></table>",
            2,
            page_number=1,
            table_rows=((" 甲 ",),),
        ),
    )

    cleaned = clean_document(document)

    assert [item.order for item in cleaned.elements] == [0, 1]
    assert cleaned.elements[0].text == "第一段 内容"
    assert cleaned.elements[0].heading_path == ("章节",)
    assert cleaned.elements[0].source_ref == "page:1/block:2"
    assert cleaned.elements[1].table_rows == (("甲",),)


def test_chunking_splits_large_table_by_rows_with_header_and_sources() -> None:
    table = ParsedElement(
        ElementKind.TABLE,
        "<table>...</table>",
        0,
        heading_path=("统计",),
        page_number=2,
        source_ref="page:2/block:3",
        table_rows=(("名称", "说明"),) + tuple((f"项目{i}", "描述" * 8) for i in range(1, 7)),
        caption="表 1",
    )

    chunks = chunk_document(_document(table), config=ChunkConfig(max_tokens=55, overlap_tokens=8))

    assert len(chunks) > 1
    assert all("名称" in chunk.content and "说明" in chunk.content for chunk in chunks)
    assert [chunk.metadata["row_start"] for chunk in chunks][0] == 2
    assert chunks[-1].metadata["row_end"] == 7
    assert all(chunk.page_number == 2 for chunk in chunks)
    assert all(chunk.source_refs == ("page:2/block:3",) for chunk in chunks)
    assert all(chunk.token_count <= 55 for chunk in chunks)


def test_chunking_recursively_splits_long_text_with_overlap_within_budget() -> None:
    text = "\n\n".join(
        ["第一节介绍背景。" * 12, "第二节说明处理步骤。" * 12, "第三节总结结果。" * 12]
    )
    element = ParsedElement(
        ElementKind.PARAGRAPH,
        text,
        0,
        heading_path=("手册", "说明"),
        page_number=3,
        source_ref="page:3/block:1",
    )

    chunks = chunk_document(
        _document(element),
        config=ChunkConfig(max_tokens=80, overlap_tokens=10, min_chunk_tokens=0),
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= 80 for chunk in chunks)
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.source_refs == ("page:3/block:1",) for chunk in chunks)
    assert all(
        not ("第一节介绍背景。" in chunk.content and "第二节说明处理步骤。" in chunk.content)
        for chunk in chunks
    )
    assert sum(chunk.content.count("第三节总结结果。") for chunk in chunks) >= 12


def test_chunking_uses_english_sentence_boundaries() -> None:
    sentences = [
        f"Step {index} checks the input values and records the expected result."
        for index in range(1, 9)
    ]
    element = ParsedElement(
        ElementKind.PARAGRAPH,
        " ".join(sentences),
        0,
        page_number=1,
    )

    chunks = chunk_document(
        _document(element),
        config=ChunkConfig(max_tokens=45, overlap_tokens=0, min_chunk_tokens=0),
    )

    assert len(chunks) > 1
    assert all(chunk.content.endswith(".") for chunk in chunks)
    assert all(chunk.token_count <= 45 for chunk in chunks)
    assert " ".join(chunk.content for chunk in chunks) == " ".join(sentences)


def test_chunking_merges_tiny_sibling_sections_only_on_same_page() -> None:
    elements = (
        ParsedElement(
            ElementKind.PARAGRAPH,
            "先检查电源。",
            0,
            heading_path=("操作手册", "准备"),
            page_number=1,
            source_ref="page:1/block:1",
        ),
        ParsedElement(
            ElementKind.PARAGRAPH,
            "再启动设备。",
            1,
            heading_path=("操作手册", "启动"),
            page_number=1,
            source_ref="page:1/block:2",
        ),
        ParsedElement(
            ElementKind.PARAGRAPH,
            "最后查看状态。",
            2,
            heading_path=("操作手册", "检查"),
            page_number=2,
            source_ref="page:2/block:1",
        ),
    )

    chunks = chunk_document(
        _document(*elements), config=ChunkConfig(max_tokens=80, overlap_tokens=8)
    )

    assert len(chunks) == 2
    assert chunks[0].metadata["merged_sections"] == ["准备", "启动"]
    assert chunks[0].source_refs == ("page:1/block:1", "page:1/block:2")
    assert chunks[0].section_title == "操作手册"
    assert chunks[1].page_number == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.token_count <= 80 for chunk in chunks)


def test_chunking_splits_oversized_table_row_without_exceeding_budget() -> None:
    table = ParsedElement(
        ElementKind.TABLE,
        "<table>...</table>",
        0,
        heading_path=("统计",),
        page_number=2,
        source_ref="page:2/block:3",
        table_rows=(("名称", "说明"), ("项目", "长" * 300)),
    )

    chunks = chunk_document(
        _document(table), config=ChunkConfig(max_tokens=55, overlap_tokens=8)
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= 55 for chunk in chunks)
    assert all(chunk.metadata["kind"] == "table" for chunk in chunks)
    assert all(chunk.metadata["table_fragment"] for chunk in chunks)
    assert all(chunk.source_refs == ("page:2/block:3",) for chunk in chunks)
    assert sum(chunk.content.count("长") for chunk in chunks) == 300


def test_heading_only_ocr_page_produces_a_chunk() -> None:
    document = _document(
        ParsedElement(
            ElementKind.HEADING,
            "Scanned OCR Test 123",
            0,
            heading_path=("Scanned OCR Test 123",),
            heading_level=2,
            page_number=1,
            source_ref="page:1/block:1",
        )
    )

    chunks = chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].content == "Scanned OCR Test 123"
    assert chunks[0].metadata["kind"] == "heading"


def test_processing_pipeline_validates_parses_cleans_and_chunks(
    monkeypatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "upload.tmp"
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    with pdf_path.open("wb") as output:
        writer.write(output)

    from zhiqing_rag.document_processing.parsers import pdf_parser

    monkeypatch.setattr(
        pdf_parser,
        "_run_mineru",
        lambda _: {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {"type": "header", "content": "公司页眉"},
                        {"type": "doc_title", "content": "合同", "level": 1},
                        {"type": "text", "content": "第一条  合同内容。"},
                        {
                            "type": "table",
                            "content": "<table><tr><td>字段</td><td>值</td></tr>"
                            "<tr><td>期限</td><td>一年</td></tr></table>",
                        },
                    ],
                }
            ]
        },
    )

    result = process_document(pdf_path, original_filename="sample.pdf")

    assert len(result.parsed.elements) == 4
    assert len(result.cleaned.elements) == 3
    assert [chunk.metadata["kind"] for chunk in result.chunks] == ["text", "table"]
    assert result.chunks[0].content == "第一条 合同内容。"
    assert result.chunks[0].section_title == "合同"
    assert result.chunks[1].content.startswith("| 字段 | 值 |")
    assert result.chunks[1].metadata["row_start"] == 1
