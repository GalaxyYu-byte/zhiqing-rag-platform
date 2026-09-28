"""使用真实 MinerU Basic 模型验证 PDF 到项目统一格式的转换。"""

import os
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from zhiqing_rag.document_processing import process_document
from zhiqing_rag.document_processing.format_validation import DocumentFormat
from zhiqing_rag.document_processing.parsers import ElementKind

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MINERU_INTEGRATION") != "1",
    reason="设置 RUN_MINERU_INTEGRATION=1 后使用本地 MinerU Basic 模型运行",
)


def _make_text_and_table_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    commands = [
        "BT /F1 20 Tf 60 740 Td (Sales Report) Tj ET",
        "0.5 w",
        *(f"60 {y} m 500 {y} l S" for y in (680, 630, 580, 530)),
        *(f"{x} 680 m {x} 530 l S" for x in (60, 280, 500)),
        "BT /F1 14 Tf 80 645 Td (Item) Tj ET",
        "BT /F1 14 Tf 300 645 Td (Value) Tj ET",
        "BT /F1 14 Tf 80 595 Td (Alpha) Tj ET",
        "BT /F1 14 Tf 300 595 Td (42) Tj ET",
        "BT /F1 14 Tf 80 545 Td (Beta) Tj ET",
        "BT /F1 14 Tf 300 545 Td (99) Tj ET",
    ]
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def test_mineru_basic_auto_converts_to_project_elements_and_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    _make_text_and_table_pdf(pdf_path)

    result = process_document(pdf_path)

    assert result.cleaned.is_complete
    assert any("Sales Report" in element.text for element in result.cleaned.elements)
    table = next(
        element for element in result.cleaned.elements if element.kind == ElementKind.TABLE
    )
    assert table.table_rows[0] == ("Item", "Value")
    assert ("Alpha", "42") in table.table_rows
    assert table.page_number == 1
    assert table.source_ref and table.source_ref.startswith("page:1/block:")
    table_chunk = next(chunk for chunk in result.chunks if chunk.metadata["kind"] == "table")
    assert "Alpha" in table_chunk.content
    assert table.source_ref in table_chunk.source_refs
    assert table_chunk.page_number == 1


@pytest.mark.skipif(
    not os.environ.get("MINERU_TEST_PDF"),
    reason="设置 MINERU_TEST_PDF 为待测试 PDF 的路径",
)
def test_user_pdf_converts_to_project_elements_and_chunks() -> None:
    raw_path = os.environ["MINERU_TEST_PDF"].strip()
    if (
        len(raw_path) >= 2
        and raw_path[0] in {'"', "“", "”"}
        and raw_path[-1] in {'"', "“", "”"}
    ) or (len(raw_path) >= 2 and raw_path[0] == raw_path[-1] == "'"):
        raw_path = raw_path[1:-1]
    pdf_path = Path(raw_path).expanduser()
    assert pdf_path.is_file(), f"PDF 文件不存在: {pdf_path}"

    result = process_document(pdf_path)
    assert result.parsed.format == DocumentFormat.PDF
    assert result.cleaned.elements, f"未提取到文字: {result.cleaned.warnings}"
    assert result.chunks, f"未生成分块: {result.cleaned.warnings}"
    assert [element.order for element in result.cleaned.elements] == list(
        range(len(result.cleaned.elements))
    )
    assert all(
        element.page_number is not None
        and element.page_number > 0
        and element.source_ref is not None
        for element in result.cleaned.elements
    )
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(len(result.chunks)))
    assert all(chunk.metadata["source_refs"] == list(chunk.source_refs) for chunk in result.chunks)

    print(f"PDF: {pdf_path}")
    print(f"elements={len(result.cleaned.elements)}, chunks={len(result.chunks)}")
    print(f"warnings={[warning.code for warning in result.cleaned.warnings]}")
    for element in result.cleaned.elements[:10]:
        print(f"page={element.page_number}, kind={element.kind}, text={element.text[:120]!r}")
