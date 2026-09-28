"""文档解析策略的结构、顺序、定位和未完成状态测试。"""

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import xlrd
from docx import Document
from openpyxl import Workbook
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from zhiqing_rag.document_processing.format_validation import DocumentFormat, ValidatedDocument
from zhiqing_rag.document_processing.parsers import ElementKind, parse_document
from zhiqing_rag.document_processing.parsers.excel_parser import XlsParser


def _add_pdf_text_page(writer: PdfWriter, text: str, *, draw_line: bool = False) -> None:
    page = writer.add_blank_page(width=300, height=400)
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
    commands = [f"BT /F1 12 Tf 40 340 Td ({text}) Tj ET"]
    if draw_line:
        commands.append("40 300 m 260 300 l S")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(commands).encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)


def test_pdf_simple_text_skips_mineru(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "simple.pdf"
    writer = PdfWriter()
    _add_pdf_text_page(writer, "Simple PDF text")
    with path.open("wb") as file:
        writer.write(file)

    from zhiqing_rag.document_processing.parsers import pdf_parser

    def unexpected_mineru(_: Path) -> None:
        raise AssertionError("简单文字页不应启动 MinerU")

    monkeypatch.setattr(pdf_parser, "_run_mineru", unexpected_mineru)
    parsed = parse_document(path)

    assert [element.text for element in parsed.elements] == ["Simple PDF text"]
    assert parsed.elements[0].source_type == "direct_text"
    assert parsed.elements[0].page_number == 1
    assert parsed.elements[0].source_ref == "page:1/block:1"
    assert parsed.is_complete


def test_pdf_mixed_pages_only_send_complex_page_to_mineru(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "mixed.pdf"
    writer = PdfWriter()
    _add_pdf_text_page(writer, "First page")
    writer.add_blank_page(width=300, height=400)
    _add_pdf_text_page(writer, "Third page")
    with path.open("wb") as file:
        writer.write(file)

    from pypdf import PdfReader

    from zhiqing_rag.document_processing.parsers import pdf_parser

    def run_mineru(subset_path: Path) -> dict:
        assert len(PdfReader(subset_path).pages) == 1
        return {"pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": "OCR page"}]}]}

    monkeypatch.setattr(pdf_parser, "_run_mineru", run_mineru)
    parsed = parse_document(path)

    assert [element.text for element in parsed.elements] == [
        "First page", "OCR page", "Third page"
    ]
    assert [element.page_number for element in parsed.elements] == [1, 2, 3]
    assert [element.source_ref for element in parsed.elements] == [
        "page:1/block:1", "page:2/block:1", "page:3/block:1"
    ]


def test_pdf_drawn_layout_uses_mineru(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "table-like.pdf"
    writer = PdfWriter()
    _add_pdf_text_page(writer, "Header", draw_line=True)
    with path.open("wb") as file:
        writer.write(file)

    from zhiqing_rag.document_processing.parsers import pdf_parser

    called = []

    def run_mineru(input_path: Path) -> dict:
        called.append(input_path)
        return {"pages": [{"page_idx": 0, "blocks": [{"type": "text", "content": "Header"}]}]}

    monkeypatch.setattr(pdf_parser, "_run_mineru", run_mineru)
    parsed = parse_document(path)

    assert called == [path]
    assert parsed.elements[0].text == "Header"


def test_pdf_uses_mineru_structure_and_reports_unreadable_image(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "pages.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=300, height=400)
    with path.open("wb") as file:
        writer.write(file)

    from zhiqing_rag.document_processing.parsers import pdf_parser

    monkeypatch.setattr(
        pdf_parser,
        "_run_mineru",
        lambda _: {
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {"type": "header", "content": "重复页眉"},
                        {"type": "doc_title", "level": 1, "content": "报告"},
                        {"type": "text", "content": "正文", "bbox": [0.1, 0.2, 0.8, 0.3]},
                        {
                            "type": "table",
                            "content": "<table><tr><th>名称</th><th>值</th></tr>"
                            "<tr><td>甲</td><td>1</td></tr></table>",
                            "captions": [{"content": "表 1"}],
                        },
                    ],
                },
                {"page_idx": 1, "blocks": [{"type": "text", "content": "扫描页 OCR 内容"}]},
                {"page_idx": 2, "blocks": [{"type": "image", "content": ""}]},
            ]
        },
    )
    parsed = parse_document(path)

    assert parsed.format == DocumentFormat.PDF
    assert [element.kind for element in parsed.elements] == [
        ElementKind.PARAGRAPH,
        ElementKind.HEADING,
        ElementKind.PARAGRAPH,
        ElementKind.TABLE,
        ElementKind.PARAGRAPH,
    ]
    assert parsed.elements[2].page_number == 1
    assert parsed.elements[2].heading_path == ("报告",)
    assert parsed.elements[3].table_rows == (("名称", "值"), ("甲", "1"))
    assert parsed.elements[3].caption == "表 1"
    assert parsed.elements[4].page_number == 2
    assert [(warning.code, warning.page_number) for warning in parsed.warnings] == [
        ("OCR_NO_TEXT", 3),
    ]
    assert not parsed.is_complete


def test_markdown_preserves_blocks_and_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# 总则\n\n正文。\n\n- 第一项\n- 第二项\n\n"
        "| 名称 | 值 |\n| --- | --- |\n| A | 1 |\n\n"
        "```python\nprint('ok')\n```\n",
        encoding="utf-8",
    )

    parsed = parse_document(path)

    assert [element.kind for element in parsed.elements] == [
        ElementKind.HEADING,
        ElementKind.PARAGRAPH,
        ElementKind.LIST,
        ElementKind.TABLE,
        ElementKind.CODE,
    ]
    assert parsed.elements[0].heading_path == ("总则",)
    assert all(element.heading_path == ("总则",) for element in parsed.elements)
    assert parsed.elements[2].line_start == 5
    assert parsed.elements[3].table_rows == (("名称", "值"), ("A", "1"))
    assert parsed.elements[4].line_end == 14
    assert parsed.is_complete


def test_txt_preserves_paragraph_line_ranges_and_encoding(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_bytes("第一行\n第二行\n\n第三行".encode("gb18030"))

    parsed = parse_document(path)

    assert parsed.text_encoding == "gb18030"
    assert [(item.line_start, item.line_end) for item in parsed.elements] == [(1, 2), (4, 4)]
    assert parsed.elements[0].text == "第一行\n第二行"
    assert all(item.heading_path == () for item in parsed.elements)


def test_docx_keeps_heading_paragraph_table_order(tmp_path: Path) -> None:
    path = tmp_path / "manual.docx"
    document = Document()
    document.add_paragraph("第一章", style="Heading 1")
    document.add_paragraph("章节内容")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "字段"
    table.cell(0, 1).text = "值"
    table.cell(1, 0).text = "编号"
    table.cell(1, 1).text = "123"
    document.add_paragraph("后续说明")
    document.save(path)

    parsed = parse_document(path)

    assert [item.kind for item in parsed.elements] == [
        ElementKind.HEADING,
        ElementKind.PARAGRAPH,
        ElementKind.TABLE,
        ElementKind.PARAGRAPH,
    ]
    assert parsed.elements[2].table_rows == (("字段", "值"), ("编号", "123"))
    assert parsed.elements[3].heading_path == ("第一章",)
    assert [item.source_ref for item in parsed.elements] == [
        "body:1",
        "body:2",
        "body:3",
        "body:4",
    ]


def test_xlsx_uploaded_temporary_path(tmp_path: Path) -> None:
    path = tmp_path / "source.tmp"
    workbook = Workbook()
    workbook.active.append(["事项", "负责人"])
    workbook.active.append(["入职准备", "人事"])
    workbook.save(path)
    workbook.close()

    parsed = parse_document(path, original_filename="入职.xlsx")

    assert parsed.is_complete
    assert parsed.elements[0].table_rows == (("事项", "负责人"), ("入职准备", "人事"))
    assert parsed.elements[0].cell_range == "A1:B2"


def test_xlsx_regions_and_missing_formula_cache(tmp_path: Path) -> None:
    path = tmp_path / "data.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "统计"
    sheet["A1"] = "项目"
    sheet["B1"] = "数量"
    sheet["A2"] = "甲"
    sheet["B2"] = "=1+2"
    sheet["C4"] = "独立区域"
    workbook.save(path)

    parsed = parse_document(path)

    assert [element.cell_range for element in parsed.elements] == ["A1:B2", "C4:C4"]
    assert parsed.elements[0].table_rows == (("项目", "数量"), ("甲", "=1+2"))
    assert parsed.elements[1].heading_path == ("统计",)
    assert [(warning.code, warning.cell_range) for warning in parsed.warnings] == [
        ("FORMULA_CACHE_MISSING", "B2")
    ]
    assert not parsed.is_complete


def test_xlsx_uses_cached_formula_value(tmp_path: Path) -> None:
    source_path = tmp_path / "source.xlsx"
    cached_path = tmp_path / "cached.xlsx"
    workbook = Workbook()
    workbook.active["A1"] = "=1+2"
    workbook.save(source_path)

    with ZipFile(source_path) as source, ZipFile(cached_path, "w") as target:
        for entry in source.infolist():
            content = source.read(entry.filename)
            if entry.filename == "xl/worksheets/sheet1.xml":
                assert b"<f>1+2</f><v></v>" in content
                content = content.replace(b"<f>1+2</f><v></v>", b"<f>1+2</f><v>3</v>")
            target.writestr(entry, content)

    parsed = parse_document(cached_path)

    assert parsed.elements[0].table_rows == (("3",),)
    assert parsed.warnings == ()
    assert parsed.is_complete


def test_xls_strategy_preserves_sheet_coordinates(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "data.xls"
    path.write_bytes(b"unused by mocked workbook")
    rows = [
        [
            SimpleNamespace(ctype=xlrd.XL_CELL_TEXT, value="名称"),
            SimpleNamespace(ctype=xlrd.XL_CELL_TEXT, value="值"),
        ],
        [
            SimpleNamespace(ctype=xlrd.XL_CELL_TEXT, value="甲"),
            SimpleNamespace(ctype=xlrd.XL_CELL_NUMBER, value=2.0),
        ],
    ]

    class Sheet:
        name = "清单"
        nrows = 2
        ncols = 2

        def cell(self, row: int, column: int):
            return rows[row][column]

    class Book:
        datemode = 0

        def sheets(self):
            return [Sheet()]

        def release_resources(self):
            pass

    from zhiqing_rag.document_processing.parsers import excel_parser

    monkeypatch.setattr(excel_parser.xlrd, "open_workbook", lambda *args, **kwargs: Book())
    validated = ValidatedDocument(DocumentFormat.XLS, "application/vnd.ms-excel", 1, "0" * 64)

    result = XlsParser().parse(path, validated)

    assert result.elements[0].sheet_name == "清单"
    assert result.elements[0].cell_range == "A1:B2"
    assert result.elements[0].table_rows == (("名称", "值"), ("甲", "2.0"))
