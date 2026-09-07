"""Word DOCX 文档解析策略。"""

from collections.abc import Iterator
from pathlib import Path
import re

from docx import Document as open_docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

from .base import DocumentParseError, DocumentParser, ParsedBlock
from .common import base_metadata, clean_text


def _iter_document_blocks(document: DocxDocument) -> Iterator[Paragraph | Table]:
    """按照正文中的真实顺序遍历段落和表格。

    ``document.paragraphs`` 与 ``document.tables`` 会把两类元素分开，无法还原
    表格原本插在正文中的位置。直接遍历底层 body 元素可以保留原始顺序。
    """

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, document)
        elif isinstance(child, CT_Tbl):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int | None:
    """根据 Word 内置的中英文标题样式返回标题级别。"""

    style_name = paragraph.style.name if paragraph.style is not None else ""
    normalized = style_name.strip().lower()
    if not (
        normalized.startswith("heading")
        or style_name.strip().startswith("标题")
    ):
        return None

    matched = re.search(r"(\d+)$", normalized)
    return int(matched.group(1)) if matched else 1


def _table_to_text(table: Table) -> str:
    """把 Word 表格转换为按行排列、竖线分隔的可检索文本。"""

    lines: list[str] = []
    for row in table.rows:
        cells = [clean_text(cell.text).replace("\n", " ") for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class DocxDocumentParser(DocumentParser):
    """解析 Office Open XML 的 ``.docx`` 文件。

    标题作为 ``section_title`` 保存，标题下的段落和表格合并为一个自然块。
    旧版二进制 ``.doc`` 与 ``.docx`` 不是同一种格式，python-docx 不支持
    ``.doc``；若以后需要，可先用 LibreOffice 转换为 ``.docx``。
    """

    supported_extensions = frozenset({".docx"})

    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        path = self.validate_file(file_path)

        try:
            document = open_docx(str(path))
        except Exception as exc:
            raise DocumentParseError(f"DOCX 文件打开失败: {path}") from exc

        blocks: list[ParsedBlock] = []
        buffer: list[str] = []
        current_section: str | None = None
        current_heading_level: int | None = None
        current_heading_path: list[str] = []
        heading_stack: dict[int, str] = {}
        metadata = base_metadata(path, type(self).__name__)

        def flush_buffer() -> None:
            """将当前章节缓存写入结果，并为下一个章节清空缓存。"""

            content = clean_text("\n".join(buffer))
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content,
                        section_title=current_section,
                        heading_level=current_heading_level,
                        heading_path=list(current_heading_path),
                        metadata={
                            **metadata,
                            "section_title": current_section,
                            "heading_level": current_heading_level,
                            "heading_path": list(current_heading_path),
                        },
                    )
                )
            buffer.clear()

        for item in _iter_document_blocks(document):
            if isinstance(item, Paragraph):
                text = clean_text(item.text)
                if not text:
                    continue

                heading_level = _heading_level(item)
                if heading_level is not None:
                    flush_buffer()
                    current_section = text
                    current_heading_level = heading_level
                    for existing_level in [
                        value for value in heading_stack if value >= heading_level
                    ]:
                        del heading_stack[existing_level]
                    heading_stack[heading_level] = text
                    current_heading_path = [
                        heading_stack[key] for key in sorted(heading_stack)
                    ]
                else:
                    buffer.append(text)
            else:
                table_text = _table_to_text(item)
                if table_text:
                    buffer.append(table_text)

        flush_buffer()
        return blocks
