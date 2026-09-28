"""保留正文顺序和标题层级的 DOCX 解析策略。"""

import re
from pathlib import Path

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from zhiqing_rag.document_processing.format_validation import (
    DocumentFormat,
    ValidatedDocument,
)

from .base import DocumentParser
from .models import ElementKind, ParsedElement, ParseResult


def _heading_level(paragraph: Paragraph) -> int | None:
    style = paragraph.style
    if style is None:
        return None
    for name in (style.name, style.style_id):
        match = re.fullmatch(r"(?:Heading|标题)\s*(\d+)", name or "", flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _table_rows(table: Table) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(cell.text.strip() for cell in row.cells) for row in table.rows)


class DocxParser(DocumentParser):
    formats = frozenset({DocumentFormat.DOCX})

    def parse(self, path: Path, validated: ValidatedDocument) -> ParseResult:
        del validated
        document = Document(path)
        elements: list[ParsedElement] = []
        heading_stack: dict[int, str] = {}

        for body_index, item in enumerate(document.iter_inner_content(), start=1):
            heading_level = None
            table_rows: tuple[tuple[str, ...], ...] = ()
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                heading_level = _heading_level(item)
                if heading_level is not None:
                    heading_stack = {
                        level: value
                        for level, value in heading_stack.items()
                        if level < heading_level
                    }
                    heading_stack[heading_level] = text
                    kind = ElementKind.HEADING
                else:
                    kind = ElementKind.PARAGRAPH
            else:
                table_rows = _table_rows(item)
                if not any(cell for row in table_rows for cell in row):
                    continue
                text = "\n".join("\t".join(row) for row in table_rows)
                kind = ElementKind.TABLE

            elements.append(
                ParsedElement(
                    kind=kind,
                    text=text,
                    order=len(elements),
                    heading_path=tuple(heading_stack[level] for level in sorted(heading_stack)),
                    heading_level=heading_level,
                    table_rows=table_rows,
                    source_ref=f"body:{body_index}",
                )
            )

        return ParseResult(tuple(elements))
