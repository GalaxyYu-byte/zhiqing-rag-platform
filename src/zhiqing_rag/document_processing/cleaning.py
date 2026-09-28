"""结构化元素的轻量清洗；保留页码和来源定位。"""

from __future__ import annotations

import re
from dataclasses import replace

from .parsers.models import ElementKind, ParsedDocument

_PAGE_ARTIFACTS = {"header", "footer", "page_number"}


def _clean_text(text: str, *, preserve_spacing: bool = False) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    if preserve_spacing:
        return text.strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_document(document: ParsedDocument) -> ParsedDocument:
    """去除页眉、页脚、页码和空块，规范文字但不破坏表格/代码结构。"""

    elements = []
    for element in document.elements:
        if element.source_type in _PAGE_ARTIFACTS:
            continue
        preserve_spacing = element.kind in {ElementKind.CODE, ElementKind.TABLE}
        text = _clean_text(element.text, preserve_spacing=preserve_spacing)
        rows = tuple(tuple(_clean_text(cell) for cell in row) for row in element.table_rows)
        caption = _clean_text(element.caption) if element.caption else None
        if not text and not rows:
            continue
        elements.append(
            replace(
                element,
                text=text,
                order=len(elements),
                table_rows=rows,
                caption=caption,
                heading_path=tuple(_clean_text(title) for title in element.heading_path),
            )
        )
    return replace(document, elements=tuple(elements))
