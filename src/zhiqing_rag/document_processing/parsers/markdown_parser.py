"""基于 Markdown AST 的结构化解析策略。"""

from pathlib import Path
from typing import Any

from markdown_it import MarkdownIt

from zhiqing_rag.document_processing.format_validation import (
    DocumentFormat,
    ValidatedDocument,
)

from .base import DocumentParser
from .models import ElementKind, ParsedElement, ParseResult


def _closing_index(tokens: list[Any], opening_index: int) -> int:
    depth = 0
    for index in range(opening_index, len(tokens)):
        depth += tokens[index].nesting
        if depth == 0:
            return index
    return opening_index


def _heading_text(inline: Any) -> str:
    parts: list[str] = []
    for child in inline.children or ():
        if child.type in {"text", "code_inline", "image", "html_inline"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return "".join(parts).strip() or inline.content.strip()


def _table_rows(tokens: list[Any], start: int, end: int) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    cells: list[str] | None = None
    for token in tokens[start : end + 1]:
        if token.type == "tr_open":
            cells = []
        elif token.type == "inline" and cells is not None:
            cells.append(token.content)
        elif token.type == "tr_close" and cells is not None:
            rows.append(tuple(cells))
            cells = None
    return tuple(rows)


class MarkdownParser(DocumentParser):
    formats = frozenset({DocumentFormat.MARKDOWN})

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark").enable("table")

    def parse(self, path: Path, validated: ValidatedDocument) -> ParseResult:
        if validated.text_encoding is None:
            raise ValueError("Markdown 编码未经过校验")
        content = path.read_text(encoding=validated.text_encoding)
        lines = content.splitlines()
        tokens = self._markdown.parse(content)
        elements: list[ParsedElement] = []
        heading_stack: dict[int, str] = {}
        index = 0

        while index < len(tokens):
            token = tokens[index]
            if token.level != 0 or token.map is None:
                index += 1
                continue

            kind: ElementKind
            table_rows: tuple[tuple[str, ...], ...] = ()
            heading_level: int | None = None
            end_index = index
            raw_text = "\n".join(lines[token.map[0] : token.map[1]]).strip()
            if token.type == "heading_open":
                kind = ElementKind.HEADING
                heading_level = int(token.tag[1:])
                title = _heading_text(tokens[index + 1])
                heading_stack = {
                    level: value for level, value in heading_stack.items() if level < heading_level
                }
                heading_stack[heading_level] = title
                text = title
                end_index = _closing_index(tokens, index)
            elif token.type == "paragraph_open":
                kind = ElementKind.PARAGRAPH
                text = raw_text
                end_index = _closing_index(tokens, index)
            elif token.type in {"bullet_list_open", "ordered_list_open"}:
                kind = ElementKind.LIST
                text = raw_text
                end_index = _closing_index(tokens, index)
            elif token.type == "table_open":
                kind = ElementKind.TABLE
                text = raw_text
                end_index = _closing_index(tokens, index)
                table_rows = _table_rows(tokens, index, end_index)
            elif token.type in {"fence", "code_block"}:
                kind = ElementKind.CODE
                text = raw_text
            elif token.type in {"blockquote_open", "html_block", "hr"}:
                kind = ElementKind.PARAGRAPH
                text = raw_text
                if token.nesting == 1:
                    end_index = _closing_index(tokens, index)
            else:
                index += 1
                continue

            if text:
                line_start = token.map[0] + 1
                line_end = token.map[1]
                elements.append(
                    ParsedElement(
                        kind=kind,
                        text=text,
                        order=len(elements),
                        heading_path=tuple(heading_stack[level] for level in sorted(heading_stack)),
                        heading_level=heading_level,
                        line_start=line_start,
                        line_end=line_end,
                        table_rows=table_rows,
                        source_ref=f"lines:{line_start}-{line_end}",
                    )
                )
            index = end_index + 1

        return ParseResult(tuple(elements))
