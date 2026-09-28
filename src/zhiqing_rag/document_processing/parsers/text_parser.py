"""纯文本解析策略。"""

from pathlib import Path

from zhiqing_rag.document_processing.format_validation import (
    DocumentFormat,
    ValidatedDocument,
)

from .base import DocumentParser
from .models import ElementKind, ParsedElement, ParseResult


class TextParser(DocumentParser):
    formats = frozenset({DocumentFormat.TXT})

    def parse(self, path: Path, validated: ValidatedDocument) -> ParseResult:
        if validated.text_encoding is None:
            raise ValueError("文本编码未经过校验")
        content = path.read_text(encoding=validated.text_encoding)
        lines = content.splitlines()
        elements: list[ParsedElement] = []
        start: int | None = None
        paragraph: list[str] = []

        def flush(end_line: int) -> None:
            nonlocal start
            if not paragraph or start is None:
                return
            elements.append(
                ParsedElement(
                    kind=ElementKind.PARAGRAPH,
                    text="\n".join(paragraph),
                    order=len(elements),
                    line_start=start,
                    line_end=end_line,
                    source_ref=f"lines:{start}-{end_line}",
                )
            )
            paragraph.clear()
            start = None

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                flush(line_number - 1)
                continue
            if start is None:
                start = line_number
            paragraph.append(line)
        flush(len(lines))
        return ParseResult(tuple(elements))
