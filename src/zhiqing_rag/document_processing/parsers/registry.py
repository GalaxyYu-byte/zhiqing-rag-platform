"""解析策略注册与分派。"""

from collections.abc import Iterable
from pathlib import Path

from zhiqing_rag.document_processing.format_validation import (
    DEFAULT_MAX_FILE_SIZE,
    DocumentFormat,
    validate_document_format,
)

from .base import DocumentParser
from .models import DocumentParseError, ParsedDocument


class ParserRegistry:
    def __init__(self, parsers: Iterable[DocumentParser] = ()) -> None:
        self._parsers: dict[DocumentFormat, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        if not parser.formats:
            raise ValueError("解析器必须声明支持的文件格式")
        for document_format in parser.formats:
            if document_format in self._parsers:
                raise ValueError(f"解析器已注册: {document_format}")
        for document_format in parser.formats:
            self._parsers[document_format] = parser

    def get(self, document_format: DocumentFormat) -> DocumentParser:
        try:
            return self._parsers[document_format]
        except KeyError as exc:
            raise DocumentParseError(
                "PARSER_NOT_FOUND", f"没有对应的解析器: {document_format}"
            ) from exc


def parse_document(
    path: str | Path,
    *,
    original_filename: str | None = None,
    declared_mime_type: str | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    registry: ParserRegistry | None = None,
) -> ParsedDocument:
    """先验证文件，再按实际格式选择解析策略。"""

    file_path = Path(path)
    validated = validate_document_format(
        file_path,
        original_filename=original_filename,
        declared_mime_type=declared_mime_type,
        max_file_size=max_file_size,
    )
    parser = (registry or default_parser_registry).get(validated.format)
    try:
        result = parser.parse(file_path, validated)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError("PARSE_FAILED", "文档内容解析失败") from exc

    return ParsedDocument(
        format=validated.format,
        original_filename=original_filename or file_path.name,
        file_sha256=validated.file_sha256,
        elements=result.elements,
        warnings=result.warnings,
        text_encoding=validated.text_encoding,
    )


def _create_default_registry() -> ParserRegistry:
    from .docx_parser import DocxParser
    from .excel_parser import XlsParser, XlsxParser
    from .markdown_parser import MarkdownParser
    from .pdf_parser import PdfParser
    from .text_parser import TextParser

    return ParserRegistry(
        (PdfParser(), MarkdownParser(), TextParser(), DocxParser(), XlsxParser(), XlsParser())
    )


default_parser_registry = _create_default_registry()
