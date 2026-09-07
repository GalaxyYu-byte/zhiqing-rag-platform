"""可扩展的文档解析策略集合。"""

from .base import (
    DocumentParseError,
    DocumentParser,
    ParsedBlock,
    UnsupportedDocumentTypeError,
)
from .docx_parser import DocxDocumentParser
from .excel_parser import ExcelDocumentParser
from .factory import DocumentParserFactory, default_parser_factory
from .markdown_parser import MarkdownDocumentParser
from .pdf_parser import PdfDocumentParser
from .txt_parser import TxtDocumentParser

# 定义模块的公共接口
__all__ = [
    "DocumentParseError",
    "DocumentParser",
    "DocumentParserFactory",
    "DocxDocumentParser",
    "ExcelDocumentParser",
    "MarkdownDocumentParser",
    "ParsedBlock",
    "PdfDocumentParser",
    "TxtDocumentParser",
    "UnsupportedDocumentTypeError",
    "default_parser_factory",
]

