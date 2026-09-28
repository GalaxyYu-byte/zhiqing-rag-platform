"""离线文档解析入口和统一结果模型。"""

from .models import (
    DocumentParseError,
    ElementKind,
    ParsedDocument,
    ParsedElement,
    ParseWarning,
)
from .registry import ParserRegistry, parse_document

__all__ = [
    "DocumentParseError",
    "ElementKind",
    "ParsedDocument",
    "ParsedElement",
    "ParseWarning",
    "ParserRegistry",
    "parse_document",
]
