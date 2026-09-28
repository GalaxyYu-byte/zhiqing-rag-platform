"""各格式解析策略共享的结构化结果。"""

from dataclasses import dataclass
from enum import StrEnum

from zhiqing_rag.document_processing.format_validation import DocumentFormat


class ElementKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class ParsedElement:
    kind: ElementKind
    text: str
    order: int
    heading_path: tuple[str, ...] = ()
    heading_level: int | None = None
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    table_rows: tuple[tuple[str, ...], ...] = ()
    source_ref: str | None = None
    source_type: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class ParseWarning:
    code: str
    message: str
    page_number: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    format: DocumentFormat
    original_filename: str
    file_sha256: str
    elements: tuple[ParsedElement, ...]
    warnings: tuple[ParseWarning, ...] = ()
    text_encoding: str | None = None

    @property
    def requires_ocr(self) -> bool:
        return any(warning.code in {"OCR_REQUIRED", "OCR_NO_TEXT"} for warning in self.warnings)

    @property
    def is_complete(self) -> bool:
        blocking_warnings = {"OCR_REQUIRED", "OCR_NO_TEXT", "FORMULA_CACHE_MISSING"}
        has_content = any(
            (element.text.strip() or element.table_rows)
            and element.source_type not in {"header", "footer", "page_number"}
            for element in self.elements
        )
        return has_content and not any(w.code in blocking_warnings for w in self.warnings)


@dataclass(frozen=True, slots=True)
class ParseResult:
    elements: tuple[ParsedElement, ...]
    warnings: tuple[ParseWarning, ...] = ()


class DocumentParseError(RuntimeError):
    """格式合法，但内容提取失败。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
