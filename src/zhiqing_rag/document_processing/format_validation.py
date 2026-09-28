"""在选择解析器之前验证文档文件的格式和基本完整性。"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pypdf import PdfReader
from xlrd import open_workbook

DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 10_000
MAX_UNCOMPRESSED_SIZE = 250 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200


class DocumentFormat(StrEnum):
    PDF = "pdf"
    MARKDOWN = "markdown"
    TXT = "txt"
    DOCX = "docx"
    XLSX = "xlsx"
    XLS = "xls"


_EXTENSIONS = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".txt": DocumentFormat.TXT,
    ".docx": DocumentFormat.DOCX,
    ".xlsx": DocumentFormat.XLSX,
    ".xls": DocumentFormat.XLS,
}

_MIME_TYPES = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.MARKDOWN: "text/markdown",
    DocumentFormat.TXT: "text/plain",
    DocumentFormat.DOCX: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    DocumentFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    DocumentFormat.XLS: "application/vnd.ms-excel",
}

_ACCEPTED_MIME_TYPES = {
    DocumentFormat.PDF: {"application/pdf"},
    DocumentFormat.MARKDOWN: {"text/markdown", "text/x-markdown", "text/plain"},
    DocumentFormat.TXT: {"text/plain"},
    DocumentFormat.DOCX: {_MIME_TYPES[DocumentFormat.DOCX]},
    DocumentFormat.XLSX: {_MIME_TYPES[DocumentFormat.XLSX]},
    DocumentFormat.XLS: {_MIME_TYPES[DocumentFormat.XLS]},
}

_BINARY_SIGNATURES = (
    b"%PDF-",
    b"PK\x03\x04",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
)


class DocumentFormatError(ValueError):
    """可供上传接口及后台任务映射的格式校验错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ValidatedDocument:
    format: DocumentFormat
    mime_type: str
    file_size: int
    file_sha256: str
    text_encoding: str | None = None


def _validate_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(_BINARY_SIGNATURES):
        raise DocumentFormatError("FILE_TYPE_MISMATCH", "文本扩展名与二进制文件内容不一致")

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            content = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if not content.strip():
            raise DocumentFormatError("FILE_EMPTY", "文本文件没有有效内容")
        if any(ord(char) < 32 and char not in "\t\n\r\f" for char in content):
            raise DocumentFormatError("INVALID_FILE", "文本文件包含二进制控制字符")
        return encoding

    raise DocumentFormatError("ENCODING_UNSUPPORTED", "文本文件需使用 UTF-8 或 GB18030 编码")


def _validate_pdf(path: Path) -> None:
    with path.open("rb") as file:
        if not file.read(5) == b"%PDF-":
            raise DocumentFormatError("FILE_TYPE_MISMATCH", "文件内容不是 PDF")
        try:
            reader = PdfReader(file, strict=False)
            if reader.is_encrypted:
                raise DocumentFormatError("ENCRYPTED_FILE", "暂不支持加密 PDF")
            if not reader.pages:
                raise DocumentFormatError("INVALID_FILE", "PDF 没有页面")
        except DocumentFormatError:
            raise
        except Exception as exc:
            raise DocumentFormatError("INVALID_FILE", "PDF 文件结构无效") from exc


def _validate_ooxml(path: Path, document_format: DocumentFormat) -> None:
    required_part = (
        "word/document.xml" if document_format == DocumentFormat.DOCX else "xl/workbook.xml"
    )
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_ENTRIES:
                raise DocumentFormatError("INVALID_FILE", "Office 文件包含过多条目")

            names: set[str] = set()
            expanded_size = 0
            for member in members:
                name = member.filename.replace("\\", "/")
                if (
                    name.startswith("/")
                    or ".." in name.split("/")
                    or name in names
                    or member.flag_bits & 0x1
                ):
                    raise DocumentFormatError("INVALID_FILE", "Office 文件包含非法或加密条目")
                names.add(name)
                expanded_size += member.file_size
                if (
                    expanded_size > MAX_UNCOMPRESSED_SIZE
                    or member.file_size > MAX_COMPRESSION_RATIO * max(member.compress_size, 1)
                ):
                    raise DocumentFormatError("INVALID_FILE", "Office 文件解压体积异常")

            if "[Content_Types].xml" not in names or required_part not in names:
                raise DocumentFormatError("FILE_TYPE_MISMATCH", "Office 文件类型与扩展名不一致")
            if archive.testzip() is not None:
                raise DocumentFormatError("INVALID_FILE", "Office 文件压缩包已损坏")
    except DocumentFormatError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        raise DocumentFormatError("INVALID_FILE", "Office 文件结构无效") from exc


def _validate_xls(path: Path) -> None:
    with path.open("rb") as file:
        if file.read(8) != b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            raise DocumentFormatError("FILE_TYPE_MISMATCH", "文件内容不是旧版 XLS")
    try:
        workbook = open_workbook(str(path), on_demand=True)
        try:
            if not workbook.sheet_names():
                raise DocumentFormatError("INVALID_FILE", "XLS 没有工作表")
        finally:
            workbook.release_resources()
    except DocumentFormatError:
        raise
    except Exception as exc:
        raise DocumentFormatError("INVALID_FILE", "XLS 文件结构无效") from exc


def validate_document_format(
    path: str | Path,
    *,
    original_filename: str | None = None,
    declared_mime_type: str | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> ValidatedDocument:
    """校验本地文件，返回解析器选择及文件修订记录所需的信息。

    `original_filename` 用于临时文件没有原始扩展名的情况。声明 MIME 只作为
    一致性检查，不能代替文件内容校验。
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise DocumentFormatError("FILE_NOT_FOUND", "待校验文件不存在或不是普通文件")
    if max_file_size <= 0:
        raise ValueError("max_file_size 必须大于 0")

    filename = original_filename if original_filename is not None else file_path.name
    if not filename or Path(filename).name != filename or "\x00" in filename:
        raise DocumentFormatError("INVALID_FILENAME", "原始文件名无效")
    document_format = _EXTENSIONS.get(Path(filename).suffix.lower())
    if document_format is None:
        raise DocumentFormatError("UNSUPPORTED_FORMAT", "暂不支持该文件扩展名")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise DocumentFormatError("FILE_EMPTY", "文件为空")
    if file_size > max_file_size:
        raise DocumentFormatError("FILE_TOO_LARGE", "文件超过大小限制")

    if declared_mime_type:
        declared = declared_mime_type.split(";", 1)[0].strip().lower()
        if declared not in _ACCEPTED_MIME_TYPES[document_format] | {"application/octet-stream"}:
            raise DocumentFormatError("FILE_TYPE_MISMATCH", "声明 MIME 与文件扩展名不一致")

    text_encoding = None
    if document_format in {DocumentFormat.MARKDOWN, DocumentFormat.TXT}:
        text_encoding = _validate_text(file_path)
    elif document_format == DocumentFormat.PDF:
        _validate_pdf(file_path)
    elif document_format in {DocumentFormat.DOCX, DocumentFormat.XLSX}:
        _validate_ooxml(file_path, document_format)
    else:
        _validate_xls(file_path)

    digest = hashlib.sha256()
    actual_size = 0
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            actual_size += len(block)
            if actual_size > max_file_size:
                raise DocumentFormatError("FILE_TOO_LARGE", "文件超过大小限制")
            digest.update(block)
    if actual_size != file_size:
        raise DocumentFormatError("INVALID_FILE", "校验期间文件大小发生变化")

    return ValidatedDocument(
        format=document_format,
        mime_type=_MIME_TYPES[document_format],
        file_size=actual_size,
        file_sha256=digest.hexdigest(),
        text_encoding=text_encoding,
    )
