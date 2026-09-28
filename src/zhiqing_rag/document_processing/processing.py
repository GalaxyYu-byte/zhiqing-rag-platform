"""格式校验、解析、清洗和分块的离线入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import ChunkConfig, DocumentChunkDraft, chunk_document
from .cleaning import clean_document
from .format_validation import DEFAULT_MAX_FILE_SIZE
from .parsers.models import ParsedDocument
from .parsers.registry import ParserRegistry, parse_document


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    parsed: ParsedDocument
    cleaned: ParsedDocument
    chunks: tuple[DocumentChunkDraft, ...]


def process_document(
    path: str | Path,
    *,
    original_filename: str | None = None,
    declared_mime_type: str | None = None,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    registry: ParserRegistry | None = None,
    chunk_config: ChunkConfig = ChunkConfig(),
) -> ProcessedDocument:
    parsed = parse_document(
        path,
        original_filename=original_filename,
        declared_mime_type=declared_mime_type,
        max_file_size=max_file_size,
        registry=registry,
    )
    cleaned = clean_document(parsed)
    return ProcessedDocument(
        parsed=parsed,
        cleaned=cleaned,
        chunks=chunk_document(cleaned, config=chunk_config),
    )
