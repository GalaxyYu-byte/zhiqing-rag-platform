"""按章节、页码和表格行生成可追溯的检索分块。"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any

import tiktoken

from .parsers.models import ElementKind, ParsedDocument, ParsedElement


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    max_tokens: int = 512
    overlap_tokens: int = 48
    encoding_name: str = "cl100k_base"
    min_chunk_tokens: int = 96

    def __post_init__(self) -> None:
        if self.max_tokens < 32:
            raise ValueError("max_tokens 至少为 32")
        if not 0 <= self.overlap_tokens < self.max_tokens // 2:
            raise ValueError("overlap_tokens 必须小于 max_tokens 的一半")
        if self.min_chunk_tokens < 0:
            raise ValueError("min_chunk_tokens 不能小于 0")


@dataclass(frozen=True, slots=True)
class DocumentChunkDraft:
    chunk_index: int
    content: str
    embedding_content: str
    content_sha256: str
    token_count: int
    page_number: int | None
    section_title: str | None
    source_refs: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def _table_markdown(rows: tuple[tuple[str, ...], ...]) -> str:
    lines = [
        "| " + " | ".join(cell.replace("|", r"\|").replace("\n", "<br>") for cell in row) + " |"
        for row in rows
    ]
    if len(rows) > 1:
        lines.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
    return "\n".join(lines)


def _separators(text: str) -> tuple[str, ...]:
    """先保留段落和句子，再根据中英文比例选择细粒度边界。"""
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    common = ("\n\n", "\n")
    zh = ("。", "！", "？", "；", "，", "、")
    en = (". ", "! ", "? ", "; ", ", ", " ")
    if chinese >= latin * 3:
        return (*common, *zh, " ")
    if latin >= chinese * 4:
        return (*common, *en)
    return (*common, *zh, *en)


def _pieces_with_separator(text: str, separator: str) -> list[str]:
    """分隔符留在上一段，避免切分时丢字或丢标点。"""
    pieces: list[str] = []
    start = 0
    while (index := text.find(separator, start)) >= 0:
        end = index + len(separator)
        pieces.append(text[start:end])
        start = end
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def chunk_document(
    document: ParsedDocument,
    *,
    config: ChunkConfig = ChunkConfig(),
) -> tuple[DocumentChunkDraft, ...]:
    """按结构组织分块，长文本递归切分，输出严格遵守 token 上限。"""

    encoder = tiktoken.get_encoding(config.encoding_name)

    def count(value: str) -> int:
        return len(encoder.encode(value))

    def context(path: tuple[str, ...]) -> str:
        title = " / ".join(path)
        cap = min(96, config.max_tokens // 4)
        while title and count(title) > cap:
            title = title[1:]
        return title

    def embedding(content: str, path: tuple[str, ...]) -> str:
        prefix = context(path)
        return f"{prefix}\n{content}" if prefix else content

    def fits(content: str, path: tuple[str, ...]) -> bool:
        return count(embedding(content, path)) <= config.max_tokens

    def content_budget(path: tuple[str, ...]) -> int:
        prefix = context(path)
        return max(1, config.max_tokens - count(prefix) - (1 if prefix else 0))

    chunks: list[DocumentChunkDraft] = []

    def hard_split(text: str, path: tuple[str, ...]) -> list[str]:
        """只有自然边界都无法容纳时才按 Unicode 字符切。"""
        result: list[str] = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if fits(text[start:middle], path):
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("单个字符加标题上下文已超过 max_tokens")
            result.append(text[start:end])
            start = end
        return result

    def recursive_split(text: str, budget: int, separators: tuple[str, ...]) -> list[str]:
        if count(text) <= budget:
            return [text]
        if not separators:
            return split_by_budget(text, budget)
        separator, *remaining = separators
        if separator not in text:
            return recursive_split(text, budget, tuple(remaining))
        result: list[str] = []
        pending = ""
        for piece in _pieces_with_separator(text, separator):
            if count(piece) > budget:
                if pending:
                    result.append(pending)
                    pending = ""
                result.extend(recursive_split(piece, budget, tuple(remaining)))
            elif pending and count(pending + piece) > budget:
                result.append(pending)
                pending = piece
            else:
                pending += piece
        if pending:
            result.append(pending)
        return result

    def split_by_budget(text: str, budget: int) -> list[str]:
        result: list[str] = []
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            end = start
            while low <= high:
                middle = (low + high) // 2
                if count(text[start:middle]) <= budget:
                    end = middle
                    low = middle + 1
                else:
                    high = middle - 1
            if end == start:
                raise ValueError("单个字符已超过可用 token 预算")
            result.append(text[start:end])
            start = end
        return result

    def split_text(text: str, path: tuple[str, ...], *, overlap: bool = True) -> list[str]:
        if fits(text, path):
            return [text]
        reserved = config.overlap_tokens if overlap else 0
        budget = content_budget(path) - reserved - 2
        if budget < 1:
            raise ValueError("标题上下文和 overlap 已占满 max_tokens")
        base_chunks = recursive_split(text, budget, _separators(text))
        # 尾部短块能装入前块时直接合并，减少碎片。
        merged: list[str] = []
        for piece in base_chunks:
            if (
                merged
                and count(piece) < min(config.min_chunk_tokens, config.max_tokens)
                and fits(merged[-1] + piece, path)
            ):
                merged[-1] += piece
            else:
                merged.append(piece)
        result: list[str] = []
        for index, base in enumerate(merged):
            if not fits(base, path):
                result.extend(hard_split(base, path))
                continue
            if index == 0 or not reserved:
                result.append(base.strip())
                continue
            previous = merged[index - 1]
            start = len(previous)
            while start > 0 and count(previous[start - 1 :]) <= reserved:
                start -= 1
            prefix = previous[start:]
            candidate = f"{prefix}\n{base}" if prefix else base
            while prefix and not fits(candidate, path):
                prefix = prefix[1:]
                candidate = f"{prefix}\n{base}" if prefix else base
            result.append(candidate.strip())
        return [piece.strip() for piece in result if piece.strip()]

    def emit(
        content: str,
        elements: list[ParsedElement],
        *,
        kind: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        content = content.strip()
        if not content:
            return
        first = elements[0]
        pieces = split_text(content, first.heading_path, overlap=kind != "table")
        source_refs = tuple(dict.fromkeys(item.source_ref for item in elements if item.source_ref))
        for index, piece in enumerate(pieces):
            value = embedding(piece, first.heading_path)
            token_count = count(value)
            if token_count > config.max_tokens:
                raise ValueError("分块结果超过 max_tokens")
            metadata: dict[str, Any] = {
                "kind": kind,
                "heading_path": list(first.heading_path),
                "source_refs": list(source_refs),
            }
            if first.bbox is not None and len(elements) == 1:
                metadata["bbox"] = list(first.bbox)
            if extra:
                metadata.update(extra)
            if len(pieces) > 1:
                metadata["split_index"] = index
                if kind == "table":
                    metadata["table_fragment"] = True
            chunks.append(
                DocumentChunkDraft(
                    chunk_index=len(chunks),
                    content=piece,
                    embedding_content=value,
                    content_sha256=hashlib.sha256(piece.encode("utf-8")).hexdigest(),
                    token_count=token_count,
                    page_number=first.page_number,
                    section_title=first.heading_path[-1] if first.heading_path else None,
                    source_refs=source_refs,
                    metadata=metadata,
                )
            )

    pending: list[ParsedElement] = []
    unmatched_headings: list[ParsedElement] = []

    def flush() -> None:
        if pending:
            emit("\n\n".join(item.text for item in pending), pending, kind="text")
            pending.clear()

    for element in document.elements:
        if element.kind == ElementKind.HEADING:
            flush()
            unmatched_headings.append(element)
            continue

        unmatched_headings.clear()

        budget = content_budget(element.heading_path)
        if element.kind == ElementKind.TABLE:
            flush()
            rows = element.table_rows
            prefix = f"{element.caption}\n" if element.caption else ""
            if not rows:
                emit(prefix + element.text, [element], kind="table")
                continue
            whole = prefix + _table_markdown(rows)
            if count(whole) <= budget or len(rows) == 1:
                emit(whole, [element], kind="table", extra={"row_start": 1, "row_end": len(rows)})
                continue
            header = rows[:1]
            group: list[tuple[str, ...]] = []
            group_start = 2
            for row_number, row in enumerate(rows[1:], start=2):
                candidate = prefix + _table_markdown(header + tuple(group) + (row,))
                if group and count(candidate) > budget:
                    emit(
                        prefix + _table_markdown(header + tuple(group)),
                        [element],
                        kind="table",
                        extra={"row_start": group_start, "row_end": row_number - 1},
                    )
                    group = []
                    group_start = row_number
                group.append(row)
            if group:
                emit(
                    prefix + _table_markdown(header + tuple(group)),
                    [element],
                    kind="table",
                    extra={"row_start": group_start, "row_end": len(rows)},
                )
            continue

        if element.kind == ElementKind.CODE:
            flush()
            emit(element.text, [element], kind="code")
            continue

        if not fits(element.text, element.heading_path):
            flush()
            emit(element.text, [element], kind=element.kind.value)
            continue

        if pending and (
            pending[0].page_number != element.page_number
            or pending[0].heading_path != element.heading_path
            or not fits(
                "\n\n".join([*(item.text for item in pending), element.text]),
                element.heading_path,
            )
        ):
            flush()
        pending.append(element)
    flush()
    if unmatched_headings:
        emit(
            "\n".join(item.text for item in unmatched_headings), unmatched_headings, kind="heading"
        )

    # 合并同页同章节，或同一父章节下相邻的小正文块；表格、代码和已有 overlap 不参与。
    merged_chunks: list[DocumentChunkDraft] = []
    for chunk in chunks:
        if not merged_chunks:
            merged_chunks.append(chunk)
            continue
        previous = merged_chunks[-1]
        left_path = tuple(previous.metadata["heading_path"])
        right_path = tuple(chunk.metadata["heading_path"])
        same_section = left_path == right_path
        sibling_sections = (
            bool(left_path and right_path)
            and left_path != right_path
            and left_path[:-1] == right_path[:-1]
            and bool(left_path[:-1])
        )
        can_merge = (
            config.min_chunk_tokens > 0
            and min(previous.token_count, chunk.token_count)
            < min(config.min_chunk_tokens, config.max_tokens)
            and previous.page_number == chunk.page_number
            and previous.metadata["kind"] in {"text", "paragraph", "list"}
            and chunk.metadata["kind"] in {"text", "paragraph", "list"}
            and "split_index" not in previous.metadata
            and "split_index" not in chunk.metadata
            and "merged_sections" not in previous.metadata
            and (same_section or sibling_sections)
        )
        if not can_merge:
            merged_chunks.append(chunk)
            continue
        path = left_path if same_section else left_path[:-1]
        if sibling_sections:
            content = (
                f"{previous.section_title}\n{previous.content}\n\n"
                f"{chunk.section_title}\n{chunk.content}"
            )
        else:
            content = f"{previous.content}\n\n{chunk.content}"
        if not fits(content, path):
            merged_chunks.append(chunk)
            continue
        refs = tuple(dict.fromkeys((*previous.source_refs, *chunk.source_refs)))
        metadata = dict(previous.metadata)
        metadata.pop("bbox", None)
        metadata["kind"] = "text"
        metadata["heading_path"] = list(path)
        metadata["source_refs"] = list(refs)
        metadata["small_chunk_merged"] = True
        if sibling_sections:
            metadata["merged_sections"] = [previous.section_title, chunk.section_title]
        merged_chunks[-1] = replace(
            previous,
            content=content,
            embedding_content=embedding(content, path),
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            token_count=count(embedding(content, path)),
            section_title=path[-1] if path else None,
            source_refs=refs,
            metadata=metadata,
        )
    return tuple(replace(chunk, chunk_index=index) for index, chunk in enumerate(merged_chunks))
