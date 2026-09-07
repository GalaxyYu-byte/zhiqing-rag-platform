"""PDF 页面级噪声清洗规则。"""

from collections import Counter
import math
import re

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


_PAGE_NUMBER_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^第\s*\d+\s*页$"),
    re.compile(r"^\d+\s*/\s*\d+$"),
    re.compile(r"^-\s*\d+\s*-$"),
    re.compile(r"^第\s*\d+\s*页\s*(?:/|共)\s*\d+\s*页?$"),
)


def _is_pdf(context: CleaningContext) -> bool:
    return context.source_format == "pdf"


def _is_page_number(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.fullmatch(stripped) for pattern in _PAGE_NUMBER_PATTERNS)


def _non_blank_indices(lines: list[str]) -> list[int]:
    return [index for index, line in enumerate(lines) if line.strip()]


def _remove_lines(block: CleanedBlock, indexes: set[int], operation: str) -> None:
    if not indexes:
        return
    before = block.clean_content
    lines = block.clean_content.split("\n")
    removed_lines = [lines[index] for index in sorted(indexes)]
    block.clean_content = "\n".join(
        line for index, line in enumerate(lines) if index not in indexes
    ).strip("\n")
    block.record_change(
        rule_name=operation,
        change_type="remove_lines",
        before=before,
        after=block.clean_content,
        details={
            "line_indexes": sorted(indexes),
            "removed_lines": removed_lines,
        },
    )


class PdfPageNumberRule(DocumentCleaningRule):
    """删除 PDF 页面首尾位置的独立页码行。"""

    name = "remove_pdf_page_number"
    priority = 30

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def supports(self, context: CleaningContext) -> bool:
        return self.enabled and _is_pdf(context)

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        for block in blocks:
            lines = block.clean_content.split("\n")
            non_blank = _non_blank_indices(lines)
            boundary = set(non_blank[:2] + non_blank[-2:])
            indexes = {
                index for index in boundary if _is_page_number(lines[index])
            }
            _remove_lines(block, indexes, self.name)
        return blocks


def _margin_signature(text: str) -> str:
    normalized = re.sub(r"\d+", "{num}", text.strip().lower())
    return re.sub(r"\s+", " ", normalized)


class PdfRepeatedMarginRule(DocumentCleaningRule):
    """通过跨页重复频率识别并删除短页眉和页脚。

    当前 PDF 解析器按页输出块但没有字符坐标，因此仅检查每页第一行和最后一行，
    并采用较保守的出现次数及比例阈值。
    """

    name = "remove_pdf_repeated_margin"
    priority = 40

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_pages: int = 3,
        min_occurrences: int = 3,
        occurrence_ratio: float = 0.6,
        max_line_length: int = 100,
    ) -> None:
        self.enabled = enabled
        self.min_pages = min_pages
        self.min_occurrences = min_occurrences
        self.occurrence_ratio = occurrence_ratio
        self.max_line_length = max_line_length

    def supports(self, context: CleaningContext) -> bool:
        return self.enabled and _is_pdf(context)

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        page_blocks = [block for block in blocks if block.page_num is not None]
        if len(page_blocks) < self.min_pages:
            return blocks

        headers: Counter[str] = Counter()
        footers: Counter[str] = Counter()
        positions: dict[int, tuple[int, int, str, str]] = {}

        for block in page_blocks:
            lines = block.clean_content.split("\n")
            indexes = _non_blank_indices(lines)
            if not indexes:
                continue
            first_index, last_index = indexes[0], indexes[-1]
            header = _margin_signature(lines[first_index])
            footer = _margin_signature(lines[last_index])
            positions[id(block)] = (first_index, last_index, header, footer)

            if header and len(lines[first_index].strip()) <= self.max_line_length:
                headers[header] += 1
            if footer and len(lines[last_index].strip()) <= self.max_line_length:
                footers[footer] += 1

        threshold = max(
            self.min_occurrences,
            math.ceil(len(page_blocks) * self.occurrence_ratio),
        )
        repeated_headers = {
            signature for signature, count in headers.items() if count >= threshold
        }
        repeated_footers = {
            signature for signature, count in footers.items() if count >= threshold
        }

        for block in page_blocks:
            position = positions.get(id(block))
            if position is None:
                continue
            first_index, last_index, header, footer = position
            indexes: set[int] = set()
            if header in repeated_headers:
                indexes.add(first_index)
            if footer in repeated_footers:
                indexes.add(last_index)
            _remove_lines(block, indexes, self.name)

        return blocks


_STRUCTURAL_LINE_PATTERNS = (
    re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+[编章节条款]"),
    re.compile(r"^[一二三四五六七八九十百千万]+、"),
    re.compile(r"^[（(][一二三四五六七八九十百千万\d]+[）)]"),
    re.compile(r"^\d+[.、)]\s*"),
    re.compile(r"^[-*+]\s+"),
)
_END_PUNCTUATION = frozenset("。！？；：.!?;:")


def _looks_structural(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.match(stripped) for pattern in _STRUCTURAL_LINE_PATTERNS)


def _contains_table_syntax(text: str) -> bool:
    return "\t" in text or "|" in text


def _join_wrapped_lines(previous: str, current: str) -> str:
    left = previous.rstrip()
    right = current.lstrip()
    if left.endswith("-") and right and right[0].isascii() and right[0].isalpha():
        return left[:-1] + right

    if left and right and left[-1].isascii() and right[0].isascii():
        if left[-1].isalnum() and right[0].isalnum():
            return left + " " + right
    return left + right


class PdfWrappedLineRule(DocumentCleaningRule):
    """保守合并 PDF 页面内被硬换行拆开的连续正文。"""

    name = "merge_pdf_wrapped_lines"
    priority = 50

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_previous_length: int = 15,
    ) -> None:
        self.enabled = enabled
        self.min_previous_length = min_previous_length

    def supports(self, context: CleaningContext) -> bool:
        return self.enabled and _is_pdf(context)

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        for block in blocks:
            if block.excluded_from_embedding:
                continue
            before = block.clean_content
            source_lines = before.split("\n")
            merged_lines: list[str] = []
            merged_pairs = 0

            for current in source_lines:
                if not merged_lines or not current.strip() or not merged_lines[-1].strip():
                    merged_lines.append(current)
                    continue

                previous = merged_lines[-1]
                should_merge = (
                    len(previous.strip()) >= self.min_previous_length
                    and previous.rstrip()[-1] not in _END_PUNCTUATION
                    and not _looks_structural(previous)
                    and not _looks_structural(current)
                    and not _contains_table_syntax(previous)
                    and not _contains_table_syntax(current)
                    and not previous.lstrip().startswith("```")
                    and not current.lstrip().startswith("```")
                )
                if should_merge:
                    merged_lines[-1] = _join_wrapped_lines(previous, current)
                    merged_pairs += 1
                else:
                    merged_lines.append(current)

            normalized = "\n".join(merged_lines)
            if normalized != before:
                block.clean_content = normalized
                block.record_change(
                    rule_name=self.name,
                    change_type="merge_lines",
                    before=before,
                    after=normalized,
                    details={"merged_pairs": merged_pairs},
                )
        return blocks
