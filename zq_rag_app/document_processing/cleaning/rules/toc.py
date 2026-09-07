"""目录块识别与标记。"""

import re

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


_TOC_TITLE_PATTERN = re.compile(r"^(?:目录|目\s*录|contents)$", re.IGNORECASE)
_TOC_ENTRY_PATTERNS = (
    re.compile(r"^.{2,}?(?:\.{2,}|…{2,}|·{2,})\s*\d+\s*$"),
    re.compile(
        r"^(?:第[一二三四五六七八九十百千万零〇\d]+[编章节]|"
        r"[一二三四五六七八九十百千万]+、|\d+(?:\.\d+)*)"
        r".{1,80}\s+\d+\s*$"
    ),
    re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)?\[[^\]]+\]\([^)]+\)\s*$"),
)


def _is_toc_entry(text: str) -> bool:
    stripped = text.strip()
    return any(pattern.fullmatch(stripped) for pattern in _TOC_ENTRY_PATTERNS)


class TableOfContentsRule(DocumentCleaningRule):
    """识别目录型块并将其排除出正文 Embedding。"""

    name = "exclude_table_of_contents"
    priority = 45

    def __init__(
        self,
        *,
        enabled: bool = True,
        minimum_entries: int = 3,
        minimum_entry_ratio: float = 0.5,
    ) -> None:
        self.enabled = enabled
        self.minimum_entries = minimum_entries
        self.minimum_entry_ratio = minimum_entry_ratio

    def supports(self, context: CleaningContext) -> bool:
        del context
        return self.enabled

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        for block in blocks:
            if block.excluded_from_embedding:
                continue
            lines = [line.strip() for line in block.clean_content.split("\n") if line.strip()]
            if not lines:
                continue

            has_title = bool(
                (block.section_title and _TOC_TITLE_PATTERN.fullmatch(block.section_title.strip()))
                or _TOC_TITLE_PATTERN.fullmatch(lines[0])
            )
            content_lines = lines[1:] if _TOC_TITLE_PATTERN.fullmatch(lines[0]) else lines
            entry_count = sum(_is_toc_entry(line) for line in content_lines)
            ratio = entry_count / max(len(content_lines), 1)

            if entry_count >= self.minimum_entries and (
                has_title or ratio >= self.minimum_entry_ratio
            ):
                block.exclude(
                    "table_of_contents",
                    self.name,
                    details={"entry_count": entry_count, "entry_ratio": ratio},
                )

        return blocks
