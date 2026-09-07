"""标题识别和层级路径恢复。"""

import re

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


_HEADING_PATTERNS = (
    (1, re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+编(?:\s|$|\S)")),
    (2, re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+章(?:\s|$|\S)")),
    (3, re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+节(?:\s|$|\S)")),
    (4, re.compile(r"^第[一二三四五六七八九十百千万零〇\d]+条(?:\s|$|\S)")),
    (5, re.compile(r"^[一二三四五六七八九十百千万]+、\S+")),
    (6, re.compile(r"^[（(][一二三四五六七八九十百千万\d]+[）)]\s*\S+")),
    (7, re.compile(r"^\d+(?:\.\d+)*[.、)]\s*\S+")),
    (1, re.compile(r"^#{1}\s+\S+")),
    (2, re.compile(r"^#{2}\s+\S+")),
    (3, re.compile(r"^#{3,6}\s+\S+")),
)


def detect_heading_level(text: str) -> int | None:
    stripped = text.strip()
    for level, pattern in _HEADING_PATTERNS:
        if pattern.match(stripped):
            return level
    return None


def _update_stack(stack: dict[int, str], level: int, title: str) -> list[str]:
    for existing_level in [value for value in stack if value >= level]:
        del stack[existing_level]
    stack[level] = title.strip().lstrip("#").strip()
    return [stack[key] for key in sorted(stack)]


class HeadingHierarchyRule(DocumentCleaningRule):
    """利用解析器标题信息和强编号模式恢复标题路径。"""

    name = "restore_heading_hierarchy"
    priority = 80

    def __init__(
        self,
        *,
        enabled: bool = True,
        detect_in_content: bool = True,
    ) -> None:
        self.enabled = enabled
        self.detect_in_content = detect_in_content

    def supports(self, context: CleaningContext) -> bool:
        del context
        return self.enabled

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        stack: dict[int, str] = {}

        for block in blocks:
            before_path = list(block.heading_path)
            if block.heading_path:
                stack = {
                    index: title
                    for index, title in enumerate(block.heading_path, start=1)
                }
            elif block.section_title:
                level = block.heading_level or detect_heading_level(block.section_title) or 1
                block.heading_level = level
                block.heading_path = _update_stack(stack, level, block.section_title)
            elif stack:
                block.heading_path = [stack[key] for key in sorted(stack)]

            detected: list[dict[str, object]] = []
            initial_path = list(block.heading_path)
            if self.detect_in_content:
                non_blank_seen = False
                for line_index, line in enumerate(block.clean_content.split("\n")):
                    if not line.strip():
                        continue
                    level = detect_heading_level(line)
                    if level is not None:
                        path = _update_stack(stack, level, line)
                        detected.append(
                            {
                                "line_index": line_index,
                                "text": line.strip(),
                                "level": level,
                                "heading_path": path,
                            }
                        )
                        if not non_blank_seen:
                            block.heading_level = level
                            block.heading_path = list(path)
                    non_blank_seen = True

            if not block.heading_path and initial_path:
                block.heading_path = initial_path
            if detected:
                block.source_metadata["detected_headings"] = detected

            if block.heading_path != before_path or detected:
                block.record_change(
                    rule_name=self.name,
                    change_type="update_heading_metadata",
                    before=" > ".join(before_path),
                    after=" > ".join(block.heading_path),
                    details={"detected_headings": detected},
                )

        return blocks
