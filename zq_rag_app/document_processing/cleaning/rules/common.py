"""所有文档格式共享的保守清洗规则。"""

import unicodedata

from ..base import ElementCleaningRule
from ...models import CleanedBlock, CleaningContext


def normalize_text(text: str) -> str:
    """规范字符、换行和空行，同时保留 Markdown 缩进及 Excel 制表符。"""

    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "".join(
        character
        for character in normalized
        if character in "\n\t" or unicodedata.category(character) != "Cc"
    )

    lines: list[str] = []
    previous_blank = False
    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        lines.append("" if is_blank else line)
        previous_blank = is_blank

    # 只删除文档首尾的空白行，不能对整串调用 strip()，否则会破坏 Markdown
    # 首行代码块等依赖前导空格的结构。
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


class NormalizeTextRule(ElementCleaningRule):
    """执行不改变文本语义的通用规范化。"""

    name = "normalize_text"
    priority = 10

    def apply_to_block(
        self,
        block: CleanedBlock,
        context: CleaningContext,
    ) -> None:
        del context
        before = block.clean_content
        normalized = normalize_text(block.clean_content)
        if normalized != block.clean_content:
            block.clean_content = normalized
            block.record_change(
                rule_name=self.name,
                change_type="normalize_content",
                before=before,
                after=normalized,
            )

        if block.section_title is not None:
            normalized_title = normalize_text(block.section_title)
            if normalized_title != block.section_title:
                block.source_metadata.setdefault(
                    "raw_section_title", block.section_title
                )
                block.section_title = normalized_title or None
                block.heading_path = [normalized_title] if normalized_title else []
                block.record_change(
                    rule_name="normalize_section_title",
                    change_type="normalize_metadata",
                    before=str(block.source_metadata["raw_section_title"]),
                    after=normalized_title,
                    details={"field": "section_title"},
                )


class EmptyContentRule(ElementCleaningRule):
    """保留空块用于审计，但阻止它进入 Embedding。"""

    name = "exclude_empty_content"
    priority = 90

    def apply_to_block(
        self,
        block: CleanedBlock,
        context: CleaningContext,
    ) -> None:
        del context
        if not block.clean_content.strip():
            block.exclude("empty_content", self.name)
