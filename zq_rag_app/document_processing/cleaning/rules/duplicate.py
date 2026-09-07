"""文档内完全重复块检测。"""

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


class ExactDuplicateBlockRule(DocumentCleaningRule):
    """标记同一文档内正文和章节完全相同的后续块。"""

    name = "exclude_exact_duplicate"
    priority = 70

    def __init__(
        self,
        *,
        enabled: bool = True,
        require_same_section: bool = True,
        minimum_content_length: int = 20,
    ) -> None:
        self.enabled = enabled
        self.require_same_section = require_same_section
        self.minimum_content_length = minimum_content_length

    def supports(self, context: CleaningContext) -> bool:
        del context
        return self.enabled

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        first_indexes: dict[tuple[str | None, str], int] = {}

        for index, block in enumerate(blocks):
            if block.excluded_from_embedding:
                continue
            content = block.clean_content.strip()
            if len(content) < self.minimum_content_length:
                continue

            section = block.section_title if self.require_same_section else None
            key = (section, content)
            first_index = first_indexes.get(key)
            if first_index is None:
                first_indexes[key] = index
                continue

            block.exclude(
                "exact_duplicate",
                self.name,
                details={
                    "duplicate_of_index": first_index,
                    "duplicate_of_page": blocks[first_index].page_num,
                },
            )

        return blocks
