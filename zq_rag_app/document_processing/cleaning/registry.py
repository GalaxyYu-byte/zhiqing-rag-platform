"""清洗规则注册表及默认规则集合。"""

from collections.abc import Iterable

from .base import DocumentCleaningRule
from .config import CleaningConfig
from .rules import (
    EmptyContentRule,
    ExactDuplicateBlockRule,
    HeadingHierarchyRule,
    NoiseCharacterQualityRule,
    NormalizeExcelRowsRule,
    NormalizeTextRule,
    PdfPageNumberRule,
    PdfRepeatedMarginRule,
    PdfWrappedLineRule,
    TableOfContentsRule,
)
from .validators import ChangeRatioValidationRule, ProtectedTokenValidationRule


class CleaningRuleRegistry:
    """保存清洗规则，并确保规则名称唯一。"""

    def __init__(self, rules: Iterable[DocumentCleaningRule] = ()) -> None:
        self._rules: dict[str, DocumentCleaningRule] = {}
        for rule in rules:
            self.register(rule)

    def register(
        self,
        rule: DocumentCleaningRule,
        *,
        replace: bool = False,
    ) -> None:
        if rule.name in self._rules and not replace:
            raise ValueError(f"清洗规则已注册: {rule.name}")
        self._rules[rule.name] = rule

    def ordered_rules(self) -> tuple[DocumentCleaningRule, ...]:
        """按优先级和名称返回稳定排序的规则。"""

        return tuple(
            sorted(self._rules.values(), key=lambda rule: (rule.priority, rule.name))
        )


def default_cleaning_registry(
    config: CleaningConfig | None = None,
) -> CleaningRuleRegistry:
    """创建互不共享可变状态的默认规则注册表。"""

    resolved = config or CleaningConfig()
    return CleaningRuleRegistry(
        [
            NormalizeTextRule(),
            NormalizeExcelRowsRule(),
            PdfPageNumberRule(enabled=resolved.pdf.remove_page_numbers),
            PdfRepeatedMarginRule(
                enabled=resolved.pdf.remove_repeated_margins,
                min_pages=resolved.pdf.repeated_margin_min_pages,
                min_occurrences=resolved.pdf.repeated_margin_min_occurrences,
                occurrence_ratio=resolved.pdf.repeated_margin_ratio,
                max_line_length=resolved.pdf.repeated_margin_max_line_length,
            ),
            PdfWrappedLineRule(
                enabled=resolved.pdf.merge_wrapped_lines,
                min_previous_length=resolved.pdf.wrapped_line_min_previous_length,
            ),
            TableOfContentsRule(
                enabled=resolved.toc.enabled,
                minimum_entries=resolved.toc.minimum_entries,
                minimum_entry_ratio=resolved.toc.minimum_entry_ratio,
            ),
            ExactDuplicateBlockRule(
                enabled=resolved.duplicate.enabled,
                require_same_section=resolved.duplicate.require_same_section,
                minimum_content_length=resolved.duplicate.minimum_content_length,
            ),
            HeadingHierarchyRule(
                enabled=resolved.heading.enabled,
                detect_in_content=resolved.heading.detect_headings_in_content,
            ),
            EmptyContentRule(),
            NoiseCharacterQualityRule(
                enabled=resolved.noise.enabled,
                replacement_character_ratio=resolved.noise.replacement_character_ratio,
                suspicious_character_ratio=resolved.noise.suspicious_character_ratio,
                repeated_character_run=resolved.noise.repeated_character_run,
            ),
            ProtectedTokenValidationRule(enabled=resolved.validation.enabled),
            ChangeRatioValidationRule(
                enabled=resolved.validation.enabled,
                maximum_change_ratio=resolved.validation.maximum_change_ratio,
            ),
        ]
    )
