"""清洗前后事实性标记的安全校验。"""

import re

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


_PROTECTED_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?(?:%|％|年|月|日|号|元|万元|亿元|米|千米|公里|"
    r"立方米|m|km|kg|t)?",
    re.IGNORECASE,
)
_ALLOWED_REMOVAL_RULES = {
    "remove_pdf_page_number",
    "remove_pdf_repeated_margin",
}


def _protected_tokens(text: str) -> list[str]:
    return _PROTECTED_TOKEN_PATTERN.findall(text)


class ProtectedTokenValidationRule(DocumentCleaningRule):
    """检查每一次正文变更是否意外修改数字、日期、金额或单位。"""

    name = "validate_protected_tokens"
    priority = 900

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def supports(self, context: CleaningContext) -> bool:
        del context
        return self.enabled

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        ignored_types = {"exclude_block", "update_heading_metadata", "normalize_metadata"}
        for block in blocks:
            for change in block.clean_changes:
                if (
                    change.rule_name in _ALLOWED_REMOVAL_RULES
                    or change.change_type in ignored_types
                ):
                    continue
                before_tokens = _protected_tokens(change.before)
                after_tokens = _protected_tokens(change.after)
                if before_tokens != after_tokens:
                    block.add_warning(
                        code="protected_token_changed",
                        message="清洗操作修改了数字、日期、金额或单位",
                        severity="error",
                        rule_name=self.name,
                        details={
                            "source_rule": change.rule_name,
                            "before_tokens": before_tokens,
                            "after_tokens": after_tokens,
                        },
                    )
        return blocks


class ChangeRatioValidationRule(DocumentCleaningRule):
    """标记清洗前后正文长度变化过大的块。"""

    name = "validate_change_ratio"
    priority = 910

    def __init__(
        self,
        *,
        enabled: bool = True,
        maximum_change_ratio: float = 0.3,
    ) -> None:
        self.enabled = enabled
        self.maximum_change_ratio = maximum_change_ratio

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
            raw_length = len(block.raw_content)
            clean_length = len(block.clean_content)
            ratio = abs(raw_length - clean_length) / max(raw_length, 1)
            if ratio > self.maximum_change_ratio:
                block.add_warning(
                    code="large_change_ratio",
                    message="清洗前后文本长度变化比例过大",
                    rule_name=self.name,
                    details={
                        "ratio": ratio,
                        "raw_length": raw_length,
                        "clean_length": clean_length,
                    },
                )
        return blocks
