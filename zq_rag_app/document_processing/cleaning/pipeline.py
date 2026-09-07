"""文档清洗流水线入口。"""

from collections.abc import Iterable

from ...utils.parsers import ParsedBlock
from ..models import CleanedBlock, CleanedDocument, CleaningContext, CleaningReport
from .base import DocumentCleaningRule
from .config import CleaningConfig
from .exceptions import CleaningRuleExecutionError, CleaningValidationError
from .registry import CleaningRuleRegistry, default_cleaning_registry


class CleaningPipeline:
    """根据文档上下文顺序执行适用的清洗规则。"""

    def __init__(
        self,
        rules: Iterable[DocumentCleaningRule] | None = None,
        config: CleaningConfig | None = None,
    ) -> None:
        self._config = config or CleaningConfig()
        self._registry = (
            CleaningRuleRegistry(rules)
            if rules is not None
            else default_cleaning_registry(self._config)
        )

    def clean(
        self,
        parsed_blocks: list[ParsedBlock],
        context: CleaningContext | None = None,
    ) -> CleanedDocument:
        resolved_context = context or CleaningContext.from_blocks(parsed_blocks)
        blocks = [CleanedBlock.from_parsed_block(block) for block in parsed_blocks]

        for rule in self._registry.ordered_rules():
            try:
                if rule.supports(resolved_context):
                    blocks = rule.apply(blocks, resolved_context)
            except Exception as exc:
                raise CleaningRuleExecutionError(rule.name, str(exc)) from exc

        report = CleaningReport.from_blocks(len(parsed_blocks), blocks)
        error_count = sum(
            warning.severity == "error"
            for block in blocks
            for warning in block.warnings
        )
        if self._config.validation.fail_on_error and error_count:
            raise CleaningValidationError(error_count)
        return CleanedDocument(
            blocks=blocks,
            context=resolved_context,
            report=report,
        )


def clean_parsed_blocks(
    parsed_blocks: list[ParsedBlock],
    context: CleaningContext | None = None,
    config: CleaningConfig | None = None,
) -> CleanedDocument:
    """使用默认清洗规则处理解析块。"""
    return CleaningPipeline(config=config).clean(parsed_blocks, context)
