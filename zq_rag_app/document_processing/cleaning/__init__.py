"""可组合、可审计的规则化文档清洗流水线。"""

from .config import CleaningConfig
from .exceptions import (
    CleaningRuleExecutionError,
    CleaningValidationError,
    DocumentCleaningError,
)
from .pipeline import CleaningPipeline, clean_parsed_blocks
from .registry import CleaningRuleRegistry, default_cleaning_registry

__all__ = [
    "CleaningPipeline",
    "CleaningConfig",
    "CleaningRuleExecutionError",
    "CleaningRuleRegistry",
    "CleaningValidationError",
    "DocumentCleaningError",
    "clean_parsed_blocks",
    "default_cleaning_registry",
]
