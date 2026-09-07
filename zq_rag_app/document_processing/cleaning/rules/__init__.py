"""内置清洗规则。"""

from .common import EmptyContentRule, NormalizeTextRule
from .duplicate import ExactDuplicateBlockRule
from .excel import NormalizeExcelRowsRule
from .pdf import PdfPageNumberRule, PdfRepeatedMarginRule, PdfWrappedLineRule
from .quality import NoiseCharacterQualityRule
from .structure import HeadingHierarchyRule
from .toc import TableOfContentsRule

__all__ = [
    "EmptyContentRule",
    "ExactDuplicateBlockRule",
    "HeadingHierarchyRule",
    "NoiseCharacterQualityRule",
    "NormalizeExcelRowsRule",
    "NormalizeTextRule",
    "PdfPageNumberRule",
    "PdfRepeatedMarginRule",
    "PdfWrappedLineRule",
    "TableOfContentsRule",
]
