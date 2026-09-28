"""文档解析策略接口。"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from zhiqing_rag.document_processing.format_validation import (
    DocumentFormat,
    ValidatedDocument,
)

from .models import ParseResult


class DocumentParser(ABC):
    formats: ClassVar[frozenset[DocumentFormat]] = frozenset()

    @abstractmethod
    def parse(self, path: Path, validated: ValidatedDocument) -> ParseResult:
        """从已校验文件中提取结构化元素，不做清洗和分块。"""
