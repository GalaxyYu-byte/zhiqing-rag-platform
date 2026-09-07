"""文档清洗阶段使用的中间模型。"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.parsers import ParsedBlock


@dataclass(slots=True, frozen=True)
class CleaningChange:
    """一条可审计的清洗变更。"""

    rule_name: str
    change_type: str
    before: str
    after: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class CleaningWarning:
    """清洗过程产生的质量或安全告警。"""

    code: str
    message: str
    severity: str = "warning"
    rule_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CleaningContext:
    """一次文档清洗任务的上下文。

    ``source_format`` 描述原始文件格式，``parser_name`` 描述实际解析器；两者
    分开保存，使格式规则和解析器补丁可以独立选择。
    """

    source_format: str
    parser_name: str
    document_category: str | None = None
    is_ocr: bool = False
    has_layout: bool = False
    has_tables: bool = False
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_format = self.source_format.lower().lstrip(".")
        self.parser_name = self.parser_name.strip()

    @classmethod
    def from_blocks(cls, blocks: list[ParsedBlock]) -> "CleaningContext":
        """根据解析块元数据构造默认上下文。"""

        if not blocks:
            return cls(source_format="unknown", parser_name="unknown")

        metadata = blocks[0].metadata
        extension = str(metadata.get("file_extension", "unknown"))
        parser_name = str(metadata.get("parser", "unknown"))
        return cls(
            source_format=extension,
            parser_name=parser_name,
            is_ocr=bool(metadata.get("is_ocr", False)),
            has_layout=bool(metadata.get("has_layout", False)),
            has_tables=bool(metadata.get("has_tables", False)),
        )


@dataclass(slots=True)
class CleanedBlock:
    """保留原始内容和清洗结果的文档自然块。"""

    raw_content: str
    clean_content: str
    page_num: int | None = None
    section_title: str | None = None
    heading_level: int | None = None
    heading_path: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    excluded_from_embedding: bool = False
    exclude_reason: str | None = None
    clean_operations: list[str] = field(default_factory=list)
    clean_changes: list[CleaningChange] = field(default_factory=list)
    warnings: list[CleaningWarning] = field(default_factory=list)

    @classmethod
    def from_parsed_block(cls, block: ParsedBlock) -> "CleanedBlock":
        """复制解析结果，避免清洗阶段修改 ``ParsedBlock``。"""

        heading_path = list(block.heading_path)
        if not heading_path and block.section_title:
            heading_path = [block.section_title]
        return cls(
            raw_content=block.content,
            clean_content=block.content,
            page_num=block.page_num,
            section_title=block.section_title,
            heading_level=block.heading_level,
            heading_path=heading_path,
            source_metadata=dict(block.metadata),
        )

    def record_operation(self, operation: str) -> None:
        """按执行顺序记录一次实际产生变化的清洗操作。"""

        if operation not in self.clean_operations:
            self.clean_operations.append(operation)

    def record_change(
        self,
        *,
        rule_name: str,
        change_type: str,
        before: str,
        after: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """记录具体变更，并同步维护操作名称列表。"""

        self.clean_changes.append(
            CleaningChange(
                rule_name=rule_name,
                change_type=change_type,
                before=before,
                after=after,
                details=dict(details or {}),
            )
        )
        self.record_operation(rule_name)

    def add_warning(
        self,
        *,
        code: str,
        message: str,
        severity: str = "warning",
        rule_name: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """追加告警；相同告警码和规则不会重复写入。"""

        if any(
            warning.code == code and warning.rule_name == rule_name
            for warning in self.warnings
        ):
            return
        self.warnings.append(
            CleaningWarning(
                code=code,
                message=message,
                severity=severity,
                rule_name=rule_name,
                details=dict(details or {}),
            )
        )

    def exclude(
        self,
        reason: str,
        operation: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        """将块标记为不参与后续向量化，但仍保留以供审计。"""

        before = str(self.excluded_from_embedding)
        self.excluded_from_embedding = True
        self.exclude_reason = reason
        self.record_change(
            rule_name=operation,
            change_type="exclude_block",
            before=before,
            after="True",
            details={"reason": reason, **dict(details or {})},
        )


@dataclass(slots=True)
class CleaningReport:
    """一次清洗任务的统计摘要。"""

    input_blocks: int
    output_blocks: int
    changed_blocks: int
    excluded_blocks: int
    operation_counts: dict[str, int]
    warning_count: int
    warning_counts: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_blocks(
        cls,
        input_blocks: int,
        blocks: list[CleanedBlock],
    ) -> "CleaningReport":
        operations = Counter(
            operation
            for block in blocks
            for operation in block.clean_operations
        )
        warnings = Counter(
            warning.code for block in blocks for warning in block.warnings
        )
        return cls(
            input_blocks=input_blocks,
            output_blocks=len(blocks),
            changed_blocks=sum(
                block.clean_content != block.raw_content
                or bool(block.clean_operations)
                for block in blocks
            ),
            excluded_blocks=sum(block.excluded_from_embedding for block in blocks),
            operation_counts=dict(operations),
            warning_count=sum(len(block.warnings) for block in blocks),
            warning_counts=dict(warnings),
        )


@dataclass(slots=True)
class CleanedDocument:
    """清洗流水线的完整输出。"""

    blocks: list[CleanedBlock]
    context: CleaningContext
    report: CleaningReport

    @property
    def indexable_blocks(self) -> list[CleanedBlock]:
        """返回有内容且未被排除的块。"""

        return [
            block
            for block in self.blocks
            if not block.excluded_from_embedding and block.clean_content.strip()
        ]

    @property
    def source_name(self) -> str | None:
        """返回解析器记录的原始文件名。"""

        if not self.blocks:
            return None
        value = self.blocks[0].source_metadata.get("file_name")
        return Path(str(value)).name if value else None
