"""Excel 解析文本的保守清洗规则。"""

from ..base import ElementCleaningRule
from ...models import CleanedBlock, CleaningContext


class NormalizeExcelRowsRule(ElementCleaningRule):
    """移除表格行尾空单元格，但保留中间空列。"""

    name = "normalize_excel_rows"
    priority = 20

    def supports(self, context: CleaningContext) -> bool:
        return context.source_format in {"xls", "xlsx", "xlsm", "xltx", "xltm"}

    def apply_to_block(
        self,
        block: CleanedBlock,
        context: CleaningContext,
    ) -> None:
        del context
        before = block.clean_content
        lines = [line.rstrip("\t") for line in block.clean_content.split("\n")]
        normalized = "\n".join(line for line in lines if line.strip("\t "))
        if normalized != block.clean_content:
            block.clean_content = normalized
            block.record_change(
                rule_name=self.name,
                change_type="normalize_table_rows",
                before=before,
                after=normalized,
            )
