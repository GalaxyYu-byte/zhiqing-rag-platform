"""清洗规则的公共接口。"""

from abc import ABC, abstractmethod

from ..models import CleanedBlock, CleaningContext


class DocumentCleaningRule(ABC):
    """一次处理整篇文档的清洗规则。

    文档级接口既能实现单块文本规范化，也能实现依赖跨页统计的页眉页脚识别。
    """

    name: str
    priority: int = 100

    def supports(self, context: CleaningContext) -> bool:
        """判断当前规则是否适用于这份文档。"""

        return True

    @abstractmethod
    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        """执行规则并返回清洗后的块列表。"""


class ElementCleaningRule(DocumentCleaningRule):
    """逐块执行的清洗规则基类。"""

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        for block in blocks:
            self.apply_to_block(block, context)
        return blocks

    @abstractmethod
    def apply_to_block(
        self,
        block: CleanedBlock,
        context: CleaningContext,
    ) -> None:
        """原地处理一个清洗块。"""
