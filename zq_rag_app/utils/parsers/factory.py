"""文档解析策略的注册和选择工厂。"""

from collections.abc import Iterable
from pathlib import Path

from .base import DocumentParser, ParsedBlock, UnsupportedDocumentTypeError
from .docx_parser import DocxDocumentParser
from .excel_parser import ExcelDocumentParser
from .markdown_parser import MarkdownDocumentParser
from .pdf_parser import PdfDocumentParser
from .txt_parser import TxtDocumentParser


class DocumentParserFactory:
    """根据文件扩展名选择具体解析策略。

    工厂负责“选择策略”，具体策略负责“执行解析”。调用方始终使用统一的
    ``parse`` 方法，因此新增策略不会影响 API 层和索引服务。
    """

    # Iterable[DocumentParser] 是一个可迭代对象，允许传入列表、元组或生成器等。None 表示使用默认内置策略。

    def __init__(self, parsers: Iterable[DocumentParser] | None = None) -> None:
        self._parsers: dict[str, DocumentParser] = {}
        # 只有未传 parsers 时才注册内置策略。显式传入空列表表示调用方希望
        # 创建一个真正的空工厂，随后完全按自己的需要注册策略。
        default_parsers = (
            parsers
            if parsers is not None
            else (
                PdfDocumentParser(),
                DocxDocumentParser(),
                TxtDocumentParser(),
                MarkdownDocumentParser(),
                ExcelDocumentParser(),
            )
        )
        for parser in default_parsers:
            self.register(parser)

    def register(self, parser: DocumentParser, *, replace: bool = False) -> None:
        """注册一个策略。

        默认禁止两个策略声明相同扩展名，避免因为注册顺序导致行为不确定。
        测试或自定义部署明确需要覆盖时，可以传入 ``replace=True``。
        """

        if not parser.supported_extensions:
            raise ValueError(f"{type(parser).__name__} 没有声明支持的扩展名")

        for extension in parser.supported_extensions:
            normalized = extension.lower()
            if not normalized.startswith("."):
                raise ValueError(f"扩展名必须以点开头: {extension}")
            if normalized in self._parsers and not replace:
                raise ValueError(f"扩展名已经注册解析策略: {normalized}")
            self._parsers[normalized] = parser

    def get_parser(self, file_path: str | Path) -> DocumentParser:
        """返回与文件扩展名匹配的解析策略。"""

        path = Path(file_path)
        extension = path.suffix.lower()
        parser = self._parsers.get(extension)
        if parser is None:
            supported = ", ".join(sorted(self._parsers))
            raise UnsupportedDocumentTypeError(
                f"不支持的文档格式: {extension or '无扩展名'}；"
                f"当前支持: {supported}"
            )
        return parser

    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        """选择策略并解析文件，是业务层推荐使用的统一方法。"""

        return self.get_parser(file_path).parse(file_path)

# property 装饰器可以让调用方像访问属性一样访问 ``supported_extensions``，而不必显式调用方法。 用处是让调用方可以直接通过 ``default_parser_factory.supported_extensions`` 获取已注册的扩展名列表，而不需要调用方法。
    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """返回已注册扩展名，便于上传接口做白名单校验。"""

        return tuple(sorted(self._parsers))


# 默认工厂中的解析器都是无状态对象，可以安全复用，无需每次解析重复创建。
default_parser_factory = DocumentParserFactory()
