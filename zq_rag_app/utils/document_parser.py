"""文档解析统一入口。

业务代码从本模块调用 ``parse_document``，具体格式由 parsers 包中的策略工厂
自动选择。保留这一层门面后，将来替换策略注册方式不会影响索引服务。
"""

from pathlib import Path

from .parsers import DocumentParser, ParsedBlock, default_parser_factory


def get_document_parser(file_path: str | Path) -> DocumentParser:
    """返回适用于当前文件的具体解析策略。"""

    return default_parser_factory.get_parser(file_path)


def parse_document(file_path: str | Path) -> list[ParsedBlock]:
    """解析本地文档并返回统一文本块列表。"""

    return default_parser_factory.parse(file_path)

