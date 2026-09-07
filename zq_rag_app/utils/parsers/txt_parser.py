"""普通文本文件解析策略。"""

from pathlib import Path

from .base import DocumentParser, ParsedBlock
from .common import base_metadata, clean_text, read_text_with_fallback


class TxtDocumentParser(DocumentParser):
    """解析 UTF-8、UTF-8 BOM 或常见中文编码的 TXT 文件。"""

    supported_extensions = frozenset({".txt"})

    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        path = self.validate_file(file_path)
        raw_text, encoding = read_text_with_fallback(path)
        content = clean_text(raw_text)

        # 空文本返回空列表，后续索引流程可据此标记“无可索引内容”。
        if not content:
            return []

        return [
            ParsedBlock(
                content=content,
                metadata={
                    **base_metadata(path, type(self).__name__),
                    "encoding": encoding,
                },
            )
        ]

