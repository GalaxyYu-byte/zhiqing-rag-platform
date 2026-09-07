"""Markdown 文档解析策略。"""

from pathlib import Path

from markdown_it import MarkdownIt

from .base import DocumentParser, ParsedBlock
from .common import base_metadata, clean_text, read_text_with_fallback


class MarkdownDocumentParser(DocumentParser):
    """按照 Markdown 标题边界拆分文档。

    使用 markdown-it 的语法树识别 ATX 标题（``# title``）和 Setext 标题，
    比单纯正则更可靠。正文仍保留原始 Markdown 标记，使列表、表格和代码块
    等结构信息不会在解析阶段丢失。
    """

    supported_extensions = frozenset({".md", ".markdown"})

    def __init__(self) -> None:
        self._markdown = MarkdownIt("commonmark")

    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        path = self.validate_file(file_path)
        raw_text, encoding = read_text_with_fallback(path)
        lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        # 标题 token 的 map 是 [开始行, 结束行)，可以同时正确处理 # 标题和
        # 两行形式的 Setext 标题。inline token 中保存去掉语法符号后的标题文本。
        tokens = self._markdown.parse(raw_text)
        headings: list[tuple[int, int, str, int]] = []
        for index, token in enumerate(tokens):
            if token.type != "heading_open" or token.map is None:
                continue
            if index + 1 >= len(tokens) or tokens[index + 1].type != "inline":
                continue

            title = clean_text(tokens[index + 1].content)
            level = int(token.tag[1:]) if token.tag.startswith("h") else 1
            headings.append((token.map[0], token.map[1], title, level))

        metadata = {
            **base_metadata(path, type(self).__name__),
            "encoding": encoding,
        }
        blocks: list[ParsedBlock] = []
        cursor = 0
        current_section: str | None = None
        current_heading_level: int | None = None
        current_heading_path: list[str] = []
        heading_stack: dict[int, str] = {}

        def append_section(end_line: int) -> None:
            content = clean_text("\n".join(lines[cursor:end_line]))
            if content:
                blocks.append(
                    ParsedBlock(
                        content=content,
                        section_title=current_section,
                        heading_level=current_heading_level,
                        heading_path=list(current_heading_path),
                        metadata={
                            **metadata,
                            "section_title": current_section,
                            "heading_level": current_heading_level,
                            "heading_path": list(current_heading_path),
                        },
                    )
                )

        for start_line, end_line, title, level in headings:
            append_section(start_line)
            current_section = title or None
            current_heading_level = level
            for existing_level in [value for value in heading_stack if value >= level]:
                del heading_stack[existing_level]
            if title:
                heading_stack[level] = title
            current_heading_path = [
                heading_stack[key] for key in sorted(heading_stack)
            ]
            cursor = end_line

        append_section(len(lines))
        return blocks
