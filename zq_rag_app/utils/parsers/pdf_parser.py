"""PDF 文档解析策略。"""

from pathlib import Path

from pypdf import PdfReader

from .base import DocumentParseError, DocumentParser, ParsedBlock
from .common import base_metadata, clean_text


class PdfDocumentParser(DocumentParser):
    """使用 pypdf 提取带文本层的 PDF。

    每一页生成一个 ``ParsedBlock``，以便检索结果能够准确回溯到原始页码。
    扫描件 PDF 只有图片而没有文本层，pypdf 无法做 OCR；这种文件会得到空
    结果，后续如果需要支持，应增加 PaddleOCR/Tesseract 作为 OCR 兜底策略。
    """

    supported_extensions = frozenset({".pdf"})

    def parse(self, file_path: str | Path) -> list[ParsedBlock]:
        path = self.validate_file(file_path)

        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise DocumentParseError(f"PDF 文件打开失败: {path}") from exc

        blocks: list[ParsedBlock] = []
        metadata = base_metadata(path, type(self).__name__)

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                content = clean_text(page.extract_text() or "")
            except Exception as exc:
                raise DocumentParseError(
                    f"PDF 第 {page_num} 页文本提取失败: {path}"
                ) from exc

            # 空白页和纯图片页不生成无效块，避免后续创建空向量。
            if not content:
                continue

            blocks.append(
                ParsedBlock(
                    content=content,
                    page_num=page_num,
                    metadata={**metadata, "page_num": page_num},
                )
            )

        return blocks

