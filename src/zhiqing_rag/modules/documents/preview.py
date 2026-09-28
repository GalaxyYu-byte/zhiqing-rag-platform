"""复用格式解析器生成只读预览，不创建文档、任务或索引。"""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from zhiqing_rag.document_processing.format_validation import DocumentFormatError
from zhiqing_rag.document_processing.parsers.models import DocumentParseError
from zhiqing_rag.document_processing.parsers.registry import parse_document

from .schemas import UploadError


def preview_content(content: bytes, filename: str, max_file_size: int) -> dict:
    with TemporaryDirectory(prefix="zhiqing-preview-") as directory:
        path = Path(directory) / "document.tmp"
        path.write_bytes(content)
        try:
            parsed = parse_document(path, original_filename=filename, max_file_size=max_file_size)
        except (DocumentFormatError, DocumentParseError) as error:
            raise UploadError(422, error.code, "文件预览解析失败，请下载原文件查看") from error
    # 限制浏览器渲染量；提示用户下载原文件查看其余内容。
    elements = []
    remaining = 200_000
    truncated = False
    for element in parsed.elements:
        if len(elements) >= 300 or remaining <= 0:
            truncated = True
            break
        rows = [list(row[:50]) for row in element.table_rows[:200]]
        text = element.text[:remaining]
        clipped = len(text) < len(element.text) or len(element.table_rows) > 200
        clipped |= any(len(row) > 50 for row in element.table_rows)
        # 表格也受字符预算限制，避免大单元格绕过正文上限。
        for row in rows:
            for index, cell in enumerate(row):
                row[index] = cell[: min(remaining, 4000)]
                remaining -= len(row[index])
                clipped |= len(row[index]) < len(cell)
        remaining = max(0, remaining - len(text))
        elements.append(
            {
                "kind": element.kind,
                "text": text,
                "sheet_name": element.sheet_name,
                "cell_range": element.cell_range,
                "table_rows": rows,
            }
        )
        truncated |= clipped
    return {
        "elements": elements,
        "warnings": [asdict(warning) for warning in parsed.warnings[:20]],
        "truncated": truncated,
    }
