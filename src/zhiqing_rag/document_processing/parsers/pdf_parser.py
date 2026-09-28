"""简单 PDF 页直接提取文字，复杂版面及扫描页交给 MinerU Basic。"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream

from zhiqing_rag.document_processing.format_validation import (
    DocumentFormat,
    ValidatedDocument,
)

from .base import DocumentParser
from .models import DocumentParseError, ElementKind, ParsedElement, ParseResult, ParseWarning

_PAGE_ARTIFACTS = {"header", "footer", "page_number"}
_HEADINGS = {"doc_title", "paragraph_title"}
_LISTS = {"list", "index"}
_VISUALS = {"image", "chart"}
_LAYOUT_OPERATORS = {
    b"Do", b"BI", b"re", b"m", b"l", b"c", b"v", b"y", b"h",
    b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*", b"sh",
}
_ALIGNED_COLUMNS = re.compile(r"\S[ \t]{4,}\S")


class _TableRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[str, ...]] = []
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.in_table = False
        self.has_spans = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.in_table and tag in {"th", "td"} and self.current_row is not None:
            self.has_spans |= any(key in {"rowspan", "colspan"} for key, _ in attrs)
            self.current_cell = []
        elif tag == "br" and self.current_cell is not None:
            self.current_cell.append("\n")

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.current_cell is not None:
            assert self.current_row is not None
            self.current_row.append("".join(self.current_cell).strip())
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            if self.current_row:
                self.rows.append(tuple(self.current_row))
            self.current_row = None
        elif tag == "table":
            self.in_table = False


def _table_rows(content: str) -> tuple[tuple[str, ...], ...]:
    if "<table" in content.lower():
        parser = _TableRows()
        parser.feed(content)
        # 跨行/跨列单元格保留原始 HTML，避免展开后改变表意。
        return () if parser.has_spans else tuple(parser.rows)
    lines = [line.strip() for line in content.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return ()
    rows = [tuple(cell.strip() for cell in line.strip("|").split("|")) for line in lines]
    if rows[1] and all(set(cell) <= {"-", ":", " "} for cell in rows[1]):
        rows.pop(1)
    return tuple(rows) if rows and all(len(row) == len(rows[0]) for row in rows) else ()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _mineru_python(root: Path) -> Path:
    configured = os.environ.get("MINERU_PYTHON")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (
        root / ".venv-mineru" / "Scripts" / "python.exe",
        root / ".venv-mineru" / "bin" / "python",
    ):
        if candidate.is_file():
            return candidate
    raise DocumentParseError(
        "MINERU_NOT_CONFIGURED", "找不到 MinerU 运行环境，请设置 MINERU_PYTHON"
    )


def _run_mineru(path: Path) -> dict[str, Any]:
    root = _project_root()
    python = _mineru_python(root)
    if not python.is_file():
        raise DocumentParseError("MINERU_NOT_CONFIGURED", "MINERU_PYTHON 指向的文件不存在")
    worker = Path(__file__).resolve().parents[1] / "mineru_worker.py"
    runtime_home = Path(os.environ.get("MINERU_HOME", root / ".mineru")).resolve()
    timeout = int(os.environ.get("MINERU_TIMEOUT_SECONDS", "600"))
    if timeout <= 0:
        raise DocumentParseError("MINERU_NOT_CONFIGURED", "MINERU_TIMEOUT_SECONDS 必须大于 0")

    env = os.environ.copy()
    env["MINERU_HOME"] = str(runtime_home)
    env["MINERU_MODEL_SOURCE"] = "local"
    env.setdefault("MINERU_MODEL_SMALL_BACKEND", "onnx")
    with tempfile.TemporaryDirectory(prefix="zhiqing-mineru-") as directory:
        input_path = Path(directory) / "source.pdf"
        output_path = Path(directory) / "result.json"
        shutil.copyfile(path, input_path)
        try:
            completed = subprocess.run(
                [str(python), str(worker), str(input_path), str(output_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise DocumentParseError("MINERU_TIMEOUT", "MinerU Basic 解析超时") from exc
        except OSError as exc:
            raise DocumentParseError("MINERU_NOT_CONFIGURED", "无法启动 MinerU 解析进程") from exc
        if completed.returncode != 0:
            raise DocumentParseError(
                "MINERU_PARSE_FAILED", "MinerU Basic 解析失败"
            ) from RuntimeError(completed.stderr[-2000:])
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DocumentParseError(
                "MINERU_INVALID_RESULT", "MinerU 未返回有效结构化结果"
            ) from exc
    if not isinstance(result, dict):
        raise DocumentParseError("MINERU_INVALID_RESULT", "MinerU 结果格式无效")
    return result


def _caption(block: dict[str, Any]) -> str | None:
    captions = block.get("captions", [])
    if not isinstance(captions, list):
        return None
    parts = [
        item.get("content", "").strip()
        for item in captions
        if isinstance(item, dict) and isinstance(item.get("content"), str)
    ]
    return "\n".join(part for part in parts if part) or None


def _bbox(block: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = block.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return values  # type: ignore[return-value]


def _simple_page_text(page: Any, reader: PdfReader) -> str | None:
    """只放行有可靠文字层、无图形和明显分栏的页面。"""

    try:
        contents = page.get_contents()
        if contents is None or any(
            operator in _LAYOUT_OPERATORS
            for _, operator in ContentStream(contents, reader).operations
        ):
            return None
        text = (page.extract_text() or "").strip()
        if not text or "\ufffd" in text or any(
            ord(char) < 32 and char not in "\n\t\r\f" for char in text
        ):
            return None
        layout = page.extract_text(extraction_mode="layout") or ""
        if sum(bool(_ALIGNED_COLUMNS.search(line)) for line in layout.splitlines()) >= 2:
            return None
        return text
    except Exception:
        # 文字层本身无法可靠读取时，由 MinerU 重新解析该页。
        return None


def _extract_pages(path: Path) -> list[dict[str, Any]]:
    reader = PdfReader(path, strict=False)
    pages: list[dict[str, Any] | None] = []
    mineru_indexes: list[int] = []
    for index, page in enumerate(reader.pages):
        text = _simple_page_text(page, reader)
        if text is None:
            mineru_indexes.append(index)
            pages.append(None)
        else:
            pages.append(
                {"page_idx": index, "blocks": [{"type": "direct_text", "content": text}]}
            )

    if mineru_indexes:
        if len(mineru_indexes) == len(pages):
            result = _run_mineru(path)
        else:
            with tempfile.TemporaryDirectory(prefix="zhiqing-pdf-pages-") as directory:
                subset_path = Path(directory) / "complex-pages.pdf"
                writer = PdfWriter()
                for index in mineru_indexes:
                    writer.add_page(reader.pages[index])
                with subset_path.open("wb") as output:
                    writer.write(output)
                result = _run_mineru(subset_path)
        mineru_pages = result.get("pages")
        if not isinstance(mineru_pages, list) or len(mineru_pages) != len(mineru_indexes):
            raise DocumentParseError("MINERU_INVALID_RESULT", "MinerU 返回的页面数量不匹配")
        for index, mineru_page in zip(mineru_indexes, mineru_pages, strict=True):
            if not isinstance(mineru_page, dict):
                raise DocumentParseError("MINERU_INVALID_RESULT", "MinerU 页面内容无效")
            mineru_page = dict(mineru_page)
            mineru_page["page_idx"] = index
            pages[index] = mineru_page

    return [page for page in pages if page is not None]


class PdfParser(DocumentParser):
    formats = frozenset({DocumentFormat.PDF})

    def parse(self, path: Path, validated: ValidatedDocument) -> ParseResult:
        del validated
        pages = _extract_pages(path)

        elements: list[ParsedElement] = []
        warnings: list[ParseWarning] = []
        heading_stack: list[str] = []
        for page in pages:
            if not isinstance(page, dict) or not isinstance(page.get("page_idx"), int):
                raise DocumentParseError("MINERU_INVALID_RESULT", "MinerU 页面编号无效")
            page_number = page["page_idx"] + 1
            blocks = page.get("blocks")
            if page_number < 1 or not isinstance(blocks, list):
                raise DocumentParseError("MINERU_INVALID_RESULT", "MinerU 页面内容无效")
            meaningful_text = False
            image_without_text = False
            for position, block in enumerate(blocks, start=1):
                if not isinstance(block, dict):
                    continue
                source_type = str(block.get("type", ""))
                raw_text = block.get("content", "")
                text = raw_text.strip() if isinstance(raw_text, str) else ""
                caption = _caption(block)
                if source_type in _VISUALS and not text:
                    image_without_text = True
                    text = caption or ""
                if not text and source_type != "table":
                    continue

                level = None
                if source_type in _HEADINGS:
                    raw_level = block.get("level")
                    level = raw_level if isinstance(raw_level, int) and 1 <= raw_level <= 6 else 1
                    heading_stack = heading_stack[: level - 1]
                    heading_stack.append(text)
                    kind = ElementKind.HEADING
                elif source_type == "table":
                    kind = ElementKind.TABLE
                elif source_type in _LISTS:
                    kind = ElementKind.LIST
                elif source_type == "code":
                    kind = ElementKind.CODE
                else:
                    kind = ElementKind.PARAGRAPH

                if source_type not in _PAGE_ARTIFACTS:
                    meaningful_text |= bool(text)
                elements.append(
                    ParsedElement(
                        kind=kind,
                        text=text,
                        order=len(elements),
                        heading_path=tuple(heading_stack),
                        heading_level=level,
                        page_number=page_number,
                        table_rows=_table_rows(text) if kind == ElementKind.TABLE else (),
                        source_ref=f"page:{page_number}/block:{position}",
                        source_type=source_type,
                        bbox=_bbox(block),
                        caption=caption,
                    )
                )
            if image_without_text and not meaningful_text:
                warnings.append(
                    ParseWarning("OCR_NO_TEXT", "页面包含图片，但 MinerU 未识别出文字", page_number)
                )
            elif image_without_text:
                warnings.append(
                    ParseWarning("IMAGE_WITHOUT_TEXT", "页面中有无法提取文字的图片", page_number)
                )
            elif not meaningful_text:
                warnings.append(ParseWarning("EMPTY_PAGE", "页面没有可索引的正文", page_number))

        return ParseResult(tuple(elements), tuple(warnings))
