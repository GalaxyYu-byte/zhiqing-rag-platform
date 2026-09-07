"""多个解析策略共享的文本处理函数。"""

from pathlib import Path

from .base import DocumentParseError


def clean_text(text: str) -> str:
    """统一换行并清理首尾空白和重复空行。

    这里只做保守清洗，不合并普通行，也不删除 Markdown 标记，避免破坏表格、
    列表以及代码块的原始结构。更激进的正文清洗应放在独立清洗步骤中完成。
    """

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    result: list[str] = []
    previous_line_is_blank = False

    for raw_line in normalized.split("\n"):
        line = raw_line.rstrip()
        is_blank = not line.strip()

        # 连续空行只保留一行，使输出更紧凑，同时仍保留段落边界。
        if is_blank and previous_line_is_blank:
            continue
        result.append("" if is_blank else line)
        previous_line_is_blank = is_blank

    return "\n".join(result).strip()


def read_text_with_fallback(
    path: Path,
    encodings: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030"),
) -> tuple[str, str]:
    """按常见编码顺序读取文本，并返回正文和实际采用的编码。

    中文 Windows 文本经常使用 GBK/GB18030，而现代文档通常使用 UTF-8。
    依次尝试可避免使用 ``errors='ignore'`` 静默丢失字符。
    """

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue

    attempted = ", ".join(encodings)
    raise DocumentParseError(
        f"无法识别文本文件编码: {path}；已尝试: {attempted}"
    )


def base_metadata(path: Path, parser_name: str) -> dict[str, str]:
    """生成每个解析块都应携带的基础溯源信息。"""

    return {
        "file_name": path.name,
        "file_extension": path.suffix.lower(),
        "parser": parser_name,
    }

