"""语言感知的递归文本切分器。"""

from collections.abc import Callable, Sequence
from functools import lru_cache
import re

import tiktoken


LengthFunction = Callable[[str], int]

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_PATTERN = re.compile(r"[A-Za-z]")

_STRUCTURE_SEPARATORS = ("\n\n", "\n")
_ZH_SEPARATORS = ("。", "！", "？", "；", "，", "、")
_EN_SEPARATORS = (". ", "! ", "? ", "; ", ", ", " ")


@lru_cache(maxsize=None)
def _get_encoding(encoding_name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


class TokenCounter:
    """使用 tiktoken 提供稳定且可复用的 token 计数。"""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = _get_encoding(encoding_name)

    def __call__(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))


def detect_language(text: str) -> str:
    """根据中英文字符占比返回 ``zh``、``en`` 或 ``mixed``。"""

    cjk_count = len(_CJK_PATTERN.findall(text))
    latin_count = len(_LATIN_PATTERN.findall(text))
    total = cjk_count + latin_count
    if total == 0:
        return "mixed"

    cjk_ratio = cjk_count / total
    if cjk_ratio >= 0.6:
        return "zh"
    if cjk_ratio <= 0.2:
        return "en"
    return "mixed"


def language_separators(language: str, source_format: str | None = None) -> tuple[str, ...]:
    """按文档格式和语言返回从强到弱的递归边界。"""

    normalized_format = (source_format or "").lower().lstrip(".")
    if normalized_format in {"xlsx", "xlsm", "xltx", "xltm", "xls"}:
        # 表格优先保留整行，只有单行超长时才继续按单元格切分。
        return ("\n", "\t", "")
    if language == "zh":
        return (*_STRUCTURE_SEPARATORS, *_ZH_SEPARATORS, " ", "")
    if language == "en":
        return (*_STRUCTURE_SEPARATORS, *_EN_SEPARATORS, "")
    return (
        *_STRUCTURE_SEPARATORS,
        *_ZH_SEPARATORS,
        *_EN_SEPARATORS,
        "",
    )


class LanguageAwareRecursiveSplitter:
    """优先沿结构和句子边界递归切分，最终才按字符安全兜底。"""

    def __init__(self, length_function: LengthFunction) -> None:
        self._length = length_function

    def split(
        self,
        text: str,
        max_size: int,
        *,
        language: str | None = None,
        source_format: str | None = None,
    ) -> list[str]:
        if max_size <= 0:
            raise ValueError("max_size 必须大于 0")

        normalized = text.strip()
        if not normalized:
            return []
        if self._length(normalized) <= max_size:
            return [normalized]

        resolved_language = language or detect_language(normalized)
        separators = language_separators(resolved_language, source_format)
        return [
            chunk.strip()
            for chunk in self._split_recursive(normalized, max_size, separators)
            if chunk.strip()
        ]

    def suffix(self, text: str, max_size: int) -> str:
        """返回不超过 token 预算且不破坏 Unicode 字符的文本后缀。"""

        if max_size <= 0 or not text:
            return ""
        if self._length(text) <= max_size:
            return text

        low, high = 0, len(text)
        while low < high:
            middle = (low + high) // 2
            if self._length(text[middle:]) <= max_size:
                high = middle
            else:
                low = middle + 1

        start = low
        while start < len(text) and self._length(text[start:]) > max_size:
            start += 1
        while start > 0 and self._length(text[start - 1 :]) <= max_size:
            start -= 1
        return text[start:]

    def boundary_suffix(
        self,
        text: str,
        max_size: int,
        *,
        language: str | None = None,
        source_format: str | None = None,
    ) -> str:
        """优先从自然边界开始选取不超过预算的连续尾部文本。

        边界按段落、行、句子、短句的优先级依次尝试。只有找不到任何合适
        边界时才退回普通 token 后缀，避免 Overlap 经常从词语中间开始。
        """

        if max_size <= 0 or not text:
            return ""
        if self._length(text) <= max_size:
            return text

        resolved_language = language or detect_language(text)
        for separator in language_separators(resolved_language, source_format):
            if not separator:
                continue

            positions: list[int] = []
            start = 0
            while True:
                position = text.find(separator, start)
                if position < 0:
                    break
                positions.append(position + len(separator))
                start = position + len(separator)

            # 从较早的边界开始检查，选出当前边界等级下最长的合法尾部。
            for position in positions:
                candidate = text[position:].lstrip()
                if candidate and self._length(candidate) <= max_size:
                    return candidate

        return self.suffix(text, max_size).lstrip()

    def prefix(self, text: str, max_size: int) -> str:
        """返回不超过 token 预算且不破坏 Unicode 字符的文本前缀。"""

        if max_size <= 0 or not text:
            return ""
        if self._length(text) <= max_size:
            return text

        low, high = 1, len(text)
        best = 1
        while low <= high:
            middle = (low + high) // 2
            if self._length(text[:middle]) <= max_size:
                best = middle
                low = middle + 1
            else:
                high = middle - 1

        while best > 0 and self._length(text[:best]) > max_size:
            best -= 1
        while best < len(text) and self._length(text[: best + 1]) <= max_size:
            best += 1
        return text[:best]

    def _split_recursive(
        self,
        text: str,
        max_size: int,
        separators: Sequence[str],
    ) -> list[str]:
        if self._length(text) <= max_size:
            return [text]
        if not separators or separators[0] == "":
            return self._hard_split(text, max_size)

        separator = separators[0]
        if separator not in text:
            return self._split_recursive(text, max_size, separators[1:])

        pieces = self._split_keep_separator(text, separator)
        chunks: list[str] = []
        buffer = ""

        for piece in pieces:
            if self._length(piece) > max_size:
                if buffer:
                    chunks.append(buffer)
                    buffer = ""
                chunks.extend(
                    self._split_recursive(piece, max_size, separators[1:])
                )
                continue

            candidate = f"{buffer}{piece}"
            if buffer and self._length(candidate) > max_size:
                chunks.append(buffer)
                buffer = piece
            else:
                buffer = candidate

        if buffer:
            chunks.append(buffer)
        return chunks

    def _hard_split(self, text: str, max_size: int) -> list[str]:
        """按字符边界二分定位 token 窗口，避免截断中文 UTF-8 字节。"""

        chunks: list[str] = []
        start = 0
        while start < len(text):
            if self._length(text[start : start + 1]) > max_size:
                raise ValueError(
                    "max_size 小于单个 Unicode 字符的 token 数，无法安全切分"
                )
            low, high = start + 1, len(text)
            best = start + 1
            while low <= high:
                middle = (low + high) // 2
                if self._length(text[start:middle]) <= max_size:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1

            chunks.append(text[start:best])
            start = best
        return chunks

    @staticmethod
    def _split_keep_separator(text: str, separator: str) -> list[str]:
        """沿分隔符切分，并把分隔符保留在前一段末尾。"""

        pieces: list[str] = []
        start = 0
        while True:
            position = text.find(separator, start)
            if position < 0:
                break
            end = position + len(separator)
            pieces.append(text[start:end])
            start = end
        if start < len(text):
            pieces.append(text[start:])
        return pieces
