"""不修改正文的噪声字符质量检查。"""

import re
import unicodedata

from ..base import DocumentCleaningRule
from ...models import CleanedBlock, CleaningContext


_MOJIBAKE_MARKERS = ("锟斤拷", "Ã", "Â", "ï»¿")


class NoiseCharacterQualityRule(DocumentCleaningRule):
    name = "detect_noise_characters"
    priority = 800

    def __init__(
        self,
        *,
        enabled: bool = True,
        replacement_character_ratio: float = 0.01,
        suspicious_character_ratio: float = 0.02,
        repeated_character_run: int = 8,
    ) -> None:
        self.enabled = enabled
        self.replacement_character_ratio = replacement_character_ratio
        self.suspicious_character_ratio = suspicious_character_ratio
        self.repeated_character_run = repeated_character_run

    def supports(self, context: CleaningContext) -> bool:
        del context
        return self.enabled

    def apply(
        self,
        blocks: list[CleanedBlock],
        context: CleaningContext,
    ) -> list[CleanedBlock]:
        del context
        for block in blocks:
            text = block.clean_content
            if not text:
                continue
            length = max(len(text), 1)
            replacement_ratio = text.count("�") / length
            if replacement_ratio >= self.replacement_character_ratio:
                block.add_warning(
                    code="replacement_character",
                    message="文本包含较多 Unicode 替换字符，疑似编码或 OCR 异常",
                    rule_name=self.name,
                    details={"ratio": replacement_ratio},
                )

            suspicious_count = sum(
                unicodedata.category(character) in {"Co", "Cs", "Cn"}
                for character in text
            )
            suspicious_ratio = suspicious_count / length
            if suspicious_ratio >= self.suspicious_character_ratio:
                block.add_warning(
                    code="suspicious_character",
                    message="文本包含较多私用区或未定义字符",
                    rule_name=self.name,
                    details={"ratio": suspicious_ratio},
                )

            markers = [marker for marker in _MOJIBAKE_MARKERS if marker in text]
            if markers:
                block.add_warning(
                    code="possible_mojibake",
                    message="文本包含常见乱码特征",
                    rule_name=self.name,
                    details={"markers": markers},
                )

            repeat_pattern = re.compile(
                rf"([^\s\W])\1{{{max(self.repeated_character_run - 1, 1)},}}"
            )
            repeated = repeat_pattern.search(text)
            if repeated:
                block.add_warning(
                    code="repeated_character_run",
                    message="文本包含异常的连续重复字符",
                    rule_name=self.name,
                    details={"sample": repeated.group(0)[:30]},
                )

        return blocks
