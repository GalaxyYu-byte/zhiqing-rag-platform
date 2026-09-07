"""清洗规则的集中配置。"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class PdfCleaningConfig:
    remove_page_numbers: bool = True
    remove_repeated_margins: bool = True
    repeated_margin_min_pages: int = 3
    repeated_margin_min_occurrences: int = 3
    repeated_margin_ratio: float = 0.6
    repeated_margin_max_line_length: int = 100
    merge_wrapped_lines: bool = True
    wrapped_line_min_previous_length: int = 15


@dataclass(slots=True)
class DuplicateCleaningConfig:
    enabled: bool = True
    require_same_section: bool = True
    minimum_content_length: int = 20


@dataclass(slots=True)
class TocCleaningConfig:
    enabled: bool = True
    minimum_entries: int = 3
    minimum_entry_ratio: float = 0.5


@dataclass(slots=True)
class NoiseCleaningConfig:
    enabled: bool = True
    replacement_character_ratio: float = 0.01
    suspicious_character_ratio: float = 0.02
    repeated_character_run: int = 8


@dataclass(slots=True)
class HeadingCleaningConfig:
    enabled: bool = True
    detect_headings_in_content: bool = True


@dataclass(slots=True)
class ValidationConfig:
    enabled: bool = True
    maximum_change_ratio: float = 0.3
    fail_on_error: bool = True


@dataclass(slots=True)
class CleaningConfig:
    """整条清洗流水线配置。"""

    pdf: PdfCleaningConfig = field(default_factory=PdfCleaningConfig)
    duplicate: DuplicateCleaningConfig = field(
        default_factory=DuplicateCleaningConfig
    )
    toc: TocCleaningConfig = field(default_factory=TocCleaningConfig)
    noise: NoiseCleaningConfig = field(default_factory=NoiseCleaningConfig)
    heading: HeadingCleaningConfig = field(default_factory=HeadingCleaningConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
