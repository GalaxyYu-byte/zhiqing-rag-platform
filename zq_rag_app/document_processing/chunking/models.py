"""分块阶段的输出模型。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextChunk:
    """一个可独立向量化和召回的文本块。"""

    chunk_index: int
    content: str
    embedding_content: str
    token_count: int
    source_block_index: int
    page_num: int | None = None
    section_title: str | None = None
    heading_level: int | None = None
    heading_path: list[str] = field(default_factory=list)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    language: str = "mixed"
    overlap_token_count: int = 0


@dataclass(slots=True, frozen=True)
class ChunkingReport:
    """一次文档分块的统计摘要。"""

    input_blocks: int
    indexable_blocks: int
    structural_blocks: int
    merged_structural_blocks: int
    output_chunks: int
    total_tokens: int


@dataclass(slots=True)
class ChunkedDocument:
    """结构感知分块的完整输出。"""

    chunks: list[TextChunk]
    report: ChunkingReport
