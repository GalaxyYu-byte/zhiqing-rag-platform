"""文档分块阶段的集中配置。"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ChunkingConfig:
    """结构感知递归分块配置。

    ``chunk_size`` 和 ``chunk_overlap`` 均以 tokenizer 计算出的 token 数为
    单位。标题上下文占用同一个 token 预算，确保真正送入 Embedding 模型的
    ``embedding_content`` 不超过 ``chunk_size``。
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 96
    merge_small_chunks: bool = True
    boundary_aware_overlap: bool = True
    encoding_name: str = "cl100k_base"
    include_heading_context: bool = True
    heading_separator: str = " > "
    heading_context_max_tokens: int = 96

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size 必须大于 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap 不能小于 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        if self.min_chunk_size < 0:
            raise ValueError("min_chunk_size 不能小于 0")
        if self.heading_context_max_tokens < 0:
            raise ValueError("heading_context_max_tokens 不能小于 0")
        if not self.encoding_name.strip():
            raise ValueError("encoding_name 不能为空")
