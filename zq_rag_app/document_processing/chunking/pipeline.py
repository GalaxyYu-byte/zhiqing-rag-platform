"""结构感知、语言感知和 Overlap 分块流水线。"""

import re

from ..cleaning.rules.structure import detect_heading_level
from ..models import CleanedBlock, CleanedDocument
from .config import ChunkingConfig
from .models import ChunkedDocument, ChunkingReport, TextChunk
from .splitter import (
    LanguageAwareRecursiveSplitter,
    LengthFunction,
    TokenCounter,
    detect_language,
)


_NUMBERED_SUBHEADING_PATTERN = re.compile(
    r"^\d+(?:\.\d+)+(?:[.、)])?\s+\S+"
)
_DOCUMENT_METADATA_MARKER = "[文档元数据]"
_DOCUMENT_RULE_PATTERN = re.compile(r"^[=-]{10,}$")
_APPENDIX_HEADING_PATTERN = re.compile(r"^\[(?:附加|建议)[^]]*]$")
_METADATA_LINE_PATTERN = re.compile(r"^([^:：]+)[:：]\s*(.*)$")
_FAQ_QUESTION_PATTERN = re.compile(r"^Q\d+[：:]\s*\S+", re.IGNORECASE)
_CANONICAL_METADATA_KEYS = {
    "文档ID": "document_id",
    "文档标题": "document_title",
    "文档类型": "document_type",
    "版本": "version",
    "生效日期": "effective_date",
    "发布日期": "published_date",
    "发布部门": "department",
    "适用范围": "scope",
    "密级": "classification",
    "状态": "status",
    "标签": "tags",
}


class ChunkingPipeline:
    """在清洗后的自然块内部递归切分，不跨章节或页制造 Overlap。"""

    strategy_name = "structure_recursive_v2"

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        length_function: LengthFunction | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._length = length_function or TokenCounter(self.config.encoding_name)
        self._splitter = LanguageAwareRecursiveSplitter(self._length)

    def chunk(self, document: CleanedDocument) -> ChunkedDocument:
        chunks: list[TextChunk] = []
        indexable_blocks = 0
        structural_units: list[tuple[CleanedBlock, list[int]]] = []

        for source_block_index, block in enumerate(document.blocks):
            if block.excluded_from_embedding or not block.clean_content.strip():
                continue
            indexable_blocks += 1
            for structural_block in self._structural_blocks(block):
                structural_units.append((structural_block, [source_block_index]))

        merged_units = self._merge_small_blocks(structural_units)
        for structural_block, source_block_indexes in merged_units:
            chunks.extend(
                self._chunk_block(
                    structural_block,
                    source_block_index=source_block_indexes[0],
                    source_block_indexes=source_block_indexes,
                    first_chunk_index=len(chunks),
                    document_source_format=document.context.source_format,
                )
            )

        report = ChunkingReport(
            input_blocks=len(document.blocks),
            indexable_blocks=indexable_blocks,
            structural_blocks=len(structural_units),
            merged_structural_blocks=len(merged_units),
            output_chunks=len(chunks),
            total_tokens=sum(chunk.token_count for chunk in chunks),
        )
        return ChunkedDocument(chunks=chunks, report=report)

    @classmethod
    def _structural_blocks(cls, block: CleanedBlock) -> list[CleanedBlock]:
        """把清洗阶段识别出的行内标题转换成独立章节块。

        TXT 和 PDF 解析器通常按整篇或整页输出，标题层级由清洗规则记录在
        ``detected_headings`` 中。这里把标题之间的正文恢复成自然块，使递归
        切分和 Overlap 都不会跨越章节边界。
        """

        multi_document_blocks = cls._multi_document_blocks(block)
        if multi_document_blocks is not None:
            return multi_document_blocks

        detected = block.source_metadata.get("detected_headings")
        if not isinstance(detected, list) or not detected:
            return [block]

        lines = block.clean_content.split("\n")
        headings: list[tuple[int, str, int, list[str]]] = []
        for item in detected:
            if not isinstance(item, dict):
                continue
            line_index = item.get("line_index")
            title = item.get("text")
            level = item.get("level")
            heading_path = item.get("heading_path")
            if (
                not isinstance(line_index, int)
                or not 0 <= line_index < len(lines)
                or not isinstance(title, str)
                or not isinstance(level, int)
                or not isinstance(heading_path, list)
                or not all(isinstance(value, str) for value in heading_path)
            ):
                continue
            # 清洗层会把 ``1. 操作步骤`` 也记作候选标题。它适合层级审计，
            # 但分块时仍应留在所属章节正文中；``2.1 小节`` 才是结构边界。
            if level >= 7 and not _NUMBERED_SUBHEADING_PATTERN.match(title):
                continue
            headings.append((line_index, title, level, list(heading_path)))

        headings.sort(key=lambda value: value[0])
        if not headings:
            return [block]

        output: list[CleanedBlock] = []
        preamble = "\n".join(lines[: headings[0][0]]).strip()
        if preamble:
            output.append(
                ChunkingPipeline._copy_structural_block(
                    block,
                    content=preamble,
                    section_title=block.section_title,
                    heading_level=block.heading_level,
                    heading_path=block.heading_path,
                )
            )

        for index, (line_index, title, level, heading_path) in enumerate(headings):
            end_line = headings[index + 1][0] if index + 1 < len(headings) else len(lines)
            content = "\n".join(lines[line_index + 1 : end_line]).strip()
            if not content:
                continue
            output.append(
                ChunkingPipeline._copy_structural_block(
                    block,
                    content=content,
                    section_title=title,
                    heading_level=level,
                    heading_path=heading_path,
                )
            )
        return output

    @classmethod
    def _multi_document_blocks(
        cls,
        block: CleanedBlock,
    ) -> list[CleanedBlock] | None:
        """解析由多个 ``[文档元数据]`` 区段拼接而成的 TXT 语料。"""

        lines = block.clean_content.split("\n")
        markers = [
            index
            for index, line in enumerate(lines)
            if line.strip() == _DOCUMENT_METADATA_MARKER
        ]
        if not markers:
            return None

        output: list[CleanedBlock] = []
        global_preamble = cls._normalized_region(lines[: markers[0]])
        if global_preamble:
            output.append(
                cls._copy_structural_block(
                    block,
                    content=global_preamble,
                    section_title="语料说明",
                    heading_level=1,
                    heading_path=["语料说明"],
                )
            )

        appendix_start: int | None = None
        for marker_index, marker_line in enumerate(markers):
            region_end = (
                markers[marker_index + 1]
                if marker_index + 1 < len(markers)
                else len(lines)
            )
            for line_index in range(marker_line + 1, region_end):
                if _APPENDIX_HEADING_PATTERN.match(lines[line_index].strip()):
                    region_end = line_index
                    appendix_start = line_index
                    break

            divider = next(
                (
                    line_index
                    for line_index in range(marker_line + 1, region_end)
                    if lines[line_index].strip().startswith("----------")
                ),
                None,
            )
            if divider is None:
                continue

            metadata_lines = lines[marker_line + 1 : divider]
            document_metadata = cls._parse_document_metadata(metadata_lines)
            document_title = str(
                document_metadata.get("document_title")
                or document_metadata.get("document_id")
                or "未命名文档"
            )
            metadata_updates = dict(document_metadata)
            metadata_updates["document_metadata"] = {
                match.group(1).strip(): match.group(2).strip()
                for line in metadata_lines
                if (match := _METADATA_LINE_PATTERN.match(line.strip()))
            }

            metadata_content = cls._normalized_region(metadata_lines)
            if metadata_content:
                output.append(
                    cls._copy_structural_block(
                        block,
                        content=metadata_content,
                        section_title="文档元数据",
                        heading_level=1,
                        heading_path=[document_title, "文档元数据"],
                        metadata_updates=metadata_updates,
                    )
                )

            body_start = divider + 1
            section_headings: list[tuple[int, str, int]] = []
            for line_index in range(body_start, region_end):
                title = lines[line_index].strip()
                if document_metadata.get("document_type") == "FAQ":
                    if _FAQ_QUESTION_PATTERN.match(title):
                        section_headings.append((line_index, title, 2))
                    continue
                level = detect_heading_level(title)
                if level is None:
                    continue
                if level >= 7 and not _NUMBERED_SUBHEADING_PATTERN.match(title):
                    continue
                section_headings.append((line_index, title, level))

            first_heading = (
                section_headings[0][0] if section_headings else region_end
            )
            body_preamble = cls._normalized_region(lines[body_start:first_heading])
            if body_preamble:
                output.append(
                    cls._copy_structural_block(
                        block,
                        content=body_preamble,
                        section_title="正文说明",
                        heading_level=1,
                        heading_path=[document_title, "正文说明"],
                        metadata_updates=metadata_updates,
                    )
                )

            for section_index, (line_index, title, level) in enumerate(
                section_headings
            ):
                end_line = (
                    section_headings[section_index + 1][0]
                    if section_index + 1 < len(section_headings)
                    else region_end
                )
                content = cls._normalized_region(lines[line_index + 1 : end_line])
                if not content:
                    continue
                output.append(
                    cls._copy_structural_block(
                        block,
                        content=content,
                        section_title=title,
                        heading_level=level,
                        heading_path=[document_title, title],
                        metadata_updates=metadata_updates,
                    )
                )

        if appendix_start is not None:
            output.extend(cls._appendix_blocks(block, lines, appendix_start))
        return output

    @classmethod
    def _appendix_blocks(
        cls,
        block: CleanedBlock,
        lines: list[str],
        start: int,
    ) -> list[CleanedBlock]:
        headings = [
            index
            for index in range(start, len(lines))
            if _APPENDIX_HEADING_PATTERN.match(lines[index].strip())
        ]
        output: list[CleanedBlock] = []
        for heading_index, line_index in enumerate(headings):
            end_line = (
                headings[heading_index + 1]
                if heading_index + 1 < len(headings)
                else len(lines)
            )
            title = lines[line_index].strip().strip("[]")
            content = cls._normalized_region(lines[line_index + 1 : end_line])
            if content:
                output.append(
                    cls._copy_structural_block(
                        block,
                        content=content,
                        section_title=title,
                        heading_level=1,
                        heading_path=[title],
                    )
                )
        return output

    @staticmethod
    def _normalized_region(lines: list[str]) -> str:
        return "\n".join(
            line.rstrip()
            for line in lines
            if not _DOCUMENT_RULE_PATTERN.fullmatch(line.strip())
            and line.strip() != "文档结束"
        ).strip()

    @staticmethod
    def _parse_document_metadata(lines: list[str]) -> dict[str, object]:
        metadata: dict[str, object] = {}
        for line in lines:
            match = _METADATA_LINE_PATTERN.match(line.strip())
            if not match:
                continue
            source_key, value = match.group(1).strip(), match.group(2).strip()
            key = _CANONICAL_METADATA_KEYS.get(source_key)
            if key is None:
                continue
            metadata[key] = (
                [item.strip() for item in value.split(",") if item.strip()]
                if key == "tags"
                else value
            )
        return metadata

    @staticmethod
    def _copy_structural_block(
        source: CleanedBlock,
        *,
        content: str,
        section_title: str | None,
        heading_level: int | None,
        heading_path: list[str],
        metadata_updates: dict[str, object] | None = None,
    ) -> CleanedBlock:
        metadata = dict(source.source_metadata)
        metadata.pop("detected_headings", None)
        metadata.update(
            {
                "section_title": section_title,
                "heading_level": heading_level,
                "heading_path": list(heading_path),
            }
        )
        metadata.update(metadata_updates or {})
        return CleanedBlock(
            raw_content=content,
            clean_content=content,
            page_num=source.page_num,
            section_title=section_title,
            heading_level=heading_level,
            heading_path=list(heading_path),
            source_metadata=metadata,
        )

    def _merge_small_blocks(
        self,
        units: list[tuple[CleanedBlock, list[int]]],
    ) -> list[tuple[CleanedBlock, list[int]]]:
        """合并同一父章节下可容纳在一个 Chunk 中的相邻小节。"""

        if not self.config.merge_small_chunks or self.config.min_chunk_size == 0:
            return units

        minimum = min(self.config.min_chunk_size, self.config.chunk_size)
        output: list[tuple[CleanedBlock, list[int]]] = []
        for block, source_indexes in units:
            if not output:
                output.append((block, source_indexes))
                continue

            previous, previous_indexes = output[-1]
            has_small_neighbor = (
                self._standalone_size(previous) < minimum
                or self._standalone_size(block) < minimum
            )
            if not has_small_neighbor or not self._are_mergeable_siblings(
                previous, block
            ):
                output.append((block, source_indexes))
                continue

            merged = self._merge_siblings(previous, block)
            if self._standalone_size(merged) > self.config.chunk_size:
                output.append((block, source_indexes))
                continue

            output[-1] = (
                merged,
                list(dict.fromkeys([*previous_indexes, *source_indexes])),
            )
        return output

    def _standalone_size(self, block: CleanedBlock) -> int:
        embedding_content = f"{self._heading_prefix(block)}{block.clean_content}"
        return self._length(embedding_content.strip())

    @staticmethod
    def _section_parent(block: CleanedBlock) -> list[str]:
        merged_sections = block.source_metadata.get("merged_sections")
        if isinstance(merged_sections, list) and merged_sections:
            return list(block.heading_path)
        if block.heading_path:
            return list(block.heading_path[:-1])
        return []

    @classmethod
    def _are_mergeable_siblings(
        cls,
        left: CleanedBlock,
        right: CleanedBlock,
    ) -> bool:
        left_parent = cls._section_parent(left)
        right_parent = cls._section_parent(right)
        if not left_parent or left_parent != right_parent:
            return False
        if left.page_num != right.page_num:
            return False
        if (
            left.source_metadata.get("document_type") == "FAQ"
            or right.source_metadata.get("document_type") == "FAQ"
        ):
            return False

        left_file = left.source_metadata.get("file_name")
        right_file = right.source_metadata.get("file_name")
        return not (left_file and right_file and left_file != right_file)

    @classmethod
    def _merge_siblings(
        cls,
        left: CleanedBlock,
        right: CleanedBlock,
    ) -> CleanedBlock:
        parent = cls._section_parent(left)
        entries = [*cls._section_entries(left), *cls._section_entries(right)]
        content = (
            f"{cls._content_with_section_heading(left)}\n\n"
            f"{cls._content_with_section_heading(right)}"
        ).strip()
        titles = [
            str(entry["section_title"])
            for entry in entries
            if entry.get("section_title")
        ]
        metadata = {
            **left.source_metadata,
            "section_title": " / ".join(titles),
            "heading_level": None,
            "heading_path": list(parent),
            "merged_sections": entries,
        }
        return CleanedBlock(
            raw_content=content,
            clean_content=content,
            page_num=left.page_num,
            section_title=" / ".join(titles),
            heading_level=None,
            heading_path=list(parent),
            source_metadata=metadata,
        )

    @staticmethod
    def _section_entries(block: CleanedBlock) -> list[dict[str, object]]:
        existing = block.source_metadata.get("merged_sections")
        if isinstance(existing, list) and existing:
            return [dict(entry) for entry in existing if isinstance(entry, dict)]
        return [
            {
                "section_title": block.section_title,
                "heading_level": block.heading_level,
                "heading_path": list(block.heading_path),
            }
        ]

    @staticmethod
    def _content_with_section_heading(block: CleanedBlock) -> str:
        if block.source_metadata.get("merged_sections"):
            return block.clean_content.strip()
        if block.section_title:
            return f"{block.section_title}\n{block.clean_content}".strip()
        return block.clean_content.strip()

    def _chunk_block(
        self,
        block: CleanedBlock,
        *,
        source_block_index: int,
        source_block_indexes: list[int],
        first_chunk_index: int,
        document_source_format: str,
    ) -> list[TextChunk]:
        heading_prefix = self._heading_prefix(block)
        prefix_tokens = self._length(heading_prefix)
        content_budget = self.config.chunk_size - prefix_tokens
        if content_budget <= 0:
            # _heading_prefix 会主动受限；这里只防御自定义 tokenizer 的异常实现。
            raise ValueError("标题上下文占满了整个 chunk token 预算")

        text = block.clean_content.strip()
        language = detect_language(text)
        source_format = str(
            block.source_metadata.get("file_extension") or document_source_format
        )

        if self._length(text) <= content_budget:
            base_chunks = [text]
        else:
            payload_budget = content_budget - self.config.chunk_overlap
            if payload_budget <= 0:
                raise ValueError("标题上下文导致正文和 overlap 没有可用 token 预算")
            base_chunks = self._splitter.split(
                text,
                payload_budget,
                language=language,
                source_format=source_format,
            )

        output: list[TextChunk] = []
        previous_base = ""
        for local_index, base_content in enumerate(base_chunks):
            overlap_prefix = ""
            if local_index > 0 and self.config.chunk_overlap:
                overlap_separator = "\n"
                separator_tokens = self._length(overlap_separator)
                overlap_budget = self.config.chunk_overlap - separator_tokens
                if overlap_budget > 0:
                    if self.config.boundary_aware_overlap:
                        overlap_context = self._splitter.boundary_suffix(
                            previous_base,
                            overlap_budget,
                            language=language,
                            source_format=source_format,
                        )
                    else:
                        overlap_context = self._splitter.suffix(
                            previous_base,
                            overlap_budget,
                        )
                    if overlap_context:
                        overlap_prefix = f"{overlap_context}{overlap_separator}"

            content = f"{overlap_prefix}{base_content}".strip()
            embedding_content = f"{heading_prefix}{content}".strip()
            token_count = self._length(embedding_content)
            if token_count > self.config.chunk_size:
                raise ValueError(
                    "分块结果超过 chunk_size："
                    f"{token_count} > {self.config.chunk_size}"
                )

            metadata = {
                **block.source_metadata,
                "chunk_strategy": self.strategy_name,
                "source_block_index": source_block_index,
                "source_block_indexes": list(source_block_indexes),
                "language": language,
                "overlap_token_count": self._length(overlap_prefix),
            }
            output.append(
                TextChunk(
                    chunk_index=first_chunk_index + local_index,
                    content=content,
                    embedding_content=embedding_content,
                    token_count=token_count,
                    source_block_index=source_block_index,
                    page_num=block.page_num,
                    section_title=block.section_title,
                    heading_level=block.heading_level,
                    heading_path=list(block.heading_path),
                    source_metadata=metadata,
                    language=language,
                    overlap_token_count=self._length(overlap_prefix),
                )
            )
            previous_base = base_content
        return output

    def _heading_prefix(self, block: CleanedBlock) -> str:
        if not self.config.include_heading_context:
            return ""

        heading_path = list(block.heading_path)
        if not heading_path and block.section_title:
            heading_path = [block.section_title]
        if not heading_path:
            return ""

        heading = self.config.heading_separator.join(heading_path)
        prefix = f"章节：{heading}\n\n"
        # 必须至少给正文和 overlap 留出 1 个 token；小 chunk 配置同样可用。
        maximum = min(
            self.config.heading_context_max_tokens,
            self.config.chunk_size - self.config.chunk_overlap - 1,
        )
        if maximum <= 0:
            return ""
        if self._length(prefix) <= maximum:
            return prefix

        label = "章节："
        available = maximum - self._length(label)
        if available <= 0:
            return ""
        shortened = self._splitter.suffix(heading, available).lstrip()
        candidate = f"{label}{shortened}\n\n" if shortened else ""
        while shortened and self._length(candidate) > maximum:
            shortened = shortened[1:].lstrip()
            candidate = f"{label}{shortened}\n\n" if shortened else ""
        return candidate


def chunk_cleaned_document(
    document: CleanedDocument,
    config: ChunkingConfig | None = None,
) -> ChunkedDocument:
    """使用默认 token 计数器对清洗后的文档进行分块。"""

    return ChunkingPipeline(config=config).chunk(document)
