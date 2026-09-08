"""结构感知递归分块测试。"""

import json
from pathlib import Path

import pytest

from zq_rag_app.document_processing.cleaning import clean_parsed_blocks
from zq_rag_app.document_processing.chunking import (
    ChunkingConfig,
    ChunkingPipeline,
    LanguageAwareRecursiveSplitter,
    TokenCounter,
    chunk_cleaned_document,
    detect_language,
)
from zq_rag_app.utils.document_parser import parse_document
from zq_rag_app.document_processing.models import (
    CleanedBlock,
    CleanedDocument,
    CleaningContext,
    CleaningReport,
)


def _cleaned_document(blocks: list[CleanedBlock]) -> CleanedDocument:
    return CleanedDocument(
        blocks=blocks,
        context=CleaningContext(source_format="md", parser_name="test"),
        report=CleaningReport.from_blocks(len(blocks), blocks),
    )


def test_short_block_preserves_structure_and_adds_heading_context() -> None:
    block = CleanedBlock(
        raw_content="安装服务。",
        clean_content="安装服务。",
        page_num=3,
        section_title="部署",
        heading_level=2,
        heading_path=["运维手册", "部署"],
        source_metadata={"file_extension": ".md", "file_name": "manual.md"},
    )

    result = chunk_cleaned_document(_cleaned_document([block]))

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.content == "安装服务。"
    assert chunk.embedding_content == "章节：运维手册 > 部署\n\n安装服务。"
    assert chunk.page_num == 3
    assert chunk.heading_path == ["运维手册", "部署"]
    assert chunk.source_metadata["chunk_strategy"] == "structure_recursive_v2"


def test_chinese_recursive_split_respects_token_limit_and_overlap() -> None:
    sentence = "这是用于验证中文递归分块边界的一句话。"
    text = sentence * 30
    config = ChunkingConfig(
        chunk_size=55,
        chunk_overlap=8,
        include_heading_context=False,
    )
    block = CleanedBlock(raw_content=text, clean_content=text)

    result = chunk_cleaned_document(_cleaned_document([block]), config)

    assert len(result.chunks) > 1
    assert all(chunk.token_count <= config.chunk_size for chunk in result.chunks)
    assert all(chunk.language == "zh" for chunk in result.chunks)
    assert all(
        chunk.overlap_token_count <= config.chunk_overlap
        for chunk in result.chunks[1:]
    )
    assert all(chunk.overlap_token_count > 0 for chunk in result.chunks[1:])


def test_overlap_is_not_carried_across_natural_blocks() -> None:
    config = ChunkingConfig(
        chunk_size=18,
        chunk_overlap=4,
        include_heading_context=False,
    )
    blocks = [
        CleanedBlock(
            raw_content="第一页内容。" * 20,
            clean_content="第一页内容。" * 20,
            page_num=1,
        ),
        CleanedBlock(
            raw_content="SECOND_PAGE_ONLY",
            clean_content="SECOND_PAGE_ONLY",
            page_num=2,
        ),
    ]

    result = chunk_cleaned_document(_cleaned_document(blocks), config)
    page_two = [chunk for chunk in result.chunks if chunk.page_num == 2]

    assert page_two[0].content == "SECOND_PAGE_ONLY"
    assert page_two[0].overlap_token_count == 0
    assert "第一页" not in page_two[0].content


def test_excluded_blocks_are_not_chunked_and_indexes_are_contiguous() -> None:
    blocks = [
        CleanedBlock(raw_content="保留一", clean_content="保留一"),
        CleanedBlock(
            raw_content="目录",
            clean_content="目录",
            excluded_from_embedding=True,
            exclude_reason="table_of_contents",
        ),
        CleanedBlock(raw_content="保留二", clean_content="保留二"),
    ]

    result = chunk_cleaned_document(_cleaned_document(blocks))

    assert [chunk.content for chunk in result.chunks] == ["保留一", "保留二"]
    assert [chunk.chunk_index for chunk in result.chunks] == [0, 1]
    assert [chunk.source_block_index for chunk in result.chunks] == [0, 2]
    assert result.report.indexable_blocks == 2


def test_excel_prefers_row_boundaries() -> None:
    splitter = LanguageAwareRecursiveSplitter(len)
    text = "名称\t价格\n标准版\t99\n企业版\t199"

    chunks = splitter.split(text, 12, source_format=".xlsx")

    assert chunks == ["名称\t价格", "标准版\t99", "企业版\t199"]


def test_english_language_detection_and_sentence_split() -> None:
    splitter = LanguageAwareRecursiveSplitter(len)
    text = "First sentence. Second sentence. Third sentence."

    chunks = splitter.split(text, 20)

    assert detect_language(text) == "en"
    assert chunks == ["First sentence.", "Second sentence.", "Third sentence."]


def test_boundary_aware_suffix_prefers_complete_trailing_line() -> None:
    splitter = LanguageAwareRecursiveSplitter(len)
    text = "第一句内容很长。\n完整尾句。"

    suffix = splitter.boundary_suffix(text, len("完整尾句。"), language="zh")

    assert suffix == "完整尾句。"


def test_unicode_hard_split_does_not_corrupt_chinese() -> None:
    counter = TokenCounter()
    splitter = LanguageAwareRecursiveSplitter(counter)
    text = "水利工程知识库检索"

    chunks = splitter.split(text, 2)

    assert "".join(chunks) == text
    assert "�" not in "".join(chunks)
    assert all(counter(chunk) <= 2 for chunk in chunks)


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunking_config_is_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_pipeline_supports_injected_length_function() -> None:
    config = ChunkingConfig(
        chunk_size=8,
        chunk_overlap=2,
        include_heading_context=False,
    )
    block = CleanedBlock(raw_content="1234567890", clean_content="1234567890")

    result = ChunkingPipeline(config, length_function=len).chunk(
        _cleaned_document([block])
    )

    assert [chunk.content for chunk in result.chunks] == ["123456", "6\n7890"]
    assert all(chunk.token_count <= 8 for chunk in result.chunks)


def test_small_sibling_sections_are_merged_with_explicit_headings() -> None:
    config = ChunkingConfig(
        chunk_size=80,
        chunk_overlap=10,
        min_chunk_size=30,
    )
    blocks = [
        CleanedBlock(
            raw_content="病假正文。",
            clean_content="病假正文。",
            section_title="3.2 病假",
            heading_level=7,
            heading_path=["第三章 假期制度", "3.2 病假"],
        ),
        CleanedBlock(
            raw_content="婚假正文。",
            clean_content="婚假正文。",
            section_title="3.3 婚假",
            heading_level=7,
            heading_path=["第三章 假期制度", "3.3 婚假"],
        ),
    ]

    result = ChunkingPipeline(config, length_function=len).chunk(
        _cleaned_document(blocks)
    )

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.section_title == "3.2 病假 / 3.3 婚假"
    assert chunk.heading_path == ["第三章 假期制度"]
    assert chunk.content == "3.2 病假\n病假正文。\n\n3.3 婚假\n婚假正文。"
    assert chunk.source_metadata["source_block_indexes"] == [0, 1]
    assert len(chunk.source_metadata["merged_sections"]) == 2


def test_small_sections_are_not_merged_across_parent_chapters() -> None:
    config = ChunkingConfig(chunk_size=80, chunk_overlap=10, min_chunk_size=30)
    blocks = [
        CleanedBlock(
            raw_content="小节一。",
            clean_content="小节一。",
            section_title="1.1 小节",
            heading_path=["第一章", "1.1 小节"],
        ),
        CleanedBlock(
            raw_content="小节二。",
            clean_content="小节二。",
            section_title="2.1 小节",
            heading_path=["第二章", "2.1 小节"],
        ),
    ]

    result = ChunkingPipeline(config, length_function=len).chunk(
        _cleaned_document(blocks)
    )

    assert len(result.chunks) == 2


def test_long_heading_is_truncated_without_consuming_content_budget() -> None:
    config = ChunkingConfig(
        chunk_size=12,
        chunk_overlap=2,
        heading_context_max_tokens=8,
    )
    block = CleanedBlock(
        raw_content="1234567890",
        clean_content="1234567890",
        heading_path=["非常长的父级标题", "非常长的当前章节标题"],
    )

    result = ChunkingPipeline(config, length_function=len).chunk(
        _cleaned_document([block])
    )

    assert len(result.chunks) > 1
    assert all(chunk.token_count <= config.chunk_size for chunk in result.chunks)
    assert all(chunk.embedding_content.startswith("章节：") for chunk in result.chunks)


def test_hr_handbook_parse_clean_and_chunk_preview() -> None:
    """完整处理 HR 手册，并用 ``pytest -s`` 输出可人工检查的分块数据。"""

    fixture = Path(__file__).parent / "fixtures" / "documents" / "hr-handbook.txt"
    parsed_blocks = parse_document(fixture)
    cleaned_document = clean_parsed_blocks(parsed_blocks)
    config = ChunkingConfig(chunk_size=180, chunk_overlap=30)
    result = chunk_cleaned_document(cleaned_document, config)

    preview = {
        "source": fixture.name,
        "parsed_blocks": len(parsed_blocks),
        "cleaned_blocks": len(cleaned_document.blocks),
        "indexable_blocks": result.report.indexable_blocks,
        "structural_blocks": result.report.structural_blocks,
        "merged_structural_blocks": result.report.merged_structural_blocks,
        "output_chunks": result.report.output_chunks,
        "total_tokens": result.report.total_tokens,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "source_block_index": chunk.source_block_index,
                "section_title": chunk.section_title,
                "heading_path": chunk.heading_path,
                "language": chunk.language,
                "token_count": chunk.token_count,
                "overlap_token_count": chunk.overlap_token_count,
                "content": chunk.content,
                "embedding_content": chunk.embedding_content,
            }
            for chunk in result.chunks
        ],
    }
    print("\nHR handbook chunk preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    assert len(parsed_blocks) == 1
    assert cleaned_document.indexable_blocks
    assert result.report.output_chunks > 1
    assert all(chunk.token_count <= config.chunk_size for chunk in result.chunks)
    assert "第一章 公司简介" in {
        chunk.section_title for chunk in result.chunks
    }
    assert "3.1 年假" in {chunk.section_title for chunk in result.chunks}
    assert any(
        "完成OA系统账号激活和基础信息填写" in chunk.content
        for chunk in result.chunks
    )
    assert any("\n4. 阅读《代码规范文档》" in chunk.content for chunk in result.chunks)
    assert any(
        chunk.section_title == "3.2 病假 / 3.3 婚假"
        for chunk in result.chunks
    )
    overlap_chunk = next(chunk for chunk in result.chunks if chunk.chunk_index == 3)
    assert overlap_chunk.content.startswith("了解团队和当前工作重点\n")


def test_enterprise_internal_docs_parse_clean_and_chunk_preview() -> None:
    """分块多文档企业语料，并输出携带文档级元数据的完整预览。"""

    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "documents"
        / "enterprise_internal_docs_rag_test.txt"
    )
    parsed_blocks = parse_document(fixture)
    cleaned_document = clean_parsed_blocks(parsed_blocks)
    config = ChunkingConfig(
        chunk_size=420,
        chunk_overlap=60,
        min_chunk_size=96,
    )
    result = chunk_cleaned_document(cleaned_document, config)

    preview = {
        "source": fixture.name,
        "parsed_blocks": len(parsed_blocks),
        "cleaned_blocks": len(cleaned_document.blocks),
        "indexable_blocks": result.report.indexable_blocks,
        "structural_blocks": result.report.structural_blocks,
        "merged_structural_blocks": result.report.merged_structural_blocks,
        "output_chunks": result.report.output_chunks,
        "total_tokens": result.report.total_tokens,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.source_metadata.get("document_id"),
                "document_title": chunk.source_metadata.get("document_title"),
                "version": chunk.source_metadata.get("version"),
                "classification": chunk.source_metadata.get("classification"),
                "section_title": chunk.section_title,
                "heading_path": chunk.heading_path,
                "token_count": chunk.token_count,
                "overlap_token_count": chunk.overlap_token_count,
                "content": chunk.content,
                "embedding_content": chunk.embedding_content,
            }
            for chunk in result.chunks
        ],
    }
    print("\nEnterprise internal docs chunk preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    expected_document_ids = {
        f"DOC-HY-{document_number:04d}" for document_number in range(1, 16)
    }
    actual_document_ids = {
        str(chunk.source_metadata["document_id"])
        for chunk in result.chunks
        if chunk.source_metadata.get("document_id")
    }
    assert len(parsed_blocks) == 1
    assert actual_document_ids == expected_document_ids
    assert all(chunk.token_count <= config.chunk_size for chunk in result.chunks)
    assert not any("[文档元数据]" in chunk.content for chunk in result.chunks)
    assert not any(
        "DOC-HY-0002" in chunk.content
        for chunk in result.chunks
        if chunk.source_metadata.get("document_id") == "DOC-HY-0001"
    )
    assert any(
        chunk.source_metadata.get("document_id") == "DOC-HY-0002"
        and "chunk_size 默认 420 tokens" in chunk.content
        for chunk in result.chunks
    )
    assert any(
        chunk.section_title == "附加测试段落：用于干扰检索"
        for chunk in result.chunks
    )
    faq_chunks = [
        chunk
        for chunk in result.chunks
        if chunk.source_metadata.get("document_id") == "DOC-HY-0015"
        and chunk.section_title
        and chunk.section_title.startswith("Q")
    ]
    assert len(faq_chunks) == 8
    assert all(len(chunk.source_metadata.get("merged_sections", [])) == 0 for chunk in faq_chunks)
    assert any(
        chunk.section_title.startswith("Q4：出差酒店最高能报多少")
        and "一线城市 600 元/晚" in chunk.content
        for chunk in faq_chunks
    )


def test_meeting_minutes_parse_clean_and_chunk_preview() -> None:
    """处理单份会议纪要，并输出章节与表格均可检查的分块预览。"""

    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "documents"
        / "Meeting_Minutes_Q2_2026.txt"
    )
    parsed_blocks = parse_document(fixture)
    cleaned_document = clean_parsed_blocks(parsed_blocks)
    config = ChunkingConfig(
        chunk_size=420,
        chunk_overlap=60,
        min_chunk_size=96,
    )
    result = chunk_cleaned_document(cleaned_document, config)

    preview = {
        "source": fixture.name,
        "parsed_blocks": len(parsed_blocks),
        "cleaned_blocks": len(cleaned_document.blocks),
        "indexable_blocks": result.report.indexable_blocks,
        "structural_blocks": result.report.structural_blocks,
        "merged_structural_blocks": result.report.merged_structural_blocks,
        "output_chunks": result.report.output_chunks,
        "total_tokens": result.report.total_tokens,
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "section_title": chunk.section_title,
                "heading_path": chunk.heading_path,
                "language": chunk.language,
                "token_count": chunk.token_count,
                "overlap_token_count": chunk.overlap_token_count,
                "content": chunk.content,
                "embedding_content": chunk.embedding_content,
            }
            for chunk in result.chunks
        ],
    }
    print("\nMeeting minutes chunk preview:")
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    assert len(parsed_blocks) == 1
    assert len(cleaned_document.blocks) == 1
    assert result.report.output_chunks == 5
    assert all(chunk.token_count <= config.chunk_size for chunk in result.chunks)
    assert {
        "一、会议议题",
        "二、主要发言摘要",
        "三、决议事项",
        "四、下次会议",
    }.issubset({chunk.section_title for chunk in result.chunks})
    assert any(
        "| 序号 | 行动项" in chunk.content
        and "| 1    | 制定云成本优化方案" in chunk.content
        and "| 4    | Q3招聘计划：后端2人，前端1人，测试1人" in chunk.content
        for chunk in result.chunks
    )
    assert any("CTO 周明（出差，委托张伟代参会）" in chunk.content for chunk in result.chunks)
