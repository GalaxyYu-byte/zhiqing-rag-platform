"""规则化文档清洗流水线测试。"""

import pytest

from zq_rag_app.document_processing.cleaning import (
    CleaningConfig,
    CleaningPipeline,
    CleaningRuleExecutionError,
    CleaningValidationError,
    clean_parsed_blocks,
)
from zq_rag_app.document_processing.cleaning.base import (
    DocumentCleaningRule,
    ElementCleaningRule,
)
from zq_rag_app.document_processing.cleaning.validators import (
    ProtectedTokenValidationRule,
)
from zq_rag_app.document_processing.models import CleanedBlock
from zq_rag_app.document_processing.models import CleaningContext
from zq_rag_app.utils.parsers import ParsedBlock


def _pdf_block(content: str, page_num: int) -> ParsedBlock:
    return ParsedBlock(
        content=content,
        page_num=page_num,
        metadata={
            "file_name": "manual.pdf",
            "file_extension": ".pdf",
            "parser": "PdfDocumentParser",
            "page_num": page_num,
        },
    )


def test_cleaning_preserves_raw_content_and_infers_context() -> None:
    block = ParsedBlock(
        content="第一段\r\n\r\n\r\n第二段  ",
        metadata={
            "file_name": "说明.txt",
            "file_extension": ".txt",
            "parser": "TxtDocumentParser",
        },
    )

    document = clean_parsed_blocks([block])

    assert document.context.source_format == "txt"
    assert document.context.parser_name == "TxtDocumentParser"
    assert document.blocks[0].raw_content == "第一段\r\n\r\n\r\n第二段  "
    assert document.blocks[0].clean_content == "第一段\n\n第二段"
    assert document.blocks[0].clean_operations == ["normalize_text"]
    assert document.blocks[0].clean_changes[0].before.endswith("第二段  ")
    assert document.blocks[0].clean_changes[0].after == "第一段\n\n第二段"
    assert document.report.changed_blocks == 1


def test_common_cleaning_preserves_indentation_and_tabs() -> None:
    block = ParsedBlock(
        content="    print('hello')\n名称\t\t价格",
        metadata={
            "file_extension": ".md",
            "parser": "MarkdownDocumentParser",
        },
    )

    cleaned = clean_parsed_blocks([block]).blocks[0]

    assert cleaned.clean_content == "    print('hello')\n名称\t\t价格"


def test_pdf_page_number_is_removed_only_at_page_boundary() -> None:
    blocks = [
        _pdf_block("第 1 页\n正文提到第 12 页的要求。", 1),
        _pdf_block("正文第二页\n2 / 10", 2),
    ]

    document = clean_parsed_blocks(blocks)

    assert document.blocks[0].clean_content == "正文提到第 12 页的要求。"
    assert document.blocks[1].clean_content == "正文第二页"
    assert document.report.operation_counts["remove_pdf_page_number"] == 2


def test_repeated_pdf_headers_and_footers_are_removed() -> None:
    blocks = [
        _pdf_block(f"水库运行管理办法\n第 {page} 页正文\n内部资料 {page}/3", page)
        for page in range(1, 4)
    ]

    document = clean_parsed_blocks(blocks)

    assert [block.clean_content for block in document.blocks] == [
        "第 1 页正文",
        "第 2 页正文",
        "第 3 页正文",
    ]
    assert document.report.operation_counts["remove_pdf_repeated_margin"] == 3


def test_pdf_specific_rules_do_not_run_for_plain_text() -> None:
    block = ParsedBlock(content="1\n正文", metadata={})
    context = CleaningContext(source_format="txt", parser_name="custom")

    document = clean_parsed_blocks([block], context)

    assert document.blocks[0].clean_content == "1\n正文"
    assert "remove_pdf_page_number" not in document.report.operation_counts


def test_empty_cleaned_block_is_excluded_but_retained() -> None:
    block = ParsedBlock(
        content="\x00\r\n",
        metadata={"file_extension": ".txt", "parser": "TxtDocumentParser"},
    )

    document = clean_parsed_blocks([block])

    assert len(document.blocks) == 1
    assert document.blocks[0].excluded_from_embedding is True
    assert document.blocks[0].exclude_reason == "empty_content"
    assert document.indexable_blocks == []


def test_excel_rule_preserves_middle_empty_cells() -> None:
    block = ParsedBlock(
        content="名称\t\t价格\t\t\n\t\t\n标准版\t\t99\t",
        section_title="产品",
        metadata={
            "file_extension": ".xlsx",
            "parser": "ExcelDocumentParser",
        },
    )

    document = clean_parsed_blocks([block])

    assert document.blocks[0].clean_content == "名称\t\t价格\n标准版\t\t99"
    assert document.blocks[0].heading_path == ["产品"]
    assert "normalize_excel_rows" in document.blocks[0].clean_operations


def test_cleaning_config_can_disable_pdf_rules() -> None:
    config = CleaningConfig()
    config.pdf.remove_page_numbers = False
    config.pdf.merge_wrapped_lines = False
    block = _pdf_block("第 1 页\n这是一行长度足够的连续正文内容\n下一行正文", 1)

    document = clean_parsed_blocks([block], config=config)

    assert document.blocks[0].clean_content == block.content
    assert "remove_pdf_page_number" not in document.blocks[0].clean_operations
    assert "merge_pdf_wrapped_lines" not in document.blocks[0].clean_operations


def test_exact_duplicate_block_is_marked_and_retained() -> None:
    content = "这是一个长度超过最小限制的完全重复正文内容，用于测试重复检测。"
    blocks = [
        ParsedBlock(content=content, section_title="说明", metadata={}),
        ParsedBlock(content=content, section_title="说明", metadata={}),
    ]

    document = clean_parsed_blocks(blocks)

    assert document.blocks[0].excluded_from_embedding is False
    assert document.blocks[1].excluded_from_embedding is True
    assert document.blocks[1].exclude_reason == "exact_duplicate"
    assert document.blocks[1].clean_changes[-1].details["duplicate_of_index"] == 0
    assert len(document.indexable_blocks) == 1


def test_table_of_contents_is_detected_and_marked() -> None:
    block = ParsedBlock(
        content=(
            "目录\n"
            "第一章 总则........1\n"
            "第二章 管理职责........3\n"
            "第三章 运行管理........8"
        ),
        metadata={},
    )

    document = clean_parsed_blocks([block])

    assert document.blocks[0].excluded_from_embedding is True
    assert document.blocks[0].exclude_reason == "table_of_contents"
    toc_change = next(
        change
        for change in document.blocks[0].clean_changes
        if change.rule_name == "exclude_table_of_contents"
    )
    assert toc_change.details["entry_count"] == 3


def test_pdf_wrapped_lines_are_merged_conservatively() -> None:
    block = _pdf_block(
        "为了保障水库安全运行需要加强日常巡查\n并及时记录巡查结果。\n第一条 适用范围",
        1,
    )

    document = clean_parsed_blocks([block])

    assert document.blocks[0].clean_content == (
        "为了保障水库安全运行需要加强日常巡查并及时记录巡查结果。\n第一条 适用范围"
    )
    change = next(
        change
        for change in document.blocks[0].clean_changes
        if change.rule_name == "merge_pdf_wrapped_lines"
    )
    assert change.details["merged_pairs"] == 1


def test_noise_characters_create_quality_warnings() -> None:
    block = ParsedBlock(content="正常内容���锟斤拷", metadata={})

    document = clean_parsed_blocks([block])

    warning_codes = {warning.code for warning in document.blocks[0].warnings}
    assert "replacement_character" in warning_codes
    assert "possible_mojibake" in warning_codes
    assert document.report.warning_count >= 2


def test_heading_hierarchy_is_restored_across_blocks() -> None:
    blocks = [
        _pdf_block("第一章 总则\n本章说明。", 1),
        _pdf_block("第一节 管理范围\n本节说明。", 2),
        _pdf_block("第一条 适用范围\n具体正文。", 3),
    ]

    document = clean_parsed_blocks(blocks)

    assert document.blocks[0].heading_path == ["第一章 总则"]
    assert document.blocks[1].heading_path == ["第一章 总则", "第一节 管理范围"]
    assert document.blocks[2].heading_path == [
        "第一章 总则",
        "第一节 管理范围",
        "第一条 适用范围",
    ]
    assert document.blocks[2].source_metadata["detected_headings"][0]["level"] == 4


def test_rule_exception_is_wrapped_with_rule_name() -> None:
    class BrokenRule(DocumentCleaningRule):
        name = "broken_rule"

        def apply(self, blocks, context):  # type: ignore[no-untyped-def]
            raise ValueError("boom")

    with pytest.raises(CleaningRuleExecutionError, match="broken_rule"):
        CleaningPipeline(rules=[BrokenRule()]).clean(
            [ParsedBlock(content="正文", metadata={})]
        )


def test_protected_number_change_fails_validation() -> None:
    class CorruptNumberRule(ElementCleaningRule):
        name = "corrupt_number"
        priority = 10

        def apply_to_block(
            self,
            block: CleanedBlock,
            context: CleaningContext,
        ) -> None:
            del context
            before = block.clean_content
            block.clean_content = before.replace("85米", "86米")
            block.record_change(
                rule_name=self.name,
                change_type="replace_text",
                before=before,
                after=block.clean_content,
            )

    with pytest.raises(CleaningValidationError):
        CleaningPipeline(
            rules=[CorruptNumberRule(), ProtectedTokenValidationRule()]
        ).clean([ParsedBlock(content="正常水位为85米。", metadata={})])
