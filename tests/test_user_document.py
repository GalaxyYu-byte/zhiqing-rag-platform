"""对指定的本地文件执行格式校验、解析、清洗和分块。"""

import os
from pathlib import Path

import pytest

from zhiqing_rag.document_processing import process_document


def test_user_document_parses_and_chunks() -> None:
    raw_path = os.environ.get("DOCUMENT_TEST_FILE")
    if not raw_path:
        pytest.skip("设置 DOCUMENT_TEST_FILE 为待测试文件的路径")

    file_path = Path(raw_path.strip().strip('"')).expanduser()
    assert file_path.is_file(), f"文件不存在: {file_path}"

    result = process_document(file_path)

    print(f"文件: {file_path}")
    print(f"格式: {result.parsed.format.value}")
    print(
        f"解析元素: {len(result.parsed.elements)}, "
        f"清洗元素: {len(result.cleaned.elements)}, 分块: {len(result.chunks)}"
    )
    print(f"解析完整: {result.cleaned.is_complete}")
    print(f"告警: {[warning.code for warning in result.cleaned.warnings]}")
    for chunk in result.chunks[:20]:
        preview = chunk.content.replace("\n", "\\n")[:200]
        print(
            f"块 {chunk.chunk_index}: tokens={chunk.token_count}, "
            f"页码={chunk.page_number}, 章节={chunk.section_title!r}, "
            f"来源={chunk.source_refs}, 内容={preview!r}"
        )
    if len(result.chunks) > 20:
        print(f"其余 {len(result.chunks) - 20} 个分块未展示")

    assert result.parsed.elements, "解析后没有结构化元素"
    assert result.cleaned.elements, "清洗后没有有效元素"
    assert result.chunks, "没有生成分块"
    assert [chunk.chunk_index for chunk in result.chunks] == list(range(len(result.chunks)))
    assert all(chunk.content and chunk.embedding_content for chunk in result.chunks)
