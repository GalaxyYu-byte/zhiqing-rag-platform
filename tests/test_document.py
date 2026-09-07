from zq_rag_app.utils.document_parser import (
    get_document_parser,
    parse_document,
)

file_path = r"tests\fixtures\documents\product-faq.txt"

parser = get_document_parser(file_path)
print(type(parser).__name__)

blocks = parse_document(file_path)
print("文本块数量:", len(blocks))

for index, block in enumerate(blocks):
    print("=" * 50)
    print("块序号:", index)
    print("页码:", block.page_num)
    print("章节:", block.section_title)
    print("元数据:", block.metadata)
    print("内容:", block.content[:300])
