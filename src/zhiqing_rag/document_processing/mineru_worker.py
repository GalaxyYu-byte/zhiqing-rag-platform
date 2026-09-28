"""在独立 MinerU 环境中执行的 PDF 解析进程。

此文件不导入主项目依赖，因为 MinerU 与主项目的 OpenAI 版本要求不同。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: mineru_worker.py INPUT_PDF OUTPUT_JSON", file=sys.stderr)
        return 2

    from mineru.parser import parse

    input_path, output_path = (Path(value) for value in sys.argv[1:])
    result = parse(input_path, tier="basic", ocr_mode="auto")
    structured = result.structured_content()
    pages = []
    for page in structured.get("pages", []):
        blocks = []
        for block in page.get("blocks", []):
            # 图片 data URI 可能很大，分块阶段只需要语义文字与来源位置。
            blocks.append(
                {
                    key: block[key]
                    for key in ("type", "content", "level", "bbox", "captions", "index")
                    if key in block
                }
            )
        pages.append({"page_idx": page["page_idx"], "blocks": blocks})

    output_path.write_text(json.dumps({"pages": pages}, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
