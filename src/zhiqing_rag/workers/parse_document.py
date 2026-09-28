"""在可终止子进程中解析文档，结果以本地 JSON 返回给索引 Worker。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from zhiqing_rag.document_processing import ChunkConfig, process_document
from zhiqing_rag.document_processing.format_validation import DocumentFormatError
from zhiqing_rag.document_processing.parsers.models import DocumentParseError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--max-file-size", type=int, required=True)
    args = parser.parse_args()
    try:
        processed = process_document(
            args.source,
            original_filename=args.filename,
            max_file_size=args.max_file_size,
            chunk_config=ChunkConfig(max_tokens=args.size, overlap_tokens=args.overlap),
        )
        if not processed.cleaned.is_complete:
            result = {"error_code": "PARSE_INCOMPLETE"}
        elif not processed.chunks:
            result = {"error_code": "NO_USABLE_CONTENT"}
        else:
            result = {"chunks": [asdict(chunk) for chunk in processed.chunks]}
    except (DocumentFormatError, DocumentParseError) as error:
        result = {"error_code": error.code}
    except Exception:
        result = {"error_code": "PARSE_FAILED"}
    args.result.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
