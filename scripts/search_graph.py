"""从命令行执行一次可审计的 Graph Chunk 召回。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.services.graph_retrieval_service import search_by_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行 Neo4j 图谱证据召回")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hops", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--full",
        action="store_true",
        help="输出完整 Chunk 正文和 Claim；默认只输出便于验收的摘要",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    async with SessionLocal() as session:
        result = await search_by_graph(
            session,
            query=args.query,
            kb_ids=[args.kb_id],
            top_k=args.top_k,
            max_hops=args.max_hops,
        )
    payload = asdict(result)
    if not args.full:
        payload["results"] = [
            {
                "rank": chunk["rank"],
                "chunk_id": chunk["chunk_id"],
                "document": chunk["document"],
                "section": chunk["section"],
                "score": chunk["score"],
                "content": chunk["content"][:300],
            }
            for chunk in payload["results"]
        ]
        payload["matches"] = [
            {
                "claim_uid": match["claim_uid"],
                "subject": match["subject"],
                "predicate": match["predicate"],
                "object": match["object"],
                "evidence_quote": match["evidence_quote"],
                "score": match["score"],
            }
            for match in payload["matches"]
        ]
    print(json.dumps(payload, ensure_ascii=False, default=str))


async def _run_and_close() -> None:
    try:
        await main()
    finally:
        await close_neo4j()
        await close_database()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(_run_and_close())
    else:
        asyncio.run(_run_and_close())
