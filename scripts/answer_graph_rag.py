"""从命令行执行一次带引用的 Graph RAG 回答。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import asdict

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.services.rag_service import RagAnswerService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行带引用的 Graph RAG 回答")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    service = RagAnswerService()
    try:
        async with SessionLocal() as session:
            result = await service.answer(
                session,
                query=args.query,
                kb_ids=[args.kb_id],
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                rerank=not args.no_rerank,
            )
    finally:
        await service.aclose()
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


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
