"""运行 Graph RAG 端到端评测并输出 JSON 报告。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

from zq_rag_app.core.database import SessionLocal, close_database
from zq_rag_app.core.neo4j import close_neo4j
from zq_rag_app.core.security import UserContext
from zq_rag_app.evaluation.graph_rag import (
    GraphRagQualityThresholds,
    evaluate_graph_rag,
    evaluate_quality_gate,
    load_graph_rag_cases,
)
from zq_rag_app.services.rag_service import RagAnswerService
from zq_rag_app.models.document import Document
from zq_rag_app.services.permission_service import list_accessible_document_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Graph RAG 端到端评测")
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("qa-docs/qa_ground_truth.jsonl"),
    )
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--only-permission",
        action="store_true",
        help="只运行 5 条文档 ACL 题，不调用检索、Reranker 或 LLM",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-skipped-permission",
        action="store_true",
        help="开发阶段允许跳过文档级权限题；正式准入不要使用",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    cases = load_graph_rag_cases(args.dataset)
    if args.only_permission:
        cases = tuple(case for case in cases if case.question_type == "permission")
        if not cases:
            raise ValueError("评测集中没有 permission 用例")
    service = RagAnswerService()
    try:
        async with SessionLocal() as session:
            public_user = UserContext(
                user_id=-1,
                username="graph-rag-public-eval",
                department_id="EVAL-PUBLIC",
                role="USER",
                clearance="内部公开",
            )

            async def check_permission(case) -> bool:
                rows = await session.execute(
                    select(Document.id).where(
                        Document.kb_id == args.kb_id,
                        Document.file_name.in_(list(case.expected_sources)),
                        Document.is_deleted.is_(False),
                    )
                )
                protected_ids = {int(row[0]) for row in rows.all()}
                if not protected_ids:
                    return False
                accessible = await list_accessible_document_ids(
                    session,
                    user=public_user,
                    kb_ids=[args.kb_id],
                    requested_doc_ids=sorted(protected_ids),
                )
                return protected_ids.isdisjoint(accessible)

            report = await evaluate_graph_rag(
                session,
                service=service,
                cases=cases,
                kb_id=args.kb_id,
                candidate_k=args.candidate_k,
                top_k=args.top_k,
                rerank=not args.no_rerank,
                permission_checker=check_permission,
            )
    finally:
        await service.aclose()
    thresholds = (
        GraphRagQualityThresholds(
            min_case_count=5,
            min_source_hit_rate=0.0,
            min_keyword_recall=0.0,
            min_valid_citation_rate=0.0,
            min_no_answer_accuracy=0.0,
            require_permission_coverage=True,
        )
        if args.only_permission
        else GraphRagQualityThresholds(
            require_permission_coverage=not args.allow_skipped_permission
        )
    )
    gate = evaluate_quality_gate(report, thresholds)
    payload = {"report": report.to_dict(), "quality_gate": gate.to_dict()}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if gate.passed else 1


async def _run_and_close() -> int:
    try:
        return await main()
    finally:
        await close_neo4j()
        await close_database()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.platform == "win32":
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            exit_code = runner.run(_run_and_close())
    else:
        exit_code = asyncio.run(_run_and_close())
    raise SystemExit(exit_code)
