"""通过真实鉴权和当前 ASGI 接口检查固定问题的 Top-5 原文证据，不启动 HTTP 服务。"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from zhiqing_rag.core.config import PROJECT_ROOT, get_settings
from zhiqing_rag.infrastructure.database.connection import close_database_pool, database_connection
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.identity.auth_repository import AuthRepository
from zhiqing_rag.modules.identity.auth_service import identity_from_access
from zhiqing_rag.modules.identity.auth_store import AuthStore, create_auth_redis
from zhiqing_rag.modules.identity.auth_tokens import decode_access_token
from zhiqing_rag.query_analysis.prompts import SYSTEM_PROMPT
from zhiqing_rag.query_answering.providers import GENERATE_PROMPT, REVIEW_PROMPT, AnswerProvider
from zhiqing_rag.query_answering.router import get_answerer
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.models import ChunkRecord
from zhiqing_rag.query_retrieval.repository import QueryRetrievalRepository
from zhiqing_rag.query_routing.models import PlanQueryBody
from zhiqing_rag.query_routing.scope import QueryScopeRepository


def load_dataset(path: Path):
    raw = path.read_bytes()
    dataset = json.loads(raw)
    source = PROJECT_ROOT / dataset["source"]
    chunks = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    by_key = {(str(row["document_id"]), row["chunk_index"]): row for row in chunks}
    documents_path = source.parent / "documents.jsonl"
    documents = (
        {
            str(row["document_id"]): row
            for row in (
                json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines()
            )
        }
        if documents_path.exists()
        else {}
    )
    for case in dataset["cases"]:
        for reference in case["references"]:
            row = by_key[(reference["document_id"], reference["chunk_index"])]
            if not all(quote in row["content"] for quote in reference["quotes"]):
                raise ValueError(f"{case['id']} 的答案依据不在冻结原文中")
            reference["content_sha256"] = row["content_sha256"]
            if document := documents.get(reference["document_id"]):
                reference["source_file_sha256"] = document["source_sha256"]
    return dataset, by_key, hashlib.sha256(raw).hexdigest()


def read_password(args):
    if args.credentials_file:
        seed = json.loads(args.credentials_file.read_text(encoding="utf-8-sig"))
        return seed[args.email]["password"]
    value = os.environ.get("QUERY_EVAL_PASSWORD")
    if not value:
        raise ValueError("通过 QUERY_EVAL_PASSWORD 或私有 --credentials-file 提供密码")
    return value


def match_reindexed_reference(reference, frozen, records, file_sha256):
    """只允许同一不可变原文件重新切块，冻结问题和逐字答案依据不变。"""
    if not reference.get("source_file_sha256") or file_sha256 != reference["source_file_sha256"]:
        raise ValueError("重建后的原文件与冻结来源不一致")
    eligible = tuple(
        record
        for record in records
        if (record.tenant_id, record.knowledge_base_id, record.document_id, record.revision_id)
        == (
            int(frozen["tenant_id"]),
            int(frozen["kb_id"]),
            int(frozen["document_id"]),
            int(frozen["revision_id"]),
        )
    )
    matches = tuple(
        record
        for record in eligible
        if all(quote in record.content for quote in reference["quotes"])
        and hashlib.sha256(record.content.encode()).hexdigest() == record.content_sha256
    )
    if not matches:
        raise ValueError("同一修订的活动索引缺少冻结逐字答案依据")
    return matches


async def verify_sources(claims, dataset, by_key, *, allow_reindexed_sources=False):
    settings = get_settings()
    with database_connection(settings) as connection:
        identity = identity_from_access(AuthRepository(connection, settings.db_schema), claims)
        scope = QueryScopeRepository(connection, settings.db_schema).resolve(
            identity, PlanQueryBody(query="固定检索评估")
        )
    if allow_reindexed_sources:
        return await verify_reindexed_sources(identity, scope, dataset, by_key, settings)
    keys = {
        (reference["document_id"], reference["chunk_index"])
        for case in dataset["cases"]
        for reference in case["references"]
    }
    records = []
    for key in sorted(keys):
        row = by_key[key]
        records.append(
            ChunkRecord(
                tenant_id=int(row["tenant_id"]),
                knowledge_base_id=int(row["kb_id"]),
                document_id=int(row["document_id"]),
                revision_id=int(row["revision_id"]),
                generation_id=int(row["generation_id"]),
                chunk_id=int(row["id"]),
                chunk_index=row["chunk_index"],
                document_title="",
                document_name="",
                content=row["content"],
                content_sha256=row["content_sha256"],
                token_count=0,
                page_number=row["page_number"],
                section_title=row["section_title"],
            )
        )
    current = await QueryRetrievalRepository(settings).revalidate(identity, scope, tuple(records))
    actual = {row.chunk_id: row for row in current}
    if any(
        row.chunk_id not in actual
        or actual[row.chunk_id].content != row.content
        or actual[row.chunk_id].content_sha256 != row.content_sha256
        for row in records
    ):
        raise ValueError("部分冻结答案来源已变更或当前账号不可读，请先更新评估集")
    return scope.summary().model_dump(), len(records), []


async def verify_reindexed_sources(identity, scope, dataset, by_key, settings):
    from zhiqing_rag.query_retrieval.repository import _CHUNK_COLUMNS, _record

    repository = QueryRetrievalRepository(settings)
    with database_connection(settings) as connection:
        fresh = repository._refresh_identity(connection, identity)
        repository._configure_planner(connection)
        authorized, parameters = repository._authorized(fresh, scope)
        rows = repository._query(
            connection,
            "SELECT " + _CHUNK_COLUMNS + ", r.file_sha256 " + authorized,
            parameters,
        )
    records = await repository.revalidate(identity, scope, tuple(_record(row) for row in rows))
    hashes = {row["revision_id"]: row["file_sha256"] for row in rows}
    transitions, verified = [], set()
    for case in dataset["cases"]:
        for reference in case["references"]:
            frozen = by_key[(reference["document_id"], reference["chunk_index"])]
            matches = match_reindexed_reference(
                reference, frozen, records, hashes.get(int(frozen["revision_id"]))
            )
            verified.update(record.chunk_id for record in matches)
            transitions.append(
                {
                    "case_id": case["id"],
                    "document_id": reference["document_id"],
                    "revision_id": str(frozen["revision_id"]),
                    "original_generation_id": str(frozen["generation_id"]),
                    "original_chunk_index": reference["chunk_index"],
                    "file_sha256": reference["source_file_sha256"],
                    "current_sources": [
                        {
                            "generation_id": str(record.generation_id),
                            "chunk_id": str(record.chunk_id),
                            "chunk_index": record.chunk_index,
                            "content_sha256": record.content_sha256,
                        }
                        for record in matches
                    ],
                }
            )
    return scope.summary().model_dump(), len(verified), transitions


def cleanup_session(claims):
    # 仅撤销本脚本新建的会话；业务 logout 会退出所有设备，不能用于评估清理。
    settings = get_settings()
    with database_connection(settings) as connection:
        row = (
            AuthRepository(connection, settings.db_schema)
            .query(
                """UPDATE {schema}.auth_session SET revoked_at = CURRENT_TIMESTAMP
               WHERE session_key = %s AND tenant_id = %s AND tenant_member_id = %s
                 AND revoked_at IS NULL RETURNING refresh_token_hash""",
                (claims["sid"], claims["tid"], claims["mid"]),
            )
            .fetchone()
        )
    if row:
        with create_auth_redis(settings) as client:
            AuthStore(client, settings.redis_key_prefix).delete_refreshes(
                [row["refresh_token_hash"]]
            )


def evaluate_case(case, body):
    evidence = body["retrieval"]["evidence"]
    matches = []
    for reference in case["references"]:
        hits = [
            item
            for item in evidence
            if item["document_id"] == reference["document_id"]
            and all(quote in item["content"] for quote in reference["quotes"])
        ]
        matches.append(
            {
                "document_id": reference["document_id"],
                "hit": bool(hits),
                "ranks": [item["rank"] for item in hits],
            }
        )
    if matches:
        passed = all(item["hit"] for item in matches)
    elif "expected_route" in case:
        passed = (
            body["plan"]["route"] == case["expected_route"]
            and body["retrieval"]["status"] == case["expected_retrieval_status"]
        )
    else:
        # 无答案题的 Top-K 通常仍有相似块，不能要求空召回来证明拒答能力。
        passed = None
    return {"passed": passed, "reference_hits": matches}


def write_report(output: Path, report):
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    cases = report["results"]
    positives = [row for row in cases if row.get("kind") == "answerable"]
    passed = sum(row["passed"] for row in positives)
    lines = [
        "# 固定问题的 Top-5 检索验收",
        "",
        f"时间：{report['created_at']}",
        "",
        f"所检正向题完整命中：**{passed}/{len(positives)}**。",
        "",
        "问题及逐字答案依据在首次模型调用前固定。来源经当前权限、活动批次及正文哈希复核。",
        "命中表示 Top-5 中包含指定原文依据，不代表回答生成或全库检索质量已通过。",
        "",
        "| 题目 | 完整命中 | 依据排名 | 端到端耗时 |",
        "| --- | --- | --- | ---: |",
    ]
    for row in cases:
        result = {True: "通过", False: "未通过", None: "待回答验收"}[row.get("passed")]
        ranks = "; ".join(
            f"文档 {hit['document_id']}: {','.join(map(str, hit['ranks'])) or '未命中'}"
            for hit in row.get("reference_hits", [])
        ) or row.get("route", "请求失败")
        lines.append(
            f"| {row['id']} {row['query']} | {result} | {ranks} | {row['elapsed_ms']} ms |"
        )
    lines.extend(
        [
            "",
            "无答案题须在生成环节检查拒答；不能用有相似检索结果推断知识库有答案。",
            "本次使用当前代码的真实 ASGI HTTP 接口、PostgreSQL、Redis 和配置的模型服务。",
            "评估只创建自己的认证会话，结束后定向撤销，不更改文档或其他设备的会话。",
            "",
        ]
    )
    if report.get("mode") == "answer":
        lines.extend(
            [
                "## 回答状态",
                "",
                "| 题号 | 状态 | 引用可定位 | 预期状态匹配 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in cases:
            lines.append(
                f"| {row['id']} | {row.get('answer_status', '请求失败')} | "
                f"{row.get('citation_valid', False)} | {row.get('answer_status_matches', False)} |"
            )
        lines.extend(
            [
                "",
                "引用可定位与模型复核通过都不能代替独立事实验收。",
                "captured_model_outputs 如存在，是排查用模型草稿，不能作为已验收回答。",
                "",
            ]
        )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


async def run(args):
    dataset, by_key, digest = load_dataset(args.cases)
    if args.only:
        dataset["cases"] = [case for case in dataset["cases"] if case["id"] in args.only]
        if len(dataset["cases"]) != len(set(args.only)):
            raise ValueError("--only 包含未知题目编号")
    settings = get_settings()
    report = {
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "dataset_sha256": digest,
        "analyzer_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "reranker_input": "document_title_name_section_content",
        "analyzer_model": settings.query_analyzer_model,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "results": [],
        "mode": "answer" if args.answer else "search",
        "answer_model": settings.chat_model if args.answer else None,
    }
    claims = None
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analyzer_prompt.txt").write_text(SYSTEM_PROMPT, encoding="utf-8")
    app = create_app()
    captured = []
    if args.answer:
        (args.output / "answer_prompt.txt").write_text(GENERATE_PROMPT, encoding="utf-8")
        (args.output / "review_prompt.txt").write_text(REVIEW_PROMPT, encoding="utf-8")
    if args.capture_model_output:
        if not args.answer:
            raise ValueError("--capture-model-output 仅用于回答评估")

        class RecordingProvider(AnswerProvider):
            async def generate(self, question, evidence):
                value = await super().generate(question, evidence)
                captured.append({"stage": "generation", "value": value.value.model_dump()})
                return value

            async def review(self, question, evidence, draft):
                value = await super().review(question, evidence, draft)
                captured.append({"stage": "review", "value": value.value.model_dump()})
                return value

        app.dependency_overrides[get_answerer] = lambda: QueryAnswerService(
            QueryRetrievalRepository(settings), RecordingProvider(settings), settings
        )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://query-eval.local",
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=120,
        ) as client:
            print("开始评估登录", flush=True)
            login = await client.post(
                "/api/auth/login",
                json={
                    "email": args.email,
                    "password": read_password(args),
                },
            )
            if login.status_code != 200:
                raise ValueError(f"评估登录失败：HTTP {login.status_code}")
            selection = login.json()
            tenant = next(row for row in selection["tenants"] if row["code"] == args.tenant_code)
            selected = await client.post(
                "/api/auth/select-tenant",
                json={
                    "selection_token": selection["selection_token"],
                    "tenant_id": int(tenant["id"]),
                },
            )
            if selected.status_code != 200:
                raise ValueError(f"评估选择租户失败：HTTP {selected.status_code}")
            token = selected.json()["access_token"]
            claims = decode_access_token(settings, token)
            client.headers["Authorization"] = f"Bearer {token}"
            print("登录完成，复核固定答案来源", flush=True)
            (
                report["scope"],
                report["verified_source_count"],
                report["source_transitions"],
            ) = await verify_sources(
                claims, dataset, by_key, allow_reindexed_sources=args.allow_reindexed_sources
            )
            report["source_mode"] = (
                "same_revision_reindexed" if args.allow_reindexed_sources else "frozen_exact"
            )
            for case in dataset["cases"]:
                print(f"开始 {case['id']}", flush=True)
                capture_start = len(captured)
                started = time.perf_counter()
                endpoint = "/api/query/answer" if args.answer else "/api/query/search"
                response = await client.post(endpoint, json={"query": case["query"]})
                row = {
                    "id": case["id"],
                    "query": case["query"],
                    "kind": case.get("kind", "answerable"),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                    "http_status": response.status_code,
                }
                if response.status_code == 200:
                    body = response.json()
                    row.update(evaluate_case(case, body))
                    row.update({"route": body["plan"]["route"], "response": body})
                    if args.answer:
                        answer = body["answer"]
                        sources = {
                            item["citation_id"]: item["content"]
                            for item in body["retrieval"]["evidence"]
                        }
                        row["answer_status"] = answer["status"]
                        row["citation_valid"] = all(
                            citation["citation_id"] in sources
                            and citation["quote"] in sources[citation["citation_id"]]
                            for statement in answer["statements"]
                            for citation in statement["citations"]
                        )
                        expected = case.get("expected_answer_status")
                        if expected is None:
                            expected = {
                                "answerable": "answered",
                                "clarify": "skipped",
                                "unsupported": "unsupported",
                            }.get(row["kind"])
                        row["answer_status_matches"] = answer["status"] == expected
                else:
                    row.update(
                        {
                            "passed": False,
                            "error_code": response.json()
                            .get("detail", {})
                            .get("code", "HTTP_ERROR"),
                        }
                    )
                report["results"].append(row)
                if args.capture_model_output:
                    row["captured_model_outputs"] = captured[capture_start:]
                write_report(args.output, report)
                print(
                    json.dumps(
                        {k: row[k] for k in ("id", "passed", "elapsed_ms")}, ensure_ascii=False
                    ),
                    flush=True,
                )
            positives = [row for row in report["results"] if row.get("kind") == "answerable"]
            report["summary"] = {
                "answerable_count": len(positives),
                "full_hit_count": sum(r["passed"] for r in positives),
                "median_ms": round(statistics.median(r["elapsed_ms"] for r in report["results"])),
            }
            if args.answer:
                controls = [row for row in report["results"] if row.get("kind") != "answerable"]
                report["summary"].update(
                    answered_count=sum(row.get("answer_status") == "answered" for row in positives),
                    answered_with_valid_citations_count=sum(
                        row.get("answer_status") == "answered" and row.get("citation_valid", False)
                        for row in positives
                    ),
                    control_count=len(controls),
                    control_status_matches_count=sum(
                        row.get("answer_status_matches", False) for row in controls
                    ),
                )
    finally:
        if claims is not None:
            cleanup_session(claims)
            report["evaluation_session_revoked"] = True
        close_database_pool(settings)
        write_report(args.output, report)
    print(json.dumps(report.get("summary", {}), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=PROJECT_ROOT / "eval/query_retrieval/v1/cases.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--email", required=True)
    parser.add_argument("--tenant-code", default="DEFAULT")
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--answer", action="store_true", help="执行回答接口并记录拒答与引用检查")
    parser.add_argument(
        "--allow-reindexed-sources",
        action="store_true",
        help="允许同一原文件修订重建索引；逐字答案依据不变，并记录新旧来源",
    )
    parser.add_argument("--only", action="append", default=[], help="仅执行指定题号，可重复")
    parser.add_argument(
        "--capture-model-output", action="store_true", help="保存排查用草稿，不能视作已审核回答"
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
