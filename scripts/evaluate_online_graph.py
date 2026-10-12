"""通过真实鉴权接口比较一跳 GraphRAG 与禁用图的基线；模型调用前核对冻结来源。"""

import argparse
import asyncio
import hashlib
import json
import os
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
from zhiqing_rag.query_retrieval.graph import OnlineGraphService
from zhiqing_rag.query_retrieval.repository import _CHUNK_COLUMNS, QueryRetrievalRepository
from zhiqing_rag.query_routing.models import GraphPlan, PlanQueryBody
from zhiqing_rag.query_routing.planning import get_graph_service
from zhiqing_rag.query_routing.scope import QueryScopeRepository

CASES = (
    {"id": "G1", "query": "转派摘要包含哪些字段？", "document_id": "76"},
    {"id": "G2", "query": "系统管理员负责什么？", "document_id": "71"},
    {"id": "G3", "query": "关闭摘要应写明什么？", "document_id": "56"},
    {
        "id": "G4",
        "query": "接收方确认后，原受理人负责什么？",
        "document_id": "76",
        "required_condition": "接收方确认后",
    },
)
FROZEN_DOCUMENTS = PROJECT_ROOT / "eval/graph_build/v1/documents.jsonl"


def selected_cases(only):
    chosen = [dict(case) for case in CASES if not only or case["id"] in only]
    if only and len(chosen) != len(set(only)):
        raise ValueError("--only 包含未知题目编号")
    return chosen


def load_frozen_documents(path, cases):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    frozen = {str(row["document_id"]): row for row in rows}
    if len(frozen) != len(rows):
        raise ValueError("冻结文档编号重复")
    selected = {}
    for document_id in sorted({case["document_id"] for case in cases}):
        document = frozen.get(document_id)
        if document is None:
            raise ValueError("固定题目没有冻结文档来源")
        suffix = Path(document["source_filename"]).suffix.lower()
        if suffix not in {".pdf", ".docx", ".md", ".txt", ".xlsx"}:
            raise ValueError("冻结来源格式不受支持")
        source = path.parent / "sources" / f"{document_id}{suffix}"
        if hashlib.sha256(source.read_bytes()).hexdigest() != document["source_sha256"]:
            raise ValueError("冻结原文件哈希不一致")
        selected[document_id] = document
    return selected, hashlib.sha256(raw).hexdigest()


def verify_source_rows(frozen, rows):
    """允许活动批次重建，只接受同一租户、知识库、文档、修订及原文件。"""
    verified = []
    for document_id, original in frozen.items():
        current = [row for row in rows if str(row["document_id"]) == document_id]
        if not current:
            raise ValueError("冻结文档没有当前可读的活动索引")
        expected = tuple(int(original[key]) for key in ("tenant_id", "kb_id", "revision_id"))
        generations = {row["generation_id"] for row in current}
        if len(generations) != 1:
            raise ValueError("同一文档出现多个活动批次")
        for row in current:
            if (
                (row["tenant_id"], row["knowledge_base_id"], row["revision_id"]) != expected
                or row["file_sha256"] != original["source_sha256"]
                or hashlib.sha256(row["content"].encode()).hexdigest() != row["content_sha256"]
            ):
                raise ValueError("当前活动来源与冻结原文件、修订或正文哈希不一致")
        verified.append(
            {
                "document_id": document_id,
                "tenant_id": str(expected[0]),
                "knowledge_base_id": str(expected[1]),
                "revision_id": str(expected[2]),
                "generation_id": str(current[0]["generation_id"]),
                "source_file_sha256": original["source_sha256"],
                "verified_chunk_count": len(current),
            }
        )
    return verified


def verify_sources(claims, settings, frozen):
    repository = QueryRetrievalRepository(settings)
    with database_connection(settings) as connection:
        identity = identity_from_access(AuthRepository(connection, settings.db_schema), claims)
        scope = QueryScopeRepository(connection, settings.db_schema).resolve(
            identity,
            PlanQueryBody(query="在线图固定来源复核", document_ids=list(frozen)),
        )
        fresh = repository._refresh_identity(connection, identity)
        repository._configure_planner(connection)
        authorized, parameters = repository._authorized(fresh, scope)
        rows = repository._query(
            connection,
            "SELECT " + _CHUNK_COLUMNS + ", r.file_sha256 " + authorized,
            parameters,
        )
    return scope.summary().model_dump(), verify_source_rows(frozen, rows)


def read_password(args):
    if args.credentials_file:
        seed = json.loads(args.credentials_file.read_text(encoding="utf-8-sig"))
        return seed[args.email]["password"]
    password = os.environ.get("QUERY_EVAL_PASSWORD")
    if not password:
        raise ValueError("通过 QUERY_EVAL_PASSWORD 或私有 --credentials-file 提供密码")
    return password


def cleanup_session(claims, settings):
    # 定向撤销本次新会话；不调用会退出其他设备的业务 logout。
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


def relation_is_grounded(relation, evidence, seeds):
    content = evidence.get("content")
    quote = relation.get("source_quote")
    start, end = relation.get("quote_start"), relation.get("quote_end")
    condition = relation.get("condition_text")
    seed = relation.get("seed_entity_uid")
    return (
        isinstance(content, str)
        and isinstance(quote, str)
        and bool(quote)
        and type(start) is int
        and type(end) is int
        and 0 <= start < end <= len(content)
        and end == start + len(quote)
        and content[start:end] == quote
        and isinstance(condition, str)
        and (not condition or condition in quote)
        and seed in seeds
        and seed in (relation.get("subject_entity_uid"), relation.get("object_entity_uid"))
    )


def citation_check(body):
    answer = body.get("answer")
    if answer is None:
        return {}
    sources = {
        item["citation_id"]: item["content"]
        for item in body.get("retrieval", {}).get("evidence", [])
    }
    statements = answer.get("statements", [])
    citations = [
        citation for statement in statements for citation in statement.get("citations", [])
    ]
    valid = all(
        isinstance(citation.get("quote"), str)
        and bool(citation["quote"])
        and citation.get("citation_id") in sources
        and citation["quote"] in sources[citation["citation_id"]]
        for citation in citations
    )
    if answer.get("status") == "answered":
        valid = (
            valid
            and bool(statements)
            and all(statement.get("citations") for statement in statements)
        )
    elif not citations:
        valid = None
    return {
        "answer_status": answer.get("status"),
        "citation_valid": valid,
        "answer_review_status": answer.get("review_status"),
        "semantic_correct": None,
    }


def evaluate_response(case, body, *, baseline=False):
    plan = body.get("plan", {})
    retrieval = body.get("retrieval", {})
    trace = retrieval.get("trace", {})
    evidence = retrieval.get("evidence", [])
    relations = [
        (item, relation) for item in evidence for relation in item.get("graph_relations", [])
    ]
    result = {
        "route": plan.get("route"),
        "graph_status": trace.get("graph_status"),
        "graph_count": trace.get("graph_count", 0),
        "returned_count": len(evidence),
        "relation_count": len(relations),
        "target_document_hit": any(
            item.get("document_id") == case["document_id"] for item in evidence
        ),
        **citation_check(body),
    }
    if baseline:
        result["passed"] = (
            plan.get("route") == "hybrid"
            and trace.get("graph_status") == "skipped"
            and trace.get("graph_count", 0) == 0
            and trace.get("graph_seed_count", 0) == 0
            and trace.get("graph_assertion_count", 0) == 0
            and not relations
        )
        return result
    try:
        graph_plan = GraphPlan.model_validate_json(json.dumps(plan))
        approved = (
            graph_plan.reason_code == "GRAPH_APPROVED"
            and graph_plan.planning_status == "planned"
            and graph_plan.params.max_hops == 1
            and all(ref.document_id == case["document_id"] for ref in graph_plan.params.projections)
        )
        seeds = set(graph_plan.params.seed_entity_uids)
    except (ValueError, TypeError):
        approved, seeds = False, set()
    trace_succeeded = (
        trace.get("graph_status") == "succeeded"
        and trace.get("graph_hops") == 1
        and all(
            type(trace.get(field)) is int and trace[field] > 0
            for field in ("graph_count", "graph_seed_count", "graph_assertion_count")
        )
    )
    provenance_valid = bool(relations) and all(
        item.get("document_id") == case["document_id"]
        and relation_is_grounded(relation, item, seeds)
        for item, relation in relations
    )
    required = case.get("required_condition")
    condition_preserved = required is None or any(
        relation_is_grounded(relation, item, seeds)
        and required in relation["condition_text"]
        and required in relation["source_quote"]
        for item, relation in relations
    )
    result.update(
        graph_plan_valid=approved,
        graph_trace_valid=trace_succeeded,
        provenance_valid=provenance_valid,
        condition_preserved=condition_preserved,
        passed=approved and trace_succeeded and provenance_valid and condition_preserved,
    )
    return result


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "results.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "results.json")
    lines = [
        "# 在线一跳 GraphRAG 验收",
        "",
        f"时间：{report['created_at']}",
        "",
        "每题限定同一文档，图模式与基线仅改变 query_graph_enabled；其他配置和接口相同。",
        "首次模型调用前核对当前权限、活动索引、同一修订及冻结原文件和正文哈希。",
        "通过仅表示服务端图计划、一跳召回及关系逐字出处有效，不表示答案语义正确或检索优于基线。",
        "完整 HTTP 响应保存在 results.json，供独立复核；语义正确性始终待人工验收。",
        "",
        "| 题号 | 模式 | HTTP | 路由 | 图状态 | 图块数 | 关系数 | 检查 | 耗时 |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            f"| {row['id']} | {row['mode']} | {row['http_status']} | {row.get('route', '-')} | "
            f"{row.get('graph_status', '-')} | {row.get('graph_count', 0)} | "
            f"{row.get('relation_count', 0)} | {row.get('passed', False)} | "
            f"{row['elapsed_ms']} ms |"
        )
    if report["mode"] == "answer":
        lines.extend(["", "| 题号 | 模式 | 回答状态 | 引用可定位 |", "| --- | --- | --- | --- |"])
        for row in report["results"]:
            lines.append(
                f"| {row['id']} | {row['mode']} | {row.get('answer_status', '-')} | "
                f"{row.get('citation_valid', '-')} |"
            )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def implementation_hashes(root=PROJECT_ROOT):
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "src/zhiqing_rag/query_retrieval/graph.py",
            "src/zhiqing_rag/query_retrieval/service.py",
            "src/zhiqing_rag/query_retrieval/fusion.py",
            "src/zhiqing_rag/query_retrieval/models.py",
            "src/zhiqing_rag/query_routing/planning.py",
            "src/zhiqing_rag/query_routing/service.py",
        )
    }


async def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("--output 必须位于当前项目内")
    cases = selected_cases(args.only)
    frozen, digest = load_frozen_documents(FROZEN_DOCUMENTS, cases)
    settings = get_settings()
    report = {
        "version": 1,
        "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "mode": "answer" if args.answer else "search",
        "cases": cases,
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "frozen_documents_sha256": digest,
        "analyzer_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "implementation_sha256": implementation_hashes(),
        "query_graph_enabled": settings.query_graph_enabled,
        "neo4j_enabled": settings.neo4j_enabled,
        "models": {
            "analyzer": settings.query_analyzer_model,
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model,
            "answer": settings.chat_model if args.answer else None,
        },
        "source_verified": False,
        "evaluation_session_revoked": False,
        "semantic_review_required": True,
        "results": [],
    }
    write_report(output, report)
    claims = None
    stage = "login"
    app = create_app()
    disabled_graph = OnlineGraphService(settings.model_copy(update={"query_graph_enabled": False}))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://online-graph-eval.local",
            headers={"X-Requested-With": "XMLHttpRequest"},
            timeout=120,
        ) as client:
            login = await client.post(
                "/api/auth/login", json={"email": args.email, "password": read_password(args)}
            )
            if login.status_code != 200:
                raise ValueError("评估登录失败")
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
                raise ValueError("评估选择租户失败")
            token = selected.json()["access_token"]
            claims = decode_access_token(settings, token)
            client.headers["Authorization"] = f"Bearer {token}"
            stage = "source_verification"
            report["scope"], report["verified_sources"] = verify_sources(claims, settings, frozen)
            report["source_verified"] = True
            write_report(output, report)
            endpoint = "/api/query/answer" if args.answer else "/api/query/search"
            for case in cases:
                for mode in ("graph", "baseline"):
                    stage = f"{case['id']}/{mode}"
                    if mode == "baseline":
                        app.dependency_overrides[get_graph_service] = lambda: disabled_graph
                    else:
                        app.dependency_overrides.pop(get_graph_service, None)
                    started = time.perf_counter()
                    row = {"id": case["id"], "mode": mode, "document_id": case["document_id"]}
                    try:
                        response = await client.post(
                            endpoint,
                            json={"query": case["query"], "document_ids": [case["document_id"]]},
                        )
                        row["http_status"] = response.status_code
                        body = response.json()
                        row["response"] = body
                        if response.status_code == 200:
                            row.update(evaluate_response(case, body, baseline=mode == "baseline"))
                        else:
                            row["passed"] = False
                    except Exception as error:
                        # 保存进度但不记录可能带凭据或模型载荷的底层异常文本。
                        row.setdefault("http_status", None)
                        row.update(passed=False, error_type=type(error).__name__)
                    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
                    report["results"].append(row)
                    write_report(output, report)
                    print(
                        json.dumps(
                            {
                                "id": f"{case['id']}/{mode}",
                                "http_status": row["http_status"],
                                "route": row.get("route"),
                                "graph_status": row.get("graph_status"),
                                "graph_count": row.get("graph_count", 0),
                                "elapsed_ms": row["elapsed_ms"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
            report["summary"] = {
                "graph_passed": sum(
                    row.get("passed", False) for row in report["results"] if row["mode"] == "graph"
                ),
                "baseline_valid": sum(
                    row.get("passed", False)
                    for row in report["results"]
                    if row["mode"] == "baseline"
                ),
                "case_count": len(cases),
                "semantic_correct": None,
            }
    except Exception as error:
        report["fatal_error"] = {"stage": stage, "type": type(error).__name__}
        raise
    finally:
        app.dependency_overrides.pop(get_graph_service, None)
        try:
            if claims is not None:
                cleanup_session(claims, settings)
                report["evaluation_session_revoked"] = True
        except Exception as error:
            report["cleanup_error"] = {"type": type(error).__name__}
            raise
        finally:
            close_database_pool(settings)
            write_report(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--tenant-code", default="DEFAULT")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--answer", action="store_true", help="执行回答并记录引用，不自动判定答案语义"
    )
    parser.add_argument("--only", action="append", default=[], help="仅执行指定题号，可重复")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as error:
        parser.exit(1, f"在线图验收未完成：{type(error).__name__}\n")
    summary = report["summary"]
    complete = summary["graph_passed"] == summary["baseline_valid"] == summary["case_count"]
    raise SystemExit(0 if complete else 1)


if __name__ == "__main__":
    main()
