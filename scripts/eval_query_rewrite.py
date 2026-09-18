"""真实 DeepSeek 预处理冒烟测试：合成输入，不访问数据库或执行检索。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService
from zq_rag_app.services.query_preprocessing_service import prepare_hyde, prepare_query
from zq_rag_app.services.query_router_service import QueryRouterService


CASES = [
    {"name": "context_relation", "query": "他是否也负责天河项目？",
     "history": [{"role": "user", "content": "张三是星河项目的负责人。"}],
     "method": "query_rewrite", "contains": ["张三", "天河项目"], "paths": ["hybrid_graph"]},
    {"name": "context_current_constraints", "query": "它去年未验收的原因是什么？",
     "history": [{"role": "user", "content": "星河项目今年验收了吗？"}],
     "method": "query_rewrite", "contains": ["星河项目", "去年", "未验收"],
     "excludes": ["今年"], "paths": ["hybrid", "hybrid_graph"]},
    {"name": "ambiguous_context", "query": "它是谁负责？",
     "history": [{"role": "user", "content": "介绍星河项目和天河项目。"}],
     "paths": ["clarify"]},
    {"name": "exact_constraint_preservation", "query": "查 HT-2025-001 合同，金额不超过 100 万元且未验收的记录。",
     "contains": ["HT-2025-001", "不超过", "100 万元", "未验收"],
     "paths": ["hybrid", "structured"]},
    {"name": "topic_switch", "query": "员工差旅报销需要哪些审批步骤？",
     "history": [{"role": "user", "content": "张三负责星河项目。"}],
     "excludes": ["张三", "星河项目"], "paths": ["hybrid"]},
    {"name": "hyde_dense_only", "query": "资料搜出来一堆但答得很散，这种情况一般怎么改善？",
     # 专项验证 HyDE 执行器；先做真实 Analyzer，再选择经过服务端安全规则校验的 HyDE。
     "select_hyde": True, "method": "hyde", "paths": ["hybrid"]},
]


async def evaluate(output: Path) -> bool:
    rows = []
    for case in CASES:
        if case.get("select_hyde"):
            class HyDEAnalyzer(QueryAnalyzerService):
                async def analyze(self, request):
                    analysis = await super().analyze(request)
                    if analysis.status == "ok":
                        analysis.retrieval_strategy.rewrite_method = "hyde"
                        analysis.retrieval_strategy.rewrite_operations = []
                        analysis.retrieval_strategy.use_query_rewrite = False
                        analysis.retrieval_strategy.use_hyde = True
                    return analysis
            analyzer_factory = HyDEAnalyzer
        else:
            analyzer_factory = QueryAnalyzerService
        prepared = await prepare_query(QueryAnalysisRequest(
            query=case["query"], history=case.get("history", []), reference_date="2026-09-17",
        ), analyzer_factory=analyzer_factory)
        if prepared.analysis.retrieval_strategy.use_hyde:
            # HyDE 仅适用于基础 Hybrid，此路由无需数据库或图谱预检查。
            plan = await QueryRouterService().route(
                None, analysis=prepared.analysis, request=prepared.request,
                kb_ids=[1], doc_ids=None, preprocessing_complete=True,
            )
            prepared = await prepare_hyde(prepared, plan)
        errors = []
        summary, analysis = prepared.rewrite, prepared.analysis
        if analysis.status != "ok":
            errors.append(f"analysis_degraded:{analysis.degradation_reason}")
        if summary.status == "degraded":
            errors.append(f"rewrite_degraded:{summary.degradation_reason}")
        if case.get("method") and summary.method != case["method"]:
            errors.append("unexpected_method")
        if case.get("method") and summary.status != "rewritten":
            errors.append("rewrite_not_completed")
        if analysis.retrieval_strategy.path not in case["paths"]:
            errors.append("unexpected_path")
        if any(text not in prepared.request.query for text in case.get("contains", [])):
            errors.append("lost_expected_subject_or_constraint")
        if any(text in prepared.request.query for text in case.get("excludes", [])):
            errors.append("inherited_unrelated_history")
        if summary.method == "hyde" and (
            prepared.request.query != case["query"] or summary.reanalyzed or not prepared.hypothetical_document
        ):
            errors.append("invalid_hyde_isolation")
        rows.append({
            "name": case["name"], "passed": not errors, "errors": errors,
            "method_source": "test_override" if case.get("select_hyde", False) else "analyzer",
            "rewrite": asdict(summary), "analysis": analysis.model_dump(mode="json"),
            "hypothetical_document": prepared.hypothetical_document,
            "timing": {"analysis_ms": prepared.analysis_ms, "rewrite_ms": prepared.rewrite_ms},
        })
        print(f"{case['name']}: {'FAIL' if errors else 'PASS'} "
              f"method={summary.method} status={summary.status} errors={errors}", flush=True)
        if analysis.degradation_reason in {"missing_api_key", "provider_error"}:
            break
    passed = sum(row["passed"] for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "total": len(CASES), "executed": len(rows), "passed": passed, "cases": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={passed}/{len(CASES)} report={output}")
    return passed == len(CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval-design/query-rewrite-smoke.json"))
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output)) else 1)


if __name__ == "__main__":
    main()
