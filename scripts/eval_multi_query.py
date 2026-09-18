"""真实 DeepSeek 等价扩展冒烟检查；模拟零结果触发，不访问数据库。"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from zq_rag_app.query_analysis.models import QueryAnalysisRequest
from zq_rag_app.services.multi_query_service import MultiQueryService
from zq_rag_app.services.query_analyzer_service import QueryAnalyzerService


CASES = [
    {"name": "analyzer_suggestion", "query": "星河项目验收卡住该怎么查原因？", "trigger": "analyzer",
     "expect_expanded": True, "contains": ["星河项目"], "allow_suggestion_override": True},
    {"name": "empty_retrieval", "query": "星河项目的验收审批流程是怎样的？", "trigger": "empty_retrieval",
     "expect_expanded": True, "contains": ["星河项目"]},
    {"name": "constraints", "query": "星河项目去年未验收的原因是什么？", "trigger": "empty_retrieval",
     "expect_expanded": True, "contains": ["星河项目", "去年", "未验收"]},
    {"name": "exact_identifier", "query": "HT-2025-001 的金额是多少？", "trigger": "empty_retrieval",
     "expect_expanded": False},
    {"name": "missing_reference", "query": "它是谁负责？", "trigger": "empty_retrieval",
     "expect_expanded": False},
    {"name": "hyde_not_combined", "query": "员工差旅报销审批步骤是什么？", "trigger": "empty_retrieval",
     "expect_expanded": False, "hyde_selected": True},
]


async def evaluate(output: Path) -> bool:
    analyzer, expander = QueryAnalyzerService(), MultiQueryService()
    rows = []
    try:
        for case in CASES:
            request = QueryAnalysisRequest(query=case["query"], reference_date="2026-09-18")
            analysis = await analyzer.analyze(request)
            method_source = "analyzer"
            analyzer_suggested = analysis.retrieval_strategy.use_multi_query
            if case.get("allow_suggestion_override") and not analyzer_suggested and analysis.status == "ok":
                # 专项验证 Analyzer 触发分支；明确记录覆盖，不宣称模型自主建议正确。
                analysis.retrieval_strategy.use_multi_query = True
                method_source = "test_override"
            result = await expander.expand(request, analysis, trigger=case["trigger"],
                                           hyde_selected=case.get("hyde_selected", False))
            errors = []
            if analysis.status != "ok":
                errors.append(f"analysis_degraded:{analysis.degradation_reason}")
            if case["expect_expanded"]:
                if result.status != "expanded" or not 2 <= len(result.queries) <= 3:
                    errors.append("expansion_not_completed")
                if any(fragment not in query for query in result.queries[1:] for fragment in case.get("contains", [])):
                    errors.append("lost_constraint")
            elif result.status != "skipped" or result.queries != (request.query,):
                errors.append("unsafe_expansion_not_skipped")
            rows.append({"name": case["name"], "passed": not errors, "errors": errors,
                         "method_source": method_source, "analyzer_suggested": analyzer_suggested,
                         "analysis": analysis.model_dump(mode="json"), "multi_query": asdict(result)})
            print(f"{case['name']}: {'FAIL' if errors else 'PASS'} status={result.status} "
                  f"queries={len(result.queries)} source={method_source} errors={errors}", flush=True)
            if analysis.degradation_reason in {"missing_api_key", "provider_error"}:
                break
    finally:
        await analyzer.aclose()
        await expander.aclose()
    passed = sum(row["passed"] for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"total": len(CASES), "executed": len(rows), "passed": passed, "cases": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={passed}/{len(CASES)} report={output}")
    return passed == len(CASES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("eval-design/multi-query-smoke.json"))
    args = parser.parse_args()
    raise SystemExit(0 if asyncio.run(evaluate(args.output)) else 1)


if __name__ == "__main__":
    main()
