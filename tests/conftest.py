"""在线编排测试使用固定分析结果，避免单元测试调用真实 LLM。"""

import pytest

from zq_rag_app.query_analysis.models import QueryAnalysisResult


@pytest.fixture
def analysis_factory():
    def make(*, query="Aurora-KB 谁负责？", path="hybrid", intent="fact",
             query_type="semantic", names=(), status="ok", reference=False,
             exhaustive=False, multi_hop=False, rewrite=False, multi_query=False):
        relation = path == "hybrid_graph" or query_type in {"relational", "multi_hop"}
        clarification = path == "clarify"
        return QueryAnalysisResult(
            intent={"primary": intent, "secondary": [], "confidence": 0.9},
            query_type={
                "primary": query_type, "confidence": 0.9,
                "has_context_reference": reference, "relation_required": relation,
                "potential_multi_hop": multi_hop,
                "requires_exhaustive": exhaustive,
                "needs_clarification": clarification,
                "clarification_question": "请明确项目名称。" if clarification else None,
            },
            entities=[{
                "name": name, "entity_type": "SYSTEM", "source": "query",
                "history_index": None, "evidence_quote": name, "confidence": 0.9,
            } for name in names],
            keywords=[], metadata=[],
            retrieval_strategy={
                "path": path, "use_query_rewrite": rewrite,
                "use_multi_query": multi_query, "use_hyde": False, "reason": "测试建议",
                "dense_weight": 0.4, "bm25_weight": 0.3, "graph_weight": 0.3,
                "graph_requires_resolution": path == "hybrid_graph",
                "requires_coverage_check": path in {"hybrid_graph", "structured"},
            },
            original_query=query, reference_date="2026-09-17", model="test",
            status=status, degradation_reason="timeout" if status == "degraded" else None,
            attempts=1, latency_ms=1, warnings=[],
        )
    return make


@pytest.fixture
def analyzer_stub_factory():
    def make(analysis, *, requests=None, closed=None):
        class Analyzer:
            async def analyze(self, request):
                if requests is not None:
                    requests.append(request)
                return analysis

            async def aclose(self):
                if closed is not None:
                    closed.append(True)
        return Analyzer
    return make
