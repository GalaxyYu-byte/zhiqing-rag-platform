import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AuthenticationError
from pydantic import ValidationError

from zq_rag_app.query_analysis.models import QueryAnalysisRequest, RetrievalProposal
from zq_rag_app.query_analysis.policy import select_strategy
from zq_rag_app.query_analysis.rewrite import RewriteCandidate, validate_candidate
from zq_rag_app.services.query_rewrite_service import QueryRewriteService


class Client:
    def __init__(self, responses, delay=0):
        self.responses = list(responses)
        self.calls = []
        self.delay = delay
        self.closed = False
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(self.delay)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=value),
        )])

    async def close(self):
        self.closed = True


def candidate(query="Aurora-KB 的负责人是谁？", **kwargs):
    return {
        "status": "rewritten", "query": query, "hypothetical_document": None,
        "clarification_question": None, "operations": ["retrieval_normalization"],
        "context_sources": [], "reason": "检索标准化", **kwargs,
    }


def audit(**kwargs):
    return {
        "equivalent": True, "intent_preserved": True, "constraints_preserved": True,
        "no_unsupported_additions": True, "context_resolved": True, "reason": "语义一致",
        **kwargs,
    }


@pytest.mark.asyncio
async def test_rewrite_is_audited_json_and_does_not_own_injected_client(analysis_factory):
    analysis = analysis_factory(rewrite=True, names=["Aurora-KB"])
    client = Client([candidate(), audit()])
    service = QueryRewriteService(client=client)
    result = await service.rewrite(QueryAnalysisRequest(query=analysis.original_query), analysis)
    assert result.summary.status == "rewritten"
    assert result.summary.retrieval_query == candidate()["query"]
    assert result.hypothetical_document is None
    assert len(client.calls) == 2
    assert all(c["response_format"] == {"type": "json_object"} for c in client.calls)
    assert all(c["extra_body"] == {"thinking": {"type": "disabled"}} for c in client.calls)
    await service.aclose()
    assert not client.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["{}", "```json\n{}\n```", '{"query":"x","query":"y"}'])
async def test_invalid_candidate_retries(analysis_factory, bad):
    analysis = analysis_factory(rewrite=True)
    client = Client([bad, candidate(), audit()])
    result = await QueryRewriteService(client=client).rewrite(
        QueryAnalysisRequest(query=analysis.original_query), analysis,
    )
    assert result.summary.attempts == 2 and result.summary.status == "rewritten"
    assert "correction" in json.loads(client.calls[1]["messages"][1]["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", ["equivalent", "intent_preserved", "constraints_preserved",
                                      "no_unsupported_additions", "context_resolved"])
async def test_audit_rejection_discards_candidate(analysis_factory, flag):
    analysis = analysis_factory(rewrite=True)
    client = Client([candidate(), audit(**{flag: False})])
    result = await QueryRewriteService(client=client, retry_attempts=1).rewrite(
        QueryAnalysisRequest(query=analysis.original_query), analysis,
    )
    assert result.summary.status == "degraded"
    assert result.summary.retrieval_query == analysis.original_query
    assert result.hypothetical_document is None


@pytest.mark.parametrize("text", [
    "HT-2025-001 的金额超过 10 美元吗？",
    "HT-2025-001 的金额超过 10 元吗？",
    "HT-2025-002 的金额不超过 10 元吗？",
    "HT-2025-001 的金额不超过 100 元吗？",
])
def test_identifier_negation_amount_and_unit_cannot_be_changed(analysis_factory, text):
    query = "HT-2025-001 的金额不超过 10 元吗？"
    analysis = analysis_factory(query=query, rewrite=True, names=["HT-2025-001"])
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(query=text)), QueryAnalysisRequest(query=query), analysis)


def test_units_are_checked_even_when_analyzer_misses_metadata(analysis_factory):
    query = "预算超过 10 元吗？"
    analysis = analysis_factory(query=query, rewrite=True)
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(query="预算超过 10 美元吗？")),
                           QueryAnalysisRequest(query=query), analysis)


def test_numeric_token_not_preserved_by_substring_of_other_number(analysis_factory):
    query = "100 号记录金额超过 10 吗？"
    analysis = analysis_factory(query=query, rewrite=True)
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(query="100 号记录金额超过 100 吗？")),
                           QueryAnalysisRequest(query=query), analysis)


@pytest.mark.asyncio
async def test_keyword_optimization_keeps_named_term(analysis_factory):
    query = "RAG 的资料分块怎么弄？"
    analysis = analysis_factory(query=query, rewrite=True, names=["RAG"])
    analysis.retrieval_strategy.rewrite_operations = ["keyword_optimization"]
    updated = "RAG 的文档分块（文本切分）方法是什么？"
    result = await QueryRewriteService(client=Client([candidate(
        query=updated, operations=["keyword_optimization"],
    ), audit()])).rewrite(QueryAnalysisRequest(query=query), analysis)
    assert result.summary.operations == ("keyword_optimization",)
    assert result.summary.retrieval_query == updated


def test_metadata_constraint_and_unsupported_operation_rejected(analysis_factory):
    query = "2025 年之前未验收的项目记录"
    analysis = analysis_factory(query=query, rewrite=True)
    from zq_rag_app.query_analysis.models import MetadataConstraint
    analysis.metadata = [MetadataConstraint(
        field="time", operator="lt", value="2025 年之前", source="query", history_index=None,
        evidence_quote="2025 年之前", confidence=0.9,
    )]
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(query="2025 年之后未验收的项目记录是什么？")),
                           QueryAnalysisRequest(query=query), analysis)
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(query=query + "是什么？", operations=["keyword_optimization"])),
                           QueryAnalysisRequest(query=query), analysis)


@pytest.mark.asyncio
async def test_context_completion_uses_user_quote_and_current_conditions(analysis_factory):
    query = "它去年未验收的情况呢？"
    analysis = analysis_factory(query=query, reference=True, rewrite=True)
    analysis.retrieval_strategy.rewrite_operations = ["context_completion", "constraint_explicitization"]
    client = Client([candidate(
        query="Aurora-KB 去年未验收的情况是什么？",
        operations=["context_completion", "constraint_explicitization"],
        context_sources=[{"history_index": 0, "evidence_quote": "Aurora-KB"}],
    ), audit()])
    request = QueryAnalysisRequest(query=query, history=[
        {"role": "user", "content": "Aurora-KB 今年验收了吗？"},
        {"role": "assistant", "content": "负责人是未经确认的甲。"},
    ])
    result = await QueryRewriteService(client=client).rewrite(request, analysis)
    assert result.summary.status == "rewritten"
    data = json.loads(client.calls[0]["messages"][1]["content"])
    assert data["request"]["history"] == [{"history_index": 0, "content": request.history[0].content}]
    assert "今年" not in result.summary.retrieval_query


@pytest.mark.parametrize("role,reference", [("assistant", True), ("user", False)])
def test_assistant_facts_and_unrelated_history_are_rejected(analysis_factory, role, reference):
    analysis = analysis_factory(query="它是谁？", reference=reference, rewrite=True)
    analysis.retrieval_strategy.rewrite_operations = ["context_completion"]
    request = QueryAnalysisRequest(query="它是谁？", history=[{"role": role, "content": "Aurora-KB"}])
    with pytest.raises(ValueError):
        validate_candidate(RewriteCandidate(**candidate(
            query="Aurora-KB 是谁？", operations=["context_completion"],
            context_sources=[{"history_index": 0, "evidence_quote": "Aurora-KB"}],
        )), request, analysis)


@pytest.mark.asyncio
async def test_missing_key_context_failure_requests_clarification(analysis_factory):
    analysis = analysis_factory(query="它是谁？", reference=True, rewrite=True)
    result = await QueryRewriteService(api_key="").rewrite(
        QueryAnalysisRequest(query="它是谁？", history=[{"role": "user", "content": "Aurora-KB"}]), analysis,
    )
    assert result.summary.degradation_reason == "missing_api_key"
    assert result.clarification_question and result.summary.attempts == 0


@pytest.mark.asyncio
async def test_total_timeout_covers_semantic_audit(analysis_factory):
    analysis = analysis_factory(rewrite=True)
    class SlowAuditClient(Client):
        async def create(self, **kwargs):
            self.delay = 1 if self.calls else 0
            return await super().create(**kwargs)
    client = SlowAuditClient([candidate(), audit()])
    result = await QueryRewriteService(client=client, timeout_seconds=0.2).rewrite(
        QueryAnalysisRequest(query=analysis.original_query), analysis,
    )
    assert len(client.calls) == 2
    assert result.summary.degradation_reason == "timeout"


@pytest.mark.asyncio
async def test_auth_error_does_not_retry_or_log_body(analysis_factory, caplog):
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.deepseek.com"))
    client = Client([AuthenticationError("secret-provider-body", response=response, body=None)])
    analysis = analysis_factory(rewrite=True)
    result = await QueryRewriteService(client=client).rewrite(QueryAnalysisRequest(query=analysis.original_query), analysis)
    assert result.summary.degradation_reason == "provider_error"
    assert len(client.calls) == 1 and "secret-provider-body" not in caplog.text


@pytest.mark.asyncio
async def test_cancellation_is_not_fallback(analysis_factory):
    analysis = analysis_factory(rewrite=True)
    service = QueryRewriteService(client=Client([candidate()], delay=10))
    task = asyncio.create_task(service.rewrite(QueryAnalysisRequest(query=analysis.original_query), analysis))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_hyde_question_unchanged_and_document_audited(analysis_factory):
    query = "知识零散导致答案偏离重点该怎么改善？"
    analysis = analysis_factory(query=query, intent="recommendation")
    analysis.retrieval_strategy.rewrite_method = "hyde"
    analysis.retrieval_strategy.use_hyde = True
    doc = "知识检索可采用语义召回、词法匹配及相关性重排，组织分散证据以聚焦问题。"
    client = Client([candidate(query=query, hypothetical_document=doc, operations=[]), audit()])
    result = await QueryRewriteService(client=client).rewrite(QueryAnalysisRequest(query=query), analysis)
    assert result.summary.method == "hyde" and result.summary.status == "rewritten"
    assert result.summary.retrieval_query == query and result.hypothetical_document == doc


@pytest.mark.parametrize("kwargs", [
    {"query_type": "exact"}, {"names": ["Aurora-KB"]}, {"intent": "statistics"},
    {"path": "hybrid_graph", "names": ["Aurora-KB"]},
    {"query": "去年为什么失败？"}, {"query": "如何排查 503 错误？"},
])
def test_hyde_disabled_for_constraints_relations_and_business_entities(analysis_factory, kwargs):
    analysis = analysis_factory(intent="explanation", **{k: v for k, v in kwargs.items() if k != "intent"})
    if "intent" in kwargs:
        analysis.intent.primary = kwargs["intent"]
    analysis.retrieval_strategy.rewrite_method = "hyde"
    analysis.retrieval_strategy.use_hyde = True
    strategy, _ = select_strategy(analysis, QueryAnalysisRequest(query=analysis.original_query))
    assert not strategy.use_hyde and strategy.rewrite_method == "none"


def test_rewrite_and_hyde_are_exclusive():
    with pytest.raises(ValidationError):
        RetrievalProposal(path="hybrid", rewrite_method="hyde", rewrite_operations=[],
                          use_query_rewrite=True, use_hyde=True, use_multi_query=False, reason="冲突")
