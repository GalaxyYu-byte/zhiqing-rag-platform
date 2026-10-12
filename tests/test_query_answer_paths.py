"""回答模型等待期间的路径撤权、投影退休及桥接出处完整性。"""

import hashlib

import pytest
from test_query_answering import Provider, draft
from test_query_graph_paths import path_hits, two_hop_plan
from test_query_retrieval_service import FakeRepository, identity, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_answering.providers import AnswerProvider, ModelResult, chain_payload
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.fusion import FusedCandidate
from zhiqing_rag.query_retrieval.models import EvidenceScores, RetrievalResult, RetrievalTrace
from zhiqing_rag.query_retrieval.service import HybridRetrievalService


def paths_retrieval():
    hits = path_hits()
    evidence = tuple(
        HybridRetrievalService._evidence(
            FusedCandidate(hit.chunk, EvidenceScores(fusion_score=0.1), hit.graph_relations), index
        )
        for index, hit in enumerate(hits, 1)
    )
    return RetrievalResult(
        status="ok", evidence=evidence, graph_paths=hits[0].graph_paths,
        trace=RetrievalTrace(graph_status="succeeded", graph_count=2,
                             graph_path_count=1, graph_hops=2),
    )


class Graph:
    def __init__(self, answers=None, *, error=None, on_check=None):
        self.answers = answers or [frozenset((10, 12))] * 3
        self.error, self.on_check = error, on_check
        self.calls = []

    async def revalidate_relations(self, user, access, params, records):
        self.calls.append(records)
        if self.on_check:
            self.on_check()
        if self.error:
            raise self.error
        return self.answers[min(len(self.calls) - 1, len(self.answers) - 1)]


async def answer_paths(*, graph=None, repository=None, retrieval=None, query_plan=None,
                       max_hops=None):
    value = draft(statements=[{
        "text": "步骤需要表单，表单包含字段。", "citations": [
            {"citation_id": "E1", "quote": "步骤需要表单。"},
            {"citation_id": "E2", "quote": "表单包含字段。"},
        ],
    }])
    provider = Provider(value)
    service = QueryAnswerService(
        repository or FakeRepository(), provider, Settings(_env_file=None),
        graph_service=graph or Graph(),
    )
    query_plan = query_plan or two_hop_plan()
    query_plan = query_plan.model_copy(update={"params": query_plan.params.model_copy(
        update={"seed_entity_uids": ("step",), **(
            {"max_hops": max_hops} if max_hops is not None else {}
        )}
    )})
    output = await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=query_plan,
        retrieval=retrieval or paths_retrieval(),
    )
    return (*output, provider)


async def test_complete_paths_have_ready_checks_before_and_after_model_calls():
    graph = Graph()
    retrieval, answer, provider = await answer_paths(graph=graph)
    assert answer.status == "answered" and len(retrieval.graph_paths) == 1
    assert len(graph.calls) == 3
    assert all({row.chunk_id for row in rows} == {10, 12} for rows in graph.calls)
    assert [name for name, _ in provider.calls] == ["generate", "review"]


@pytest.mark.parametrize("at", [0, 1, 2])
async def test_projection_retired_during_answer_drops_path_but_keeps_readable_fact_answer(at):
    graph = Graph([frozenset((10, 12))] * at + [frozenset((10,))])
    retrieval, answer, _ = await answer_paths(graph=graph)
    assert answer.status == "answered" and len(retrieval.evidence) == 2
    assert retrieval.status == "degraded" and not retrieval.graph_paths
    assert retrieval.trace.graph_path_count == 0 and retrieval.trace.graph_hops == 1
    assert not retrieval.evidence[1].graph_relations
    assert "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes


async def test_graph_provenance_failure_strips_metadata_and_retains_authorized_raw_sources():
    retrieval, answer, _ = await answer_paths(graph=Graph(error=TimeoutError()))
    assert answer.status == "answered" and len(retrieval.evidence) == 2
    assert not retrieval.graph_paths
    assert all(not item.graph_relations for item in retrieval.evidence)
    assert retrieval.trace.graph_status == "failed" and retrieval.trace.graph_hops == 0
    assert "GRAPH_PROVENANCE_UNAVAILABLE" in retrieval.degradation_codes


async def test_graph_authorization_failure_is_not_a_soft_metadata_failure():
    with pytest.raises(AuthError) as error:
        await answer_paths(graph=Graph(error=AuthError(401, "SESSION_REVOKED", "退出")))
    assert error.value.code == "SESSION_REVOKED"


async def test_raw_revocation_while_ready_check_waits_is_checked_before_generation():
    class Repository(FakeRepository):
        revoked = False

        async def revalidate(self, user, access, records):
            current = await super().revalidate(user, access, records)
            return tuple(row for row in current if row.chunk_id != 12) if self.revoked else current

    repo = Repository()
    graph = Graph(on_check=lambda: setattr(repo, "revoked", True))
    retrieval, answer, _ = await answer_paths(graph=graph, repository=repo)
    assert not retrieval.graph_paths and {item.chunk_id for item in retrieval.evidence} == {"10"}
    assert answer.status in ("rejected", "insufficient") and not answer.statements
    assert "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes


def test_removed_hop_source_cannot_survive_in_final_with_evidence_path_metadata():
    value = paths_retrieval()
    result = QueryAnswerService._with_evidence(value, value.evidence[:1], tenant_id=1)
    assert not result.graph_paths and result.trace.graph_path_count == 0
    assert "GRAPH_PROVENANCE_CHANGED" in result.degradation_codes


def test_hash_or_quote_forgery_drops_path_even_if_required_ids_are_present():
    value = paths_retrieval()
    path = value.graph_paths[0]
    bad = path.model_copy(update={"hops": (
        path.hops[0].model_copy(update={"source": path.hops[0].source.model_copy(
            update={"source_quote": "隐藏桥接内容"}
        )}), path.hops[1],
    )})
    result = QueryAnswerService._with_evidence(
        value.model_copy(update={"graph_paths": (bad,)}), value.evidence, tenant_id=1
    )
    assert not result.graph_paths


async def test_unplanned_projection_id_cannot_be_kept_using_same_readable_chunk_id():
    value = paths_retrieval()
    path = value.graph_paths[0]
    bad = path.model_copy(update={"hops": (
        path.hops[0].model_copy(update={"source": path.hops[0].source.model_copy(
            update={"projection_id": "999"}
        )}), path.hops[1],
    )})
    graph = Graph()
    result, answer, _ = await answer_paths(
        retrieval=value.model_copy(update={"graph_paths": (bad,)}), graph=graph
    )
    assert answer.status == "answered" and not result.graph_paths
    assert not graph.calls
    assert "GRAPH_PROVENANCE_CHANGED" in result.degradation_codes


def three_source_retrieval():
    value = paths_retrieval()
    path = value.graph_paths[0]
    content = "申请表是该步骤所需表单的正式名称。"
    bridge_source = path.hops[1].source.model_copy(update={
        "chunk_id": "14", "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "source_quote": content, "quote_start": 0, "quote_end": len(content),
    })
    path = path.model_copy(update={
        "bridges": (path.bridges[0].model_copy(update={
            "sources": (*path.bridges[0].sources, bridge_source),
        }),),
        "required_chunk_ids": ("10", "12", "14"),
    })
    alias = value.evidence[1].model_copy(update={
        "chunk_id": "14", "citation_id": "E3", "content": content,
        "content_sha256": bridge_source.content_sha256, "graph_relations": (),
    })
    return value.model_copy(update={"graph_paths": (path,), "evidence": (*value.evidence, alias)})


def path_draft(value, *, include_alias):
    return draft(statements=[{
        "text": "步骤需要表单，表单包含字段。",
        "citations": [{"citation_id": item.citation_id, "quote": item.content}
                      for item in value.evidence[:3 if include_alias else 2]],
    }])


class PathProvider(Provider):
    def __init__(self, value, repaired, on_repair=None):
        super().__init__(value)
        self.repaired, self.on_repair = repaired, on_repair

    async def generate_with_paths(self, question, evidence, paths):
        self.calls.append(("generate_paths", paths))
        return ModelResult(self.value)

    async def repair(self, *args):
        raise AssertionError("路径修复应调用受限来源组接口")

    async def repair_with_paths(self, question, evidence, value, code, paths):
        assert code == "ANSWER_GRAPH_PATH_NOT_CITED" and len(paths) == 1
        self.calls.append(("repair_paths", paths))
        if self.on_repair:
            self.on_repair()
        return ModelResult(self.repaired)

    async def review_with_paths(self, question, evidence, value, paths):
        self.calls.append(("review_paths", paths))
        return ModelResult(self.review_value)


async def execute_three_source(provider, repository=None):
    value = three_source_retrieval()
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    graph = Graph([frozenset((10, 12, 14))] * 6)
    service = QueryAnswerService(
        repository or FakeRepository(), provider, Settings(_env_file=None), graph_service=graph
    )
    return await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=value,
    )


async def test_missing_third_bridge_quote_requires_one_repair_before_review():
    value = three_source_retrieval()
    provider = PathProvider(path_draft(value, include_alias=False),
                            path_draft(value, include_alias=True))
    retrieval, answer = await execute_three_source(provider)
    assert answer.status == "answered" and answer.trace.repair_count == 1
    assert retrieval.graph_paths and answer.statements[0].citations[2].citation_id == "E3"
    assert [name for name, _ in provider.calls] == [
        "generate_paths", "repair_paths", "review_paths",
    ]


async def test_incomplete_bridge_repair_is_rejected_even_if_review_would_support_it():
    value = three_source_retrieval()
    incomplete = path_draft(value, include_alias=False)
    provider = PathProvider(incomplete, incomplete)
    _, answer = await execute_three_source(provider)
    assert answer.status == "rejected" and answer.codes == ("ANSWER_GRAPH_PATH_NOT_CITED",)
    assert not answer.statements and answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate_paths", "repair_paths"]


async def test_revoked_alias_source_during_path_repair_cannot_be_published():
    class Repository(FakeRepository):
        revoked = False

        async def revalidate(self, user, access, records):
            current = await super().revalidate(user, access, records)
            return tuple(row for row in current if row.chunk_id != 14) if self.revoked else current

    value, repo = three_source_retrieval(), Repository()
    provider = PathProvider(
        path_draft(value, include_alias=False), path_draft(value, include_alias=True),
        on_repair=lambda: setattr(repo, "revoked", True),
    )
    retrieval, answer = await execute_three_source(provider, repo)
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",) and not answer.statements
    assert not retrieval.graph_paths
    assert {item.chunk_id for item in retrieval.evidence} == {"10", "12"}
    assert [name for name, _ in provider.calls] == ["generate_paths", "repair_paths"]


async def test_one_hop_provider_keeps_original_interfaces_without_graph_path_io():
    graph = Graph(error=AssertionError("一跳回答不应复查两跳路径"))
    value = paths_retrieval().model_copy(update={"graph_paths": ()})
    _, answer, provider = await answer_paths(graph=graph, retrieval=value)
    assert answer.status == "answered" and not graph.calls
    assert [name for name, _ in provider.calls] == ["generate", "review"]


async def test_path_provider_sends_only_citation_groups_and_current_original_evidence():
    class Capturing(AnswerProvider):
        calls = None

        async def _call(self, prompt, payload, schema, temperature):
            self.calls = self.calls or []
            self.calls.append(payload)
            if schema.__name__ == "SelectedRepairAnswer":
                return ModelResult(schema.model_validate({
                    "status": "answered", "statements": [{
                        "text": "步骤需要表单，表单包含字段。", "citations": [
                            {"citation_id": item.citation_id, "quote_candidate_index": -1}
                            for item in value.evidence
                        ],
                    }],
                }))
            return ModelResult(path_draft(value, include_alias=True))

    value = three_source_retrieval()
    provider = Capturing(Settings(_env_file=None))
    generated = await provider.generate_with_paths("问题", value.evidence, value.graph_paths)
    await provider.review_with_paths("问题", value.evidence, generated.value, value.graph_paths)
    await provider.repair_with_paths(
        "问题", value.evidence, generated.value, "ANSWER_GRAPH_PATH_NOT_CITED", value.graph_paths
    )
    assert chain_payload(value.evidence, value.graph_paths) == [
        {"required_citation_ids": ["E1", "E2", "E3"]},
    ]
    for payload in provider.calls:
        assert payload["evidence_chains"] == [{"required_citation_ids": ["E1", "E2", "E3"]}]
        assert all("graph_relations" not in item for item in payload["evidence"])


def test_citation_ids_without_alias_original_quote_do_not_prove_path_coverage():
    from zhiqing_rag.query_answering.support import path_citations_complete

    value = three_source_retrieval()
    answer = path_draft(value, include_alias=True)
    answer.statements[0].citations[2] = answer.statements[0].citations[2].model_copy(
        update={"quote": "正式名称"}
    )
    assert not path_citations_complete(answer, value.evidence, value.graph_paths)


async def test_one_hop_plan_cannot_authorize_supplied_two_hop_metadata():
    graph = Graph()
    retrieval, answer, _ = await answer_paths(graph=graph, max_hops=1)
    assert answer.status == "answered" and not retrieval.graph_paths
    assert not graph.calls and "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes


async def test_last_ready_check_cannot_keep_an_uncited_replacement_path():
    value = three_source_retrieval()
    original = value.graph_paths[0]
    replacement_source = original.bridges[0].sources[-1].model_copy(update={"chunk_id": "16"})
    replacement = original.model_copy(update={
        "path_id": "replacement", "required_chunk_ids": ("10", "12", "16"),
        "bridges": (original.bridges[0].model_copy(update={
            "sources": (*original.bridges[0].sources[:-1], replacement_source),
        }),),
    })
    alias = value.evidence[-1].model_copy(update={"chunk_id": "16", "citation_id": "E4"})
    retrieval = value.model_copy(update={
        "evidence": (*value.evidence, alias), "graph_paths": (original, replacement),
    })
    provider = Provider(path_draft(value, include_alias=True))
    graph = Graph([frozenset((10, 12, 14, 16))] * 2 + [frozenset((10, 12, 16))])
    service = QueryAnswerService(
        FakeRepository(), provider, Settings(_env_file=None), graph_service=graph
    )
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    final, answer = await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=retrieval
    )
    assert answer.status == "answered" and not final.graph_paths
    assert len(final.evidence) == 4 and final.trace.graph_path_count == 0
    assert "GRAPH_PROVENANCE_CHANGED" in final.degradation_codes


async def test_review_rejection_also_strips_retired_bridge_metadata():
    from zhiqing_rag.query_answering.models import AnswerReview

    value = three_source_retrieval()
    review = AnswerReview.model_validate({
        "sufficient": False, "missing_aspects": ["其他必要信息"],
        "statements": [{"index": 0, "supported": False, "reason": "不足"}],
    })
    provider = Provider(path_draft(value, include_alias=True), review=review)
    graph = Graph([frozenset((10, 12, 14))] * 2 + [frozenset((10, 12))])
    service = QueryAnswerService(
        FakeRepository(), provider, Settings(_env_file=None), graph_service=graph
    )
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    final, answer = await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=value
    )
    assert answer.codes == ("ANSWER_REVIEW_FAILED",) and not final.graph_paths
    assert final.trace.graph_path_count == 0 and len(final.evidence) == 3


def terse_path_draft(value):
    return draft(statements=[{
        "text": "步骤需要表单，表单包含字段。", "citations": [
            {"citation_id": "E1", "quote": "步骤"},
            {"citation_id": "E2", "quote": "字段"},
            {"citation_id": "E3", "quote": "正式名称"},
        ],
    }])


def test_explicit_complete_source_group_expands_only_selected_quotes_not_facts_or_ids():
    value = three_source_retrieval()
    original = terse_path_draft(value)
    checked = QueryAnswerService._validate_draft(original, value.evidence, paths=value.graph_paths)
    assert checked.statements[0].text == original.statements[0].text
    assert [c.citation_id for c in checked.statements[0].citations] == ["E1", "E2", "E3"]
    assert [c.quote for c in checked.statements[0].citations] == [
        item.content for item in value.evidence
    ]


def test_a_missing_bridge_cid_is_never_added_by_reference_expansion():
    from zhiqing_rag.query_answering.support import expand_selected_path_quotes

    value = three_source_retrieval()
    original = path_draft(value, include_alias=False)
    expanded = expand_selected_path_quotes(original, value.evidence, value.graph_paths)
    assert expanded == original


def test_invalid_original_quote_cannot_be_rescued_by_valid_graph_source_ranges():
    from zhiqing_rag.query_answering.providers import InvalidAnswer

    value = three_source_retrieval()
    original = terse_path_draft(value)
    original.statements[0].citations[0] = original.statements[0].citations[0].model_copy(
        update={"quote": "伪造步骤引用"}
    )
    with pytest.raises(InvalidAnswer) as error:
        QueryAnswerService._validate_draft(original, value.evidence, paths=value.graph_paths)
    assert error.value.code == "ANSWER_QUOTE_NOT_VERBATIM"


def long_alias_retrieval(length):
    value = three_source_retrieval()
    content = "开头说明。" + "中" * length + "。不能把发送通知视为完成交接。申请表是正式名称。"
    alias = value.evidence[2].model_copy(update={
        "content": content, "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
    })
    path = value.graph_paths[0]
    source = path.bridges[0].sources[2].model_copy(update={
        "content_sha256": alias.content_sha256, "source_quote": content,
        "quote_start": 0, "quote_end": len(content),
    })
    path = path.model_copy(update={"bridges": (path.bridges[0].model_copy(update={
        "sources": (*path.bridges[0].sources[:2], source),
    }),)})
    return value.model_copy(update={
        "evidence": (*value.evidence[:2], alias), "graph_paths": (path,),
    })


def test_contiguous_bridge_reference_expansion_preserves_intervening_negation():
    value = long_alias_retrieval(20)
    checked = QueryAnswerService._validate_draft(
        terse_path_draft(value), value.evidence, paths=value.graph_paths
    )
    quote = checked.statements[0].citations[2].quote
    assert quote == value.evidence[2].content
    assert "不能把发送通知视为完成交接" in quote


def test_bridge_span_above_two_thousand_characters_is_not_silently_cropped():
    from zhiqing_rag.query_answering.providers import InvalidAnswer

    value = long_alias_retrieval(2000)
    with pytest.raises(InvalidAnswer) as error:
        QueryAnswerService._validate_draft(
            terse_path_draft(value), value.evidence, paths=value.graph_paths
        )
    assert error.value.code == "ANSWER_GRAPH_PATH_NOT_CITED"


def test_path_quote_expansion_never_moves_citations_between_statements():
    from zhiqing_rag.query_answering.support import expand_selected_path_quotes

    value = three_source_retrieval()
    original = draft(statements=[{
        "text": "步骤需要表单。", "citations": [{"citation_id": "E1", "quote": "步骤"}],
    }, {
        "text": "表单包含字段。", "citations": [
            {"citation_id": "E2", "quote": "字段"}, {"citation_id": "E3", "quote": "正式名称"},
        ],
    }])
    checked = expand_selected_path_quotes(original, value.evidence, value.graph_paths)
    assert [[c.citation_id for c in statement.citations] for statement in checked.statements] == [
        ["E1"], ["E2", "E3"],
    ]
    assert [statement.text for statement in checked.statements] == [
        statement.text for statement in original.statements
    ]


@pytest.mark.parametrize("bad", ["offset", "hash"])
def test_unverified_path_ranges_cannot_expand_an_existing_quote(bad):
    from zhiqing_rag.query_answering.support import expand_selected_path_quotes

    value = three_source_retrieval()
    path = value.graph_paths[0]
    source = path.bridges[0].sources[2].model_copy(update={
        "quote_start": 1,
    } if bad == "offset" else {"content_sha256": "0" * 64})
    path = path.model_copy(update={"bridges": (path.bridges[0].model_copy(update={
        "sources": (*path.bridges[0].sources[:2], source),
    }),)})
    original = terse_path_draft(value)
    assert expand_selected_path_quotes(original, value.evidence, (path,)) == original


async def test_review_input_budget_counts_server_expanded_references_before_external_call():
    value = long_alias_retrieval(1600)
    provider = Provider(terse_path_draft(value))
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    service = QueryAnswerService(FakeRepository(), provider, Settings(
        _env_file=None, query_answer_context_tokens=2500,
    ), graph_service=Graph([frozenset((10, 12, 14))] * 3))
    _, answer = await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=value
    )
    assert answer.codes == ("ANSWER_CONTEXT_LIMIT",) and not answer.statements
    assert [name for name, _ in provider.calls] == ["generate"]


async def test_expanded_supported_answer_is_discarded_when_alias_acl_changes_during_review():
    class Repository(FakeRepository):
        revoked = False

        async def revalidate(self, user, access, records):
            current = await super().revalidate(user, access, records)
            return tuple(row for row in current if row.chunk_id != 14) if self.revoked else current

    class Reviewer(Provider):
        async def review(self, query, evidence, value):
            assert value.statements[0].citations[2].quote == evidence[2].content
            repo.revoked = True
            return await super().review(query, evidence, value)

    value, repo = three_source_retrieval(), Repository()
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    service = QueryAnswerService(repo, Reviewer(terse_path_draft(value)), Settings(_env_file=None),
                                 graph_service=Graph([frozenset((10, 12, 14))] * 3))
    retrieval, answer = await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=value
    )
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",) and not answer.statements
    assert not retrieval.graph_paths
    assert {item.chunk_id for item in retrieval.evidence} == {"10", "12"}
