"""完整路径引用不能掩盖正文漏条件；条件可分句，但不能借别来源或改变否定/时序。"""

import hashlib

import pytest
from test_query_answer_paths import Graph, PathProvider, paths_retrieval
from test_query_answering import draft
from test_query_graph_paths import two_hop_plan
from test_query_retrieval_service import FakeRepository, identity, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.query_answering.providers import build_repair_payload, chain_payload
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_answering.support import (
    graph_condition_diagnostics,
    verified_path_conditions,
)

FIRST = "审批完成后"
SECOND = "接收确认后"


def conditional_retrieval(*, second_condition="", alternative=False):
    value = paths_retrieval()
    path = value.graph_paths[0]
    first = FIRST + "，步骤需要表单。"
    if alternative:
        first += "复核通过后，步骤也可使用表单。"
    content = (first, (second_condition + "，" if second_condition else "") + "表单包含字段。")
    sources, hops, evidence = [], [], []
    for index, (hop, item, text, condition) in enumerate(zip(
        path.hops, value.evidence, content, (FIRST, second_condition), strict=True
    )):
        source = hop.source.model_copy(update={
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(), "source_quote": text,
            "quote_start": 0, "quote_end": len(text),
        })
        relation = hop.relation.model_copy(update={
            "condition_text": condition, "source_quote": text, "quote_start": 0,
            "quote_end": len(text),
        })
        sources.append(source)
        hops.append(hop.model_copy(update={"source": source, "relation": relation}))
        evidence.append(item.model_copy(update={"content": text, "graph_relations": (relation,)}))
    path = path.model_copy(update={
        "hops": tuple(hops),
        "bridges": (path.bridges[0].model_copy(update={"sources": tuple(sources)}),),
    })
    paths = [path]
    if alternative:
        relation = hops[0].relation.model_copy(update={"condition_text": "复核通过后"})
        other = path.model_copy(update={"path_id": "alternative", "hops": (
            hops[0].model_copy(update={"relation": relation}), hops[1],
        )})
        evidence[0] = evidence[0].model_copy(update={
            "graph_relations": (*evidence[0].graph_relations, relation),
        })
        paths.append(other)
    return value.model_copy(update={"evidence": tuple(evidence), "graph_paths": tuple(paths)})


def full_draft(value, text):
    return draft(statements=[{"text": text, "citations": [
        {"citation_id": item.citation_id, "quote": item.content} for item in value.evidence
    ]}])


def test_payload_conditions_are_only_literal_current_source_metadata():
    value = conditional_retrieval(second_condition=SECOND)
    conditions = verified_path_conditions(value.evidence, value.graph_paths[0])
    assert [row["condition_text"] for row in conditions] == [FIRST, SECOND]
    assert [row["citation_id"] for row in conditions] == ["E1", "E2"]
    by_id = {item.citation_id: item for item in value.evidence}
    assert all(by_id[row["citation_id"]].content[row["start"]:row["end"]]
               == row["condition_text"] for row in conditions)
    payload = chain_payload(value.evidence, value.graph_paths)
    assert payload[0]["conditions"] == list(conditions)
    assert "predicate" not in str(payload) and "subject_entity_uid" not in str(payload)


@pytest.mark.parametrize("change", [
    {"content_sha256": "0" * 64}, {"document_id": "99"}, {"revision_id": "99"},
    {"generation_id": "99"}, {"knowledge_base_id": "99"}, {"quote_start": 1},
])
def test_foreign_or_stale_source_cannot_provide_conditions(change):
    value = conditional_retrieval()
    path = value.graph_paths[0]
    path = path.model_copy(update={"hops": (
        path.hops[0].model_copy(update={"source": path.hops[0].source.model_copy(update=change)}),
        path.hops[1],
    )})
    assert verified_path_conditions(value.evidence, path) is None
    assert not chain_payload(value.evidence, (path,))


@pytest.mark.parametrize("text", [
    "步骤需要表单，表单包含字段。",
    "无需审批完成后，步骤需要表单，表单包含字段。",
    "并非审批完成后，步骤需要表单，表单包含字段。",
    "未在审批完成后，步骤需要表单，表单包含字段。",
    "审批完成后之前，步骤需要表单，表单包含字段。",
    "审批完成后不是必要条件，步骤需要表单，表单包含字段。",
    "审批完成后并不是适用前提，步骤需要表单，表单包含字段。",
    "审批完成后这一要求不适用，步骤需要表单，表单包含字段。",
    "审批完成后不用再受此条件限制，步骤需要表单，表单包含字段。",
    "并不要求审批完成后才能填写表单，表单包含字段。",
])
def test_missing_negated_or_changed_timing_condition_is_rejected_despite_full_quotes(text):
    from zhiqing_rag.query_answering.providers import InvalidAnswer

    value = conditional_retrieval()
    answer = full_draft(value, text)
    diagnostics = graph_condition_diagnostics(answer, value.evidence, value.graph_paths)
    assert diagnostics[0]["missing_conditions"][0]["condition_text"] == FIRST
    assert diagnostics[0]["missing_conditions"][0]["source_statement_indices"] == [0]
    with pytest.raises(InvalidAnswer) as error:
        QueryAnswerService._validate_draft(answer, value.evidence, paths=value.graph_paths)
    assert error.value.code == "ANSWER_GRAPH_CONDITION_NOT_PRESERVED"


def test_conditional_second_hop_cannot_be_omitted_when_first_condition_is_present():
    value = conditional_retrieval(second_condition=SECOND)
    answer = full_draft(value, FIRST + "，步骤需要表单，表单包含字段。")
    rows = graph_condition_diagnostics(answer, value.evidence, value.graph_paths)
    assert [row["condition_text"] for row in rows[0]["missing_conditions"]] == [SECOND]
    assert not graph_condition_diagnostics(
        full_draft(value, FIRST + "，步骤需要表单；" + SECOND + "，表单包含字段。"),
        value.evidence, value.graph_paths,
    )


def test_condition_may_be_in_separate_statement_with_own_hop_source():
    value = conditional_retrieval()
    answer = draft(statements=[
        {"text": FIRST + "，步骤需要表单。", "citations": [
            {"citation_id": "E1", "quote": value.evidence[0].content}]},
        {"text": "表单包含字段。", "citations": [
            {"citation_id": "E2", "quote": value.evidence[1].content}]},
    ])
    assert not graph_condition_diagnostics(answer, value.evidence, value.graph_paths)
    assert QueryAnswerService._validate_draft(answer, value.evidence, paths=value.graph_paths)
    foreign = answer.model_copy(update={"statements": (
        answer.statements[0].model_copy(update={"text": "步骤需要表单。"}),
        answer.statements[1].model_copy(update={"text": FIRST + "，表单包含字段。"}),
    )})
    assert graph_condition_diagnostics(foreign, value.evidence, value.graph_paths)


def test_unrelated_third_statement_condition_cannot_replace_own_hop_source():
    value = conditional_retrieval()
    unrelated = value.evidence[1].model_copy(update={
        "citation_id": "E3", "chunk_id": "14", "document_id": "99",
        "content": FIRST + "，另一事项需要核对。", "graph_relations": (),
    })
    answer = full_draft(value, "步骤需要表单，表单包含字段。")
    extra = draft(statements=[{"text": unrelated.content, "citations": [
        {"citation_id": "E3", "quote": unrelated.content}]}]).statements[0]
    answer = answer.model_copy(update={"statements": (*answer.statements, extra)})
    assert graph_condition_diagnostics(answer, (*value.evidence, unrelated), value.graph_paths)


def test_one_complete_alternative_path_is_enough_and_diagnostics_never_union_conditions():
    value = conditional_retrieval(alternative=True)
    answer = full_draft(value, "复核通过后，步骤需要表单，表单包含字段。")
    assert not graph_condition_diagnostics(answer, value.evidence, value.graph_paths)
    missing = graph_condition_diagnostics(full_draft(value, "步骤需要表单，表单包含字段。"),
                                          value.evidence, value.graph_paths)
    assert len(missing) == 1 and len(missing[0]["missing_conditions"]) == 1
    assert missing[0]["missing_conditions"][0]["condition_text"] in {FIRST, "复核通过后"}


def test_direct_fields_without_used_path_keep_existing_behavior():
    value = conditional_retrieval()
    answer = draft(statements=[{"text": "表单包含字段。", "citations": [
        {"citation_id": "E2", "quote": value.evidence[1].content}]}])
    assert not graph_condition_diagnostics(answer, value.evidence, value.graph_paths)
    assert QueryAnswerService._validate_draft(answer, value.evidence)


async def execute_conditional(provider, *, repository=None, retrieval=None, graph=None):
    value = retrieval or conditional_retrieval()
    plan = two_hop_plan()
    plan = plan.model_copy(update={"params": plan.params.model_copy(
        update={"seed_entity_uids": ("step",)}
    )})
    service = QueryAnswerService(repository or FakeRepository(), provider,
                                 Settings(_env_file=None), graph_service=graph or Graph())
    return await service.execute(
        identity=identity(), scope=scope(second_document=True), plan=plan, retrieval=value,
    )


class ConditionProvider(PathProvider):
    async def repair_with_paths(self, question, evidence, value, code, paths):
        assert code == "ANSWER_GRAPH_CONDITION_NOT_PRESERVED"
        self.payload = build_repair_payload(question, evidence, value, code, paths)
        self.calls.append(("repair_paths", paths))
        if self.on_repair:
            self.on_repair()
        from zhiqing_rag.query_answering.providers import ModelResult

        return ModelResult(self.repaired)


@pytest.mark.parametrize("fixed", [True, False])
async def test_condition_failure_gets_only_one_repair_and_final_guard(fixed):
    value = conditional_retrieval()
    missing = full_draft(value, "步骤需要表单，表单包含字段。")
    repaired = full_draft(value, FIRST + "，步骤需要表单，表单包含字段。") if fixed else missing
    provider = ConditionProvider(missing, repaired)
    _, answer = await execute_conditional(provider)
    assert answer.status == ("answered" if fixed else "rejected")
    assert answer.trace.repair_count == 1
    assert provider.payload["graph_condition_diagnostics"] == graph_condition_diagnostics(
        missing, value.evidence, value.graph_paths
    )
    assert [name for name, _ in provider.calls] == ["generate_paths", "repair_paths"] + (
        ["review_paths"] if fixed else []
    )


async def test_revocation_during_condition_repair_never_publishes_answer():
    value = conditional_retrieval()
    repository = FakeRepository()
    provider = ConditionProvider(full_draft(value, "步骤需要表单，表单包含字段。"),
                                 full_draft(value, FIRST + "，步骤需要表单，表单包含字段。"),
                                 on_repair=lambda: setattr(repository, "validations", [()]))
    retrieval, answer = await execute_conditional(provider, repository=repository)
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",)
    assert not answer.statements and not retrieval.evidence and not retrieval.graph_paths
    assert [name for name, _ in provider.calls] == ["generate_paths", "repair_paths"]


async def test_final_retired_chosen_path_cannot_leave_unfulfilled_alternative_metadata():
    value = conditional_retrieval()
    original = value.graph_paths[0]
    text = SECOND + "，步骤需要表单。"
    source = original.hops[0].source.model_copy(update={
        "chunk_id": "14", "source_quote": text, "quote_end": len(text),
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    })
    relation = original.hops[0].relation.model_copy(update={
        "assertion_id": "other-assertion", "condition_text": SECOND,
        "source_quote": text, "quote_end": len(text),
    })
    other = original.model_copy(update={
        "path_id": "other-path", "required_chunk_ids": ("14", "12"),
        "hops": (original.hops[0].model_copy(update={"source": source, "relation": relation}),
                 original.hops[1]),
        "bridges": (original.bridges[0].model_copy(update={
            "sources": (source, original.hops[1].source),
        }),),
    })
    item = value.evidence[0].model_copy(update={
        "chunk_id": "14", "citation_id": "E3", "content": text,
        "graph_relations": (relation,),
    })
    value = value.model_copy(update={"evidence": (*value.evidence, item),
                                      "graph_paths": (original, other)})
    answer = full_draft(value, FIRST + "，步骤需要表单，表单包含字段。")
    provider = ConditionProvider(answer, answer)
    graph = Graph([frozenset((10, 12, 14))] * 2 + [frozenset((12, 14))])
    retrieval, output = await execute_conditional(provider, retrieval=value, graph=graph)
    assert output.status == "answered" and output.trace.repair_count == 0
    assert retrieval.status == "degraded" and not retrieval.graph_paths
    assert "GRAPH_PROVENANCE_CHANGED" in retrieval.degradation_codes
    assert [name for name, _ in provider.calls] == ["generate_paths", "review_paths"]
