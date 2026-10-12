"""状态与职责负例使用真实语义差异，模型 supported=true 不能放行。"""

import json
from pathlib import Path

import pytest
from test_query_answering import Provider, draft, execute, retrieved

from zhiqing_rag.query_answering.models import AnswerReview, DraftAnswer
from zhiqing_rag.query_answering.providers import evidence_payload
from zhiqing_rag.query_answering.support import (
    protected_clauses,
    semantic_guard_failure,
    source_scope_ambiguous,
)
from zhiqing_rag.query_retrieval.models import Evidence


def claimed(text, quote, *, citation_id="E1"):
    return draft(
        statements=[
            {"text": text, "citations": [{"citation_id": citation_id, "quote": quote}]}
        ]
    )


@pytest.mark.parametrize(
    "source,text",
    [
        (
            "未经确认的转派不能把工单视为已解决。",
            "未经确认的转派不能视为完成交接。",
        ),
        (
            "未经确认的领用不能把资产视为已移交。",
            "未经确认的领用不能视为审批完成。",
        ),
        ("报销审核通过不代表付款完成。", "报销审核通过代表付款完成。"),
        ("报销审核通过不代表付款完成。", "报销审核通过，付款已完成。"),
        ("报销审核通过不代表付款完成。", "报销审核通过，付款没有完成。"),
        ("报销审核通过不代表付款完成。", "报销审核通过后已经付款。"),
        ("报销审核通过不代表付款完成。", "报销审核通过后尚未付款。"),
        ("发送通知不能把事件视为已关闭。", "发送通知后事件已关闭。"),
    ],
)
async def test_state_replacement_or_non_implication_cannot_reach_model_review(source, text):
    provider = Provider(claimed(text, source))
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "rejected" and not answer.statements
    assert answer.codes == ("ANSWER_SEMANTIC_GUARD_FAILED",)
    assert [name for name, _ in provider.calls] == ["generate"]


@pytest.mark.parametrize(
    "text",
    [
        "原受理人仍负责向客户通知状态。",
        "接收方确认前，原受理人仍负责向客户通知状态。",
        "接收方确认后，原受理人不再负责向客户通知状态。",
        "接收方确认后，接手团队仍负责向客户通知状态。",
        "原受理人继续向客户通知状态。",
    ],
)
async def test_conditional_responsibility_retains_condition_polarity_and_actor(text):
    source = "接收方确认后，原受理人仍负责向客户通知状态。"
    provider = Provider(claimed(text, source))
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "rejected" and not answer.statements
    assert answer.codes == ("ANSWER_SEMANTIC_GUARD_FAILED",)


@pytest.mark.parametrize(
    "source,quote,text",
    [
        (
            "如果责任角色没有确定，不要把发送通知当成已经完成交接。",
            "不要把发送通知当成已经完成交接",
            "不要把发送通知当成已经完成交接。",
        ),
        (
            "审批完成后，部门管理员负责开通账号。",
            "部门管理员负责开通账号",
            "部门管理员负责开通账号。",
        ),
    ],
)
async def test_verbatim_suffix_citation_cannot_hide_the_source_condition(source, quote, text):
    provider = Provider(claimed(text, quote))
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "rejected" and not answer.statements
    assert answer.codes == ("ANSWER_SEMANTIC_GUARD_FAILED",)


@pytest.mark.parametrize(
    "source",
    [
        "未经确认的转派不能把工单视为已解决。",
        "报销审核通过不代表付款完成。",
        "接收方确认后，原受理人仍负责向客户通知状态。",
        "如果责任角色没有确定，不要把发送通知当成已经完成交接。",
        "审批完成后，部门管理员负责开通账号。",
    ],
)
async def test_complete_protected_source_clause_is_reviewed_and_published(source):
    provider = Provider(claimed(source, source))
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "answered" and answer.review_status == "passed"
    assert [name for name, _ in provider.calls] == ["generate", "review"]


async def test_unrelated_non_implication_need_not_be_recited_in_information_answer():
    source = "申请表包含用途、金额和收款人。报销审核通过不代表付款完成。"
    value = claimed("申请表包含用途、金额和收款人。", "申请表包含用途、金额和收款人。")
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"


async def test_positive_status_has_separate_explicit_evidence_and_is_not_inferred_from_rule():
    source = "报销审核通过不代表付款完成。财务记录显示付款已完成。"
    value = claimed("财务记录显示付款已完成。", "财务记录显示付款已完成。")
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"


async def test_safe_clause_cannot_cover_an_extra_unsupported_status_in_same_sentence():
    source = "未经确认的转派不能把工单视为已解决。"
    value = claimed("未经确认的转派不能把工单视为已解决，所以转派已经完成。", source)
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "rejected" and not answer.statements


def test_protected_clause_offsets_are_locatable_in_source_and_payload_preserves_raw_text():
    source = "说明\n\n接收方确认后，原受理人仍负责向客户通知状态。报销审核通过不代表付款完成。"
    evidence = retrieved(source).evidence[0]
    clauses = protected_clauses(evidence)
    assert [item.kind for item in clauses] == ["conditional_responsibility", "non_implication"]
    for item in clauses:
        assert source[item.start:item.end] == item.text
    payload = evidence_payload((evidence,))[0]
    assert payload["content"] == source
    assert payload["protected_clauses"] == [item.payload() for item in clauses]


@pytest.mark.parametrize("case_id", ["Q11", "Q12"])
def test_historical_supported_model_review_cannot_mask_state_replacement(case_id):
    # 复放真正发布过的草稿；该历史文件不可改写成新的成功样本。
    path = (
        Path(__file__).resolve().parents[1]
        / "eval/query_retrieval/runs/20261010-delivery-acceptance/results.json"
    )
    row = next(
        item for item in json.loads(path.read_text(encoding="utf-8"))["results"]
        if item["id"] == case_id
    )
    evidence = tuple(
        Evidence.model_validate(item) for item in row["response"]["retrieval"]["evidence"]
    )
    value = DraftAnswer.model_validate(row["captured_model_outputs"][0]["value"])
    review = AnswerReview.model_validate(row["captured_model_outputs"][1]["value"])
    assert review.sufficient and all(item.supported for item in review.statements)
    assert semantic_guard_failure(value, evidence)


def test_no_cross_statement_or_uncited_source_support_for_state_claim():
    evidence = retrieved("发送通知不能把事件视为已关闭。").evidence[0]
    other = evidence.model_copy(update={"citation_id": "E2", "content": "事件已经关闭。"})
    value = claimed("事件已经关闭。", evidence.content)
    assert semantic_guard_failure(value, (evidence, other))


async def test_source_condition_guard_survives_unique_pdf_line_break_restoration():
    source = "接收方确认后，原受理人仍负责\n向客户通知状态。"
    quote = source.replace("\n", "")
    value = claimed(quote, quote)
    retrieval, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"
    assert answer.statements[0].citations[0].quote == source
    assert retrieval.evidence[0].content == source


async def test_condition_guard_does_not_bypass_final_authorization():
    from test_query_retrieval_service import FakeRepository

    source = "接收方确认后，原受理人仍负责向客户通知状态。"
    repository = FakeRepository(validations=[lambda records: records, lambda records: records, ()])
    retrieval, answer = await execute(
        repository=repository,
        provider=Provider(claimed(source, source)),
        retrieval=retrieved(source),
    )
    assert answer.status == "rejected" and not answer.statements and not retrieval.evidence
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",)


async def test_independent_recording_action_need_not_be_added_to_protected_state_rule():
    source = "尚缺少必要确认时，记录当前进展和下一步动作，不要把发送邮件直接视为办结。"
    value = claimed("尚缺少必要确认时，不能把发送邮件直接视为办结。", source)
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"
    assert "记录" not in answer.statements[0].text


def test_table_quote_candidates_are_individual_verbatim_rows_never_header_row_splices():
    source = "| 项目 | 状态 |\n| --- | --- |\n| 甲 | 待确认 |\n| 乙 | 已完成 |"
    payload = evidence_payload(retrieved(source).evidence)[0]
    candidates = payload["quote_candidates"]
    assert len(candidates) == 4
    for item in candidates:
        assert source[item["start"]:item["end"]] == item["text"]
        assert "\n" not in item["text"]


async def test_prior_result_received_is_not_same_as_handover_completed():
    source = (
        "如果责任角色没有确定，不要把发送通知当成已经完成交接。"
        "交接动作进入下一步前，应确认上一环节的结果已经被接收。"
    )
    value = claimed("交接动作进入下一步前，应确认上一环节的结果已经被接收。", source)
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"


async def test_determine_who_is_responsible_does_not_assign_specific_conditional_actor():
    source = "接收前应看\n----------\n启动这项工作时，先确认事项范围，再确定谁负责发起和核对。"
    value = claimed("开始工作前需要明确发起和核对的责任角色。", source)
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(source))
    assert answer.status == "answered"
    assert not protected_clauses(retrieved(source).evidence[0])


def test_protected_clause_does_not_force_a_markdown_heading_into_an_answer():
    source = "接收后做什么\n----------\n接收方确认后，原受理人仍负责向客户通知状态。"
    item = retrieved(source).evidence[0]
    rule = protected_clauses(item)[0]
    assert rule.answer_text == "接收方确认后，原受理人仍负责向客户通知状态"
    assert source[rule.start:rule.end] == rule.text
    assert not semantic_guard_failure(claimed(rule.answer_text, rule.text), (item,))


def test_protected_payload_is_bounded_without_three_duplicate_source_copies():
    source = "审批完成后，部门管理员负责开通账号。" * 20
    payload = evidence_payload(retrieved(source).evidence)[0]
    assert all(set(rule) == {"kind", "start", "end", "answer_text"}
               for rule in payload["protected_clauses"])
    assert sum(len(item["text"]) for item in payload["quote_candidates"]) <= 600


def test_table_does_not_borrow_a_different_rows_condition_for_unassigned_role_question():
    source = (
        "| 工作状态 | 进入条件 | 离开该状态前核对 |\n| --- | --- | --- |\n"
        "| 待客户确认 | 确认时间待定时 | 是否取得确认或有后续安排 |\n"
        "| 已关闭 | 处理结果被接收 | 遗留项是否有人负责 |"
    )
    item = retrieved(source).evidence[0]
    assert not protected_clauses(item)
    value = claimed(
        "需要核对遗留项是否有人负责。", "| 已关闭 | 处理结果被接收 | 遗留项是否有人负责 |"
    )
    assert not semantic_guard_failure(value, (item,))


def test_table_cell_own_conditional_responsibility_and_state_judgement_stay_protected():
    source = (
        "| 工作状态 | 要求 |\n| --- | --- |\n"
        "| 移交 | 接收方确认后，原受理人仍负责通知客户 |\n"
        "| 注意 | 审核通过不代表付款完成 |"
    )
    item = retrieved(source).evidence[0]
    assert [rule.kind for rule in protected_clauses(item)] == [
        "conditional_responsibility", "non_implication",
    ]
    assert semantic_guard_failure(claimed("原受理人仍负责通知客户。", source), (item,))
    for rule in protected_clauses(item):
        assert source[rule.start:rule.end] == rule.text


def test_prose_attached_to_table_does_not_lose_its_own_condition_or_negation():
    source = "| 项目 | 状态 |\n| --- | --- |\n| 甲 | 待确认 |\n审核通过不代表\n付款完成"
    item = retrieved(source).evidence[0]
    assert [rule.kind for rule in protected_clauses(item)] == ["non_implication"]
    assert semantic_guard_failure(claimed("付款已完成。", source), (item,))


class RepairProvider(Provider):
    def __init__(self, value, repaired, *, error=None):
        super().__init__(value)
        self.repaired, self.repair_error = repaired, error

    async def repair(self, query, evidence, value, code):
        from zhiqing_rag.query_answering.providers import ModelResult

        self.calls.append(("repair", evidence))
        if self.repair_error:
            raise self.repair_error
        assert code in (
            "ANSWER_QUOTE_NOT_VERBATIM", "ANSWER_SEMANTIC_GUARD_FAILED",
            "ANSWER_SOURCE_SCOPE_AMBIGUOUS",
            "ANSWER_CONFIRMATION_GUARD_FAILED",
        )
        return ModelResult(self.repaired, 8, 6)


@pytest.mark.parametrize("bad_quote", [False, True])
async def test_one_repair_attempt_keeps_all_checks_and_counts_usage(bad_quote):
    source = "未经确认的转派不能把工单视为已解决。"
    value = claimed("未经确认的转派不能视为完成交接。", "拼接无效引用" if bad_quote else source)
    provider = RepairProvider(value, claimed(source, source))
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "answered" and answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate", "repair", "review"]
    assert (answer.trace.input_tokens, answer.trace.output_tokens) == (40, 24)


async def test_repair_is_not_retried_and_a_second_bad_state_never_reaches_review():
    source = "未经确认的转派不能把工单视为已解决。"
    value = claimed("未经确认的转派不能视为完成交接。", source)
    provider = RepairProvider(value, value)
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "rejected" and not answer.statements
    assert answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate", "repair"]


async def test_source_revoked_before_repair_prevents_second_external_call():
    from test_query_retrieval_service import FakeRepository

    source = "未经确认的转派不能把工单视为已解决。"
    provider = RepairProvider(claimed("转派完成。", "无效引用"), claimed(source, source))
    repo = FakeRepository(validations=[lambda rows: rows, lambda rows: rows, ()])
    retrieval, answer = await execute(
        repository=repo, provider=provider, retrieval=retrieved(source)
    )
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",) and not retrieval.evidence
    assert answer.trace.repair_count == 0
    assert [name for name, _ in provider.calls] == ["generate"]


def list_evidence():
    first = retrieved("新增版本时，至少说明变更位置、影响对象和需要谁确认。")
    first_item = first.evidence[0].model_copy(update={
        "document_title": "文档命名与版本约定.pdf", "document_name": "文档命名与版本约定.pdf",
    })
    second_item = first_item.model_copy(update={
        "citation_id": "E2", "chunk_id": "12", "document_id": "6",
        "generation_id": "9", "document_title": "发布记录编写指南.md",
        "document_name": "发布记录编写指南.md",
        "content": "记录包含构建标识、适用环境、负责人角色、变更说明和已知限制。",
    })
    return first_item, second_item


def mixed_list(evidence):
    return draft(statements=[{
        "text": "版本记录包含变更位置、影响对象、确认人、构建标识、适用环境、负责人和限制。",
        "citations": [{"citation_id": item.citation_id, "quote": item.content}
                      for item in evidence],
    }])


def test_independent_cross_topic_lists_do_not_become_one_generic_list():
    evidence = list_evidence()
    assert source_scope_ambiguous("文档版本需要包含哪些信息？", mixed_list(evidence), evidence)


def test_splitting_same_objects_incompatible_lists_into_statements_cannot_bypass_guard():
    first, second = list_evidence()
    evidence = (first, second)
    value = draft(statements=[{
        "text": "文档版本记录应至少说明变更位置、影响对象和需要谁确认。",
        "citations": [{"citation_id": "E1", "quote": first.content}],
    }, {
        "text": "文档版本记录还应保存构建标识、适用环境、负责人角色、变更说明和限制。",
        "citations": [{"citation_id": "E2", "quote": second.content}],
    }])
    assert source_scope_ambiguous("文档版本记录包含哪些信息？", value, evidence)


def test_same_named_list_cross_statement_supplements_keep_same_topic_sources():
    first, second = list_evidence()
    second = second.model_copy(update={"document_title": "文档命名与版本约定补充.md"})
    value = draft(statements=[{
        "text": "文档版本记录应至少说明变更位置、影响对象和需要谁确认。",
        "citations": [{"citation_id": "E1", "quote": first.content}],
    }, {
        "text": "文档版本记录也应保存构建标识、适用环境、负责人角色、变更说明和限制。",
        "citations": [{"citation_id": "E2", "quote": second.content}],
    }])
    assert not source_scope_ambiguous("文档版本记录包含哪些信息？", value, (first, second))


def test_review_evidence_keeps_full_facts_and_protection_without_selection_duplicates():
    from zhiqing_rag.query_answering.providers import review_evidence_payload

    source = "确认接收后，原受理人仍负责通知客户。审核通过不代表付款完成。"
    payload = review_evidence_payload(retrieved(source).evidence)[0]
    assert payload["content"] == source and len(payload["protected_clauses"]) == 2
    assert "quote_candidates" not in payload and "full_source_available" not in payload


@pytest.mark.parametrize("question", [
    "汇总这两份文档规定的全部信息项。", "对照不同文档包含的信息项。",
])
def test_explicit_aggregate_information_question_allows_cross_topic_lists(question):
    evidence = list_evidence()
    assert not source_scope_ambiguous(question, mixed_list(evidence), evidence)


def test_same_topic_supplement_from_another_document_can_extend_a_list():
    first, second = list_evidence()
    evidence = (first, second.model_copy(update={
        "document_title": "文档版本字段补充说明.md",
        "content": "版本记录包含发布人员、修改日期、适用期限。",
    }))
    evidence = (evidence[0].model_copy(update={
        "document_title": "文档版本字段要求.pdf",
    }), evidence[1])
    assert not source_scope_ambiguous(
        "文档版本包含哪些信息？", mixed_list(evidence), evidence
    )


def test_parallel_questions_keep_each_independent_topic_in_its_own_statement():
    evidence = list_evidence()
    value = draft(statements=[{
        "text": item.content,
        "citations": [{"citation_id": item.citation_id, "quote": item.content}],
    } for item in evidence])
    assert not source_scope_ambiguous("文档版本和发布记录各包含哪些信息？", value, evidence)


def test_complete_path_in_another_statement_cannot_bypass_cross_topic_list_guard():
    from test_query_answer_paths import path_draft, paths_retrieval

    graph = paths_retrieval()
    unrelated = tuple(item.model_copy(update={
        "citation_id": f"E{index}", "chunk_id": str(20 + index),
        "document_id": str(30 + index),
    }) for index, item in enumerate(list_evidence(), 3))
    graph_answer = path_draft(graph, include_alias=False)
    value = draft(statements=[*graph_answer.statements, *mixed_list(unrelated).statements])
    assert source_scope_ambiguous(
        "步骤表单和文档版本包含哪些信息？", value, (*graph.evidence, *unrelated), graph.graph_paths
    )


def test_extra_unrelated_list_in_a_complete_path_statement_is_still_checked():
    from test_query_answer_paths import paths_retrieval

    graph = paths_retrieval()
    first, extra = list_evidence()
    source = graph.evidence[0].model_copy(update={"content": first.content,
        "document_title": first.document_title})
    extra = extra.model_copy(update={"citation_id": "E3", "chunk_id": "14", "document_id": "8"})
    path = graph.graph_paths[0]
    first_source = path.hops[0].source.model_copy(update={"source_quote": first.content})
    path = path.model_copy(update={
        "hops": (path.hops[0].model_copy(update={"source": first_source}), path.hops[1]),
        "bridges": (path.bridges[0].model_copy(update={
            "sources": (first_source, path.hops[1].source),
        }),),
    })
    evidence = (source, graph.evidence[1], extra)
    value = mixed_list(evidence)
    assert source_scope_ambiguous("步骤材料包含哪些信息？", value, evidence, (path,))


def test_same_statement_full_path_permits_its_own_cross_document_lists():
    from test_query_answer_paths import paths_retrieval

    graph = paths_retrieval()
    evidence = tuple(original.model_copy(update={
        "content": listed.content, "document_title": listed.document_title,
    }) for original, listed in zip(graph.evidence, list_evidence(), strict=True))
    path = graph.graph_paths[0]
    sources = tuple(hop.source.model_copy(update={"source_quote": item.content})
                    for hop, item in zip(path.hops, evidence, strict=True))
    path = path.model_copy(update={
        "hops": tuple(hop.model_copy(update={"source": source})
                      for hop, source in zip(path.hops, sources, strict=True)),
        "bridges": (path.bridges[0].model_copy(update={"sources": sources}),),
    })
    assert not source_scope_ambiguous(
        "步骤材料包含哪些信息？", mixed_list(evidence), evidence, (path,)
    )


async def test_mixed_scope_list_gets_only_one_repair_and_does_not_trust_review_flag():
    from test_query_retrieval_service import FakeRepository, identity, plan, scope

    from zhiqing_rag.core.config import Settings
    from zhiqing_rag.query_answering.service import QueryAnswerService
    from zhiqing_rag.query_retrieval.models import RetrievalResult

    evidence = list_evidence()
    provider = RepairProvider(
        mixed_list(evidence), claimed(evidence[0].content, evidence[0].content)
    )
    service = QueryAnswerService(FakeRepository(), provider, Settings(_env_file=None))
    _, answer = await service.execute(
        identity=identity(), scope=scope(second_document=True),
        plan=plan().model_copy(update={"query": "文档版本需要包含哪些信息？"}),
        retrieval=RetrievalResult(status="ok", evidence=evidence),
    )
    assert answer.status == "answered" and answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate", "repair", "review"]


async def test_source_revoked_during_repair_discards_repaired_draft():
    from test_query_retrieval_service import FakeRepository

    source = "未经确认的转派不能把工单视为已解决。"
    provider = RepairProvider(claimed("转派完成。", "无效引用"), claimed(source, source))
    repo = FakeRepository(validations=[lambda rows: rows] * 3 + [()])
    retrieval, answer = await execute(
        repository=repo, provider=provider, retrieval=retrieved(source)
    )
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",) and not retrieval.evidence
    assert answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate", "repair"]


async def test_repair_context_budget_prevents_second_call_without_loosening_checks():
    source = "未经确认的转派不能把工单视为已解决。"
    value = claimed("责任" * 700, "无效引用")
    provider = RepairProvider(value, claimed(source, source))
    _, answer = await execute(
        provider=provider, retrieval=retrieved(source), query_answer_context_tokens=1000
    )
    assert answer.status == "rejected" and answer.trace.repair_count == 0
    assert [name for name, _ in provider.calls] == ["generate"]


def selected_repair(index, citation_id="E1", text="原文规则"):
    from zhiqing_rag.query_answering.models import SelectedRepairAnswer

    return SelectedRepairAnswer.model_validate({
        "status": "answered", "statements": [{
            "text": text, "citations": [{
                "citation_id": citation_id, "quote_candidate_index": index,
            }],
        }],
    })


@pytest.mark.parametrize("index", [-2, 10, 0.0, True, "0"])
def test_repair_selection_protocol_requires_valid_strict_integer(index):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        selected_repair(index)


def test_repair_selection_materializes_current_contiguous_candidate_only():
    from zhiqing_rag.query_answering.providers import ModelResult, materialize_repair

    evidence = retrieved("第一项已经确认。第二项仍待核对。").evidence
    result = materialize_repair(ModelResult(selected_repair(1), 4, 3), evidence)
    assert result.value.statements[0].citations[0].quote == "第二项仍待核对"
    assert (result.input_tokens, result.output_tokens) == (4, 3)


@pytest.mark.parametrize("index,citation_id", [(8, "E1"), (0, "E2")])
def test_repair_cannot_select_missing_candidate_or_unavailable_citation(index, citation_id):
    from zhiqing_rag.query_answering.providers import InvalidAnswer, ModelResult, materialize_repair

    with pytest.raises(InvalidAnswer):
        materialize_repair(
            ModelResult(selected_repair(index, citation_id)), retrieved("一段当前原文。").evidence
        )


@pytest.mark.parametrize("content", ["", "一", "长" * 2001])
def test_whole_source_fallback_rejects_empty_or_overlong_text(content):
    from zhiqing_rag.query_answering.providers import InvalidAnswer, ModelResult, materialize_repair

    source = retrieved("可读原文").evidence[0].model_copy(update={"content": content})
    with pytest.raises(InvalidAnswer):
        materialize_repair(ModelResult(selected_repair(-1)), (source,))


def selected_indices(citations):
    from zhiqing_rag.query_answering.models import SelectedRepairAnswer

    return SelectedRepairAnswer.model_validate({
        "status": "answered", "statements": [{"text": "原文中的信息项", "citations": citations}],
    })


def test_repair_repeated_indices_are_rejected_before_materializing():
    from pydantic import ValidationError

    citation = {"citation_id": "E1", "quote_candidate_index": 0}
    with pytest.raises(ValidationError):
        selected_indices([citation, citation])


def test_more_than_five_selected_spans_keep_intervening_negation_in_one_verbatim_quote():
    from zhiqing_rag.query_answering.providers import ModelResult, materialize_repair

    source = "甲项目。审核通过不代表付款完成。乙项目。丙项目。丁项目。戊项目。己项目。"
    indices = [{"citation_id": "E1", "quote_candidate_index": index}
               for index in (0, 2, 3, 4, 5, 6)]
    value = materialize_repair(ModelResult(selected_indices(indices)), retrieved(source).evidence)
    citations = value.value.statements[0].citations
    assert len(citations) == 1 and citations[0].quote in source
    assert "审核通过不代表付款完成" in citations[0].quote


def test_merging_candidate_ranges_cannot_bypass_final_quote_length_limit():
    from zhiqing_rag.query_answering.providers import InvalidAnswer, ModelResult, materialize_repair

    source = "甲项目。乙项目。" + "中间原文" * 600 + "。丙项目。丁项目。戊项目。己项目。"
    selected = selected_indices([{"citation_id": "E1", "quote_candidate_index": index}
                                 for index in range(6)])
    with pytest.raises(InvalidAnswer):
        materialize_repair(ModelResult(selected), retrieved(source).evidence)


def test_different_citation_sources_cannot_be_merged_to_avoid_final_limit():
    from zhiqing_rag.query_answering.providers import InvalidAnswer, ModelResult, materialize_repair

    original = retrieved("当前原文内容。").evidence[0]
    evidence = tuple(original.model_copy(update={"citation_id": f"E{index}"})
                     for index in range(1, 7))
    selected = selected_indices([{"citation_id": item.citation_id, "quote_candidate_index": 0}
                                 for item in evidence])
    with pytest.raises(InvalidAnswer):
        materialize_repair(ModelResult(selected), evidence)


async def test_selected_repair_preserves_state_guard_after_quote_materialization():
    from zhiqing_rag.query_answering.providers import ModelResult, materialize_repair

    source = "未经确认的转派不能把工单视为已解决。"
    evidence = retrieved(source).evidence
    repaired = materialize_repair(
        ModelResult(selected_repair(-1, text="未经确认的转派不能视为完成交接。")), evidence
    ).value
    provider = RepairProvider(claimed("错误引用", "无效引用"), repaired)
    _, answer = await execute(provider=provider, retrieval=retrieved(source))
    assert answer.status == "rejected" and answer.codes == ("ANSWER_SEMANTIC_GUARD_FAILED",)
    assert [name for name, _ in provider.calls] == ["generate", "repair"]


async def test_selected_repair_quote_is_not_published_after_source_revocation():
    from test_query_retrieval_service import FakeRepository

    from zhiqing_rag.query_answering.providers import ModelResult, materialize_repair

    source = "当前资料已经经过确认。"
    evidence = retrieved(source).evidence
    repaired = materialize_repair(
        ModelResult(selected_repair(-1, text=source)), evidence
    ).value
    provider = RepairProvider(claimed(source, "无效引用"), repaired)
    repo = FakeRepository(validations=[lambda rows: rows] * 3 + [()])
    _, answer = await execute(repository=repo, provider=provider, retrieval=retrieved(source))
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",) and not answer.statements
