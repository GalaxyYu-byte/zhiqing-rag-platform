"""同文档阶段原句与进度等待的真实边界，不由模型 supported 标记豁免。"""

import asyncio

import pytest
from test_query_answer_support import RepairProvider
from test_query_answering import Provider, draft, execute, retrieved
from test_query_retrieval_service import FakeRepository, identity, plan, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.query_answering.providers import evidence_payload
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_answering.support import (
    confirmation_clauses,
    confirmation_diagnostics,
    confirmation_guard_failure,
    general_confirmation_question,
)
from zhiqing_rag.query_retrieval.models import RetrievalResult

START = ("启动这项工作时，先确认事项确实落在办理人与接手团队的职责范围内，"
         "再确定谁负责发起、谁负责核对、谁负责接收最终结果。")
STEP = "第三步，选择接手团队并确认接收。"
TRANSITION = "进入下一步前，先确认上一环节的结果已经被接收。"
QUESTION = "事项转派需要满足什么确认条件？"


def source(text, cid="E1", **changes):
    return retrieved(text).evidence[0].model_copy(update={
        "citation_id": cid, "chunk_id": "10" if cid == "E1" else "11",
        "section_title": None, **changes,
    })


def confirmation_draft(text, evidence):
    return draft(statements=[{"text": text, "citations": [
        {"citation_id": item.citation_id, "quote": item.content} for item in evidence
    ]}])


async def run_confirmation(provider, evidence, *, question=QUESTION, repository=None):
    return await QueryAnswerService(
        repository or FakeRepository(), provider, Settings(_env_file=None)
    ).execute(identity=identity(), scope=scope(), plan=plan(query=question),
              retrieval=RetrievalResult(status="ok", evidence=evidence))


def test_stage_units_have_real_contiguous_offsets_and_exclude_failure_or_negative_rules():
    text = (START + "\n\n交接动作\n--------\n3) " + STEP + TRANSITION
            + "如果资料无法核对，应说明卡点，不能先标为完成。"
            + "未经确认的转派不能把事项视为已解决。"
            + "本事项以客户确认作为最后一道核对。"
            + "经办人应查看接收方能否继续工作，材料是否可找到，差异是否已分派。")
    evidence = source(text)
    units = confirmation_clauses(evidence)
    assert [unit.stage for unit in units] == ["start", "step", "transition", "finish", "finish"]
    assert all(evidence.content[unit.start:unit.end] == unit.text for unit in units)
    assert all("无法" not in unit.text and "不能" not in unit.text for unit in units)
    assert units[1].text == STEP.rstrip("。")


@pytest.mark.parametrize("question", [
    "事项转派需要满足什么确认条件？", "交接和关闭分别需要满足哪些确认要求？",
])
def test_general_confirmation_questions_trigger_stage_contract(question):
    assert general_confirmation_question(question)


@pytest.mark.parametrize("question", [
    "转派第三步需要满足什么确认条件？", "启动转派需要满足什么确认条件？",
    "工单关闭要满足什么客户确认或通知条件？", "交接后需要满足哪些确认要求？",
    "状态表各状态需要满足什么确认条件？", "申请表应包含哪些字段？",
])
def test_specific_stage_or_confirmation_kind_or_fields_do_not_require_full_workflow(question):
    assert not general_confirmation_question(question)


def test_payload_document_identity_is_exact_and_only_generic_question_has_stage_units():
    item = source(START)
    payload = evidence_payload((item,), QUESTION)[0]
    assert payload["source_document"] == [
        item.knowledge_base_id, item.document_id, item.revision_id, item.generation_id,
    ]
    assert payload["confirmation_clauses"][0]["stage"] == "start"
    assert not evidence_payload((item,), "第三步如何确认接收？")[0]["confirmation_clauses"]


async def test_same_document_other_chunk_startup_cannot_be_omitted_even_if_model_would_support():
    evidence = (source(START), source(STEP + TRANSITION, "E2"))
    value = confirmation_draft(STEP + TRANSITION, (evidence[1],))
    provider = Provider(value)
    _, answer = await run_confirmation(provider, evidence)
    assert answer.codes == ("ANSWER_CONFIRMATION_GUARD_FAILED",)
    assert [name for name, _ in provider.calls] == ["generate"]


async def test_one_repair_explicitly_selects_missing_phase_source_not_added_by_server():
    evidence = (source(START), source(STEP + TRANSITION, "E2"))
    old = confirmation_draft(STEP + TRANSITION, (evidence[1],))
    complete = confirmation_draft(START + STEP + TRANSITION, evidence)
    provider = RepairProvider(old, complete)
    _, answer = await run_confirmation(provider, evidence)
    assert answer.status == "answered" and answer.trace.repair_count == 1
    assert {c.citation_id for s in answer.statements for c in s.citations} == {"E1", "E2"}
    assert [name for name, _ in provider.calls] == ["generate", "repair", "review"]


async def test_repair_still_missing_phase_is_rejected_without_another_call():
    evidence = (source(START), source(STEP + TRANSITION, "E2"))
    old = confirmation_draft(STEP + TRANSITION, (evidence[1],))
    provider = RepairProvider(old, old)
    _, answer = await run_confirmation(provider, evidence)
    assert answer.codes == ("ANSWER_CONFIRMATION_GUARD_FAILED",)
    assert answer.trace.repair_count == 1
    assert [name for name, _ in provider.calls] == ["generate", "repair"]


@pytest.mark.parametrize("changed", [
    {"document_id": "6"}, {"knowledge_base_id": "99"}, {"revision_id": "12"},
    {"generation_id": "6"},
])
def test_same_title_other_source_identity_does_not_supply_or_force_roles(changed):
    first, other = source(STEP), source(START, "E2", **changed)
    value = confirmation_draft(STEP, (first,))
    assert not confirmation_guard_failure(QUESTION, value, (first, other))


def test_full_quote_without_startup_in_statement_still_fails_coverage():
    evidence = (source(START + STEP),)
    assert confirmation_guard_failure(QUESTION, confirmation_draft(STEP, evidence), evidence)


def test_same_document_explicit_different_workflow_does_not_force_other_roles():
    first = source("启动这项工作时，先确认采购预算。", section_title="采购流程")
    other = source("启动这项工作时，先确认休假额度。", "E2", section_title="请假流程")
    value = confirmation_draft("采购流程：" + first.content, (first,))
    assert not confirmation_guard_failure("采购需要满足什么确认条件？", value, (first, other))


def test_parallel_topics_in_same_document_keep_each_statement_requirements_separate():
    first = source("启动这项工作时，先确认采购预算。", section_title="采购流程")
    other = source("启动这项工作时，先确认休假额度。", "E2", section_title="请假流程")
    value = draft(statements=[
        {"text": "采购流程：" + first.content,
         "citations": [{"citation_id": "E1", "quote": first.content}]},
        {"text": "请假流程：" + other.content,
         "citations": [{"citation_id": "E2", "quote": other.content}]},
    ])
    assert not confirmation_guard_failure("采购和请假各自需要满足什么确认条件？",
                                          value, (first, other))


@pytest.mark.parametrize("section", [None, "采购流程"])
def test_inline_flow_headings_in_one_chunk_do_not_merge_other_workflow_startup(section):
    item = source("采购流程\n--------\n启动这项工作时，先确认采购预算。\n\n"
                  "请假流程\n--------\n启动这项工作时，先确认休假额度。", section_title=section)
    value = confirmation_draft("采购流程：启动这项工作时，先确认采购预算。", (item,))
    assert {unit.topic for unit in confirmation_clauses(item)} == {"采购流程", "请假流程"}
    assert not confirmation_guard_failure("采购需要满足什么确认条件？", value, (item,))


def test_full_quote_incidental_state_table_does_not_add_requirements_to_text():
    table = ("\n| 状态 | 进入条件 | 离开前核对 |\n| --- | --- | --- |\n"
             "| 待验证 | 已实施处理 | 客户侧是否能使用结果 |")
    evidence = (source(START + STEP + table),)
    value = confirmation_draft(START + STEP, evidence)
    assert not confirmation_guard_failure(QUESTION, value, evidence)


def test_other_state_table_check_cannot_expand_complete_prose_confirmation():
    table = source("| 状态 | 进入条件 | 离开前核对 |\n| --- | --- | --- |\n"
                   "| 待验证 | 已实施处理 | 客户侧是否能使用结果 |",
                   "E2", document_id="6", generation_id="9")
    prose = source("本事项以客户确认作为最后一道核对。", section_title="关闭条件")
    value = confirmation_draft(prose.content + "关闭确认还包括客户侧能使用结果。", (prose, table))
    assert confirmation_guard_failure("工单关闭需要满足什么确认条件？", value, (prose, table))


def test_negative_status_clause_may_be_separate_limit_but_not_positive_semicolon_item():
    item = source(STEP + "未经确认的转派不能把事项视为已解决。")
    unsafe = confirmation_draft("确认条件是：" + STEP.rstrip("。")
                               + "；未经确认的转派不能把事项视为已解决。", (item,))
    safe = confirmation_draft(STEP + "限制：未经确认的转派不能把事项视为已解决。", (item,))
    assert confirmation_guard_failure(QUESTION, unsafe, (item,))
    assert not confirmation_guard_failure(QUESTION, safe, (item,))


@pytest.mark.parametrize("stage,expected_calls", [("generation", []), ("review", ["generate"])])
async def test_progress_callback_wait_revocation_stops_next_external_call(stage, expected_calls):
    repository = FakeRepository()
    provider = Provider()
    entered, release = asyncio.Event(), asyncio.Event()

    async def progress(current):
        if current == stage:
            entered.set()
            await release.wait()

    task = asyncio.create_task(execute(repository=repository, provider=provider,
                                       on_progress=progress))
    await entered.wait()
    repository.validations = [()]
    release.set()
    retrieval, answer = await task
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",)
    assert not retrieval.evidence and not answer.statements
    assert [name for name, _ in provider.calls] == expected_calls


async def test_authorized_progress_callbacks_continue_and_skipped_branch_has_no_fake_stage():
    events = []

    async def progress(stage):
        events.append(stage)

    _, answer = await execute(on_progress=progress)
    assert answer.status == "answered" and events == ["generation", "review"]
    events.clear()
    retrieval = retrieved().model_copy(update={"status": "skipped", "evidence": ()})
    _, answer = await execute(retrieval=retrieval, on_progress=progress)
    assert answer.status == "skipped" and not events


@pytest.mark.parametrize("stage,expected_calls", [("generation", []), ("review", ["generate"])])
async def test_progress_fresh_payload_budget_blocks_external_call(stage, expected_calls):
    from dataclasses import replace

    repository, provider = FakeRepository(), Provider()

    async def progress(current):
        if current == stage:
            repository.validations = [lambda rows: tuple(
                replace(row, document_title="大标题" * 4000) for row in rows
            )]

    _, answer = await execute(repository=repository, provider=provider, on_progress=progress)
    assert answer.codes == ("ANSWER_CONTEXT_LIMIT",)
    assert answer.trace.context_tokens > 6000
    assert [name for name, _ in provider.calls] == expected_calls


async def test_repair_fresh_metadata_payload_cannot_bypass_budget_before_external_call():
    from dataclasses import replace

    repository = FakeRepository(validations=[lambda rows: rows, lambda rows: rows,
                                             lambda rows: tuple(replace(
                                                 row, document_title="大标题" * 4000
                                             ) for row in rows)])
    source_text = "有明确可读的连续原文。"
    provider = RepairProvider(confirmation_draft("直接答案", (source(source_text),)),
                              confirmation_draft("直接答案", (source(source_text),)))
    provider.value = draft(statements=[{"text": "直接答案", "citations": [
        {"citation_id": "E1", "quote": "不在原文中的引文"}
    ]}])
    _, answer = await execute(repository=repository, provider=provider,
                              retrieval=retrieved(source_text))
    assert answer.trace.context_tokens > 6000 and answer.trace.repair_count == 0
    assert [name for name, _ in provider.calls] == ["generate"]


def test_another_document_explicit_topic_does_not_hide_generic_transfer_startup():
    closing = source("本事项以客户确认作为最后一道核对。", section_title="关闭条件")
    transfer = source(STEP + TRANSITION, "E2", document_id="6", generation_id="9")
    startup = source(START, "E3", document_id="6", generation_id="9", chunk_id="12")
    value = confirmation_draft(closing.content + STEP + TRANSITION, (closing, transfer))
    assert confirmation_guard_failure("转派和关闭分别需要满足什么确认条件？", value,
                                      (closing, transfer, startup))


def test_finite_phase_frames_keep_core_conditions_roles_and_stage():
    text = START + STEP + TRANSITION + "本事项以客户确认作为最后一道核对。"
    item = source(text)
    answer_text = text.replace("启动这项工作时，", "启动时").replace("第三步，", "第三步为")
    answer_text = answer_text.replace("本事项以", "以")
    assert not confirmation_guard_failure(QUESTION, confirmation_draft(answer_text, (item,)),
                                          (item,))


@pytest.mark.parametrize("old,new", [
    ("启动时", "结束时"), ("第三步为", "第四步为"), ("先确认", "不用确认"),
    ("再确定谁负责发起", "不必确定谁负责发起"), ("办理人与接手团队", "管理员"),
    ("已经被接收", "尚未被接收"),
])
def test_finite_frames_never_remove_phase_order_polarity_or_roles(old, new):
    item = source(START + STEP + TRANSITION)
    safe = item.content.replace("启动这项工作时，", "启动时").replace("第三步，", "第三步为")
    value = confirmation_draft(safe.replace(old, new), (item,))
    assert confirmation_guard_failure(QUESTION, value, (item,))


def test_normative_material_requirement_is_not_treated_as_negative_state_constraint():
    item = source(STEP + "接收方应能找到输入和未结事项，交接不能只停留在发送文件。")
    text = "确认条件是：" + STEP.rstrip("。")
    text += "；接收方应能找到输入和未结事项，交接不能只停留在发送文件。"
    value = confirmation_draft(text, (item,))
    assert not confirmation_guard_failure(QUESTION, value, (item,))


def test_repair_payload_short_sources_only_offer_minus_one_long_sources_offer_real_indices():
    from zhiqing_rag.query_answering.providers import repair_evidence_payload

    short = source(START)
    long = source("可定位的首句。" + "长内容" * 800, "E2")
    first, second = repair_evidence_payload((short, long), QUESTION)
    assert first["allowed_quote_candidate_indices"] == [-1] and "quote_candidates" not in first
    assert second["allowed_quote_candidate_indices"] == [item["index"]
                                                       for item in second["quote_candidates"]]
    assert -1 not in second["allowed_quote_candidate_indices"]


async def test_provider_repair_uses_same_current_payload_helper_as_service_budget():
    from test_query_answer_support import selected_repair

    from zhiqing_rag.query_answering.providers import (
        AnswerProvider,
        ModelResult,
        build_repair_payload,
    )

    class RecordingProvider(AnswerProvider):
        async def _call(self, prompt, payload, schema, temperature):
            self.payload = payload
            return ModelResult(selected_repair(-1, text=START), 1, 1)

    evidence = (source(START),)
    value = confirmation_draft(STEP, evidence)
    provider = RecordingProvider(Settings(_env_file=None))
    result = await provider.repair(QUESTION, evidence, value, "ANSWER_CONFIRMATION_GUARD_FAILED")
    assert provider.payload == build_repair_payload(
        QUESTION, evidence, value, "ANSWER_CONFIRMATION_GUARD_FAILED"
    )
    assert result.value.statements[0].citations[0].quote == START


def test_scope_diagnostic_is_same_guard_with_specific_sources_not_delete_instruction():
    from test_query_answer_support import list_evidence, mixed_list

    from zhiqing_rag.query_answering.providers import build_repair_payload
    from zhiqing_rag.query_answering.support import source_scope_ambiguous, source_scope_conflicts

    evidence = list_evidence()
    value = mixed_list(evidence)
    question = "文档版本记录包含哪些信息？"
    conflicts = source_scope_conflicts(question, value, evidence)
    assert bool(conflicts) == source_scope_ambiguous(question, value, evidence)
    assert len(conflicts) == 1 and conflicts[0]["statement_index"] == 0
    assert {item["citation_id"] for item in conflicts[0]["sources"]} == {"E1", "E2"}
    assert {item["document_title"] for item in conflicts[0]["sources"]} == {
        item.document_title for item in evidence
    }
    assert all(len(item["source_document"]) == 4 for item in conflicts[0]["sources"])
    assert "delete_cid" not in str(conflicts)
    assert build_repair_payload(question, evidence, value,
                                "ANSWER_SOURCE_SCOPE_AMBIGUOUS")["scope_conflicts"] == conflicts
    assert "scope_conflicts" not in build_repair_payload(question, evidence, value,
                                                        "ANSWER_QUOTE_NOT_VERBATIM")


def test_confirmation_diagnostics_collect_all_missing_phases_and_wrong_state_cell():
    from zhiqing_rag.query_answering.providers import build_repair_payload

    start, step = source(START), source(STEP + TRANSITION, "E2")
    table = source("| 状态 | 进入条件 | 离开前核对 |\n| --- | --- | --- |\n"
                   "| 已关闭 | 客户已确认 | 遗留项是否有人负责 |",
                   "E3", document_id="6", generation_id="9", chunk_id="12")
    value = confirmation_draft("确认条件是：" + STEP
                               + "关闭前还核对遗留项是否有人负责。", (step, table))
    evidence = (start, step, table)
    diagnostics = confirmation_diagnostics(QUESTION, value, evidence)
    assert confirmation_guard_failure(QUESTION, value, evidence) == bool(diagnostics)
    phases = [row for row in diagnostics if row["kind"] == "missing_literal_phase"]
    assert {row["required_clause"]["stage"] for row in phases} == {"start", "transition"}
    assert {row["citation_id"] for row in phases} == {"E1", "E2"}
    by_id = {item.citation_id: item for item in evidence}
    for row in phases:
        clause = row["required_clause"]
        source_text = by_id[row["citation_id"]].content[clause["start"]:clause["end"]]
        assert source_text == clause["answer_text"]
    state = [row for row in diagnostics if row["kind"] == "non_phase_state_cell"]
    assert len(state) == 1 and state[0]["citation_id"] == "E3"
    cell = state[0]["matched_cell"]
    assert cell["state"] == "已关闭" and cell["column"] == "离开前核对"
    assert table.content[cell["start"]:cell["end"]] == "遗留项是否有人负责"
    payload = build_repair_payload(QUESTION, evidence, value, "ANSWER_CONFIRMATION_GUARD_FAILED")
    assert payload["confirmation_diagnostics"] == diagnostics
    assert "confirmation_diagnostics" not in build_repair_payload(
        QUESTION, evidence, value, "ANSWER_QUOTE_NOT_VERBATIM"
    )


def test_confirmation_diagnostics_do_not_add_unrelated_source_requirements():
    first = source(STEP)
    other = source(START, "E2", revision_id="12")
    value = confirmation_draft(STEP, (first,))
    assert not confirmation_diagnostics(QUESTION, value, (first, other))
    assert not confirmation_diagnostics("第三步如何确认？", value, (first, other))


async def test_failed_review_remains_failed_even_with_complete_literal_confirmation():
    from zhiqing_rag.query_answering.models import AnswerReview

    item = source(START + STEP + TRANSITION)
    value = confirmation_draft(item.content, (item,))
    review = AnswerReview(sufficient=False, missing_aspects=["原文不支持额外确认形式"],
                          statements=[{"index": 0, "supported": True, "reason": "原句阶段完整"}])
    provider = Provider(value, review)
    _, answer = await run_confirmation(provider, (item,))
    assert answer.status == "insufficient" and answer.codes == ("ANSWER_REVIEW_FAILED",)
    assert not answer.statements
