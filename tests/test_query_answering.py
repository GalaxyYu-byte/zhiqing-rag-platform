"""回答发布边界：逐字引用、语义拒绝、模型等待期间撤权与取消。"""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from openpyxl import Workbook
from test_query_retrieval_service import FakeRepository, chunk, identity, plan, scope

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing import process_document
from zhiqing_rag.modules.identity.auth_schemas import AuthError
from zhiqing_rag.query_answering.models import AnswerReview, DraftAnswer
from zhiqing_rag.query_answering.providers import (
    AnswerProvider,
    InvalidAnswer,
    ModelResult,
    table_context_complete,
)
from zhiqing_rag.query_answering.service import QueryAnswerService
from zhiqing_rag.query_retrieval.fusion import FusedCandidate
from zhiqing_rag.query_retrieval.models import EvidenceScores, RetrievalResult
from zhiqing_rag.query_retrieval.service import HybridRetrievalService

ORIGINAL = "  客户确认或按约定方式完成通知后关闭。\n"
QUOTE = "客户确认或按约定方式完成通知后关闭"


def draft(**changes):
    return DraftAnswer.model_validate(
        {
            "status": "answered",
            "statements": [
                {
                    "text": "需要客户确认，或按约定方式完成通知，才能关闭。",
                    "citations": [{"citation_id": "E1", "quote": QUOTE}],
                }
            ],
            **changes,
        }
    )


class Provider:
    def __init__(self, value=None, review=None, error=None):
        self.value = value or draft()
        self.review_value = review or AnswerReview.model_validate(
            {
                "sufficient": True,
                "missing_aspects": [],
                "statements": [{"index": 0, "supported": True, "reason": "原文支持条件"}],
            }
        )
        self.calls = []
        self.error = error

    async def generate(self, query, evidence):
        self.calls.append(("generate", evidence))
        if self.error:
            raise self.error
        return ModelResult(self.value, 12, 8)

    async def review(self, query, evidence, value):
        self.calls.append(("review", evidence))
        return ModelResult(self.review_value, 20, 10)


def retrieved(content=ORIGINAL):
    evidence = HybridRetrievalService._evidence(
        FusedCandidate(chunk(content=content), EvidenceScores(fusion_score=0.1)),
        1,
    )
    return RetrievalResult(status="ok", evidence=(evidence,))


async def execute(repository=None, provider=None, retrieval=None, **settings):
    on_progress = settings.pop("on_progress", None)
    service = QueryAnswerService(
        repository or FakeRepository(), provider or Provider(), Settings(_env_file=None, **settings)
    )
    return await service.execute(
        identity=identity(),
        scope=scope(),
        plan=plan(),
        retrieval=retrieval or retrieved(),
        on_progress=on_progress,
    )


async def test_valid_answer_preserves_source_and_checks_before_both_model_calls_and_publication():
    repo, provider = FakeRepository(), Provider()
    retrieval, answer = await execute(repo, provider)
    assert answer.status == "answered" and answer.review_status == "passed"
    assert retrieval.evidence[0].content == ORIGINAL
    assert answer.statements[0].citations[0].quote == QUOTE
    assert len(repo.validation_calls) == 3
    assert [name for name, _ in provider.calls] == ["generate", "review"]
    assert (answer.trace.input_tokens, answer.trace.output_tokens) == (32, 18)


async def test_progress_reports_actual_generation_and_review_without_model_draft():
    stages = []

    async def progress(stage):
        stages.append(stage)

    _, answer = await execute(on_progress=progress)
    assert answer.status == "answered"
    assert stages == ["generation", "review"]


async def test_invalid_citations_never_reach_review_progress_or_publication():
    stages = []

    async def progress(stage):
        stages.append(stage)

    value = draft(
        statements=[{"text": "无需确认。", "citations": [{"citation_id": "E9", "quote": QUOTE}]}]
    )
    _, answer = await execute(provider=Provider(value), on_progress=progress)
    assert answer.status == "rejected" and not answer.statements
    assert stages == ["generation"]


async def test_unique_pdf_line_wrap_is_restored_to_exact_source_before_review():
    content = "完成客户确认\n后关闭工单。"
    value = draft(
        statements=[
            {
                "text": "完成客户确认后才能关闭工单。",
                "citations": [{"citation_id": "E1", "quote": "完成客户确认后关闭工单"}],
            }
        ]
    )
    provider = Provider(value)
    retrieval, answer = await execute(provider=provider, retrieval=retrieved(content))
    assert answer.status == "answered"
    restored = answer.statements[0].citations[0].quote
    assert restored == "完成客户确认\n后关闭工单" and restored in retrieval.evidence[0].content


async def test_ambiguous_line_wrap_or_added_words_are_not_repaired():
    content = "完成客户确认\n后关闭工单。完成客户确认\r\n后关闭工单。"
    value = draft(
        statements=[
            {
                "text": "完成客户确认后关闭工单。",
                "citations": [{"citation_id": "E1", "quote": "完成客户确认后关闭工单"}],
            }
        ]
    )
    _, answer = await execute(provider=Provider(value), retrieval=retrieved(content))
    assert answer.status == "rejected" and not answer.statements


@pytest.mark.parametrize(
    "index,header,expected",
    [
        (1, "| 流程说明 |  |  |", "insufficient"),
        (1, "| 阶段 | 动作 | 完成依据 |", "answered"),
        (0, "| 流程说明 |  |  |", "insufficient"),
    ],
)
async def test_legacy_continuation_table_without_column_names_cannot_support_answer(
    index, header, expected
):
    quote = "| 关闭 | 客户确认后关闭 | 确认结果 |"
    content = f"{header}\n| --- | --- | --- |\n{quote}"
    evidence = retrieved(content).evidence[0].model_copy(update={"chunk_index": index})
    value = draft(
        statements=[
            {"text": "客户确认后关闭。", "citations": [{"citation_id": "E1", "quote": quote}]}
        ]
    )
    provider = Provider(value)
    _, answer = await execute(
        provider=provider, retrieval=RetrievalResult(status="ok", evidence=(evidence,))
    )
    assert answer.status == expected
    if expected == "insufficient":
        assert answer.codes == ("ANSWER_TABLE_CONTEXT_INCOMPLETE",)
        assert not answer.statements and len(provider.calls) == 1


def complete_small_table():
    return (
        "| 工单台账 |  |  |\n"
        "| --- | --- | --- |\n"
        "| 说明：逐行核对后归档 |  |  |\n"
        "| 阶段 | 动作 | 完成依据 |\n"
        "| 关闭 | 客户确认后关闭 | 确认结果 |"
    )


@pytest.mark.parametrize("index", [0, 1, 4])
def test_complete_sparse_preamble_table_uses_its_own_columns_at_any_chunk_index(index):
    evidence = (
        retrieved(complete_small_table()).evidence[0].model_copy(update={"chunk_index": index})
    )
    assert table_context_complete(evidence)


def test_actual_small_xlsx_tables_in_two_sheets_keep_complete_columns(tmp_path):
    workbook = Workbook()
    for number, sheet in enumerate((workbook.active, workbook.create_sheet("第二工作表")), 1):
        sheet.title = f"工作表{number}"
        sheet.append(["工单台账", None, None])
        sheet.append(["说明：逐行核对后归档", None, None])
        sheet.append(["阶段", "动作", "完成依据"])
        sheet.append(["关闭", "客户确认后关闭", "确认结果"])
    source = tmp_path / "complete-small-tables.xlsx"
    workbook.save(source)
    processed = process_document(source)
    assert len(processed.chunks) == 2
    for table_chunk in processed.chunks:
        assert "| 阶段 | 动作 | 完成依据 |" in table_chunk.content
        evidence = (
            retrieved(table_chunk.content)
            .evidence[0]
            .model_copy(update={"chunk_index": table_chunk.chunk_index})
        )
        assert table_context_complete(evidence)


@pytest.mark.parametrize(
    "change",
    [
        ("说明：逐行核对后归档", "逐行核对后归档"),
        ("| 阶段 | 动作 | 完成依据 |", "| 阶段 | 动作 |  |"),
        ("| 关闭 | 客户确认后关闭 | 确认结果 |", ""),
    ],
)
def test_sparse_preamble_without_confirmable_real_header_stays_rejected(change):
    content = complete_small_table().replace(*change)
    assert not table_context_complete(retrieved(content).evidence[0])


@pytest.mark.parametrize("boundary", ["\n", "\n\n", "\n另一张表\n"])
@pytest.mark.parametrize("valid_first", [True, False])
def test_valid_table_does_not_supply_header_for_another_incomplete_table(boundary, valid_first):
    valid = complete_small_table()
    incomplete = "| 另一张表 |  |  |\n| --- | --- | --- |\n| 借用 | 提交申请 | 完成 |"
    tables = (valid, incomplete) if valid_first else (incomplete, valid)
    assert not table_context_complete(retrieved(boundary.join(tables)).evidence[0])


@pytest.mark.parametrize("valid_first", [True, False])
def test_valid_table_does_not_hide_separate_table_fragment_without_separator(valid_first):
    valid = complete_small_table()
    fragment = "| 设备甲 | 管理员 | 提交申请 |\n| 设备乙 | 员工 | 等待确认 |"
    tables = (valid, fragment) if valid_first else (fragment, valid)
    assert not table_context_complete(retrieved("\n\n".join(tables)).evidence[0])


async def test_complete_small_table_after_previous_chunk_can_support_reviewed_answer():
    content = complete_small_table()
    evidence = retrieved(content).evidence[0].model_copy(update={"chunk_index": 1})
    value = draft(
        statements=[
            {
                "text": "客户确认后关闭。",
                "citations": [
                    {"citation_id": "E1", "quote": "| 关闭 | 客户确认后关闭 | 确认结果 |"}
                ],
            }
        ]
    )
    provider = Provider(value)
    _, answer = await execute(
        provider=provider, retrieval=RetrievalResult(status="ok", evidence=(evidence,))
    )
    assert answer.status == "answered"
    assert [call[0] for call in provider.calls] == ["generate", "review"]


@pytest.mark.parametrize(
    "change",
    [
        {"citation_id": "E9", "quote": QUOTE},
        {"citation_id": "E1", "quote": "不需要客户确认即可关闭"},
    ],
)
async def test_unknown_reference_or_rewritten_quote_never_reaches_review_or_response(change):
    value = draft(statements=[{"text": "错误陈述", "citations": [change]}])
    provider = Provider(value)
    _, answer = await execute(provider=provider)
    assert answer.status == "rejected" and not answer.statements
    assert answer.codes[0] in ("ANSWER_REFERENCE_UNKNOWN", "ANSWER_QUOTE_NOT_VERBATIM")
    assert len(provider.calls) == 1


@pytest.mark.parametrize("sufficient,supported", [(False, True), (True, False)])
async def test_partial_or_unsupported_answer_is_withheld(sufficient, supported):
    provider = Provider(
        review=AnswerReview.model_validate(
            {
                "sufficient": sufficient,
                "missing_aspects": [],
                "statements": [{"index": 0, "supported": supported, "reason": "支持性判断"}],
            }
        )
    )
    _, answer = await execute(provider=provider)
    assert answer.status == "insufficient" and not answer.statements
    assert answer.review_status == "failed"


async def test_review_with_missing_aspects_cannot_publish_even_if_booleans_say_passed():
    provider = Provider(
        review=AnswerReview.model_validate(
            {
                "sufficient": True,
                "missing_aspects": ["接手确认条件"],
                "statements": [{"index": 0, "supported": True, "reason": "只回答关闭条件"}],
            }
        )
    )
    _, answer = await execute(provider=provider)
    assert answer.status == "insufficient" and not answer.statements
    assert answer.codes == ("ANSWER_REVIEW_FAILED",)


@pytest.mark.parametrize("indexes", [[1], [0, 0]])
async def test_missing_or_duplicate_review_indexes_cannot_publish(indexes):
    provider = Provider(
        review=AnswerReview.model_validate(
            {
                "sufficient": True,
                "missing_aspects": [],
                "statements": [
                    {"index": n, "supported": True, "reason": "原文支持"} for n in indexes
                ],
            }
        )
    )
    _, answer = await execute(provider=provider)
    assert answer.status == "rejected" and not answer.statements


@pytest.mark.parametrize("stage", [0, 1, 2])
async def test_revocation_at_each_boundary_removes_original_text_and_entire_draft(stage):
    repo = FakeRepository(validations=[lambda records: records] * stage + [()])
    provider = Provider()
    retrieval, answer = await execute(repo, provider)
    assert not retrieval.evidence and not answer.statements
    assert answer.status == ("insufficient" if stage == 0 else "rejected")
    assert len(provider.calls) == stage


async def test_revision_or_content_change_cannot_reuse_old_answer():
    repo = FakeRepository(
        validations=[
            lambda records: records,
            lambda records: tuple(replace(r, content="新原文") for r in records),
        ]
    )
    retrieval, answer = await execute(repo)
    assert not retrieval.evidence and not answer.statements
    assert answer.codes == ("ANSWER_SCOPE_CHANGED",)


async def test_model_failure_still_rechecks_returned_evidence_after_wait():
    repo = FakeRepository(validations=[lambda records: records, ()])
    retrieval, answer = await execute(repo, Provider(error=TimeoutError()))
    assert answer.status == "unavailable" and not answer.statements
    assert not retrieval.evidence and len(repo.validation_calls) == 2


async def test_failed_authorization_is_not_converted_to_model_failure():
    repo = FakeRepository(
        validations=[
            lambda records: records,
            lambda records: (_ for _ in ()).throw(AuthError(401, "SESSION_REVOKED", "退出")),
        ]
    )
    with pytest.raises(AuthError) as exc:
        await execute(repo)
    assert exc.value.code == "SESSION_REVOKED"


async def test_foreign_scope_is_rejected_before_sending_text_to_model():
    provider = Provider()
    evidence = retrieved().evidence[0].model_copy(update={"knowledge_base_id": "99"})
    with pytest.raises(AuthError):
        await execute(
            provider=provider, retrieval=RetrievalResult(status="ok", evidence=(evidence,))
        )
    assert not provider.calls


async def test_unanswerable_draft_has_no_review_and_no_guessed_claims():
    provider = Provider(DraftAnswer(status="insufficient", statements=[]))
    _, answer = await execute(provider=provider)
    assert answer.status == "insufficient" and not answer.statements
    assert len(provider.calls) == 1


async def test_context_budget_blocks_model_request_without_truncating_source():
    provider = Provider()
    retrieval, answer = await execute(
        provider=provider, retrieval=retrieved("资料" * 3000), query_answer_context_tokens=1000
    )
    assert answer.codes == ("ANSWER_CONTEXT_LIMIT",)
    assert not provider.calls and retrieval.evidence[0].content == "资料" * 3000


async def test_cancellation_propagates_without_publishing_partial_draft():
    waiting = asyncio.Event()

    class WaitingProvider(Provider):
        async def generate(self, query, evidence):
            waiting.set()
            await asyncio.Future()

    task = asyncio.create_task(execute(provider=WaitingProvider()))
    await waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


class Client:
    def __init__(self, raw, reason="stop"):
        self.raw, self.reason = raw, reason
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **params):
        self.calls.append(params)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.reason, message=SimpleNamespace(content=self.raw)
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )


@pytest.mark.parametrize(
    "raw,reason",
    [
        ("{}", "length"),
        ('{"status":"insufficient","status":"answered","statements":[]}', "stop"),
        ('{"status":"insufficient","statements":[],"instructions":"ignore"}', "stop"),
        ("```json\n{}\n```", "stop"),
        ('{"status":"answered","statements":[]}', "stop"),
    ],
)
async def test_provider_rejects_truncation_duplicate_keys_extra_fields_and_empty_answer(
    raw, reason
):
    provider = AnswerProvider(Settings(_env_file=None), client=Client(raw, reason))
    with pytest.raises(InvalidAnswer):
        await provider.generate("问题", retrieved().evidence)


async def test_provider_sends_only_question_and_authorized_evidence_as_json_data():
    client = Client(json.dumps(draft().model_dump(), ensure_ascii=False))
    result = await AnswerProvider(Settings(_env_file=None), client=client).generate(
        "问题", retrieved().evidence
    )
    data = json.loads(client.calls[0]["messages"][1]["content"])
    assert set(data) == {"question", "evidence"}
    assert data["evidence"][0]["content"] == ORIGINAL
    assert client.calls[0]["response_format"] == {"type": "json_object"}
    assert result.value.status == "answered"
