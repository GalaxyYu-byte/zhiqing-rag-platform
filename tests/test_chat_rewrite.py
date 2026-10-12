"""多轮改写的逐字依据、条件保留、权限边界与模型故障降级。"""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.modules.chat.rewrite import QueryRewriteService, depends_on_history


class FakeClient:
    def __init__(self, payload=None, *, delay=0, error=None, finish_reason="stop", usage=None):
        self.payload = payload
        self.delay = delay
        self.error = error
        self.finish_reason = finish_reason
        self.usage = usage or SimpleNamespace(prompt_tokens=120, completion_tokens=40)
        self.closed = False
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        content = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=self.finish_reason,
                message=SimpleNamespace(content=content, refusal=None),
            )],
            usage=self.usage,
        )

    async def close(self):
        self.closed = True


def settings(**changes):
    values = {
        "deepseek_api_key": "",
        "query_analyzer_enabled": True,
        "query_analyzer_timeout_seconds": 15,
    }
    return Settings(_env_file=None, app_env="test", **{**values, **changes})


def history(*, answer=None, question="工单由谁确认？", run_id="1"):
    return [{"run_id": run_id, "question": question, "answer": answer}]


def rewritten(query="它由谁确认？", *, text="工单", quote="工单由谁确认？"):
    return {
        "original_question": query,
        "mode": "rewritten",
        "question": text + query[1:],
        "clarification": None,
        "terms": [{
            "text": text,
            "evidence": {
                "history_index": 0, "run_id": "1", "source": "question",
                "statement_index": None, "quote": quote,
            },
        }],
        "replacements": [{"start": 0, "end": 1, "term_index": 0}],
    }


def unchanged(query):
    return {
        "original_question": query, "mode": "unchanged", "question": query,
        "clarification": None, "terms": [], "replacements": [],
    }


async def test_without_history_preserves_query_and_leaves_clarification_to_analyzer():
    client = FakeClient("must not call")
    result = await QueryRewriteService(settings(), client=client).rewrite("它怎么办？", [])
    assert result.question == "它怎么办？"
    assert result.clarification is None
    assert (result.input_tokens, result.output_tokens) == (0, 0)
    assert client.calls == []


async def test_grounded_reference_replaced_and_owned_original_sent_to_model():
    client = FakeClient(rewritten())
    result = await QueryRewriteService(settings(), client=client).rewrite(
        "它由谁确认？", history(),
    )
    assert result.question == "工单由谁确认？"
    assert result.clarification is None
    assert (result.input_tokens, result.output_tokens) == (120, 40)
    assert result.used_run_ids == ("1",)
    assert not client.closed
    call = client.calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["temperature"] == 0
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert call["model"] == settings().query_analyzer_model
    assert json.loads(call["messages"][1]["content"]) == {
        "original_question": "它由谁确认？",
        "history": [{
            "run_id": "1", "question": "工单由谁确认？", "assistant_statements": [],
        }],
    }


async def test_explicit_current_conditions_and_negation_remain_verbatim():
    query = "它在2025年财务部不是PDF的资料中有说明吗？"
    client = FakeClient(rewritten(query))
    result = await QueryRewriteService(settings(), client=client).rewrite(query, history())
    assert result.question == "工单在2025年财务部不是PDF的资料中有说明吗？"


async def test_original_whitespace_is_preserved_and_offsets_are_not_normalized():
    query = " 它由谁确认？ "
    payload = rewritten(query)
    payload["question"] = " 工单由谁确认？ "
    payload["replacements"][0].update(start=1, end=2)
    result = await QueryRewriteService(settings(), client=FakeClient(payload)).rewrite(
        query, history(),
    )
    assert result.question == " 工单由谁确认？ "


def ordinal_payload():
    payload = rewritten("第二条展开一下", text="合同归档流程", quote="合同归档流程要求保留签字记录")
    payload["question"] = "合同归档流程展开一下"
    payload["replacements"][0]["end"] = 3
    payload["terms"][0]["evidence"].update(source="assistant_statement", statement_index=1)
    return payload


def reviewed_answer():
    return {
        "status": "answered", "review_status": "passed",
        "statements": [
            {"text": "工单审批流程包括责任人确认", "citations": [{"quote": "private raw quote"}]},
            {"text": "合同归档流程要求保留签字记录", "citations": [{"quote": "private raw quote"}]},
        ],
        "message": "do not send message", "trace": {"input_tokens": 1},
    }


async def test_second_item_uses_authorized_statement_topic_without_old_facts_or_citations():
    client = FakeClient(ordinal_payload())
    result = await QueryRewriteService(settings(), client=client).rewrite(
        "第二条展开一下", history(answer=reviewed_answer()),
    )
    assert result.question == "合同归档流程展开一下"
    assert result.used_run_ids == ("1",)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["history"][0]["assistant_statements"] == [
        "工单审批流程包括责任人确认", "合同归档流程要求保留签字记录",
    ]
    assert "private raw quote" not in client.calls[0]["messages"][1]["content"]
    assert "do not send message" not in client.calls[0]["messages"][1]["content"]


@pytest.mark.parametrize("answer", [
    None,
    {**reviewed_answer(), "status": "rejected"},
    {**reviewed_answer(), "review_status": "failed"},
])
async def test_missing_or_currently_unavailable_answer_cannot_ground_an_ordinal(answer):
    result = await QueryRewriteService(settings(), client=FakeClient(ordinal_payload())).rewrite(
        "第二条展开一下", history(answer=answer),
    )
    assert result.question is None
    assert result.clarification


async def test_ambiguous_model_clarification_does_not_execute_retrieval():
    payload = {
        **unchanged("它和那个有何区别？"), "mode": "clarify", "question": None,
        "clarification": "请说明要比较的两个流程名称。",
    }
    result = await QueryRewriteService(settings(), client=FakeClient(payload)).rewrite(
        "它和那个有何区别？", history(),
    )
    assert result.question is None
    assert result.clarification
    assert result.clarification != "请说明要比较的两个流程名称。"
    assert result.used_run_ids == ()


@pytest.mark.parametrize("text,source", [
    ("2025年工单", "2025年工单由谁确认？"),
    ("去年", "去年上传的工单有哪些？"),
    ("财务部", "财务部工单由谁确认？"),
    ("PDF", "PDF工单由谁确认？"),
    ("PDF文档", "PDF文档由谁确认？"),
    ("DOCX合同", "DOCX合同由谁确认？"),
    ("财务部工单", "财务部工单由谁确认？"),
    ("技术部的合同", "技术部的合同由谁确认？"),
    ("运营部门资料", "运营部门资料由谁确认？"),
    ("业务科合同", "业务科合同由谁确认？"),
    ("小组操作流程", "小组操作流程由谁确认？"),
    ("工单必须经负责人确认", "工单必须经负责人确认"),
    ("工单由经理负责", "工单由经理负责"),
])
async def test_quoted_old_filters_numbers_and_assertions_are_not_inherited(text, source):
    result = await QueryRewriteService(
        settings(), client=FakeClient(rewritten(text=text, quote=source)),
    ).rewrite("它由谁确认？", history(question=source))
    assert result.question is None
    assert result.clarification


async def test_department_detection_does_not_misclassify_processing_as_a_department():
    text, source = "工单处理流程", "工单处理流程由谁确认？"
    result = await QueryRewriteService(
        settings(), client=FakeClient(rewritten(text=text, quote=source)),
    ).rewrite("它由谁确认？", history(question=source))
    assert result.question == "工单处理流程由谁确认？"


def invalid_payloads():
    ungrounded = rewritten(text="合同")
    changed_original = rewritten()
    changed_original["original_question"] = "工单由谁确认？"
    invented_question = rewritten()
    invented_question["question"] = "工单由张三在2025年确认吗？"
    changed_current_condition = rewritten("它不是PDF吗？")
    changed_current_condition["question"] = "工单是PDF吗？"
    wrong_run = rewritten()
    wrong_run["terms"][0]["evidence"]["run_id"] = "2"
    wrong_index = rewritten()
    wrong_index["terms"][0]["evidence"]["history_index"] = 1
    extra = rewritten()
    extra["sql"] = "select * from secret"
    wrong_type = rewritten()
    wrong_type["replacements"][0]["start"] = False
    wrong_span = rewritten()
    wrong_span["replacements"][0]["end"] = 2
    overlap = rewritten()
    overlap["replacements"].append(deepcopy(overlap["replacements"][0]))
    unused_term = rewritten()
    unused_term["terms"].append(deepcopy(unused_term["terms"][0]))
    no_grounding = {**rewritten(), "terms": [], "replacements": []}
    return [
        ungrounded, changed_original, invented_question, changed_current_condition,
        wrong_run, wrong_index, extra, wrong_type, wrong_span, overlap, unused_term,
        no_grounding, unchanged("它由谁确认？"), "```json\n{}\n```", "[]",
        '{"original_question":"它由谁确认？","original_question":"它由谁确认？"}',
        json.dumps(rewritten()).replace('"run_id": "1"', '"run_id": "1", "run_id": "1"'),
    ]


@pytest.mark.parametrize("payload", invalid_payloads())
async def test_invalid_ungrounded_or_condition_changing_model_output_clarifies(payload):
    query = payload["original_question"] if isinstance(payload, dict) else "它由谁确认？"
    if query == "工单由谁确认？":
        query = "它由谁确认？"
    result = await QueryRewriteService(settings(), client=FakeClient(payload)).rewrite(
        query, history(),
    )
    assert result.question is None
    assert result.clarification
    assert result.input_tokens == 120  # 失效响应依然消耗模型费用。


async def test_ordinal_must_match_latest_answer_and_exact_statement_index():
    for changes in ({"statement_index": 0}, {"history_index": 0, "run_id": "1"}):
        payload = ordinal_payload()
        histories = history(answer=reviewed_answer())
        if "history_index" in changes:
            histories += history(answer=reviewed_answer(), run_id="2")
        payload["terms"][0]["evidence"].update(changes)
        result = await QueryRewriteService(settings(), client=FakeClient(payload)).rewrite(
            "第二条展开一下", histories,
        )
        assert result.question is None


async def test_recent_six_rounds_only_and_earlier_references_are_not_guessed():
    query = "它由谁确认？"
    payload = rewritten()
    payload["terms"][0]["evidence"].update(history_index=5, run_id="7")
    client = FakeClient(payload)
    histories = [history(run_id=str(index))[0] for index in range(1, 8)]
    result = await QueryRewriteService(settings(), client=client).rewrite(query, histories)
    assert result.question == "工单由谁确认？"
    sent = json.loads(client.calls[0]["messages"][1]["content"])["history"]
    assert [item["run_id"] for item in sent] == ["2", "3", "4", "5", "6", "7"]
    payload["terms"][0]["evidence"].update(history_index=0, run_id="2")
    result = await QueryRewriteService(settings(), client=FakeClient(payload)).rewrite(
        query, histories,
    )
    assert result.question is None


@pytest.mark.parametrize("query", ["它怎么办？", "第二条展开一下", "展开一下", "继续", "然后呢？"])
@pytest.mark.parametrize("config", [{"query_analyzer_enabled": False}, {"deepseek_api_key": ""}])
async def test_disabled_or_missing_key_never_retrieves_a_context_dependent_query(query, config):
    result = await QueryRewriteService(settings(**config)).rewrite(query, history())
    assert result.question is None
    assert result.clarification


@pytest.mark.parametrize("fault", ["timeout", "provider", "invalid", "incomplete"])
async def test_model_faults_clarify_dependent_query_and_preserve_independent_query(fault):
    for query in ("它怎么办？", "合同如何归档？"):
        kwargs = {
            "timeout": {"delay": 1},
            "provider": {"error": httpx.ConnectError("secret payload")},
            "invalid": {"payload": "garbage secret payload"},
            "incomplete": {"finish_reason": "length"},
        }[fault]
        client = FakeClient(**kwargs)
        result = await QueryRewriteService(
            settings(query_analyzer_timeout_seconds=0.02), client=client,
        ).rewrite(query, history())
        assert result.question == (None if depends_on_history(query) else query)
        assert bool(result.clarification) == depends_on_history(query)


async def test_owned_client_has_no_retries_and_closes_after_timeout(monkeypatch):
    client = FakeClient(delay=1)
    arguments = []

    def factory(**kwargs):
        arguments.append(kwargs)
        return client

    monkeypatch.setattr("zhiqing_rag.modules.chat.rewrite.AsyncOpenAI", factory)
    result = await QueryRewriteService(settings(
        deepseek_api_key="test-key", query_analyzer_timeout_seconds=0.02,
    )).rewrite("它怎么办？", history())
    assert result.question is None
    assert client.closed
    assert arguments[0]["max_retries"] == 0
    assert arguments[0]["timeout"] == 0.02


async def test_owned_client_closes_on_cancellation_and_cancellation_propagates(monkeypatch):
    client = FakeClient(delay=10)
    monkeypatch.setattr("zhiqing_rag.modules.chat.rewrite.AsyncOpenAI", lambda **kwargs: client)
    task = asyncio.create_task(QueryRewriteService(
        settings(deepseek_api_key="test-key"),
    ).rewrite("它怎么办？", history()))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert client.closed


async def test_negative_or_boolean_usage_is_not_counted():
    client = FakeClient(
        rewritten(), usage=SimpleNamespace(prompt_tokens=True, completion_tokens=-2),
    )
    result = await QueryRewriteService(settings(), client=client).rewrite("它由谁确认？", history())
    assert (result.input_tokens, result.output_tokens) == (0, 0)


async def test_independent_query_does_not_depend_on_historical_permissions():
    query = "这个月的合同如何归档？"
    client = FakeClient(unchanged(query))
    result = await QueryRewriteService(settings(), client=client).rewrite(query, history())
    assert result.question == query
    assert result.used_run_ids == ()


def large_answer():
    return {
        "status": "answered", "review_status": "passed",
        "statements": [{"text": "甲" * 1500} for _ in range(8)],
    }


async def test_context_budget_drops_oldest_complete_rounds_and_renumbers_sources():
    payload = rewritten()
    payload["terms"][0]["evidence"]["run_id"] = "2"
    client = FakeClient(payload)
    histories = history(answer=large_answer()) + history(run_id="2")
    result = await QueryRewriteService(settings(), client=client).rewrite("它由谁确认？", histories)
    assert result.question == "工单由谁确认？"
    assert result.used_run_ids == ("2",)
    sent = json.loads(client.calls[0]["messages"][1]["content"])["history"]
    assert len(sent) == 1
    assert sent[0]["run_id"] == "2"


async def test_latest_round_over_context_budget_clarifies_without_using_older_round():
    client = FakeClient(rewritten())
    histories = history() + history(run_id="2", answer=large_answer())
    result = await QueryRewriteService(settings(), client=client).rewrite("它怎么办？", histories)
    assert result.question is None
    assert result.clarification
    assert result.used_run_ids == ()
    assert client.calls == []
    query = "合同归档流程是什么？"
    result = await QueryRewriteService(settings(), client=client).rewrite(query, histories)
    assert result.question == query
    assert result.used_run_ids == ()


async def test_logs_do_not_contain_questions_histories_responses_or_errors(caplog):
    client = FakeClient(error=httpx.ConnectError("secret-password"))
    with caplog.at_level("INFO"):
        await QueryRewriteService(settings(), client=client).rewrite("它怎么办？", history())
    assert "secret-password" not in caplog.text
    assert "它怎么办" not in caplog.text
    assert "工单" not in caplog.text


@pytest.mark.parametrize("bad_history", [
    [{"run_id": "1", "question": ""}],
    [{"run_id": "1", "question": "x" * 2001}],
    [{"run_id": None, "question": "工单"}],
])
async def test_invalid_internal_history_is_not_misreported_as_a_model_failure(bad_history):
    with pytest.raises(ValueError):
        await QueryRewriteService(settings()).rewrite("它怎么办？", bad_history)


@pytest.mark.parametrize("query", [
    "关闭前呢", "用表格展示", "谁负责？", "补充一下", "这两份有区别吗", "tell me more",
])
async def test_obvious_ellipsis_also_requires_clarification_when_model_is_disabled(query):
    result = await QueryRewriteService(settings(query_analyzer_enabled=False)).rewrite(
        query, history(),
    )
    assert result.question is None
    assert result.clarification


@pytest.mark.parametrize("query", ["这两年合同如何归档？", "这个月工单关闭规定是什么？"])
async def test_explicit_current_time_does_not_become_a_history_dependency(query):
    result = await QueryRewriteService(settings(query_analyzer_enabled=False)).rewrite(
        query, history(),
    )
    assert result.question == query
    assert result.clarification is None
