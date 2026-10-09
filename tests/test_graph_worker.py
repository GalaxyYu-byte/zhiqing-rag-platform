"""图抽取的结构约束、证据校验与可重放图写入。"""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError
from pydantic import ValidationError

from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph import GraphWorker
from zhiqing_rag.workers.graph_cleanup import CleanupLease
from zhiqing_rag.workers.graph_extraction import (
    Extraction,
    GraphExtractionError,
    extract_chunk,
    normalize_name,
    stable_key,
    validate_evidence,
    validated_endpoint_subset,
)
from zhiqing_rag.workers.graph_storage import GraphStore


def sample():
    return {
        "entities": [
            {
                "id": "a",
                "type": "Actor",
                "name": "财务",
                "canonical_name": "财务",
                "evidence_quote": "财务审核报销申请",
            },
            {
                "id": "s",
                "type": "Step",
                "name": "审核报销申请",
                "canonical_name": "审核报销申请",
                "evidence_quote": "财务审核报销申请",
            },
        ],
        "relations": [
            {
                "type": "REVIEWS",
                "source_id": "a",
                "target_id": "s",
                "evidence_quote": "财务审核报销申请",
                "condition_text": None,
            },
        ],
    }


def test_schema_rejects_unlisted_relation_and_wrong_endpoints():
    data = sample()
    data["relations"][0]["type"] = "RELATED_TO"
    with pytest.raises(ValidationError):
        Extraction.model_validate(data)
    data["relations"][0].update(type="REVIEWS", source_id="s", target_id="a")
    with pytest.raises(ValidationError):
        Extraction.model_validate(data)


def test_evidence_and_normalization_are_deterministic():
    result = Extraction.model_validate(sample())
    validate_evidence(result, "财务审核报销申请")
    assert normalize_name(" ＡＢＣ   财务。") == normalize_name("abc 财务")
    assert stable_key("tenant", 1, "财务") == stable_key("tenant", 1, "财务")
    data = sample()
    data["relations"][0]["evidence_quote"] = "不存在的原文"
    with pytest.raises(ValueError, match="证据原文"):
        validate_evidence(Extraction.model_validate(data), "财务审核报销申请")


def test_invalid_relation_endpoint_keeps_only_valid_evidenced_facts():
    import json

    data = sample()
    data["relations"].append(
        {
            "type": "REVIEWS",
            "source_id": "s",
            "target_id": "a",
            "evidence_quote": "财务审核报销申请",
            "condition_text": None,
        }
    )
    result = validated_endpoint_subset(json.dumps(data), "财务审核报销申请")
    assert len(result.entities) == 2
    assert len(result.relations) == 1


def test_relation_condition_must_be_in_its_own_evidence():
    import json

    content = "主管批准后，财务审核报销申请。"
    data = sample()
    data["relations"][0]["condition_text"] = "主管批准后"
    with pytest.raises(ValueError, match="证据未包含对应条件"):
        validate_evidence(Extraction.model_validate(data), content)

    # 端点容错也不能把不含条件的关系重新放行。
    assert not validated_endpoint_subset(json.dumps(data), content).relations
    data["relations"][0]["evidence_quote"] = content
    result = Extraction.model_validate(data)
    validate_evidence(result, content)
    assert result.relations[0].condition_text == "主管批准后"


def test_rule_quote_alone_cannot_support_an_unmentioned_endpoint():
    data = sample()
    data["relations"][0]["evidence_quote"] = "审核报销申请"
    with pytest.raises(ValueError, match="两端实体名称"):
        validate_evidence(Extraction.model_validate(data), "财务审核报销申请")


@pytest.mark.parametrize("condition", ["材料齐全时", "主管批准后", "如果申请有效"])
def test_conditional_clause_cannot_be_reduced_to_an_unconditional_action(condition):
    content = f"{condition}，财务审核报销申请。"
    data = sample()
    # 即使模型把前置条件从引文剪掉，也按原句校验。
    with pytest.raises(ValueError, match="遗漏原句前置条件"):
        validate_evidence(Extraction.model_validate(data), content)
    data["relations"][0].update(evidence_quote=content, condition_text=condition)
    validate_evidence(Extraction.model_validate(data), content)


def test_unrelated_previous_condition_does_not_apply_to_next_sentence():
    data = Extraction.model_validate(sample())
    validate_evidence(data, "主管批准后，系统发送通知。财务审核报销申请。")
    validate_evidence(data, "若干事项待核实；财务审核报销申请。")


def test_contact_recipient_is_not_the_actor_responsible_for_contacting():
    data = sample()
    data["entities"][1].update(name="联系财务", canonical_name="联系财务")
    for entity in data["entities"]:
        entity["evidence_quote"] = "联系财务"
    data["relations"][0].update(type="RESPONSIBLE_FOR", evidence_quote="联系财务")
    with pytest.raises(ValueError, match="误当执行人"):
        validate_evidence(Extraction.model_validate(data), "联系财务")


def test_forbidden_action_cannot_be_written_as_positive_operation():
    content = "而不是删除旧附件。"
    data = sample()
    data["entities"][0].update(type="Step", name="删除旧附件", canonical_name="删除旧附件")
    data["entities"][1].update(type="Material", name="旧附件", canonical_name="旧附件")
    for entity in data["entities"]:
        entity["evidence_quote"] = content
    data["relations"][0].update(type="OPERATES_ON", evidence_quote=content)
    with pytest.raises(ValueError, match="否定动作"):
        validate_evidence(Extraction.model_validate(data), content)
    # 模型裁掉否定文字的引文也不能绕过原句校验。
    data["relations"][0]["evidence_quote"] = "删除旧附件"
    with pytest.raises(ValueError, match="否定动作"):
        validate_evidence(Extraction.model_validate(data), content)


def test_checking_an_attribute_does_not_define_a_record_field():
    data = sample()
    data["entities"][0].update(type="Material", name="清单", canonical_name="清单")
    data["entities"][1].update(type="Field", name="形成时间", canonical_name="形成时间")
    for entity in data["entities"]:
        entity["evidence_quote"] = "核对清单的形成时间"
    data["relations"][0].update(type="HAS_FIELD", evidence_quote="核对清单的形成时间")
    with pytest.raises(ValueError, match="内容归属"):
        validate_evidence(Extraction.model_validate(data), "核对清单的形成时间")
    content = "清单包含形成时间"
    for entity in data["entities"]:
        entity["evidence_quote"] = content
    data["relations"][0]["evidence_quote"] = content
    validate_evidence(Extraction.model_validate(data), content)


@pytest.mark.asyncio
async def test_condition_evidence_failure_requests_corrected_quote():
    import copy
    import json

    content = "主管批准后，财务审核报销申请。"
    invalid = sample()
    invalid["relations"][0]["condition_text"] = "主管批准后"
    corrected = copy.deepcopy(invalid)
    corrected["relations"][0]["evidence_quote"] = content
    messages = []

    async def create(**kwargs):
        messages.append(copy.deepcopy(kwargs["messages"]))
        payload = invalid if len(messages) == 1 else corrected
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await extract_chunk(
        Settings(_env_file=None, deepseek_api_key="test"), content, client=client
    )
    assert len(messages) == 2
    assert "关系证据未包含对应条件" in messages[1][-1]["content"]
    assert result.relations[0].evidence_quote == content


@pytest.mark.asyncio
async def test_deepseek_json_mode_is_validated_before_write():
    class Client:
        def __init__(self, content):
            async def create(**kwargs):
                assert kwargs["response_format"] == {"type": "json_object"}
                assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop", message=SimpleNamespace(content=content)
                        )
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    settings = Settings(_env_file=None, deepseek_api_key="test")
    import json

    result = await extract_chunk(settings, "财务审核报销申请", client=Client(json.dumps(sample())))
    assert len(result.relations) == 1
    with pytest.raises(GraphExtractionError, match="SCHEMA_INVALID"):
        await extract_chunk(
            settings, "财务审核报销申请", client=Client('{"entities":[],"relations":[],"extra":1}')
        )


@pytest.mark.asyncio
async def test_deepseek_empty_output_is_retried_with_json_feedback():
    import json

    calls = []

    class Client:
        def __init__(self):
            async def create(**kwargs):
                calls.append(kwargs)
                content = "" if len(calls) == 1 else json.dumps(sample())
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop", message=SimpleNamespace(content=content)
                        )
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    result = await extract_chunk(
        Settings(_env_file=None, deepseek_api_key="test"), "财务审核报销申请", client=Client()
    )
    assert len(result.relations) == 1
    assert len(calls) == 2
    assert "返回为空" in calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_invalid_nonempty_candidate_cannot_become_successful_empty_graph():
    import json

    invalid = sample()
    invalid["entities"][0]["evidence_quote"] = "原文没有这句证据"
    responses = [json.dumps(invalid), '{"entities":[],"relations":[]}']
    calls = []

    class Client:
        def __init__(self):
            async def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(content=responses[min(len(calls) - 1, 1)]),
                        )
                    ]
                )

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    with pytest.raises(GraphExtractionError, match="OUTPUT_EMPTY_AFTER_INVALID"):
        await extract_chunk(
            Settings(_env_file=None, deepseek_api_key="test"),
            "财务审核报销申请",
            client=Client(),
        )
    assert len(calls) == 3
    assert "非空候选" in calls[2]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_payment_required_stops_graph_task_without_retry():
    failures = []
    worker = GraphWorker.__new__(GraphWorker)
    worker.settings = Settings(_env_file=None)
    worker.queue = SimpleNamespace(
        fail=lambda lease, code, retryable: failures.append((code, retryable))
    )

    async def pipeline(_lease):
        response = httpx.Response(402, request=httpx.Request("POST", "https://api.deepseek.com"))
        raise APIStatusError("Payment Required", response=response, body=None)

    worker.pipeline = pipeline
    lease = SimpleNamespace(task_id=1)
    await worker.process(lease)
    assert failures == [("MODEL_QUOTA_EXHAUSTED", False)]
    assert worker.model_paused


@pytest.mark.asyncio
async def test_network_failure_sets_worker_cooldown():
    import time

    failures = []
    worker = GraphWorker.__new__(GraphWorker)
    worker.settings = Settings(_env_file=None)
    worker.queue = SimpleNamespace(
        fail=lambda lease, code, retryable: failures.append((code, retryable))
    )

    async def pipeline(_lease):
        response = httpx.Response(503, request=httpx.Request("POST", "https://api.deepseek.com"))
        raise APIStatusError("Unavailable", response=response, body=None)

    worker.pipeline = pipeline
    await worker.process(SimpleNamespace(task_id=1))
    assert failures == [("MODEL_NETWORK_UNAVAILABLE", True)]
    assert worker.cooldown_until > time.monotonic()


@pytest.mark.asyncio
async def test_graph_cleanup_dispatches_and_acknowledges():
    calls = []

    async def delete_generation(*args):
        calls.append(args)

    worker = GraphWorker.__new__(GraphWorker)
    worker.store = SimpleNamespace(delete_generation=delete_generation)
    worker.cleanup = SimpleNamespace(complete=lambda lease: calls.append(lease.event_id))
    lease = CleanupLease(
        8,
        1,
        "GRAPH_PURGE_GENERATION",
        {"document_id": 2, "generation_id": 3, "namespace": "n"},
        "w",
    )
    await worker.process_cleanup(lease)
    assert calls == [(1, 2, 3, "n"), 8]


@pytest.mark.asyncio
async def test_graph_retry_resumes_after_completed_chunks(monkeypatch):
    from zhiqing_rag.workers import graph

    extracted = []
    checkpointed = []
    completed = []

    async def fake_extract(_settings, content):
        extracted.append(content)
        return Extraction(entities=[], relations=[])

    async def fake_write(_scope, chunk, _result):
        return {f"entity-{chunk['chunk_index']}"}, set()

    worker = GraphWorker.__new__(GraphWorker)
    worker.settings = Settings(_env_file=None)
    worker.queue = SimpleNamespace(
        scope=lambda _lease: ({"document_id": 1}, 4),
        checkpoint_state=lambda _lease: (2, {"entity-0", "entity-1"}, set()),
        chunk=lambda _lease, index: {"chunk_index": index, "content": f"chunk-{index}"},
        cached_extraction=lambda _lease, _chunk: None,
        save_extraction=lambda _lease, _chunk, _result: None,
        checkpoint=lambda _lease, index, count, entities, relations: checkpointed.append(
            (index, count, set(entities), set(relations))
        ),
        finish=lambda _lease, entities, relations: completed.append((entities, relations)),
    )
    worker.store = SimpleNamespace(write_chunk=fake_write)
    monkeypatch.setattr(graph, "extract_chunk", fake_extract)
    await worker.pipeline(SimpleNamespace(task_id=1))
    assert extracted == ["chunk-2", "chunk-3"]
    assert [row[0] for row in checkpointed] == [3, 4]
    assert completed == [(4, 0)]


@pytest.mark.asyncio
async def test_unchanged_chunk_reuses_extraction_but_writes_new_provenance(monkeypatch):
    from zhiqing_rag.workers import graph

    old_result = Extraction.model_validate(sample())
    written = []
    extracted = []
    saved = []
    first = {"id": 101, "chunk_index": 0, "content": "财务审核报销申请"}
    second = {"id": 102, "chunk_index": 1, "content": "新增的步骤"}

    async def fake_extract(_settings, content):
        extracted.append(content)
        return Extraction(entities=[], relations=[])

    async def fake_write(scope, chunk, result):
        written.append((scope["generation_id"], chunk["id"], result))
        return set(), set()

    worker = GraphWorker.__new__(GraphWorker)
    worker.settings = Settings(_env_file=None)
    worker.queue = SimpleNamespace(
        scope=lambda _lease: ({"generation_id": 9}, 2),
        checkpoint_state=lambda _lease: (0, set(), set()),
        chunk=lambda _lease, index: (first, second)[index],
        cached_extraction=lambda _lease, chunk: old_result if chunk["id"] == 101 else None,
        save_extraction=lambda _lease, chunk, _result: saved.append(chunk["id"]),
        checkpoint=lambda *_args: None,
        finish=lambda *_args: None,
    )
    worker.store = SimpleNamespace(write_chunk=fake_write)
    monkeypatch.setattr(graph, "extract_chunk", fake_extract)

    await worker.pipeline(SimpleNamespace(task_id=1))

    assert extracted == ["新增的步骤"]
    assert saved == [102]
    assert [(generation, chunk_id) for generation, chunk_id, _ in written] == [(9, 101), (9, 102)]
    assert written[0][2] == old_result


@pytest.mark.asyncio
async def test_graph_write_replays_same_keys_and_keeps_provenance():
    class Result:
        async def consume(self):
            return None

        async def single(self):
            return {"deleted": False, "retired": []}

    class Transaction:
        def __init__(self):
            self.calls = []

        async def run(self, query, **parameters):
            self.calls.append((query, parameters))
            return Result()

    class Session:
        def __init__(self, calls):
            self.calls = calls

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute_write(self, callback, *args):
            tx = Transaction()
            await callback(tx, *args)
            self.calls.append(tx.calls)

    class Driver:
        def __init__(self):
            self.calls = []

        def session(self, **kwargs):
            return Session(self.calls)

    store = GraphStore.__new__(GraphStore)
    store.settings = Settings(_env_file=None)
    store.driver = Driver()
    scope = {"tenant_id": 1, "kb_id": 2, "document_id": 3, "revision_id": 4, "generation_id": 5}
    chunk = {
        "id": 6,
        "chunk_index": 0,
        "content": "财务审核报销申请",
        "content_sha256": "a" * 64,
        "page_number": 2,
        "section_title": "报销",
        "start_offset": 10,
        "end_offset": 18,
        "metadata_": {"source_refs": [{"page": 2}]},
    }
    extraction = Extraction.model_validate(sample())
    first_keys = await store.write_chunk(scope, chunk, extraction)
    second_keys = await store.write_chunk(scope, chunk, extraction)
    assert first_keys == second_keys
    assert store.driver.calls[0] == store.driver.calls[1]
    queries = store.driver.calls[0]
    assert "GraphFence" in queries[0][0]
    assert "DETACH DELETE" in queries[1][0]
    entity_evidence = queries[3][1]["evidence"]
    assert entity_evidence["quote"] == "财务审核报销申请"
    assert entity_evidence["chunk_id"] == 6
    assert entity_evidence["revision_id"] == 4
    assert entity_evidence["quote_start"] == 0
    assert '"page": 2' in entity_evidence["source_refs_json"]
    await store.write_chunk(scope, chunk, Extraction(entities=[], relations=[]))
    assert len(store.driver.calls[-1]) == 3
    await store.delete_generation(1, 3, 5, store.settings.graph_namespace)
    cleanup = store.driver.calls[-1]
    assert "retired_generations" in cleanup[0][0]
    assert "DETACH DELETE a" in cleanup[1][0]
    assert "DETACH DELETE e" in cleanup[2][0]
    assert "NOT EXISTS" in cleanup[3][0]
