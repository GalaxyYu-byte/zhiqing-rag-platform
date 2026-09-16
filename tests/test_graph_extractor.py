from dataclasses import dataclass

import pytest

from zq_rag_app.graph.extractor import QwenGraphExtractor
from zq_rag_app.graph.models import GraphChunkContext


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]


class _Completions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _Response([_Choice(_Message(self.content))])


class _Chat:
    def __init__(self, content: str) -> None:
        self.completions = _Completions(content)


class _Client:
    def __init__(self, content: str) -> None:
        self.chat = _Chat(content)


@pytest.mark.asyncio
async def test_qwen_extractor_uses_json_mode_and_validates_result() -> None:
    content = """{
      "schema_version": "1.0",
      "entities": [{
        "local_id": "e1",
        "entity_type": "TECHNOLOGY",
        "name": "Redis",
        "aliases": [],
        "description": null,
        "external_id_source": null,
        "external_id": null,
        "evidence_quote": "Redis"
      }],
      "relations": []
    }"""
    client = _Client(content)
    extractor = QwenGraphExtractor(
        client=client,
        model="qwen-test",
        retry_attempts=1,
    )
    context = GraphChunkContext(
        kb_id=1,
        doc_id=2,
        doc_version=1,
        chunk_index=0,
        file_name="manual.md",
        content="Redis 是缓存组件。",
    )

    result = await extractor.extract(context)

    assert result.entities[0].name == "Redis"
    kwargs = client.chat.completions.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["extra_body"] == {"enable_thinking": False}
    assert "JSON Schema" in kwargs["messages"][0]["content"]


@pytest.mark.asyncio
async def test_qwen_extractor_normalizes_owns_and_reverses_direction() -> None:
    content = """{
      "schema_version": "1.0",
      "entities": [
        {
          "local_id": "e1",
          "entity_type": "ORGANIZATION",
          "name": "甲公司",
          "aliases": [],
          "description": null,
          "external_id_source": null,
          "external_id": null,
          "evidence_quote": "甲公司"
        },
        {
          "local_id": "e2",
          "entity_type": "SYSTEM",
          "name": "订单系统",
          "aliases": [],
          "description": null,
          "external_id_source": null,
          "external_id": null,
          "evidence_quote": "订单系统"
        }
      ],
      "relations": [{
        "source_local_id": "e1",
        "target_local_id": "e2",
        "predicate": "OWNS",
        "polarity": "POSITIVE",
        "confidence": 0.9,
        "evidence_quote": "甲公司拥有订单系统",
        "valid_from": null,
        "valid_to": null
      }]
    }"""
    extractor = QwenGraphExtractor(
        client=_Client(content),
        model="qwen-test",
        retry_attempts=1,
    )
    context = GraphChunkContext(
        kb_id=1,
        doc_id=2,
        doc_version=1,
        chunk_index=0,
        file_name="manual.md",
        content="甲公司拥有订单系统。",
    )

    result = await extractor.extract(context)

    relation = result.relations[0]
    assert relation.source_local_id == "e2"
    assert relation.predicate == "OWNED_BY"
    assert relation.target_local_id == "e1"


def test_payload_normalization_drops_document_entities_and_normalizes_aliases() -> None:
    from zq_rag_app.graph.extractor import _normalize_model_payload

    payload = _normalize_model_payload(
        '{"schema_version":"1.0",'
        '"entities":[{"local_id":"e1","entity_type":"DOCUMENT",'
        '"name":"x","evidence_quote":"x"},'
        '{"local_id":"e2","entity_type":"SERVICE",'
        '"name":"y","evidence_quote":"y"}],'
        '"relations":[{"source_local_id":"e1","target_local_id":"e2",'
        '"predicate":"USED_BY"},{"source_local_id":"e2",'
        '"target_local_id":"e2","predicate":"APPLIES_TO"}]}'
    )

    assert [entity["local_id"] for entity in payload["entities"]] == ["e2"]
    assert payload["relations"] == []
