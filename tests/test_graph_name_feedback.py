"""名称反馈可引导重试；失败候选不被自动改名或放宽校验。"""

import json
from types import SimpleNamespace

import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import GraphExtractionError, extract_chunk


def candidate(name="变更清单", canonical_name="变更清单", entity_type="Material"):
    return {"entities": [{"id": "e1", "type": entity_type, "name": name,
                          "canonical_name": canonical_name, "evidence_quote": "变更清单"}],
            "relations": []}


class Client:
    def __init__(self, responses):
        self.responses = responses
        self.messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.messages.append([dict(message) for message in kwargs["messages"]])
        response = self.responses[min(len(self.messages) - 1, len(self.responses) - 1)]
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(response)))])


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["name", "canonical_name"])
async def test_retry_names_invalid_field_and_quote_then_accepts_model_correction(field):
    invalid = candidate(entity_type="Step")
    invalid["entities"][0][field] = "核对变更清单"
    client = Client([invalid, candidate()])

    result = await extract_chunk(Settings(_env_file=None, deepseek_api_key="test"),
                                 "| 核对事项 |\n| 变更清单 |", client=client)

    assert len(client.messages) == 2
    feedback = client.messages[1][-1]["content"]
    assert f"entities[0].{field}" in feedback
    assert "核对变更清单" in feedback
    assert '逐字证据："变更清单"' in feedback
    assert "禁止自行补上" in feedback
    assert "重新判断名称和类型" in feedback
    assert result.entities[0].name == "变更清单"
    assert result.entities[0].type == "Material"


@pytest.mark.asyncio
async def test_invalid_name_is_not_automatically_stripped_or_accepted():
    invalid = candidate(name="核对变更清单", canonical_name="核对变更清单", entity_type="Step")
    client = Client([invalid])

    with pytest.raises(GraphExtractionError) as error:
        await extract_chunk(Settings(_env_file=None, deepseek_api_key="test"),
                            "变更清单", client=client)

    assert len(client.messages) == 3
    assert error.value.code == "EVIDENCE_INVALID"
    assert error.value.detail == "实体名称未出现在当前分块"
    assert invalid["entities"][0]["name"] == "核对变更清单"
