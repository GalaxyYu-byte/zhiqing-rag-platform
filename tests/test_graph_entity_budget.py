"""密集表格预算与严格校验的回归测试，不调用外部模型。"""

import json
from types import SimpleNamespace

import pytest

from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import extract_chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [24576, 16384])
async def test_dense_table_keeps_all_entities_and_passes_configured_output_budget(budget):
    names = [f"字段{i:02d}" for i in range(52)]
    content = "\n".join(names)
    payload = {"entities": [
        {"id": f"e{i}", "type": "Field", "name": name, "canonical_name": name,
         "evidence_quote": name} for i, name in enumerate(names)
    ], "relations": []}
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = await extract_chunk(
        Settings(_env_file=None, deepseek_api_key="test", graph_llm_max_output_tokens=budget),
        content, client=client,
    )
    assert requests[0]["max_tokens"] == budget
    assert len(requests) == 1
    assert [entity.name for entity in result.entities] == names


def test_output_budget_cannot_be_unbounded():
    with pytest.raises(ValueError):
        Settings(_env_file=None, graph_llm_max_output_tokens=1000000)
