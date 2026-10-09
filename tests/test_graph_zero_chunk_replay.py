"""Keep raw model output when the current extractor accepts an empty graph."""

from types import SimpleNamespace

import pytest

from scripts.replay_graph_zero_chunks import RecordingClient
from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers.graph_extraction import extract_chunk


@pytest.mark.asyncio
async def test_empty_extraction_is_valid_and_raw_response_is_recorded():
    class FakeCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                model="test-model",
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content='{"entities":[],"relations":[]}'),
                    )
                ],
                usage=None,
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    trace = []
    settings = Settings(_env_file=None, deepseek_api_key="test-key")

    result = await extract_chunk(
        settings, "用人部门确认岗位安排。", client=RecordingClient(fake_client, trace)
    )

    assert result.entities == []
    assert result.relations == []
    assert len(trace) == 1
    assert trace[0]["raw_output"] == '{"entities":[],"relations":[]}'
    assert trace[0]["finish_reason"] == "stop"
