"""The committed graph evaluation input must remain complete and immutable."""

import shutil
from pathlib import Path

import pytest

from scripts.freeze_graph_eval import verify

SNAPSHOT = Path(__file__).resolve().parents[1] / "eval" / "graph_build" / "v1"


def test_frozen_graph_input_integrity(tmp_path: Path):
    verify(SNAPSHOT)

    changed = tmp_path / "changed"
    shutil.copytree(SNAPSHOT, changed)
    chunks = changed / "chunks.jsonl"
    chunks.write_bytes(chunks.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="chunks.jsonl"):
        verify(changed)
