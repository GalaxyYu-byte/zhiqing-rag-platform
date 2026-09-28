"""永久删除只清理精确对象名，包括启用版本控制时的历史版本。"""

from types import SimpleNamespace

import pytest

from zhiqing_rag.infrastructure.object_storage import DocumentObjectStorage


@pytest.mark.parametrize("versions", [[], [None], ["v1", "v2", "delete-marker"]])
def test_remove_exact_object_and_all_versions(versions):
    removed = []

    def list_objects(bucket, *, prefix, include_version):
        assert (bucket, prefix, include_version) == ("documents", "owned-file", True)
        return [
            *(SimpleNamespace(object_name="owned-file", version_id=v) for v in versions),
            SimpleNamespace(object_name="owned-file-other", version_id="keep"),
        ]

    client = SimpleNamespace(
        list_objects=list_objects,
        remove_object=lambda bucket, key, *, version_id: removed.append((bucket, key, version_id)),
    )
    DocumentObjectStorage(client, "documents").remove("documents", "owned-file")
    assert removed == [("documents", "owned-file", v) for v in versions]
