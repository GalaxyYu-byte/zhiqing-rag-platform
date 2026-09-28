"""上传接口契约、文件校验、失败边界及本机开发身份限制。"""

from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.main import create_app
from zhiqing_rag.modules.documents.router import (
    get_upload_identity,
    get_upload_service,
    require_local_development,
)
from zhiqing_rag.modules.documents.schemas import (
    KnowledgeBaseOption,
    UploadError,
    UploadIdentity,
    UploadParameters,
    UploadResult,
)
from zhiqing_rag.modules.documents.service import DocumentUploadService

IDENTITY = UploadIdentity(tenant_id=1, member_id=2, user_id=3, department_id=None, clearance=2)


class FakeRepository:
    def __init__(self):
        self.saved = []
        self.commits = 0
        self.forbidden = False
        self.fail_save = False
        self.fail_commit = False
        self.connection = SimpleNamespace(commit=self.commit)

    def commit(self):
        if self.fail_commit:
            raise RuntimeError("secret database password")
        self.commits += 1

    def writable_bases(self, identity):
        return [
            KnowledgeBaseOption(id="11", code="TECH", name="技术知识库", description="技术资料")
        ]

    def require_write(self, identity, kb_id):
        if self.forbidden or kb_id != 11:
            raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "没有写入权限")

    def embedding_profile(self, settings):
        return 7

    def persist(
        self, identity, parameters, filename, validated, bucket, key, profile_id, max_attempts
    ):
        if self.fail_save:
            raise RuntimeError("secret database password")
        self.saved.append((identity, parameters, filename, validated, bucket, key))
        return UploadResult(
            document_id="9007199254740993",
            revision_id="42",
            generation_id="43",
            task_id="44",
            knowledge_base_id="11",
            original_filename=filename,
            file_size=validated.file_size,
            file_sha256=validated.file_sha256,
            mime_type=validated.mime_type,
            created_at=datetime.now(UTC),
        )


class FakeStorage:
    bucket = "documents"

    def __init__(self):
        self.objects = []
        self.removed = []
        self.fail = False

    def put(self, key, path, mime):
        if self.fail:
            raise RuntimeError("secret storage password")
        self.objects.append((key, path.read_bytes(), mime, path))

    def remove(self, bucket, key):
        if self.fail:
            raise RuntimeError("secret storage password")
        self.removed.append((bucket, key))


@pytest.fixture
def api():
    settings = Settings(_env_file=None, app_env="test", upload_max_file_size=64)
    repository = FakeRepository()
    storage = FakeStorage()
    service = DocumentUploadService(repository, storage, settings)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_upload_service] = lambda: service
    app.dependency_overrides[get_upload_identity] = lambda: IDENTITY
    with TestClient(app) as client:
        yield client, repository, storage


def upload(
    client, name="手册.md", content=b"# Handbook\n\nKnowledge.", mime="text/markdown", **data
):
    return client.post(
        "/api/documents/upload",
        files={"file": (name, content, mime)},
        data={"knowledge_base_id": "11", **data},
    )


def test_upload_persists_validated_file_and_returns_string_ids(api):
    client, repository, storage = api
    response = upload(client, chunk_size="256", chunk_overlap="32")
    assert response.status_code == 202
    body = response.json()
    assert body["document_id"] == "9007199254740993"
    assert body["status"] == "STORED" and body["processing_status"] == "PENDING"
    assert body["mime_type"] == "text/markdown"
    assert len(body["file_sha256"]) == 64
    assert repository.commits == 1
    assert repository.saved[0][1].chunk_size == 512
    assert repository.saved[0][1].chunk_overlap == 48
    key, data, mime, path = storage.objects[0]
    assert key.startswith("zhiqing-rag-platform/1/11/uploads/")
    assert "手册" not in key
    assert data == b"# Handbook\n\nKnowledge."
    assert not path.exists(), "临时目录必须释放"


def test_writable_knowledge_bases(api):
    response = api[0].get("/api/knowledge-bases")
    assert response.status_code == 200
    assert response.json()[0]["id"] == "11"


def test_delete_document_returns_empty_success(api, monkeypatch):
    client, repository, _ = api
    calls = []

    def prepare(identity, doc_id):
        calls.append((identity, doc_id))
        return [{"bucket": "documents", "object_key": "test-key"}]

    monkeypatch.setattr(repository, "delete_document", prepare, raising=False)
    purged = []
    monkeypatch.setattr(
        repository, "purge_document", lambda *args: purged.append(args), raising=False
    )
    response = client.delete("/api/documents/9007199254740993")
    assert response.status_code == 204 and not response.content
    assert calls == [(IDENTITY, 9007199254740993)]
    assert purged == calls
    assert api[2].removed == [("documents", "test-key")]


def test_delete_storage_failure_retains_database_records_for_retry(api, monkeypatch):
    client, repository, storage = api
    monkeypatch.setattr(
        repository,
        "delete_document",
        lambda *args: [
            {"bucket": "documents", "object_key": "test-key"},
        ],
        raising=False,
    )
    purged = []
    monkeypatch.setattr(
        repository, "purge_document", lambda *args: purged.append(args), raising=False
    )
    storage.fail = True
    response = client.delete("/api/documents/33")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DELETE_STORAGE_FAILED"
    assert "secret" not in response.text
    assert not purged
    storage.fail = False
    assert client.delete("/api/documents/33").status_code == 204
    assert purged == [(IDENTITY, 33)]


@pytest.mark.parametrize("status", [403, 404, 409])
def test_delete_document_preserves_error_response(api, monkeypatch, status):
    def reject(identity, doc_id):
        raise UploadError(status, "DELETE_REJECTED", "文档无法移除")

    monkeypatch.setattr(api[1], "delete_document", reject, raising=False)
    response = api[0].delete("/api/documents/33")
    assert response.status_code == status
    assert response.json()["detail"] == {"code": "DELETE_REJECTED", "message": "文档无法移除"}


def test_delete_cors_preflight(api):
    response = api[0].options(
        "/api/documents/33",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert response.status_code == 200
    assert "DELETE" in response.headers["access-control-allow-methods"]


@pytest.mark.parametrize(
    ("name", "content", "mime", "code", "status"),
    [
        ("empty.txt", b"", "text/plain", "FILE_EMPTY", 422),
        ("large.txt", b"a" * 65, "text/plain", "FILE_TOO_LARGE", 413),
        ("old.doc", b"data", "application/octet-stream", "UNSUPPORTED_FORMAT", 422),
        ("fake.pdf", b"not a pdf", "application/pdf", "FILE_TYPE_MISMATCH", 422),
        ("fake.txt", b"%PDF-1.7", "text/plain", "FILE_TYPE_MISMATCH", 422),
        ("wrong.md", b"# test", "image/png", "FILE_TYPE_MISMATCH", 422),
        ("../test.md", b"# test", "text/markdown", "INVALID_FILENAME", 422),
        ("blank.txt", b"   ", "text/plain", "FILE_EMPTY", 422),
    ],
)
def test_invalid_files_never_reach_storage(api, name, content, mime, code, status):
    client, repository, storage = api
    response = upload(client, name, content, mime)
    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
    assert not storage.objects and not repository.saved


@pytest.mark.parametrize(
    "data",
    [
        {"knowledge_base_id": "QA_TECH"},
        {"knowledge_base_id": "0"},
    ],
)
def test_invalid_parameters(api, data):
    client, repository, storage = api
    assert upload(client, **data).status_code == 422
    assert not storage.objects and not repository.saved


@pytest.mark.parametrize(
    "data",
    [
        {"chunk_size": "127"},
        {"chunk_size": "2049"},
        {"chunk_overlap": "-1"},
        {"chunk_size": "256", "chunk_overlap": "256"},
    ],
)
def test_upload_ignores_user_chunk_settings(api, data):
    client, repository, _ = api
    assert upload(client, **data).status_code == 202
    assert repository.saved[0][1].chunk_size == 512
    assert repository.saved[0][1].chunk_overlap == 48


def test_upload_uses_admin_server_chunk_configuration(api):
    client, repository, _ = api
    settings = client.app.dependency_overrides[get_upload_service]().settings
    settings.upload_chunk_size = 256
    settings.upload_chunk_overlap = 32
    assert upload(client, chunk_size="2048", chunk_overlap="100").status_code == 202
    assert repository.saved[0][1].chunk_size == 256
    assert repository.saved[0][1].chunk_overlap == 32


def test_forbidden_base_does_not_write(api):
    client, repository, storage = api
    repository.forbidden = True
    assert upload(client).status_code == 403
    assert not storage.objects and not repository.saved


def test_storage_failure_does_not_persist_or_expose_credentials(api):
    client, repository, storage = api
    storage.fail = True
    response = upload(client)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "STORAGE_UNAVAILABLE"
    assert "secret" not in response.text
    assert not repository.saved


@pytest.mark.parametrize("failure", ["fail_save", "fail_commit"])
def test_database_failure_never_claims_success(api, failure):
    client, repository, storage = api
    setattr(repository, failure, True)
    response = upload(client)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DATABASE_UNAVAILABLE"
    assert "secret" not in response.text
    assert not storage.objects[0][3].exists()


def test_identical_uploads_use_distinct_keys(api):
    client, _, storage = api
    assert upload(client).status_code == 202
    assert upload(client).status_code == 202
    assert storage.objects[0][0] != storage.objects[1][0]


@pytest.mark.parametrize("filename", [r"..\test.md", "a\x00.md", "a\n.md", "a" * 501])
def test_invalid_filename_service(api, filename):
    _, repository, storage = api
    service = DocumentUploadService(repository, storage, Settings(_env_file=None))
    with pytest.raises(UploadError, match="文件名"):
        service.upload(
            BytesIO(b"# test"),
            filename,
            "text/markdown",
            IDENTITY,
            UploadParameters(knowledge_base_id=11),
        )
    assert not storage.objects


@pytest.mark.parametrize(
    "host,env,enabled,origin",
    [
        ("192.168.1.2", "dev", True, None),
        ("127.0.0.1", "test", True, None),
        ("127.0.0.1", "dev", False, None),
        ("127.0.0.1", "dev", True, "https://untrusted.example"),
    ],
)
def test_development_identity_is_restricted(host, env, enabled, origin):
    headers = [(b"origin", origin.encode())] if origin else []
    request = Request({"type": "http", "client": (host, 1234), "headers": headers})
    settings = Settings(_env_file=None, app_env=env, dev_upload_identity_enabled=enabled)
    with pytest.raises(UploadError):
        require_local_development(request, settings)


def test_loopback_development_identity_allowed():
    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})
    require_local_development(request, Settings(_env_file=None))


def test_cors_preflight(api):
    response = api[0].options(
        "/api/documents/upload",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_request_body_limit_before_multipart_parsing(api):
    response = api[0].post(
        "/api/documents/upload", content=b"test", headers={"Content-Length": str(100 * 1024 * 1024)}
    )
    assert response.status_code == 413


def test_streamed_body_limit_without_content_length(monkeypatch):
    import zhiqing_rag.main as main_module

    monkeypatch.setattr(
        main_module, "get_settings", lambda: Settings(_env_file=None, upload_max_file_size=64)
    )
    app = create_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/documents/upload",
            headers={"Content-Type": "multipart/form-data; boundary=upload-test"},
            content=iter(
                [
                    b'--upload-test\r\nContent-Disposition: form-data; name="file"; '
                    b'filename="big.txt"\r\nContent-Type: text/plain\r\n\r\n',
                    b"a" * (1024 * 1024 + 65),
                    b"\r\n--upload-test--\r\n",
                ]
            ),
        )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"


def test_development_identity_rejects_public_binding():
    request = Request({"type": "http", "client": ("127.0.0.1", 1234), "headers": []})
    settings = Settings(_env_file=None, app_host="0.0.0.0")
    with pytest.raises(UploadError):
        require_local_development(request, settings)
