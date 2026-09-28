"""流式接收、内容校验、对象存储和文档修订持久化。"""

import logging
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from uuid import uuid4

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.format_validation import (
    DocumentFormatError,
    validate_document_format,
)
from zhiqing_rag.infrastructure.object_storage import DocumentObjectStorage

from .repository import UploadRepository
from .schemas import UploadError, UploadIdentity, UploadParameters, UploadResult

logger = logging.getLogger(__name__)


class DocumentUploadService:
    def __init__(
        self,
        repository: UploadRepository,
        storage: DocumentObjectStorage,
        settings: Settings,
    ):
        self.repository = repository
        self.storage = storage
        self.settings = settings

    def delete_document(self, identity: UploadIdentity, document_id: int) -> None:
        # 先提交任务取消和发布撤销；原文件清理失败时保留修订，供下一次请求重试。
        objects = self.repository.delete_document(identity, document_id)
        try:
            for item in objects:
                self.storage.remove(item["bucket"], item["object_key"])
        except Exception as error:
            logger.error("文档原文件清理失败：%s", type(error).__name__)
            raise UploadError(
                503, "DELETE_STORAGE_FAILED", "文档已停止处理，但原文件清理失败，请重试移除"
            ) from error
        self.repository.purge_document(identity, document_id)

    def original_document(self, identity: UploadIdentity, document_id: int, revision_id: int):
        row = self.repository.preview_revision(identity, document_id, revision_id)
        if row["file_size"] > self.settings.upload_max_file_size:
            raise UploadError(413, "FILE_TOO_LARGE", "文件超过预览大小限制")
        try:
            source = self.storage.client.get_object(row["bucket"], row["object_key"])
            try:
                content = source.read(self.settings.upload_max_file_size + 1)
            finally:
                source.close()
                source.release_conn()
        except Exception as error:
            raise UploadError(503, "STORAGE_UNAVAILABLE", "原文件读取失败，请稍后重试") from error
        if len(content) != row["file_size"] or sha256(content).hexdigest() != row["file_sha256"]:
            raise UploadError(409, "FILE_INTEGRITY_FAILED", "原文件完整性校验失败")
        return row, content

    def upload(
        self,
        source: BinaryIO,
        filename: str,
        mime_type: str | None,
        identity: UploadIdentity,
        parameters: UploadParameters,
    ) -> UploadResult:
        if (
            not filename
            or len(filename) > 500
            or filename in {".", ".."}
            or any(char in filename for char in ("/", "\\", "\x00"))
            or any(ord(char) < 32 for char in filename)
        ):
            raise UploadError(422, "INVALID_FILENAME", "文件名无效或超过 500 字符")
        self.repository.require_write(identity, parameters.knowledge_base_id)
        profile_id = self.repository.embedding_profile(self.settings)
        with TemporaryDirectory(prefix="zhiqing-upload-") as directory:
            path = Path(directory) / "document.tmp"
            size = 0
            with path.open("wb") as target:
                while block := source.read(1024 * 1024):
                    size += len(block)
                    if size > self.settings.upload_max_file_size:
                        raise UploadError(413, "FILE_TOO_LARGE", "文件超过大小限制")
                    target.write(block)
            try:
                validated = validate_document_format(
                    path,
                    original_filename=filename,
                    declared_mime_type=mime_type,
                    max_file_size=self.settings.upload_max_file_size,
                )
            except DocumentFormatError as error:
                raise UploadError(
                    413 if error.code == "FILE_TOO_LARGE" else 422, error.code, str(error)
                ) from error
            # UUID 保证同名及同内容的独立上传不会覆盖对象；不使用用户文件名构造路径。
            key = (
                f"{self.settings.minio_object_prefix.rstrip('/')}/{identity.tenant_id}/"
                f"{parameters.knowledge_base_id}/uploads/{uuid4().hex}/"
                f"{validated.file_sha256}.{validated.format.value}"
            )
            try:
                self.storage.put(key, path, validated.mime_type)
            except Exception as error:
                logger.error("文档对象写入失败：%s", type(error).__name__)
                raise UploadError(
                    503, "STORAGE_UNAVAILABLE", "文件存储暂不可用，请稍后重试"
                ) from error
            try:
                result = self.repository.persist(
                    identity,
                    parameters,
                    filename,
                    validated,
                    self.storage.bucket,
                    key,
                    profile_id,
                    self.settings.ingestion_max_attempts,
                )
                # 退出数据库连接前显式提交，确认文件、修订与任务已持久化才返回 202。
                self.repository.connection.commit()
                return result
            except UploadError:
                raise
            except Exception as error:
                # 提交结果可能未知；不要删掉可能已被已提交修订引用的对象。
                # 后续孤儿清理由对象前缀与已提交修订核对后处理。
                logger.error("文档记录保存失败：%s", type(error).__name__)
                raise UploadError(
                    503, "DATABASE_UNAVAILABLE", "文档记录保存失败，请稍后重试"
                ) from error
