"""上传用例的数据库访问；所有查询显式约束租户与当前授权。"""

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.format_validation import ValidatedDocument

from .schemas import (
    DocumentTaskPage,
    DocumentTaskStatus,
    KnowledgeBaseOption,
    UploadError,
    UploadIdentity,
    UploadParameters,
    UploadResult,
)


class UploadRepository:
    def __init__(self, connection: Connection, schema: str):
        self.connection = connection
        self.schema = schema

    def query(self, statement: str, parameters=()):
        return self.connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.schema)), parameters
        )

    def identity(self, email: str, tenant_code: str) -> UploadIdentity:
        row = self.query(
            """SELECT m.tenant_id, m.id AS member_id, m.user_id, m.department_id, m.clearance
               FROM {schema}.tenant_member m
               JOIN {schema}.tenant t ON t.id = m.tenant_id
               JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE t.code = %s AND u.email = %s AND t.status = 'ACTIVE'
                 AND m.status = 'ACTIVE' AND u.is_active
                 AND (m.department_id IS NULL OR d.is_active)""",
            (tenant_code, email),
        ).fetchone()
        if row is None:
            raise UploadError(403, "IDENTITY_UNAVAILABLE", "开发身份不存在或已停用")
        return UploadIdentity(**row)

    def writable_bases(self, identity: UploadIdentity) -> list[KnowledgeBaseOption]:
        rows = self.query(
            """SELECT kb.id, kb.code, kb.name, kb.description
               FROM {schema}.knowledge_base kb
               WHERE kb.tenant_id = %s AND kb.status = 'ACTIVE'
                 AND kb.deleted_at IS NULL AND EXISTS (
                   SELECT 1 FROM {schema}.knowledge_base_grant g
                   WHERE g.tenant_id = kb.tenant_id AND g.kb_id = kb.id
                     AND g.permission IN ('WRITE', 'ADMIN')
                     AND (g.user_id = %s OR g.department_id = %s))
               ORDER BY kb.id""",
            (identity.tenant_id, identity.user_id, identity.department_id),
        ).fetchall()
        return [
            KnowledgeBaseOption(
                id=str(row["id"]),
                code=row["code"],
                name=row["name"],
                description=row["description"] or "",
            )
            for row in rows
        ]

    def require_write(self, identity: UploadIdentity, kb_id: int) -> None:
        if not any(base.id == str(kb_id) for base in self.writable_bases(identity)):
            raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库不可用或没有写入权限")

    def embedding_profile(self, settings: Settings) -> int:
        row = self.query(
            """SELECT id FROM {schema}.embedding_profile
               WHERE provider = 'dashscope' AND model_name = %s AND dimensions = %s
                 AND distance_metric = 'cosine' ORDER BY id DESC LIMIT 1""",
            (settings.embedding_model, settings.embedding_dimensions),
        ).fetchone()
        if row is None or settings.embedding_dimensions != 1024:
            raise UploadError(
                503, "EMBEDDING_PROFILE_UNAVAILABLE", "没有匹配的 1024 维向量模型配置"
            )
        return row["id"]

    def persist(
        self,
        identity: UploadIdentity,
        parameters: UploadParameters,
        filename: str,
        validated: ValidatedDocument,
        bucket: str,
        object_key: str,
        profile_id: int,
        max_attempts: int,
    ) -> UploadResult:
        # 对象已保存后再次检查身份与权限，避免前置校验过期。
        with self.connection.transaction():
            member = self.query(
                """SELECT m.id FROM {schema}.tenant_member m
                   JOIN {schema}.tenant t ON t.id = m.tenant_id
                   JOIN {schema}.user_account u ON u.id = m.user_id
                   LEFT JOIN {schema}.department d
                     ON d.tenant_id = m.tenant_id AND d.id = m.department_id
                   WHERE m.tenant_id = %s AND m.id = %s AND m.user_id = %s
                     AND m.status = 'ACTIVE' AND t.status = 'ACTIVE' AND u.is_active
                     AND (m.department_id IS NULL OR d.is_active)
                     AND m.department_id IS NOT DISTINCT FROM %s
                     AND m.clearance = %s
                   FOR SHARE OF m, t, u""",
                (
                    identity.tenant_id,
                    identity.member_id,
                    identity.user_id,
                    identity.department_id,
                    identity.clearance,
                ),
            ).fetchone()
            if member is None:
                raise UploadError(403, "IDENTITY_UNAVAILABLE", "身份或授权已变更，请刷新重试")
            if identity.department_id is not None:
                department = self.query(
                    """SELECT id FROM {schema}.department
                       WHERE tenant_id = %s AND id = %s AND is_active FOR SHARE""",
                    (identity.tenant_id, identity.department_id),
                ).fetchone()
                if department is None:
                    raise UploadError(403, "IDENTITY_UNAVAILABLE", "所属部门已停用")
            kb = self.query(
                """SELECT id FROM {schema}.knowledge_base
                   WHERE tenant_id = %s AND id = %s AND status = 'ACTIVE'
                   AND deleted_at IS NULL FOR SHARE""",
                (identity.tenant_id, parameters.knowledge_base_id),
            ).fetchone()
            if kb is None:
                raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库不可用")
            grants = self.query(
                """SELECT id FROM {schema}.knowledge_base_grant
                   WHERE tenant_id = %s AND kb_id = %s
                     AND permission IN ('WRITE', 'ADMIN')
                     AND (user_id = %s OR department_id = %s) FOR SHARE""",
                (
                    identity.tenant_id,
                    parameters.knowledge_base_id,
                    identity.user_id,
                    identity.department_id,
                ),
            ).fetchall()
            if not grants:
                raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库写入权限已撤销")
            metadata = {
                "chunk_config": {
                    "max_tokens": parameters.chunk_size,
                    "overlap_tokens": parameters.chunk_overlap,
                }
            }
            document = self.query(
                """INSERT INTO {schema}.document
                   (tenant_id, kb_id, title, department_id, confidentiality, created_by, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                (
                    identity.tenant_id,
                    parameters.knowledge_base_id,
                    filename,
                    identity.department_id,
                    identity.clearance,
                    identity.member_id,
                    Jsonb(metadata),
                ),
            ).fetchone()
            revision = self.query(
                """INSERT INTO {schema}.document_revision
                   (tenant_id, document_id, revision_no, original_filename, mime_type,
                    file_size, file_sha256, bucket, object_key, uploaded_by)
                   VALUES (%s, %s, 1, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id, created_at""",
                (
                    identity.tenant_id,
                    document["id"],
                    filename,
                    validated.mime_type,
                    validated.file_size,
                    validated.file_sha256,
                    bucket,
                    object_key,
                    identity.member_id,
                ),
            ).fetchone()
            self.query(
                """UPDATE {schema}.document SET desired_revision_id = %s
                   WHERE tenant_id = %s AND id = %s""",
                (revision["id"], identity.tenant_id, document["id"]),
            )
            generation = self.query(
                """INSERT INTO {schema}.index_generation
                   (tenant_id, document_id, revision_id, embedding_profile_id, generation_no,
                    parser_version, cleaner_version, chunker_version, chunk_config)
                   VALUES (%s, %s, %s, %s, 1, 'document-processing-v1', 'cleaning-v1',
                           'chunking-v1', %s) RETURNING id""",
                (
                    identity.tenant_id,
                    document["id"],
                    revision["id"],
                    profile_id,
                    Jsonb(metadata["chunk_config"]),
                ),
            ).fetchone()
            task = self.query(
                """INSERT INTO {schema}.background_task
                   (tenant_id, task_type, document_id, revision_id, generation_id,
                    idempotency_key, stage, max_attempts, payload)
                   VALUES (%s, 'INDEX', %s, %s, %s, %s, 'QUEUED', %s, %s) RETURNING id""",
                (
                    identity.tenant_id,
                    document["id"],
                    revision["id"],
                    generation["id"],
                    f"ingestion:{generation['id']}",
                    max_attempts,
                    Jsonb({"pipeline": "document-ingestion-v1"}),
                ),
            ).fetchone()
        return UploadResult(
            document_id=str(document["id"]),
            revision_id=str(revision["id"]),
            generation_id=str(generation["id"]),
            task_id=str(task["id"]),
            knowledge_base_id=str(parameters.knowledge_base_id),
            original_filename=filename,
            file_size=validated.file_size,
            file_sha256=validated.file_sha256,
            mime_type=validated.mime_type,
            created_at=revision["created_at"],
        )

    def preview_revision(self, identity: UploadIdentity, document_id: int, revision_id: int):
        row = self.query(
            """SELECT rev.original_filename, rev.mime_type, rev.file_size, rev.file_sha256,
                      rev.bucket, rev.object_key
               FROM {schema}.document_revision rev
               JOIN {schema}.document doc ON doc.tenant_id = rev.tenant_id
                 AND doc.id = rev.document_id
               JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id
                 AND kb.id = doc.kb_id
               WHERE doc.tenant_id = %s AND doc.id = %s AND rev.id = %s
                 AND doc.deleted_at IS NULL AND doc.status <> 'WITHDRAWN'
                 AND doc.confidentiality <= %s AND kb.status = 'ACTIVE'
                 AND kb.deleted_at IS NULL AND EXISTS (
                   SELECT 1 FROM {schema}.knowledge_base_grant g
                   WHERE g.tenant_id = kb.tenant_id AND g.kb_id = kb.id
                     AND g.permission IN ('WRITE', 'ADMIN')
                     AND (g.user_id = %s OR g.department_id = %s))""",
            (
                identity.tenant_id,
                document_id,
                revision_id,
                identity.clearance,
                identity.user_id,
                identity.department_id,
            ),
        ).fetchone()
        if row is None:
            raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
        return row

    def task_rows(self, identity: UploadIdentity, task_id: int | None = None):
        return self.query(
            """SELECT task.id AS task_id, task.document_id, task.revision_id, task.generation_id,
                      doc.kb_id AS knowledge_base_id, rev.original_filename, rev.file_size,
                      task.status, task.stage, task.progress, task.attempt_count, task.max_attempts,
                      task.error_code, gen.chunk_count, doc.status AS document_status,
                      task.created_at, task.finished_at
               FROM {schema}.background_task task
               JOIN {schema}.document doc ON doc.tenant_id = task.tenant_id
                 AND doc.id = task.document_id
               JOIN {schema}.document_revision rev ON rev.tenant_id = task.tenant_id
                 AND rev.id = task.revision_id AND rev.document_id = doc.id
               JOIN {schema}.index_generation gen ON gen.tenant_id = task.tenant_id
                 AND gen.id = task.generation_id AND gen.document_id = doc.id
               JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE task.tenant_id = %s AND task.payload->>'pipeline' = 'document-ingestion-v1'
                 AND (%s::bigint IS NULL OR task.id = %s)
                 AND doc.confidentiality <= %s AND kb.status = 'ACTIVE'
                 AND kb.deleted_at IS NULL AND EXISTS (
                   SELECT 1 FROM {schema}.knowledge_base_grant grant_row
                   WHERE grant_row.tenant_id = kb.tenant_id AND grant_row.kb_id = kb.id
                     AND grant_row.permission IN ('WRITE', 'ADMIN')
                     AND (grant_row.user_id = %s OR grant_row.department_id = %s))
               ORDER BY task.id DESC LIMIT 50""",
            (
                identity.tenant_id,
                task_id,
                task_id,
                identity.clearance,
                identity.user_id,
                identity.department_id,
            ),
        ).fetchall()

    @staticmethod
    def task_response(row) -> DocumentTaskStatus:
        return DocumentTaskStatus(
            **{
                key: str(value)
                if key
                in {"task_id", "document_id", "revision_id", "generation_id", "knowledge_base_id"}
                else value
                for key, value in row.items()
            }
        )

    def task_status(self, identity: UploadIdentity, task_id: int) -> DocumentTaskStatus:
        rows = self.task_rows(identity, task_id)
        if not rows:
            raise UploadError(404, "TASK_NOT_FOUND", "任务不存在或没有访问权限")
        return self.task_response(rows[0])

    def list_tasks(self, identity: UploadIdentity) -> list[DocumentTaskStatus]:
        return [self.task_response(row) for row in self.task_rows(identity)]

    def paginate_tasks(
        self,
        identity: UploadIdentity,
        *,
        offset: int,
        limit: int,
        knowledge_base_id: int | None,
        search: str,
        status: str,
        start,
        end,
    ) -> DocumentTaskPage:
        row = self.query(
            """WITH scoped AS (
               SELECT task.id AS task_id, task.document_id, task.revision_id, task.generation_id,
                      doc.kb_id AS knowledge_base_id, rev.original_filename, rev.file_size,
                      task.status, task.stage, task.progress, task.attempt_count, task.max_attempts,
                      task.error_code, gen.chunk_count, doc.status AS document_status,
                      task.created_at, task.finished_at,
                      CASE WHEN doc.status = 'DELETED' OR task.status = 'FAILED' THEN 'error'
                           WHEN task.status = 'SUCCEEDED' THEN 'done'
                           WHEN task.status = 'CANCELLED' THEN 'cancelled'
                           ELSE 'processing' END AS queue_status
               FROM {schema}.background_task task
               JOIN {schema}.document doc ON doc.tenant_id = task.tenant_id
                 AND doc.id = task.document_id
               JOIN {schema}.document_revision rev ON rev.tenant_id = task.tenant_id
                 AND rev.id = task.revision_id AND rev.document_id = doc.id
               JOIN {schema}.index_generation gen ON gen.tenant_id = task.tenant_id
                 AND gen.id = task.generation_id AND gen.document_id = doc.id
               JOIN {schema}.knowledge_base kb ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE task.tenant_id = %s AND task.payload->>'pipeline' = 'document-ingestion-v1'
                 AND doc.confidentiality <= %s AND kb.status = 'ACTIVE'
                 AND kb.deleted_at IS NULL AND EXISTS (
                   SELECT 1 FROM {schema}.knowledge_base_grant g
                   WHERE g.tenant_id = kb.tenant_id AND g.kb_id = kb.id
                     AND g.permission IN ('WRITE', 'ADMIN')
                     AND (g.user_id = %s OR g.department_id = %s))
            ), matching AS (
               SELECT * FROM scoped
               WHERE (%s::bigint IS NULL OR knowledge_base_id = %s)
                 AND strpos(lower(original_filename), lower(%s)) > 0
                 AND (%s::timestamptz IS NULL OR created_at >= %s)
                 AND (%s::timestamptz IS NULL OR created_at < %s)
            ), filtered AS (
               SELECT * FROM matching WHERE %s = 'all' OR queue_status = %s
            ), page_rows AS (
               SELECT * FROM filtered ORDER BY created_at DESC, task_id DESC LIMIT %s OFFSET %s
            )
            SELECT (SELECT coalesce(jsonb_agg(to_jsonb(page_rows)
                           ORDER BY created_at DESC, task_id DESC), '[]'::jsonb)
                    FROM page_rows) items,
                   (SELECT count(*) FROM filtered) total,
                   (SELECT count(*) FROM scoped) overall_total,
                   (SELECT coalesce(sum(file_size), 0) FROM scoped) overall_size,
                   count(*) AS all_count,
                   count(*) FILTER (WHERE queue_status = 'done') AS done_count,
                   count(*) FILTER (WHERE queue_status = 'processing') AS processing_count,
                   count(*) FILTER (WHERE queue_status = 'error') AS error_count,
                   count(*) FILTER (WHERE queue_status = 'cancelled') AS cancelled_count
            FROM matching""",
            (
                identity.tenant_id,
                identity.clearance,
                identity.user_id,
                identity.department_id,
                knowledge_base_id,
                knowledge_base_id,
                search,
                start,
                start,
                end,
                end,
                status,
                status,
                limit,
                offset,
            ),
        ).fetchone()
        return DocumentTaskPage(
            items=[self.task_response(item) for item in row["items"]],
            total=row["total"],
            overall_total=row["overall_total"],
            overall_size=row["overall_size"],
            counts={
                name: row[f"{name}_count"]
                for name in ("all", "done", "processing", "error", "cancelled")
            },
            offset=offset,
            limit=limit,
        )

    def delete_document(self, identity: UploadIdentity, document_id: int) -> list[dict]:
        with self.connection.transaction():
            # 与重试、Worker 发布保持任务 → 文档的锁顺序，防止删除后重新发布。
            self.query(
                """SELECT id FROM {schema}.background_task
                   WHERE tenant_id = %s AND document_id = %s ORDER BY id FOR UPDATE""",
                (identity.tenant_id, document_id),
            ).fetchall()
            document = self.query(
                """SELECT kb_id, deleted_at FROM {schema}.document
                   WHERE tenant_id = %s AND id = %s AND confidentiality <= %s FOR UPDATE""",
                (identity.tenant_id, document_id, identity.clearance),
            ).fetchone()
            if document is None:
                raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
            kb = self.query(
                """SELECT id FROM {schema}.knowledge_base
                   WHERE tenant_id = %s AND id = %s AND status = 'ACTIVE'
                     AND deleted_at IS NULL FOR SHARE""",
                (identity.tenant_id, document["kb_id"]),
            ).fetchone()
            grants = self.query(
                """SELECT id FROM {schema}.knowledge_base_grant
                   WHERE tenant_id = %s AND kb_id = %s AND permission IN ('WRITE', 'ADMIN')
                     AND (user_id = %s OR department_id = %s) FOR SHARE""",
                (identity.tenant_id, document["kb_id"], identity.user_id, identity.department_id),
            ).fetchall()
            if kb is None or not grants:
                raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库不可用或没有写入权限")
            objects = self.query(
                """SELECT DISTINCT rev.bucket, rev.object_key FROM {schema}.document_revision rev
                   WHERE rev.tenant_id = %s AND rev.document_id = %s""",
                (identity.tenant_id, document_id),
            ).fetchall()
            shared = self.query(
                """SELECT 1 FROM {schema}.document_revision other
                   JOIN {schema}.document_revision rev
                     ON rev.bucket = other.bucket AND rev.object_key = other.object_key
                   WHERE rev.tenant_id = %s AND rev.document_id = %s
                     AND (other.tenant_id <> rev.tenant_id OR other.document_id <> rev.document_id)
                   LIMIT 1""",
                (identity.tenant_id, document_id),
            ).fetchone()
            if shared:
                raise UploadError(409, "OBJECT_SHARED", "原文件被其他文档引用，无法彻底删除")
            if document["deleted_at"] is None:
                self.query(
                    """UPDATE {schema}.document SET status = 'DELETED',
                       deleted_at = clock_timestamp(), active_generation_id = NULL,
                       desired_revision_id = NULL WHERE tenant_id = %s AND id = %s""",
                    (identity.tenant_id, document_id),
                )
                self.query(
                    """UPDATE {schema}.background_task SET status = 'CANCELLED',
                       stage = 'CANCELLED', error_code = 'DOCUMENT_DELETED',
                       finished_at = clock_timestamp(), lease_owner = NULL,
                       lease_expires_at = NULL, lease_epoch = lease_epoch + 1
                       WHERE tenant_id = %s AND document_id = %s
                         AND status IN ('PENDING', 'RUNNING', 'RETRY_WAIT')""",
                    (identity.tenant_id, document_id),
                )
                self.query(
                    """UPDATE {schema}.task_attempt SET status = 'CANCELLED',
                       error_code = 'DOCUMENT_DELETED', finished_at = clock_timestamp()
                       WHERE tenant_id = %s AND status = 'RUNNING' AND task_id IN (
                         SELECT id FROM {schema}.background_task
                         WHERE tenant_id = %s AND document_id = %s)""",
                    (identity.tenant_id, identity.tenant_id, document_id),
                )
                self.query(
                    """UPDATE {schema}.index_generation SET status = 'FAILED'
                       WHERE tenant_id = %s AND document_id = %s AND status = 'BUILDING'""",
                    (identity.tenant_id, document_id),
                )
        self.connection.commit()
        return objects

    def purge_document(self, identity: UploadIdentity, document_id: int) -> None:
        """原文件清理成功后，按外键依赖顺序原子删除全部文档数据。"""
        with self.connection.transaction():
            self.query(
                """SELECT id FROM {schema}.background_task
                   WHERE tenant_id = %s AND document_id = %s ORDER BY id FOR UPDATE""",
                (identity.tenant_id, document_id),
            ).fetchall()
            document = self.query(
                """SELECT status FROM {schema}.document
                   WHERE tenant_id = %s AND id = %s FOR UPDATE""",
                (identity.tenant_id, document_id),
            ).fetchone()
            if document is not None:
                if document["status"] != "DELETED":
                    raise UploadError(409, "DELETE_NOT_PREPARED", "文档尚未停止处理，无法彻底删除")
                self.query(
                    """DELETE FROM {schema}.outbox_event WHERE tenant_id = %s AND (
                       task_id IN (SELECT id FROM {schema}.background_task
                         WHERE tenant_id = %s AND document_id = %s)
                       OR (lower(aggregate_type) = 'document' AND aggregate_id = %s))""",
                    (identity.tenant_id, identity.tenant_id, document_id, str(document_id)),
                )
                self.query(
                    """DELETE FROM {schema}.task_attempt WHERE tenant_id = %s AND task_id IN (
                       SELECT id FROM {schema}.background_task
                       WHERE tenant_id = %s AND document_id = %s)""",
                    (identity.tenant_id, identity.tenant_id, document_id),
                )
                for table in (
                    "answer_citation",
                    "graph_projection",
                    "background_task",
                    "document_chunk",
                    "index_generation",
                    "document_revision",
                ):
                    self.query(
                        f"DELETE FROM {{schema}}.{table} WHERE tenant_id = %s AND document_id = %s",
                        (identity.tenant_id, document_id),
                    )
                self.query(
                    "DELETE FROM {schema}.document WHERE tenant_id = %s AND id = %s",
                    (identity.tenant_id, document_id),
                )
        self.connection.commit()

    def retry_task(self, identity: UploadIdentity, task_id: int) -> DocumentTaskStatus:
        self.task_status(identity, task_id)
        with self.connection.transaction():
            row = self.query(
                """SELECT status, generation_id, attempt_count, document_id, revision_id
                   FROM {schema}.background_task
                   WHERE tenant_id = %s AND id = %s FOR UPDATE""",
                (identity.tenant_id, task_id),
            ).fetchone()
            if row is None:
                raise UploadError(404, "TASK_NOT_FOUND", "任务已删除或没有访问权限")
            if row["status"] != "FAILED" or row["attempt_count"] >= 100:
                raise UploadError(409, "TASK_NOT_RETRYABLE", "任务未失败或已达到最大尝试次数")
            document = self.query(
                """SELECT id FROM {schema}.document
                   WHERE tenant_id = %s AND id = %s AND desired_revision_id = %s
                     AND status IN ('DRAFT', 'PUBLISHED') AND deleted_at IS NULL FOR UPDATE""",
                (identity.tenant_id, row["document_id"], row["revision_id"]),
            ).fetchone()
            if document is None:
                raise UploadError(409, "REVISION_UNAVAILABLE", "文档或期望修订已变更，无法重试")
            # FAILED/READY 批次不可再改为 BUILDING；重试创建配置相同的新批次。
            generation = self.query(
                """INSERT INTO {schema}.index_generation
                   (tenant_id, document_id, revision_id, embedding_profile_id, generation_no,
                    parser_version, cleaner_version, chunker_version, chunk_config)
                   SELECT old.tenant_id, old.document_id, old.revision_id, old.embedding_profile_id,
                     (SELECT max(generation_no) + 1 FROM {schema}.index_generation
                      WHERE tenant_id = old.tenant_id AND document_id = old.document_id),
                     old.parser_version, old.cleaner_version, old.chunker_version, old.chunk_config
                   FROM {schema}.index_generation old WHERE old.tenant_id = %s AND old.id = %s
                   RETURNING id""",
                (identity.tenant_id, row["generation_id"]),
            ).fetchone()
            self.query(
                """UPDATE {schema}.background_task SET status = 'PENDING', stage = 'QUEUED',
                   progress = 0, error_code = NULL, finished_at = NULL,
                   next_run_at = clock_timestamp(),
                   max_attempts = LEAST(100, attempt_count + 3), generation_id = %s
                   WHERE tenant_id = %s AND id = %s""",
                (generation["id"], identity.tenant_id, task_id),
            )
        self.connection.commit()
        return self.task_status(identity, task_id)
