"""上传用例的数据库访问；所有查询显式约束租户与当前授权。"""

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from zhiqing_rag.core.config import Settings
from zhiqing_rag.document_processing.format_validation import ValidatedDocument
from zhiqing_rag.modules.knowledge.permissions import (
    document_access_sql,
    knowledge_base_access_sql,
)

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

    def accessible_bases(
        self, identity: UploadIdentity, permission: str = "WRITE",
    ) -> list[KnowledgeBaseOption]:
        access, args = knowledge_base_access_sql(identity, permission=permission)
        rows = self.query(
            f"""SELECT kb.id, kb.code, kb.name, kb.description,
                 (SELECT g.permission FROM {{schema}}.knowledge_base_grant g
                  WHERE g.tenant_id = kb.tenant_id AND g.kb_id = kb.id
                    AND (g.user_id = %s OR g.department_id = %s)
                  ORDER BY CASE g.permission WHEN 'ADMIN' THEN 3 WHEN 'WRITE' THEN 2 ELSE 1 END
                    DESC LIMIT 1) AS permission
               FROM {{schema}}.knowledge_base kb WHERE {access}
               ORDER BY kb.id""",
            (identity.user_id, identity.department_id, *args),
        ).fetchall()
        return [
            KnowledgeBaseOption(
                id=str(row["id"]),
                code=row["code"],
                name=row["name"],
                description=row["description"] or "",
                permission=row["permission"],
            )
            for row in rows
        ]

    def writable_bases(self, identity: UploadIdentity) -> list[KnowledgeBaseOption]:
        return self.accessible_bases(identity, "WRITE")

    def require_write(self, identity: UploadIdentity, kb_id: int) -> None:
        if not any(base.id == str(kb_id) for base in self.writable_bases(identity)):
            raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库不可用或没有写入权限")

    def update_target(
        self, identity: UploadIdentity, document_id: int, expected_revision_id: int
    ) -> int:
        """上传文件前做快速校验，并从原文档读取知识库 ID。

        这里只减少无效请求产生的对象存储写入，不持有文档锁。文件上传期间状态可能
        变化，因此 persist() 还必须在事务内重新校验权限和期望修订。
        """
        access, args = document_access_sql(identity, permission="WRITE")
        row = self.query(
            f"""SELECT doc.kb_id, doc.desired_revision_id FROM {{schema}}.document doc
               JOIN {{schema}}.knowledge_base kb
                 ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE doc.id = %s AND {access}""",
            (document_id, *args),
        ).fetchone()
        if row is None:
            raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
        if row["desired_revision_id"] != expected_revision_id:
            raise UploadError(409, "REVISION_CONFLICT", "文档已被更新，请刷新后重试")
        return row["kb_id"]

    def lock_write_base(self, identity: UploadIdentity, kb_id: int) -> None:
        """串行化同一知识库的授权变更，并在事务内重新确认有效身份。"""
        kb = self.query(
            """SELECT id FROM {schema}.knowledge_base
               WHERE tenant_id = %s AND id = %s AND status = 'ACTIVE'
               AND deleted_at IS NULL FOR SHARE""",
            (identity.tenant_id, kb_id),
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
                kb_id,
                identity.user_id,
                identity.department_id,
            ),
        ).fetchall()
        if not grants:
            raise UploadError(403, "KNOWLEDGE_BASE_FORBIDDEN", "知识库写入权限已撤销")
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

    def lock_document_base(self, identity: UploadIdentity, document_id: int) -> int:
        row = self.query(
            """SELECT kb_id FROM {schema}.document WHERE tenant_id = %s AND id = %s""",
            (identity.tenant_id, document_id),
        ).fetchone()
        if row is None:
            raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
        self.lock_write_base(identity, row["kb_id"])
        return row["kb_id"]

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
        document_id: int | None = None,
        expected_revision_id: int | None = None,
    ) -> UploadResult:
        """原子创建文件修订、索引批次和处理任务。

        document_id 为空时创建业务文档；更新时保留原 document_id，仅新增不可变
        修订。此事务只设置 desired_revision_id，成功构建后由 Worker 切换
        active_generation_id，因此处理期间旧索引仍是当前发布版本。
        """
        # MinIO 写入先于数据库事务。这里重新确认身份和授权，防止上传期间权限被撤销。
        with self.connection.transaction():
            self.lock_write_base(identity, parameters.knowledge_base_id)
            if document_id is None and parameters.confidentiality > identity.clearance:
                raise UploadError(
                    403, "DOCUMENT_CLEARANCE_FORBIDDEN", "文档密级超过当前账号的访问级别",
                )
            metadata = {
                "chunk_config": {
                    "max_tokens": parameters.chunk_size,
                    "overlap_tokens": parameters.chunk_overlap,
                }
            }
            if document_id is None:
                # 首次上传创建业务文档；该文档的修订号和构建批次号均从 1 开始。
                document = self.query(
                    """INSERT INTO {schema}.document
                       (tenant_id, kb_id, title, department_id, confidentiality,
                        created_by, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (
                        identity.tenant_id,
                        parameters.knowledge_base_id,
                        filename,
                        identity.department_id,
                        parameters.confidentiality,
                        identity.member_id,
                        Jsonb(metadata),
                    ),
                ).fetchone()
                revision_no = generation_no = 1
            else:
                # Worker 发布、重试和删除都先锁任务再锁文档；统一锁顺序避免相互等待。
                # 已领取任务也会被锁住，待其当前数据库操作结束后才决定哪个修订生效。
                self.query(
                    """SELECT id FROM {schema}.background_task
                       WHERE tenant_id = %s AND document_id = %s ORDER BY id FOR UPDATE""",
                    (identity.tenant_id, document_id),
                ).fetchall()
                access, args = document_access_sql(identity, permission="WRITE")
                document = self.query(
                    f"""SELECT doc.id, doc.desired_revision_id FROM {{schema}}.document doc
                       JOIN {{schema}}.knowledge_base kb
                         ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
                       WHERE doc.id = %s AND doc.kb_id = %s AND {access} FOR UPDATE OF doc""",
                    (document_id, parameters.knowledge_base_id, *args),
                ).fetchone()
                if document is None:
                    raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
                # 前端提交它看到的修订 ID，相当于乐观锁。两个并发更新只能有一个成功。
                if document["desired_revision_id"] != expected_revision_id:
                    raise UploadError(409, "REVISION_CONFLICT", "文档已被更新，请刷新后重试")
                current = self.query(
                    """SELECT file_sha256 FROM {schema}.document_revision
                       WHERE tenant_id = %s AND document_id = %s AND id = %s""",
                    (identity.tenant_id, document_id, expected_revision_id),
                ).fetchone()
                if current is None:
                    raise UploadError(409, "REVISION_CONFLICT", "当前修订不存在，请刷新后重试")
                # 完全相同的原文件不生成重复修订；处理失败应重试任务而不是再次上传。
                if current["file_sha256"] == validated.file_sha256:
                    raise UploadError(
                        409, "FILE_UNCHANGED", "文件内容未变化；处理失败可直接重试任务"
                    )
                # 文档行已锁定，修订号和批次号在同一文档的并发更新间不会重复。
                # 重试可能新增批次，所以两个序号分别从各自历史记录的最大值递增。
                revision_no = self.query(
                    """SELECT COALESCE(MAX(revision_no), 0) + 1 AS next_no
                       FROM {schema}.document_revision WHERE tenant_id = %s AND document_id = %s""",
                    (identity.tenant_id, document_id),
                ).fetchone()["next_no"]
                generation_no = self.query(
                    """SELECT COALESCE(MAX(generation_no), 0) + 1 AS next_no
                       FROM {schema}.index_generation WHERE tenant_id = %s AND document_id = %s""",
                    (identity.tenant_id, document_id),
                ).fetchone()["next_no"]
            # 每次上传都写新的文件修订，旧修订及其 MinIO 对象保留供历史预览。
            revision = self.query(
                """INSERT INTO {schema}.document_revision
                   (tenant_id, document_id, revision_no, original_filename, mime_type,
                    file_size, file_sha256, bucket, object_key, uploaded_by)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id, created_at""",
                (
                    identity.tenant_id,
                    document["id"],
                    revision_no,
                    filename,
                    validated.mime_type,
                    validated.file_size,
                    validated.file_sha256,
                    bucket,
                    object_key,
                    identity.member_id,
                ),
            ).fetchone()
            # 只更新期望修订指针。活动索引指针要等新批次完整入库后才切换。
            self.query(
                """UPDATE {schema}.document SET desired_revision_id = %s
                   WHERE tenant_id = %s AND id = %s""",
                (revision["id"], identity.tenant_id, document["id"]),
            )
            if document_id is not None:
                # 新修订替代尚未结束的旧任务。递增 lease_epoch 并清空租约后，
                # 旧 Worker 即使继续运行，也无法凭过期租约提交或发布结果。
                self.query(
                    """UPDATE {schema}.background_task SET status = 'CANCELLED',
                       stage = 'CANCELLED', error_code = 'REVISION_SUPERSEDED',
                       finished_at = clock_timestamp(), lease_owner = NULL,
                       lease_expires_at = NULL, lease_epoch = lease_epoch + 1
                       WHERE tenant_id = %s AND document_id = %s
                         AND task_type IN ('INDEX', 'GRAPH')
                         AND status IN ('PENDING', 'RUNNING', 'RETRY_WAIT')""",
                    (identity.tenant_id, document_id),
                )
                # 同步结束正在运行的尝试记录，保留历史供诊断和审计。
                self.query(
                    """UPDATE {schema}.task_attempt SET status = 'CANCELLED',
                       error_code = 'REVISION_SUPERSEDED', finished_at = clock_timestamp()
                       WHERE tenant_id = %s AND status = 'RUNNING' AND task_id IN (
                         SELECT id FROM {schema}.background_task
                         WHERE tenant_id = %s AND document_id = %s
                           AND task_type IN ('INDEX', 'GRAPH'))""",
                    (identity.tenant_id, identity.tenant_id, document_id),
                )
                superseded_graphs = self.query(
                    """SELECT generation_id, graph_namespace, projection_version
                       FROM {schema}.graph_projection
                       WHERE tenant_id = %s AND document_id = %s
                         AND status IN ('BUILDING', 'FAILED') AND revision_id <> %s""",
                    (identity.tenant_id, document_id, revision["id"]),
                ).fetchall()
                for graph in superseded_graphs:
                    self.query(
                        """INSERT INTO {schema}.outbox_event
                           (tenant_id, event_type, aggregate_type, aggregate_id,
                            deduplication_key, payload)
                           VALUES (%s, 'GRAPH_PURGE_GENERATION', 'graph_cleanup', %s, %s, %s)
                           ON CONFLICT (tenant_id, deduplication_key) DO NOTHING""",
                        (
                            identity.tenant_id,
                            str(document_id),
                            f"graph:purge:{graph['generation_id']}:v{graph['projection_version']}",
                            Jsonb(
                                {
                                    "document_id": document_id,
                                    "generation_id": graph["generation_id"],
                                    "namespace": graph["graph_namespace"],
                                }
                            ),
                        ),
                    )
                self.query(
                    """UPDATE {schema}.graph_projection SET status = 'STALE'
                       WHERE tenant_id = %s AND document_id = %s
                         AND status IN ('BUILDING', 'FAILED') AND revision_id <> %s""",
                    (identity.tenant_id, document_id, revision["id"]),
                )
                # 仅取消 BUILDING 批次；READY 的旧活动批次继续提供检索内容。
                self.query(
                    """UPDATE {schema}.index_generation SET status = 'CANCELLED'
                       WHERE tenant_id = %s AND document_id = %s AND status = 'BUILDING'""",
                    (identity.tenant_id, document_id),
                )
            # 新批次记录解析、分块和模型配置快照。Worker 会重新切片，
            # 对相同 embedding_content 复用现有 Redis 向量缓存，再写入新批次分块。
            generation = self.query(
                """INSERT INTO {schema}.index_generation
                   (tenant_id, document_id, revision_id, embedding_profile_id, generation_no,
                    parser_version, cleaner_version, chunker_version, chunk_config)
                   VALUES (%s, %s, %s, %s, %s, 'document-processing-v1', 'cleaning-v1',
                           'chunking-v2', %s) RETURNING id""",
                (
                    identity.tenant_id,
                    document["id"],
                    revision["id"],
                    profile_id,
                    generation_no,
                    Jsonb(metadata["chunk_config"]),
                ),
            ).fetchone()
            # 与修订、批次同事务落库，避免只有文件记录却没有可恢复处理任务。
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
        access, args = document_access_sql(identity)
        row = self.query(
            f"""SELECT rev.original_filename, rev.mime_type, rev.file_size, rev.file_sha256,
                      rev.bucket, rev.object_key
               FROM {{schema}}.document_revision rev
               JOIN {{schema}}.document doc ON doc.tenant_id = rev.tenant_id
                 AND doc.id = rev.document_id
               JOIN {{schema}}.knowledge_base kb ON kb.tenant_id = doc.tenant_id
                 AND kb.id = doc.kb_id
               WHERE doc.id = %s AND rev.id = %s AND {access}""",
            (document_id, revision_id, *args),
        ).fetchone()
        if row is None:
            raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
        return row

    def task_rows(
        self,
        identity: UploadIdentity,
        task_id: int | None = None,
        document_id: int | None = None,
        limit: int | None = 50,
    ):
        access, args = document_access_sql(identity, include_deleted=True)
        write, write_args = document_access_sql(identity, permission="WRITE", include_deleted=True)
        manage, manage_args = document_access_sql(identity, permission="ADMIN")
        return self.query(
            f"""SELECT task.id AS task_id, task.document_id, task.revision_id, task.generation_id,
                      rev.revision_no, doc.desired_revision_id, doc.active_generation_id,
                      doc.kb_id AS knowledge_base_id, rev.original_filename, rev.file_size,
                      task.status, task.stage, task.progress, task.attempt_count, task.max_attempts,
                      task.error_code, gen.chunk_count, doc.status AS document_status,
                      doc.department_id, doc.confidentiality,
                      {write} AS can_write, {manage} AS can_manage_permissions,
                      task.created_at, task.finished_at
               FROM {{schema}}.background_task task
               JOIN {{schema}}.document doc ON doc.tenant_id = task.tenant_id
                 AND doc.id = task.document_id
               JOIN {{schema}}.document_revision rev ON rev.tenant_id = task.tenant_id
                 AND rev.id = task.revision_id AND rev.document_id = doc.id
               JOIN {{schema}}.index_generation gen ON gen.tenant_id = task.tenant_id
                 AND gen.id = task.generation_id AND gen.document_id = doc.id
               JOIN {{schema}}.knowledge_base kb
                 ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE task.tenant_id = %s AND task.payload->>'pipeline' = 'document-ingestion-v1'
                 AND (%s::bigint IS NULL OR task.id = %s)
                 AND (%s::bigint IS NULL OR task.document_id = %s)
                 AND {access}
               ORDER BY task.id DESC LIMIT %s""",
            (*write_args, *manage_args, identity.tenant_id, task_id, task_id,
             document_id, document_id, *args, limit),
        ).fetchall()

    @staticmethod
    def task_response(row) -> DocumentTaskStatus:
        # bigint 传给浏览器时转为字符串，避免 JavaScript 整数精度丢失；空指针保留 null。
        return DocumentTaskStatus(
            **{
                key: str(value)
                if key
                in {
                    "task_id",
                    "document_id",
                    "revision_id",
                    "generation_id",
                    "knowledge_base_id",
                    "desired_revision_id",
                    "active_generation_id",
                    "department_id",
                }
                and value is not None
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

    def revision_history(
        self, identity: UploadIdentity, document_id: int
    ) -> list[DocumentTaskStatus]:
        rows = self.task_rows(identity, document_id=document_id, limit=None)
        if not rows:
            raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
        return [self.task_response(row) for row in rows]

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
        access, args = document_access_sql(identity, include_deleted=True)
        write, write_args = document_access_sql(identity, permission="WRITE", include_deleted=True)
        manage, manage_args = document_access_sql(identity, permission="ADMIN")
        row = self.query(
            f"""WITH scoped_tasks AS (
               SELECT task.id AS task_id, task.document_id, task.revision_id, task.generation_id,
                      rev.revision_no, doc.desired_revision_id, doc.active_generation_id,
                      doc.kb_id AS knowledge_base_id, rev.original_filename, rev.file_size,
                      task.status, task.stage, task.progress, task.attempt_count, task.max_attempts,
                      task.error_code, gen.chunk_count, doc.status AS document_status,
                      doc.department_id, doc.confidentiality,
                      {write} AS can_write, {manage} AS can_manage_permissions,
                      task.created_at, task.finished_at,
                      CASE WHEN doc.status = 'DELETED' OR task.status = 'FAILED' THEN 'error'
                           WHEN task.status = 'SUCCEEDED' THEN 'done'
                           WHEN task.status = 'CANCELLED' THEN 'cancelled'
                           ELSE 'processing' END AS queue_status
               FROM {{schema}}.background_task task
               JOIN {{schema}}.document doc ON doc.tenant_id = task.tenant_id
                 AND doc.id = task.document_id
               JOIN {{schema}}.document_revision rev ON rev.tenant_id = task.tenant_id
                 AND rev.id = task.revision_id AND rev.document_id = doc.id
               JOIN {{schema}}.index_generation gen ON gen.tenant_id = task.tenant_id
                 AND gen.id = task.generation_id AND gen.document_id = doc.id
               JOIN {{schema}}.knowledge_base kb
                 ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE task.tenant_id = %s AND task.payload->>'pipeline' = 'document-ingestion-v1'
                 AND {access}
            ), scoped AS (
               SELECT DISTINCT ON (document_id) * FROM scoped_tasks
               ORDER BY document_id, revision_no DESC, task_id DESC
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
                *write_args,
                *manage_args,
                identity.tenant_id,
                *args,
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
            self.lock_document_base(identity, document_id)
            access, args = document_access_sql(identity, permission="WRITE", include_deleted=True)
            # 与重试、Worker 发布保持任务 → 文档的锁顺序，防止删除后重新发布。
            self.query(
                """SELECT id FROM {schema}.background_task
                   WHERE tenant_id = %s AND document_id = %s ORDER BY id FOR UPDATE""",
                (identity.tenant_id, document_id),
            ).fetchall()
            document = self.query(
                f"""SELECT doc.kb_id, doc.deleted_at FROM {{schema}}.document doc
                   JOIN {{schema}}.knowledge_base kb
                     ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
                   WHERE doc.id = %s AND {access} FOR UPDATE OF doc""",
                (document_id, *args),
            ).fetchone()
            if document is None:
                raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
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
                namespaces = self.query(
                    """SELECT DISTINCT graph_namespace FROM {schema}.graph_projection
                       WHERE tenant_id = %s AND document_id = %s""",
                    (identity.tenant_id, document_id),
                ).fetchall()
                if namespaces:
                    self.query(
                        """INSERT INTO {schema}.outbox_event
                           (tenant_id, event_type, aggregate_type, aggregate_id,
                            deduplication_key, payload)
                           VALUES (%s, 'GRAPH_PURGE_DOCUMENT', 'graph_cleanup', %s, %s, %s)
                           ON CONFLICT (tenant_id, deduplication_key) DO NOTHING""",
                        (
                            identity.tenant_id,
                            str(document_id),
                            f"graph:delete:{document_id}",
                            Jsonb(
                                {
                                    "document_id": document_id,
                                    "namespaces": [row["graph_namespace"] for row in namespaces],
                                }
                            ),
                        ),
                    )
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
                self.query(
                    """UPDATE {schema}.graph_projection SET status = 'STALE'
                       WHERE tenant_id = %s AND document_id = %s""",
                    (identity.tenant_id, document_id),
                )
        self.connection.commit()
        return objects

    def purge_document(self, identity: UploadIdentity, document_id: int) -> None:
        """原文件清理成功后，按外键依赖顺序原子删除全部文档数据。"""
        with self.connection.transaction():
            self.lock_document_base(identity, document_id)
            access, args = document_access_sql(identity, permission="WRITE", include_deleted=True)
            self.query(
                """SELECT id FROM {schema}.background_task
                   WHERE tenant_id = %s AND document_id = %s ORDER BY id FOR UPDATE""",
                (identity.tenant_id, document_id),
            ).fetchall()
            document = self.query(
                f"""SELECT doc.status FROM {{schema}}.document doc
                   JOIN {{schema}}.knowledge_base kb
                     ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
                   WHERE doc.id = %s AND {access} FOR UPDATE OF doc""",
                (document_id, *args),
            ).fetchone()
            if document is None:
                raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有访问权限")
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
        task = self.task_status(identity, task_id)
        with self.connection.transaction():
            self.lock_document_base(identity, int(task.document_id))
            access, args = document_access_sql(identity, permission="WRITE")
            row = self.query(
                """SELECT status, generation_id, attempt_count, document_id, revision_id
                   FROM {schema}.background_task
                   WHERE tenant_id = %s AND id = %s FOR UPDATE""",
                (identity.tenant_id, task_id),
            ).fetchone()
            if row is None:
                raise UploadError(404, "TASK_NOT_FOUND", "任务已删除或没有访问权限")
            document = self.query(
                f"""SELECT doc.id, doc.desired_revision_id FROM {{schema}}.document doc
                   JOIN {{schema}}.knowledge_base kb
                     ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
                   WHERE doc.id = %s AND {access}
                   FOR UPDATE OF doc""",
                (row["document_id"], *args),
            ).fetchone()
            if document is None:
                raise UploadError(404, "DOCUMENT_NOT_FOUND", "文档不存在或没有写入权限")
            if document["desired_revision_id"] != row["revision_id"]:
                raise UploadError(409, "REVISION_UNAVAILABLE", "文档或期望修订已变更，无法重试")
            if row["status"] != "FAILED" or row["attempt_count"] >= 100:
                raise UploadError(409, "TASK_NOT_RETRYABLE", "任务未失败或已达到最大尝试次数")
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
