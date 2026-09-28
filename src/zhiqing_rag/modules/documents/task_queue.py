"""PostgreSQL 持久化任务队列；领取、心跳与失败写入均校验租约代次。"""

from dataclasses import dataclass

from psycopg.types.json import Jsonb

from zhiqing_rag.core.config import Settings
from zhiqing_rag.infrastructure.database.connection import database_connection

from .repository import UploadRepository


class LeaseLostError(RuntimeError):
    pass


class ProcessingError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class TaskLease:
    task_id: int
    tenant_id: int
    document_id: int
    revision_id: int
    generation_id: int
    worker_id: str
    epoch: int


class IngestionQueue:
    def __init__(self, settings: Settings):
        self.settings = settings

    def claim(self, worker_id: str, task_id: int | None = None) -> TaskLease | None:
        with database_connection(self.settings) as connection:
            repository = UploadRepository(connection, self.settings.db_schema)
            row = repository.query(
                """SELECT * FROM {schema}.background_task
                   WHERE task_type = 'INDEX' AND payload->>'pipeline' = 'document-ingestion-v1'
                     AND (%s::bigint IS NULL OR id = %s) AND (
                       (status IN ('PENDING', 'RETRY_WAIT') AND next_run_at <= clock_timestamp())
                       OR (status = 'RUNNING' AND lease_expires_at <= clock_timestamp()))
                   ORDER BY next_run_at, id FOR UPDATE SKIP LOCKED LIMIT 1""",
                (task_id, task_id),
            ).fetchone()
            if row is None:
                return None
            repository.query(
                """UPDATE {schema}.task_attempt SET status = 'LOST',
                   finished_at = clock_timestamp(), error_code = 'LEASE_EXPIRED'
                   WHERE tenant_id = %s AND task_id = %s AND status = 'RUNNING'""",
                (row["tenant_id"], row["id"]),
            )
            if row["attempt_count"] >= row["max_attempts"]:
                repository.query(
                    """UPDATE {schema}.background_task SET status = 'FAILED', stage = 'FAILED',
                       finished_at = clock_timestamp(), error_code = 'ATTEMPTS_EXHAUSTED',
                       lease_owner = NULL, lease_expires_at = NULL
                       WHERE tenant_id = %s AND id = %s""",
                    (row["tenant_id"], row["id"]),
                )
                repository.query(
                    """UPDATE {schema}.index_generation SET status = 'FAILED'
                       WHERE tenant_id = %s AND id = %s AND status = 'BUILDING'""",
                    (row["tenant_id"], row["generation_id"]),
                )
                return None
            claimed = repository.query(
                """UPDATE {schema}.background_task SET status = 'RUNNING', stage = 'DOWNLOADING',
                   progress = 5, attempt_count = attempt_count + 1, lease_epoch = lease_epoch + 1,
                   lease_owner = %s,
                   lease_expires_at = clock_timestamp() + %s * interval '1 second',
                   heartbeat_at = clock_timestamp(), error_code = NULL
                   WHERE tenant_id = %s AND id = %s RETURNING *""",
                (worker_id, self.settings.ingestion_lease_seconds, row["tenant_id"], row["id"]),
            ).fetchone()
            repository.query(
                """INSERT INTO {schema}.task_attempt
                   (tenant_id, task_id, attempt_no, lease_epoch, worker_id, stage)
                   VALUES (%s, %s, %s, %s, %s, 'DOWNLOADING')""",
                (
                    row["tenant_id"],
                    row["id"],
                    claimed["attempt_count"],
                    claimed["lease_epoch"],
                    worker_id,
                ),
            )
            return TaskLease(
                task_id=row["id"],
                tenant_id=row["tenant_id"],
                document_id=row["document_id"],
                revision_id=row["revision_id"],
                generation_id=row["generation_id"],
                worker_id=worker_id,
                epoch=claimed["lease_epoch"],
            )

    def heartbeat(self, lease: TaskLease, stage: str | None = None, progress: int = 0) -> None:
        with database_connection(self.settings) as connection:
            repository = UploadRepository(connection, self.settings.db_schema)
            row = repository.query(
                """UPDATE {schema}.background_task SET heartbeat_at = clock_timestamp(),
                   lease_expires_at = clock_timestamp() + %s * interval '1 second',
                   stage = COALESCE(%s, stage), progress = GREATEST(progress, %s)
                   WHERE tenant_id = %s AND id = %s AND status = 'RUNNING'
                     AND lease_owner = %s AND lease_epoch = %s
                     AND lease_expires_at > clock_timestamp() RETURNING id""",
                (
                    self.settings.ingestion_lease_seconds,
                    stage,
                    progress,
                    lease.tenant_id,
                    lease.task_id,
                    lease.worker_id,
                    lease.epoch,
                ),
            ).fetchone()
            if row is None:
                raise LeaseLostError("任务租约已失效")
            if stage:
                repository.query(
                    """UPDATE {schema}.task_attempt SET stage = %s
                       WHERE tenant_id = %s AND task_id = %s AND lease_epoch = %s
                         AND status = 'RUNNING'""",
                    (stage, lease.tenant_id, lease.task_id, lease.epoch),
                )

    def context(self, lease: TaskLease):
        with database_connection(self.settings) as connection:
            repository = UploadRepository(connection, self.settings.db_schema)
            row = repository.query(
                """SELECT rev.bucket, rev.object_key, rev.original_filename, rev.file_sha256,
                          rev.file_size, rev.mime_type, gen.chunk_config, profile.model_name,
                          profile.dimensions, profile.provider, doc.kb_id
                   FROM {schema}.background_task task
                   JOIN {schema}.document doc ON doc.tenant_id = task.tenant_id
                     AND doc.id = task.document_id
                   JOIN {schema}.document_revision rev ON rev.tenant_id = task.tenant_id
                     AND rev.id = task.revision_id AND rev.document_id = doc.id
                   JOIN {schema}.index_generation gen ON gen.tenant_id = task.tenant_id
                     AND gen.id = task.generation_id AND gen.revision_id = rev.id
                   JOIN {schema}.embedding_profile profile ON profile.id = gen.embedding_profile_id
                   WHERE task.tenant_id = %s AND task.id = %s AND task.status = 'RUNNING'
                     AND task.lease_owner = %s AND task.lease_epoch = %s
                     AND task.lease_expires_at > clock_timestamp()
                     AND doc.deleted_at IS NULL AND doc.status IN ('DRAFT', 'PUBLISHED')
                     AND doc.desired_revision_id = rev.id""",
                (lease.tenant_id, lease.task_id, lease.worker_id, lease.epoch),
            ).fetchone()
            if row is None:
                raise ProcessingError("REVISION_UNAVAILABLE")
            if (
                row["provider"] != "dashscope"
                or row["dimensions"] != 1024
                or row["model_name"] != self.settings.embedding_model
                or self.settings.embedding_dimensions != 1024
            ):
                raise ProcessingError("MODEL_CONFIG_CHANGED")
            return row

    def fail(self, lease: TaskLease, code: str, retryable: bool, metrics: dict) -> None:
        with database_connection(self.settings) as connection:
            repository = UploadRepository(connection, self.settings.db_schema)
            row = repository.query(
                """SELECT attempt_count, max_attempts FROM {schema}.background_task
                   WHERE tenant_id = %s AND id = %s AND status = 'RUNNING'
                     AND lease_owner = %s AND lease_epoch = %s
                     AND lease_expires_at > clock_timestamp() FOR UPDATE""",
                (lease.tenant_id, lease.task_id, lease.worker_id, lease.epoch),
            ).fetchone()
            if row is None:
                return
            retry = retryable and row["attempt_count"] < row["max_attempts"]
            repository.query(
                """UPDATE {schema}.background_task SET status = %s, stage = %s, error_code = %s,
                   lease_owner = NULL, lease_expires_at = NULL,
                   finished_at = CASE WHEN %s THEN NULL ELSE clock_timestamp() END,
                   next_run_at = clock_timestamp() + %s * interval '1 second'
                   WHERE tenant_id = %s AND id = %s""",
                (
                    "RETRY_WAIT" if retry else "FAILED",
                    "RETRY_WAIT" if retry else "FAILED",
                    code,
                    retry,
                    min(300, 15 * 2 ** (row["attempt_count"] - 1)),
                    lease.tenant_id,
                    lease.task_id,
                ),
            )
            repository.query(
                """UPDATE {schema}.task_attempt SET status = 'FAILED', error_code = %s,
                   finished_at = clock_timestamp(), metrics = %s
                   WHERE tenant_id = %s AND task_id = %s AND lease_epoch = %s
                     AND status = 'RUNNING'""",
                (code, Jsonb(metrics), lease.tenant_id, lease.task_id, lease.epoch),
            )
            if not retry:
                repository.query(
                    """UPDATE {schema}.index_generation SET status = 'FAILED'
                       WHERE tenant_id = %s AND id = %s AND status = 'BUILDING'""",
                    (lease.tenant_id, lease.generation_id),
                )
