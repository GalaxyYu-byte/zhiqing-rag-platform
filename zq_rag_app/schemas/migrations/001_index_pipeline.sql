-- 为已经初始化过的数据库补充幂等索引、任务进度、心跳和多节点租约字段。
-- 本脚本可以重复执行；创建唯一索引前请先处理历史重复分块或活跃任务。

ALTER TABLE kb_index_task
    ADD COLUMN IF NOT EXISTS stage VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    ADD COLUMN IF NOT EXISTS progress_percent INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_chunks INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS embedded_chunks INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS persisted_chunks INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cache_hit_chunks INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS total_tokens INT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS worker_id VARCHAR(200),
    ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMP,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT NOW();

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'ck_task_progress_percent'
    ) THEN
        ALTER TABLE kb_index_task
            ADD CONSTRAINT ck_task_progress_percent
            CHECK (progress_percent BETWEEN 0 AND 100);
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_chunk_doc_version_index
    ON kb_doc_chunk(doc_id, doc_version, chunk_index);

CREATE UNIQUE INDEX IF NOT EXISTS uq_task_active_doc
    ON kb_index_task(doc_id)
    WHERE status IN ('PENDING', 'PROCESSING');
