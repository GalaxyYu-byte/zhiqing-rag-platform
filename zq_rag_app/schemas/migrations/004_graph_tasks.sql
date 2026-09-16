-- 独立 Graph Worker 的任务、租约、重试和候选版本状态。

CREATE TABLE IF NOT EXISTS kb_graph_task (
    id                  BIGSERIAL PRIMARY KEY,
    doc_id              BIGINT       NOT NULL,
    kb_id               BIGINT       NOT NULL,
    doc_version         INT          NOT NULL,
    extractor_version   VARCHAR(100) NOT NULL,
    status              VARCHAR(20)  NOT NULL DEFAULT 'PENDING',
    stage               VARCHAR(30)  NOT NULL DEFAULT 'PENDING',
    progress_percent    INT          NOT NULL DEFAULT 0,
    total_chunks        INT          NOT NULL DEFAULT 0,
    processed_chunks    INT          NOT NULL DEFAULT 0,
    extracted_entities  INT          NOT NULL DEFAULT 0,
    extracted_claims    INT          NOT NULL DEFAULT 0,
    retry_count         INT          NOT NULL DEFAULT 0,
    max_retry           INT          NOT NULL DEFAULT 3,
    error_msg           TEXT,
    worker_id           VARCHAR(200),
    heartbeat_at        TIMESTAMP,
    lease_expires_at    TIMESTAMP,
    created_at          TIMESTAMP    NOT NULL DEFAULT NOW(),
    started_at          TIMESTAMP,
    finished_at         TIMESTAMP,
    updated_at          TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_graph_task_doc_version UNIQUE (doc_id, doc_version),
    CONSTRAINT ck_graph_task_progress_percent
        CHECK (progress_percent BETWEEN 0 AND 100)
);

CREATE INDEX IF NOT EXISTS idx_graph_task_status
    ON kb_graph_task(status, created_at);
CREATE INDEX IF NOT EXISTS idx_graph_task_doc_id
    ON kb_graph_task(doc_id, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS uq_graph_task_active_doc
    ON kb_graph_task(doc_id)
    WHERE status IN ('PENDING', 'PROCESSING');

COMMENT ON TABLE kb_graph_task IS
    '独立 Graph Worker 任务；候选图全部完成后才切换活动版本';
