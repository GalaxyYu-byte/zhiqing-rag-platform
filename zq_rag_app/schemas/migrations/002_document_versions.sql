-- 文档更新第一版：候选版本独立索引，成功后再切换正式版本。

CREATE TABLE IF NOT EXISTS kb_document_version (
    id              BIGSERIAL PRIMARY KEY,
    doc_id          BIGINT          NOT NULL,
    version         INT             NOT NULL,
    file_name       VARCHAR(255)    NOT NULL,
    file_type       VARCHAR(20)     NOT NULL,
    file_size       BIGINT          NOT NULL,
    file_hash       VARCHAR(64),
    minio_path      VARCHAR(500)    NOT NULL,
    status          VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    uploaded_by     BIGINT          NOT NULL,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),
    indexed_at      TIMESTAMP,
    error_msg       TEXT,
    CONSTRAINT uq_doc_version UNIQUE (doc_id, version)
);

CREATE INDEX IF NOT EXISTS idx_doc_version_status
    ON kb_document_version(doc_id, status);

-- 为已有文档补一条当前版本记录；历史数据没有文件 Hash，首次更新时跳过整文件去重。
INSERT INTO kb_document_version (
    doc_id, version, file_name, file_type, file_size, minio_path,
    status, uploaded_by, created_at, indexed_at, error_msg
)
SELECT
    d.id, d.version, d.file_name, d.file_type, d.file_size, d.minio_path,
    CASE WHEN d.status = 'DONE' THEN 'READY' ELSE d.status END,
    d.uploaded_by, d.uploaded_at, d.indexed_at, d.error_msg
FROM kb_document d
WHERE NOT EXISTS (
    SELECT 1 FROM kb_document_version v
    WHERE v.doc_id = d.id AND v.version = d.version
);

ALTER TABLE kb_index_task
    ADD COLUMN IF NOT EXISTS doc_version INT;

UPDATE kb_index_task t
SET doc_version = d.version
FROM kb_document d
WHERE t.doc_id = d.id AND t.doc_version IS NULL;

ALTER TABLE kb_index_task
    ALTER COLUMN doc_version SET NOT NULL;
