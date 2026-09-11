-- 文档历史版本与恢复支持。

ALTER TABLE kb_document_version
    ADD COLUMN IF NOT EXISTS operation_type VARCHAR(20) NOT NULL DEFAULT 'UPLOAD';

ALTER TABLE kb_document_version
    ADD COLUMN IF NOT EXISTS source_version INT;

-- 尽量根据历史任务还原已有版本的产生方式。没有对应任务的 V1 视为首次上传，
-- 其他版本按重建处理；恢复功能上线后会明确写入 RESTORE 和来源版本。
UPDATE kb_document_version AS version
SET operation_type = COALESCE(
    (
        SELECT task.task_type
        FROM kb_index_task AS task
        WHERE task.doc_id = version.doc_id
          AND task.doc_version = version.version
          AND task.task_type IN ('INDEX', 'REINDEX', 'UPDATE', 'RESTORE')
        ORDER BY task.id DESC
        LIMIT 1
    ),
    CASE WHEN version.version = 1 THEN 'UPLOAD' ELSE 'REINDEX' END
);

UPDATE kb_document_version
SET operation_type = 'UPLOAD'
WHERE operation_type = 'INDEX';

COMMENT ON COLUMN kb_document_version.operation_type IS
    '版本产生方式：UPLOAD、UPDATE、REINDEX 或 RESTORE。';
COMMENT ON COLUMN kb_document_version.source_version IS
    'RESTORE 操作引用的历史版本号；其他操作为空。';
