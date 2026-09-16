-- 文档级检索权限和业务版本状态元数据。

ALTER TABLE kb_document
    ADD COLUMN IF NOT EXISTS document_code VARCHAR(50),
    ADD COLUMN IF NOT EXISTS department_id VARCHAR(50),
    ADD COLUMN IF NOT EXISTS confidentiality VARCHAR(20) NOT NULL DEFAULT '机密',
    ADD COLUMN IF NOT EXISTS business_status VARCHAR(20) NOT NULL DEFAULT '未知';

CREATE INDEX IF NOT EXISTS idx_doc_access_scope
    ON kb_document(kb_id, confidentiality, department_id)
    WHERE is_deleted = FALSE AND status = 'DONE';

CREATE INDEX IF NOT EXISTS idx_doc_document_code
    ON kb_document(document_code)
    WHERE document_code IS NOT NULL AND is_deleted = FALSE;

COMMENT ON COLUMN kb_document.document_code IS '业务文档稳定编号，例如 DOC-013';
COMMENT ON COLUMN kb_document.department_id IS '文档归属部门，用于检索前 ACL';
COMMENT ON COLUMN kb_document.confidentiality IS '内部公开、部门内部或机密';
COMMENT ON COLUMN kb_document.business_status IS '生效、草案、已归档、已废止或未知';
