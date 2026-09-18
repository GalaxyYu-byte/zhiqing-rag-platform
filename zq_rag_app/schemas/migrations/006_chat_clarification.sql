ALTER TABLE kb_chat_session
    ADD COLUMN IF NOT EXISTS pending_clarification JSONB;
COMMENT ON COLUMN kb_chat_session.pending_clarification IS
    '待澄清任务：原问题、缺失槽位、授权候选文档、补充和过期时间；使用前重新校验权限。';
