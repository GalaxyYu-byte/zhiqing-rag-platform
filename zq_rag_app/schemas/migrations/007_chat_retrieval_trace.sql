ALTER TABLE kb_chat_message
    ADD COLUMN IF NOT EXISTS retrieval_trace JSONB;
COMMENT ON COLUMN kb_chat_message.retrieval_trace IS
    '检索诊断：模型建议、Router 规则决策、Executor 实际路径、最终参数及降级原因。';
