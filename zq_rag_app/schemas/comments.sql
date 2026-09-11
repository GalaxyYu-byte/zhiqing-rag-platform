-- zq-rag-kb 数据库表及字段注释
-- 仅添加 COMMENT，不修改数据、字段、索引或约束。

COMMENT ON TABLE kb_knowledge_base IS '知识库主表：保存知识库基本信息、所属部门及软删除状态。';
COMMENT ON COLUMN kb_knowledge_base.id IS '知识库主键。';
COMMENT ON COLUMN kb_knowledge_base.name IS '知识库名称。';
COMMENT ON COLUMN kb_knowledge_base.description IS '知识库描述。';
COMMENT ON COLUMN kb_knowledge_base.department_id IS '所属部门 ID，来自外部组织/部门系统。';
COMMENT ON COLUMN kb_knowledge_base.is_public IS '是否公开：TRUE 表示允许所有符合条件的用户访问。';
COMMENT ON COLUMN kb_knowledge_base.created_by IS '创建人用户 ID，来自外部用户系统。';
COMMENT ON COLUMN kb_knowledge_base.created_at IS '创建时间。';
COMMENT ON COLUMN kb_knowledge_base.updated_at IS '最后更新时间。';
COMMENT ON COLUMN kb_knowledge_base.is_deleted IS '软删除标记：FALSE 未删除，TRUE 已删除。';

COMMENT ON TABLE kb_permission IS '知识库权限表：记录部门或用户对知识库的访问和管理权限。kb_id 逻辑关联 kb_knowledge_base.id。';
COMMENT ON COLUMN kb_permission.id IS '权限记录主键。';
COMMENT ON COLUMN kb_permission.kb_id IS '知识库 ID，逻辑关联 kb_knowledge_base.id。';
COMMENT ON COLUMN kb_permission.subject_type IS '授权主体类型：DEPARTMENT 部门，USER 用户。';
COMMENT ON COLUMN kb_permission.subject_id IS '授权主体 ID：部门编码或用户 ID 的字符串形式。';
COMMENT ON COLUMN kb_permission.permission IS '权限级别：READ 读，WRITE 写，ADMIN 管理。';
COMMENT ON COLUMN kb_permission.granted_by IS '授权操作人用户 ID。';
COMMENT ON COLUMN kb_permission.granted_at IS '授权时间。';

COMMENT ON TABLE kb_document IS '文档表：记录上传到知识库的原始文件及其索引状态。一个文档可对应多个文档分块。';
COMMENT ON COLUMN kb_document.id IS '文档主键。';
COMMENT ON COLUMN kb_document.kb_id IS '所属知识库 ID，逻辑关联 kb_knowledge_base.id。';
COMMENT ON COLUMN kb_document.file_name IS '原始文件名。';
COMMENT ON COLUMN kb_document.file_type IS '文件类型，例如 PDF、DOCX、MD、TXT、XLS 或 XLSX。';
COMMENT ON COLUMN kb_document.file_size IS '文件大小，单位为字节。';
COMMENT ON COLUMN kb_document.minio_path IS '文件在 MinIO 对象存储中的对象路径。';
COMMENT ON COLUMN kb_document.status IS '文档索引状态：PENDING 待处理，PROCESSING 处理中，DONE 完成，FAILED 失败。';
COMMENT ON COLUMN kb_document.error_msg IS '最近一次索引失败的错误信息。';
COMMENT ON COLUMN kb_document.chunk_count IS '当前版本文档切分后的分块数量。';
COMMENT ON COLUMN kb_document.token_count IS '当前文档向量化消耗或估算的 Token 数量。';
COMMENT ON COLUMN kb_document.version IS '当前正式文档版本号；候选版本索引成功后才切换。';
COMMENT ON COLUMN kb_document.uploaded_by IS '上传人用户 ID。';
COMMENT ON COLUMN kb_document.uploaded_at IS '上传时间。';
COMMENT ON COLUMN kb_document.indexed_at IS '最近一次索引完成时间。';
COMMENT ON COLUMN kb_document.is_deleted IS '软删除标记：FALSE 未删除，TRUE 已删除。';

COMMENT ON TABLE kb_document_version IS '文档版本表：保存每次上传或重建使用的候选文件，READY 表示已完成索引。';
COMMENT ON COLUMN kb_document_version.id IS '文档版本主键。';
COMMENT ON COLUMN kb_document_version.doc_id IS '逻辑文档 ID。';
COMMENT ON COLUMN kb_document_version.version IS '文档版本号。';
COMMENT ON COLUMN kb_document_version.file_hash IS '原始文件 SHA256，用于识别内容未变化的更新。';
COMMENT ON COLUMN kb_document_version.minio_path IS '该版本文件的 MinIO 路径。';
COMMENT ON COLUMN kb_document_version.operation_type IS '版本产生方式：UPLOAD、UPDATE、REINDEX 或 RESTORE。';
COMMENT ON COLUMN kb_document_version.source_version IS 'RESTORE 操作引用的历史版本号；其他操作为空。';
COMMENT ON COLUMN kb_document_version.status IS '版本状态：PENDING、PROCESSING、READY 或 FAILED。';

COMMENT ON TABLE kb_doc_chunk IS '文档分块表：保存文档切分后的可检索文本、全文检索向量及语义向量。';
COMMENT ON COLUMN kb_doc_chunk.id IS '文档分块主键。';
COMMENT ON COLUMN kb_doc_chunk.doc_id IS '所属文档 ID，逻辑关联 kb_document.id。';
COMMENT ON COLUMN kb_doc_chunk.kb_id IS '所属知识库 ID；从文档冗余保存，用于检索时快速过滤知识库。';
COMMENT ON COLUMN kb_doc_chunk.chunk_index IS '分块在文档中的顺序编号，从 0 开始。';
COMMENT ON COLUMN kb_doc_chunk.content IS '分块后的原始文本内容。';
COMMENT ON COLUMN kb_doc_chunk.content_tsv IS '由触发器根据 content 自动生成的 PostgreSQL 全文检索向量。';
COMMENT ON COLUMN kb_doc_chunk.embedding IS '文本语义向量，当前为 1024 维，用于 pgvector 相似度检索。';
COMMENT ON COLUMN kb_doc_chunk.page_num IS '分块所在页码，主要用于 PDF 文档。';
COMMENT ON COLUMN kb_doc_chunk.section_title IS '分块所属章节或标题。';
COMMENT ON COLUMN kb_doc_chunk.token_count IS '分块 Token 数量或估算值。';
COMMENT ON COLUMN kb_doc_chunk.doc_version IS '分块对应的文档索引版本号。';
COMMENT ON COLUMN kb_doc_chunk.created_at IS '分块记录创建时间。';

COMMENT ON TABLE kb_index_task IS '索引任务表：记录文档解析、切分、向量化和入库的异步任务状态及进度。';
COMMENT ON COLUMN kb_index_task.id IS '索引任务主键。';
COMMENT ON COLUMN kb_index_task.doc_id IS '待索引文档 ID，逻辑关联 kb_document.id。';
COMMENT ON COLUMN kb_index_task.doc_version IS '任务正在构建的目标文档版本。';
COMMENT ON COLUMN kb_index_task.task_type IS '任务类型：INDEX 首次索引，REINDEX 重建索引，UPDATE 更新文件，RESTORE 恢复历史版本。';
COMMENT ON COLUMN kb_index_task.status IS '任务状态：PENDING 待执行，PROCESSING 执行中，DONE 完成，FAILED 失败。';
COMMENT ON COLUMN kb_index_task.retry_count IS '已重试次数。';
COMMENT ON COLUMN kb_index_task.max_retry IS '允许的最大重试次数。';
COMMENT ON COLUMN kb_index_task.error_msg IS '任务失败或异常时的错误信息。';
COMMENT ON COLUMN kb_index_task.created_at IS '任务创建时间。';
COMMENT ON COLUMN kb_index_task.started_at IS '任务开始执行时间。';
COMMENT ON COLUMN kb_index_task.finished_at IS '任务完成或最终失败时间。';

-- 这些字段由 001_index_pipeline.sql 为旧数据库补充。
-- 采用条件注释，避免旧数据库因字段不存在而中断整个脚本。
DO $$
DECLARE
    item RECORD;
BEGIN
    FOR item IN
        SELECT * FROM (VALUES
            ('stage', '当前处理阶段，例如 PARSING、CLEANING、CHUNKING、EMBEDDING、PERSISTING。'),
            ('progress_percent', '任务完成百分比，取值范围 0 到 100。'),
            ('total_chunks', '本次任务产生的分块总数。'),
            ('embedded_chunks', '已完成向量化的分块数量。'),
            ('persisted_chunks', '已成功写入数据库的分块数量。'),
            ('cache_hit_chunks', '命中 Embedding 缓存的分块数量。'),
            ('total_tokens', '本次任务处理的 Token 总数。'),
            ('worker_id', '当前执行任务的 Worker 标识。'),
            ('heartbeat_at', 'Worker 最近一次心跳时间。'),
            ('lease_expires_at', '任务租约到期时间。'),
            ('updated_at', '任务最后更新时间。')
        ) AS columns(column_name, column_comment)
    LOOP
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'kb_index_task'
              AND column_name = item.column_name
        ) THEN
            EXECUTE format(
                'COMMENT ON COLUMN %I.%I IS %L',
                'kb_index_task', item.column_name, item.column_comment
            );
        END IF;
    END LOOP;
END $$;

COMMENT ON TABLE kb_chat_session IS '对话会话表：记录用户与 RAG 系统的一次会话及其查询的知识库范围。';
COMMENT ON COLUMN kb_chat_session.id IS '会话主键，UUID 字符串。';
COMMENT ON COLUMN kb_chat_session.user_id IS '会话所属用户 ID，来自外部用户系统。';
COMMENT ON COLUMN kb_chat_session.kb_ids IS '会话查询的知识库 ID 列表，保存为 JSON 数组字符串。';
COMMENT ON COLUMN kb_chat_session.title IS '会话标题，通常取第一条消息生成。';
COMMENT ON COLUMN kb_chat_session.message_count IS '会话消息数量。';
COMMENT ON COLUMN kb_chat_session.created_at IS '会话创建时间。';
COMMENT ON COLUMN kb_chat_session.last_active_at IS '最后活跃时间。';
COMMENT ON COLUMN kb_chat_session.is_deleted IS '软删除标记：FALSE 未删除，TRUE 已删除。';

COMMENT ON TABLE kb_chat_message IS '对话消息表：保存用户提问和系统回答，以及回答的引用来源和性能信息。';
COMMENT ON COLUMN kb_chat_message.id IS '消息主键。';
COMMENT ON COLUMN kb_chat_message.session_id IS '所属会话 ID，逻辑关联 kb_chat_session.id。';
COMMENT ON COLUMN kb_chat_message.role IS '消息角色：USER 用户消息，ASSISTANT 助手消息。';
COMMENT ON COLUMN kb_chat_message.content IS '消息正文。';
COMMENT ON COLUMN kb_chat_message.sources IS '助手回答引用的来源列表，JSON 格式，通常包含 docId、docName、chunkId、pageNum、excerpt、score。';
COMMENT ON COLUMN kb_chat_message.token_count IS '本条消息消耗或估算的 Token 数量。';
COMMENT ON COLUMN kb_chat_message.latency_ms IS '生成本条消息耗时，单位为毫秒。';
COMMENT ON COLUMN kb_chat_message.feedback IS '消息级反馈汇总：1 好评，-1 差评，NULL 未反馈。';
COMMENT ON COLUMN kb_chat_message.created_at IS '消息创建时间。';

COMMENT ON TABLE kb_answer_feedback IS '回答反馈表：记录用户对助手回答的点赞、点踩及文字评价。';
COMMENT ON COLUMN kb_answer_feedback.id IS '反馈记录主键。';
COMMENT ON COLUMN kb_answer_feedback.message_id IS '被评价的助手消息 ID，逻辑关联 kb_chat_message.id。';
COMMENT ON COLUMN kb_answer_feedback.user_id IS '提交反馈的用户 ID，来自外部用户系统。';
COMMENT ON COLUMN kb_answer_feedback.feedback IS '反馈结果：1 有用，-1 无用。';
COMMENT ON COLUMN kb_answer_feedback.comment IS '用户填写的补充意见。';
COMMENT ON COLUMN kb_answer_feedback.created_at IS '反馈提交时间。';

COMMENT ON TABLE kb_eval_dataset IS 'RAG 评估数据集表：保存问题、期望答案及期望召回的文档分块。';
COMMENT ON COLUMN kb_eval_dataset.id IS '评估数据集记录主键。';
COMMENT ON COLUMN kb_eval_dataset.kb_id IS '问题所属知识库 ID，逻辑关联 kb_knowledge_base.id。';
COMMENT ON COLUMN kb_eval_dataset.question IS '待评估的问题。';
COMMENT ON COLUMN kb_eval_dataset.expected_answer IS '预期答案，用于评估回答质量。';
COMMENT ON COLUMN kb_eval_dataset.expected_chunk_ids IS '期望召回的文档分块 ID 数组，逻辑引用 kb_doc_chunk.id。';
COMMENT ON COLUMN kb_eval_dataset.created_by IS '数据集记录创建人用户 ID。';
COMMENT ON COLUMN kb_eval_dataset.created_at IS '数据集记录创建时间。';

COMMENT ON TABLE kb_eval_result IS 'RAG 评估结果表：记录某个评估问题在指定评估版本下的召回和回答质量指标。';
COMMENT ON COLUMN kb_eval_result.id IS '评估结果主键。';
COMMENT ON COLUMN kb_eval_result.dataset_id IS '评估数据集记录 ID，逻辑关联 kb_eval_dataset.id。';
COMMENT ON COLUMN kb_eval_result.eval_version IS '评估版本标识，例如 v1_chunk512_hybrid。';
COMMENT ON COLUMN kb_eval_result.hit IS '是否命中期望召回的文档分块。';
COMMENT ON COLUMN kb_eval_result.rank IS '命中分块的排名，用于计算 MRR 等指标。';
COMMENT ON COLUMN kb_eval_result.actual_answer IS '系统实际生成的答案。';
COMMENT ON COLUMN kb_eval_result.faithfulness IS 'RAGAS Faithfulness 忠实度评分。';
COMMENT ON COLUMN kb_eval_result.answer_relevancy IS 'RAGAS Answer Relevancy 答案相关性评分。';
COMMENT ON COLUMN kb_eval_result.eval_at IS '评估执行时间。';
