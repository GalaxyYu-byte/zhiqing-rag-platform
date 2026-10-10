# 历史表格索引重建

旧 XLSX 分块可能把稀疏的标题行当列头，导致后续分块缺少真实列名。现行切块已修复，但已发布 Chunk 的正文与向量保持不可变，需要同一文件修订下的新 generation。

默认命令只读取明确指定文档的当前发布批次和原始对象，重新解析并比较表格；不会调用向量/回答模型，不写业务数据，也不会因其他内容差异重建文档。报告仅包含文档、批次、内容指纹及受影响表格位置，省略对象键与连接配置。明确指定同一租户的 1～20 份文档：

```powershell
.venv/Scripts/python.exe scripts/reindex_table_documents.py --tenant-id 1 --document-id 123
```

确认报告 `needs_reindex: true` 后，入队仍会再次解析检查，只为实际受影响的文档创建批次与标准 INDEX 任务：

```powershell
.venv/Scripts/python.exe scripts/reindex_table_documents.py --tenant-id 1 --document-id 123 --enqueue
.venv/Scripts/python.exe -m zhiqing_rag.workers.ingestion --once --task-id 456
```

将示例 ID 替换为目标租户、文档和入队返回的 `task_id`。如果常驻 ingestion Worker 正在运行，它会自动领取新任务，无需第二条命令。重复执行相同计划只返回已有任务；已修好的当前文档会被预览跳过。失败任务可通过已有文档任务重试接口重试，新批次保留相同发布前置条件。

入队要求源上传者仍有实时 WRITE 权限、当前 READY 批次与 desired revision 一致、内容指纹未改变，并拒绝已有活跃 INDEX/GRAPH 任务。解析、embedding 与保存复用现有租约 Worker。暂存和最终发布都核验预览中的输入指纹与旧发布指针，租约过期、权限撤销、修订更新或同修订下其他批次抢先发布都会阻止切换。构建期间旧索引持续可检索；失败不修改旧发布指针与旧 Chunk。

发布在单个事务中完成：检查完整向量批次、切换 `active_generation_id`、取消旧 generation 未完成的图任务租约，将旧投影标记 STALE 并写入去重的图清理 Outbox。旧 revision、generation 与 Chunk 均保留，新批次的 GRAPH 任务按现有流程重新构建，避免继续使用对应旧 Chunk 的图证据。

2026-10-10 已在项目测试库完成 8 份 XLSX 文档的真实重建，共发布 25 个新分块；旧分块和同租户其他文档指针保持不变，发布后重新预览均无需再次重建。执行证据与验收范围见 [本轮真实重建记录](eval/table_reindex/runs/20261010/README.md)。
