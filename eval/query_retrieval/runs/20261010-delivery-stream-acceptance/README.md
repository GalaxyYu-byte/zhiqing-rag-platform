# 提交版真实 SSE 回答验收

固定问题 Q03，执行当前 ASGI `/api/query/answer/stream`，使用真实 PostgreSQL、Redis、DashScope 与 DeepSeek。结果见 [results.json](results.json)。

已核对四阶段 progress、核验后 answer 和唯一终止 done；answer 为最终回答的累计完整陈述，全部引用在原文逐字可定位，最终 Q03 关闭条件保留客户确认或按约定沟通方式通知的条件。无 error、草稿或模型 token 发布，Cache-Control 包含 no-store，X-Accel-Buffering 为 no。验收会话结束后已定向撤销。

ASGI 验证不证明部署代理已关闭缓冲；生产代理配置仍需按单轮回答文档设置。本次不测断网重连，断流和取消边界由本地前后端测试覆盖。该 SSE 个案不能替代完整题集事实验收，当前仍有三道正向题未通过。
