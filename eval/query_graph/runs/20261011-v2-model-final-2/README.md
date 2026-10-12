# 中断的第二版真实模型组件运行

使用冻结五份虚构来源和五题，比较三种模式；本次只完成 4/15 组。
Neo4j 在后续预检连接返回 `Neo.ClientError.Security.AuthenticationRateLimit`，过程未完成，不能报整组通过。
结果保存已完成组及 13 份业务代码 SHA-256。三份基线答案发布但完整回答证据检查失败，已完成的两跳正题因路径引用不全被拒绝。

`cleanup=true`、`cleanup_errors=[]`；复查本次五个精确文档 namespace，残留节点均为零。
限流后单次核对凭据的健康检查通过。没有更改数据库密码、删除正式图或修改认证限制。

`pytest-full.txt` 是同一冻结实现的完整后端测试日志：1334 passed、75 skipped、4 项修改前已复现的离线评估夹具失败。
后续实现和最终验收见 [第二版验收记录](../../v2/ACCEPTANCE.md)。
