# 在线一跳 GraphRAG 验收

时间：2026-10-11T13:55:16.418952+08:00

每题限定同一文档，图模式与基线仅改变 query_graph_enabled；其他配置和接口相同。
首次模型调用前核对当前权限、活动索引、同一修订及冻结原文件和正文哈希。
通过仅表示服务端图计划、一跳召回及关系逐字出处有效，不表示答案语义正确或检索优于基线。
完整 HTTP 响应保存在 results.json，供独立复核；语义正确性始终待人工验收。

| 题号 | 模式 | HTTP | 路由 | 图状态 | 图块数 | 关系数 | 检查 | 耗时 |
| --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: |
| G1 | graph | 200 | hybrid_graph | succeeded | 1 | 5 | True | 16436 ms |
| G1 | baseline | 200 | hybrid | skipped | 0 | 0 | True | 12682 ms |
| G2 | graph | 200 | hybrid_graph | succeeded | 1 | 1 | True | 15897 ms |
| G2 | baseline | 200 | hybrid | skipped | 0 | 0 | True | 12119 ms |
| G3 | graph | 200 | hybrid_graph | succeeded | 1 | 2 | True | 15565 ms |
| G3 | baseline | 200 | hybrid | skipped | 0 | 0 | True | 10769 ms |
| G4 | graph | 200 | hybrid_graph | succeeded | 2 | 2 | True | 14995 ms |
| G4 | baseline | 200 | hybrid | skipped | 0 | 0 | True | 13866 ms |

| 题号 | 模式 | 回答状态 | 引用可定位 |
| --- | --- | --- | --- |
| G1 | graph | answered | True |
| G1 | baseline | answered | True |
| G2 | graph | answered | True |
| G2 | baseline | answered | True |
| G3 | graph | answered | True |
| G3 | baseline | answered | True |
| G4 | graph | answered | True |
| G4 | baseline | answered | True |
