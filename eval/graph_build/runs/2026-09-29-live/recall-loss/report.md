# GraphRAG 召回损失归因

Gold 实体 165 个，命中 87 个，漏召回 78 个。

## ① Entity Recall 低的可观察原因

| 实体类型 | 命中 / Gold | 召回 |
| --- | ---: | ---: |
| Actor | 32/46 | 0.6957 |
| Field | 0/4 | 0.0 |
| Material | 8/16 | 0.5 |
| Process | 1/2 | 0.5 |
| Rule | 9/30 | 0.3 |
| State | 6/6 | 1.0 |
| Step | 31/61 | 0.5082 |

78 个漏召回实体的互斥分组：

- `NO_OBVIOUS_CANDIDATE`：7。
- `SAME_NAME_WRONG_TYPE_CANDIDATE`：3。
- `SAME_PASSAGE_DIFFERENT_ENTITY_CANDIDATE`：30。
- `SAME_TYPE_NAME_BOUNDARY_CANDIDATE`：15。
- `ZERO_GRAPH_OUTPUT_CHUNK`：23。

这些分组由名称、类型及证据重叠自动生成，属于待人工确认的原因线索。逐条候选和相近系统实体见 entity_outcomes.jsonl。

59 个原始 Chunk 中有 9 个没有任何系统实体提及；其中 7 个有正向 Gold，直接对应 23 个实体漏召回和 16 条关系漏召回。这 9 个 Chunk 也没有任何 Evidence 节点（有 Evidence 的数量为 0）。由于原始抽取缓存缺失，不能判定是模型返回空结果、任务处理还是落图环节造成。
其余 Chunk 还有 55 个实体未严格命中；Rule 仅命中 9/30，Step 命中 31/61。Field 的 4 个 Gold 均位于同一个零输出 Chunk，不能据此断定所有字段抽取都失败。
另有 4 个 Chunk 的投影实体数达到或超过提示词的 25 个上限建议，涉及 16 个 Gold 实体漏召回；这是容量压力线索，并非已证实的截断原因。

## 文档差异

| 文档 | Entity 命中/Gold | Relation 命中/Gold |
| --- | ---: | ---: |
| 35 员工入职资料准备说明.pdf | 1/15 | 0/10 |
| 45 采购需求与审批边界.pdf | 4/15 | 0/11 |
| 56 工单处理与关闭步骤.docx | 15/22 | 8/11 |
| 58 发布变更准备与回退指引.docx | 13/18 | 9/9 |
| 63 文档命名与版本约定.md | 10/14 | 5/5 |
| 71 访问权限定期复核.md | 17/22 | 9/13 |
| 73 财务单据退回原因说明.txt | 9/12 | 4/6 |
| 76 服务请求转派提醒.txt | 2/10 | 0/6 |
| 83 入职事项与责任分工.xlsx | 4/19 | 0/11 |
| 89 内部资料分类与保管约定.xlsx | 12/18 | 5/9 |

其中 4 份文档的 Gold 关系完全未命中，合计贡献 38/51 条关系漏召回和 48/78 个实体漏召回。

## ② Relation Recall 损失拆解

Gold 关系 91 条，命中 40 条，漏召回 51 条。

| 结果 | 条数 |
| --- | ---: |
| `BOTH_ENDPOINTS_MISSING` | 28 |
| `MATCH` | 40 |
| `SOURCE_MISSING` | 8 |
| `TARGET_MISSING` | 15 |

按严格实体类型和规范名称匹配，51 条漏召回关系均至少缺一个 Gold 端点；两端都已命中的 40 条关系，其关系类型和方向也全部命中。这不等于 51 条关系的概念在语义上完全没有被抽出，名称或类型差异仍需裁决。
其中 16 条位于零实体输出 Chunk，其余 35 条位于有其他实体输出的 Chunk。

按关系类型统计漏召回：

- `GOVERNED_BY`：4 条。
- `HAS_FIELD`：4 条。
- `HAS_STATE`：6 条。
- `NEXT_STEP`：11 条。
- `RESPONSIBLE_FOR`：26 条。

漏召回关系共涉及 55 个不同的未命中 Gold 实体；按 source/target 位置计，有 79 次端点缺失。逐条关系和端点状态见 relation_loss_outcomes.jsonl。

阻塞关系最多的 Gold 实体：

| Gold 实体 | 类型 | 受影响关系数 |
| --- | --- | ---: |
| 采购需求单 | Material | 4 |
| 复核结论 | Material | 4 |
| 告知资料清单与提交渠道 | Step | 3 |
| 逐项核对姓名、联系方式和证件信息 | Step | 3 |
| 核对可替代方案与供应方式 | Step | 3 |

同一条关系两端都缺失时，只算 1 条关系损失，但算 2 次端点缺失。条件文本不一致另见六层评估报告，不并入三元组漏召回。
