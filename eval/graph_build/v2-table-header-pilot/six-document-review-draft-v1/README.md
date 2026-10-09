# 六份文档的 Gold 扩展草稿与逐条裁决材料

本次使用已冻结的 13 Chunk 和已保存的同一份 v2 名称反馈预测，没有重新请求模型。
所有数据来源是虚构企业测试文档。本版本在看过预测后补标，属于诊断/回归资料，不能作为独立留出验收集。

## 新 Gold 范围

覆盖六份文档全部数据行，草稿 405 个实体、245 条关系。
列和关系按原文与显式口径建模：权限职责、状态进入/离开约束、资料检查约束、发布事项及通过依据、
费用 Field/材料/审核 Rule、采购职责/输入/结果/暂停条件。范围说明与未建模列保留在 row_annotations。
没有把范围说明当负例，没有按模型输出删去漏抽目标，没有将所有共现列强建关系。

这不是“所有可能实体和关系均穷尽”的正式 Gold。费用 Field 与材料的语义在当前关系白名单中
没有直接对应关系，因此保留实体与关系缺口说明；未扩展生产 schema。
发布事项逐行类型、权限环节 Process、费用审核 Rule 等新增口径均待人工审核。
采购暂停条件的处置效果保存在 assertions 中，与裸三元组命中分开；未用结构命中代表语义通过。

## 草稿结构指标（尚未人工验收）

| 文档 | Entity 命中/目标 | Triple 命中/目标 |
| --- | --- | --- |
| 权限申请与复核责任矩阵 | 50/92 | 53/126 |
| 客户工单状态与处理要求 | 42/70 | 0/28 |
| 项目交付物核对清单 | 25/45 | 5/15 |
| 版本发布前核对清单 | 8/64 | 1/16 |
| 费用凭证核验要点 | 19/62 | 0/0 |
| 采购节点与交接说明 | 58/72 | 47/60 |

- Entity：202/405。
- Directed Triple：106/245。
- 未命中的关系中，至少一个端点缺失：139。
- 端点已齐但关系类型/方向/边未命中：0。
- 两端已抽到的条件下，Triple 命中：106/106。
  这是结构 Conditional Recall，未包含暂停/退回等语义效果是否正确。
- 旧 45/57、19/30 保留在原人工口径版本；本版本分母和建模范围扩大，不能直接当作模型性能变化。

## 初审与人工审核

预测实体：266；预测关系：202。
均有逐条审核记录，人工状态全部待确认。
关系的 Codex 初审建议：{"SUPPORT": 122, "INSUFFICIENT": 66, "REVIEW_REQUIRED": 14}。
引用本身的初审建议：{"SUPPORT": 118, "INSUFFICIENT": 70, "REVIEW_REQUIRED": 14}。
Chunk 支持 Fact 与预测引用支持 Fact 分开保存；引用错位不自动等于该 Chunk 无法支持 Fact。
引用支持包括所引文本所在行及表格列头；不得用其他行替换模型原引用来认定引用正确。
REVIEW_REQUIRED 代表未决，最终三分类仍是 SUPPORT / CONTRADICT / INSUFFICIENT。
当前无已确认的人工证据标签。
Evidence 支持、实体类型正确、主实体边界命中、条件完整是不同判断，不能互相代替。

先看 [第一批 8 条跨行错配](第一批裁决-跨行错配.md)，再按 [完整清单](人工裁决清单.md) 继续。
未命中 Gold 的预测并未自动判为 FP。
15 对资料类别/内容描述按批准规则列出“不可自动合并”候选，但没有系统全局合并轨迹，
因此 Duplicate Entity Rate / Merge Error Rate 仍不可计算；跨 Chunk 同名角色也未直接判作重复错误。
Precision、最终 Recall/F1、正式 Evidence Precision 和条件语义 Recall 仍为 null。
最终发布指标需人工确认新增 Gold 口径、逐条裁决和完整性范围；本材料不宣告离线测评通过。

## 可复核文件

- cases.jsonl：冻结原文、Gold 草稿、证据字符偏移、条件语义和未建模说明。
- entity_recall_outcomes.jsonl：命中/同名错误类型/名称未抽到。
- relation_loss_outcomes.jsonl：三元组命中与端点损失拆分、反向及其他关系类型线索。
- entity_review.jsonl / evidence_review.jsonl：全部预测的初审与 PENDING 人工状态。
- normalization_review.jsonl：15 对名称与内容描述的不可自动合并规则。
- manifest.json：父级 Gold、预测、生成脚本及输出文件哈希。

```powershell
cd D:\code\zhiqing-rag-platform
uv run python -m scripts.prepare_graph_holdout_review verify
```

本次只创建本地评估草稿与裁决材料，未改生产抽取、未外发数据、未写 PostgreSQL 或 Neo4j。
