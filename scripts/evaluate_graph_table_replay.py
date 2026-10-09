"""Freeze relocated table Gold targets, replay four chunks, and diagnose recall locally."""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openai import AsyncOpenAI

from scripts.replay_graph_zero_chunks import RecordingClient, compare_to_gold
from zhiqing_rag.core.config import Settings
from zhiqing_rag.workers import graph_extraction
from zhiqing_rag.workers.graph_extraction import (
    EXTRACTOR_VERSION,
    SYSTEM_PROMPT,
    Extraction,
    GraphExtractionError,
    extract_chunk,
    normalize_name,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "eval/graph_build/v2-table-header-pilot"
GOLD = ROOT / "eval/graph_build/gold-v1/cases.jsonl"
TARGETS = PILOT / "replay-targets"

TABLE_GUIDANCE = """
补充：输入为 Markdown 表格时，每行的数据含义由本 Chunk 的列头决定。
1. 将责任角色与其具体执行的动作分别建为 Actor、Step；Step 名称取动作单元格中的
连续动词短语，去除角色前缀。任务/事项列是主题，不能用主题名称替代另一列明确写出的动作。
多个角色共同承担一个动作时，保留原文的联合角色表达，不擅自改变主体边界。
2. 明确的保管要求、访问限制、版本约束、禁止事项等建为 Rule；资料对象建为 Material。
本行的资料受到本行要求约束时，建立 Material->Rule 的 GOVERNED_BY。
主要读者或受众不自动成为执行者；原文没有对应动作就不建立 RESPONSIBLE_FOR。
3. 在 25 实体预算内优先保留这些明确关系的两端，再抽其他孤立主题或读者。
4. 条件、否定和时间限制必须保留；不得把完成依据当成已经发生的事实。
重新核查的触发情形不等于资料当前状态，不自动生成 HAS_STATE。
5. evidence_quote 使用本行连续逐字原文，禁止重排单元格、拼接片段或改写标点。
仅为明确事实生成关系，表中相邻或共现本身不证明关系。
"""


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def relocate(chunks, old_cases):
    """Preserve original target identities; auxiliary endpoints do not inflate recall."""
    cases = {}
    for chunk in chunks:
        if not chunk["metadata"].get("table_preamble"):
            cases[chunk["local_chunk_id"]] = {
                **chunk,
                "case_id": chunk["local_chunk_id"],
                "review_status": "codex_relocated_draft_awaiting_human_review",
                "gold_scope": "selected_targets_not_exhaustive",
                "entities": [],
                "relations": [],
                "evidence": [],
            }

    def locate(document_id, quote):
        found = [
            case for case in cases.values()
            if case["document_id"] == document_id and quote in case["content"]
        ]
        if len(found) != 1:
            raise ValueError("Gold 行证据无法唯一定位到新 Chunk")
        return found[0]

    def add_entity(case, entity, origin=None, quote=None):
        key = (entity["type"], normalize_name(entity["name"]))
        saved = next((e for e in case["entities"]
                      if (e["type"], normalize_name(e["name"])) == key), None)
        if saved is None:
            saved = {**entity, "id": f"e{len(case['entities']) + 1}",
                     "evidence_quote": quote or entity["evidence_quote"],
                     "scored_target_ids": []}
            case["entities"].append(saved)
        if origin and origin not in saved["scored_target_ids"]:
            saved["scored_target_ids"].append(origin)
        return saved["id"]

    for old in old_cases:
        for entity in old["entities"]:
            case = locate(old["document_id"], entity["evidence_quote"])
            add_entity(case, entity, f"{old['case_id']}:{entity['id']}")
    for old in old_cases:
        by_id = {entity["id"]: entity for entity in old["entities"]}
        for index, relation in enumerate(old["relations"], 1):
            case = locate(old["document_id"], relation["evidence_quote"])
            quote = relation["evidence_quote"]
            source = by_id[relation["source_id"]]
            target = by_id[relation["target_id"]]
            if source["name"] not in quote or target["name"] not in quote:
                raise ValueError("Gold 关系的行证据未同时包含两端，需人工裁决")
            case["relations"].append({
                **relation,
                "id": f"r{len(case['relations']) + 1}",
                "source_id": add_entity(case, source, quote=quote),
                "target_id": add_entity(case, target, quote=quote),
                "origin_target_id": f"{old['case_id']}:r{index}",
            })
    for case in cases.values():
        for kind, records in (("entity", case["entities"]), ("relation", case["relations"])):
            for record in records:
                quote = record["evidence_quote"]
                start = case["content"].index(quote)
                case["evidence"].append({"target_kind": kind, "target_id": record["id"],
                                         "quote": quote, "start": start,
                                         "end": start + len(quote)})
    return list(cases.values())


def prepare():
    if TARGETS.exists():
        raise FileExistsError("评估目标已冻结；使用 verify，不覆盖")
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    if sha(PILOT / "chunks.jsonl") != manifest["files"]["chunks.jsonl"]:
        raise ValueError("试验 Chunk 哈希变化")
    if sha(GOLD) != manifest["gold_cases_sha256"]:
        raise ValueError("原 Gold 哈希变化")
    old = [case for case in read_jsonl(GOLD) if case["document_id"] in (83, 89)]
    cases = relocate(read_jsonl(PILOT / "chunks.jsonl"), old)
    TARGETS.mkdir(exist_ok=False)
    write_jsonl(TARGETS / "cases.jsonl", cases)
    write_json(TARGETS / "manifest.json", {
        "schema_version": "graph-table-replay-targets-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "pilot_manifest_sha256": sha(PILOT / "manifest.json"),
        "source_gold_sha256": sha(GOLD),
        "cases_sha256": sha(TARGETS / "cases.jsonl"),
        "case_count": len(cases),
        "scored_entity_targets": sum(len(e["scored_target_ids"])
                                     for c in cases for e in c["entities"]),
        "endpoint_entity_count": sum(len(c["entities"]) for c in cases),
        "relation_targets": sum(len(c["relations"]) for c in cases),
        "review_status": "codex_relocated_draft_awaiting_human_review",
    })
    (TARGETS / "README.md").write_text(
        "# 修复后表格的隔离抽取评估目标\n\n"
        "保留原 Gold 的 37 个实体目标、20 个关系目标、名称、类型、方向及条件；"
        "只重新定位到 4 个新数据 Chunk 并重算证据字符偏移。"
        "主管同时成为两个新 Chunk 的关系端点，因此共有 38 个端点实体记录；"
        "新增的辅助端点不进入 37 个实体目标的 Recall 分母。\n\n"
        "这是 Codex 迁移草稿，未获得独立人工裁决。Gold 为选择性标注，"
        "未命中目标的预测须逐条裁决，不直接计为 FP。"
        "原 v1 和 Gold v1 未改动。只向模型发送正文，不发送 Gold。\n",
        encoding="utf-8",
    )
    verify()


def verify():
    manifest = json.loads((TARGETS / "manifest.json").read_text(encoding="utf-8"))
    if (sha(PILOT / "manifest.json") != manifest["pilot_manifest_sha256"]
            or sha(GOLD) != manifest["source_gold_sha256"]
            or sha(TARGETS / "cases.jsonl") != manifest["cases_sha256"]):
        raise ValueError("目标集或来源哈希变化")
    pilot = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))
    if sha(PILOT / "chunks.jsonl") != pilot["files"]["chunks.jsonl"]:
        raise ValueError("试验 Chunk 文件已变化")
    cases = read_jsonl(TARGETS / "cases.jsonl")
    old = [case for case in read_jsonl(GOLD) if case["document_id"] in (83, 89)]
    if cases != relocate(read_jsonl(PILOT / "chunks.jsonl"), old):
        raise ValueError("Gold 迁移无法复现")
    if len(cases) != 4 or manifest["scored_entity_targets"] != 37:
        raise ValueError("授权范围或实体目标数量不符")
    if manifest["relation_targets"] != 20:
        raise ValueError("关系目标数量不符")
    for case in cases:
        for evidence in case["evidence"]:
            if case["content"][evidence["start"]:evidence["end"]] != evidence["quote"]:
                raise ValueError("证据偏移无效")
    print("评估目标校验通过：4 Chunk，37 实体目标，20 关系目标", flush=True)
    return cases


async def replay(output, table_guidance=False):
    cases = verify()
    if output.exists():
        raise FileExistsError("结果目录已存在；使用 report，不覆盖")
    settings = Settings()
    if not settings.deepseek_api_key.get_secret_value():
        raise ValueError("DeepSeek 未配置")
    effective_prompt = SYSTEM_PROMPT + TABLE_GUIDANCE if table_guidance else SYSTEM_PROMPT
    # Process-local experimental override; production source and graph remain untouched.
    graph_extraction.SYSTEM_PROMPT = effective_prompt
    output.mkdir(parents=True, exist_ok=False)
    (output / "system_prompt.txt").write_text(effective_prompt, encoding="utf-8")
    write_json(output / "metadata.json", {
        "started_at_utc": datetime.now(UTC).isoformat(),
        "targets_manifest_sha256": sha(TARGETS / "manifest.json"),
        "model": settings.graph_extraction_model,
        "extractor_version": EXTRACTOR_VERSION,
        "extractor_source_sha256": sha(ROOT / "src/zhiqing_rag/workers/graph_extraction.py"),
        "system_prompt_sha256": hashlib.sha256(effective_prompt.encode()).hexdigest(),
        "table_guidance_experiment": table_guidance,
        "mode": "isolated_current_extractor_no_database_writes",
        "authorized_chunk_ids": [c["case_id"] for c in cases],
    })
    async with AsyncOpenAI(api_key=settings.deepseek_api_key.get_secret_value(),
                           base_url=settings.deepseek_base_url,
                           timeout=settings.graph_llm_timeout_seconds,
                           max_retries=0) as client:
        with (output / "replay.jsonl").open("w", encoding="utf-8") as stream:
            for case in cases:
                trace = []
                row = {"case_id": case["case_id"], "document_id": case["document_id"],
                       "content_sha256": case["content_sha256"], "attempts": trace}
                try:
                    extraction = await extract_chunk(
                        settings, case["content"], client=RecordingClient(client, trace)
                    )
                    row.update(status="VALID_EXTRACTION",
                               extraction=extraction.model_dump(mode="json"))
                except Exception as error:
                    row.update(status="ERROR", error_type=type(error).__name__)
                    if isinstance(error, GraphExtractionError):
                        row["error_code"] = error.code
                row["finished_at_utc"] = datetime.now(UTC).isoformat()
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                print(f"{case['case_id']}: {row['status']}, {len(trace)} 次请求", flush=True)
    report(output)


def report(output):
    cases = {case["case_id"]: case for case in verify()}
    meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    if meta["targets_manifest_sha256"] != sha(TARGETS / "manifest.json"):
        raise ValueError("抽取结果对应的目标版本不一致")
    rows = read_jsonl(output / "replay.jsonl")
    if len(rows) != 4 or {r["case_id"] for r in rows} != set(cases):
        raise ValueError("结果不完整或 Chunk 不在授权范围")
    totals = Counter()
    diagnostics = []
    queue = []
    for row in rows:
        case = cases[row["case_id"]]
        if row["content_sha256"] != case["content_sha256"]:
            raise ValueError("抽取输入哈希与目标不一致")
        unavailable = row["status"] == "ERROR" and not any(
            attempt.get("raw_output") is not None for attempt in row["attempts"]
        )
        totals["unavailable_chunks"] += unavailable
        extraction = Extraction.model_validate(row.get("extraction", {"entities": [],
                                                                     "relations": []}))
        validate_evidence(extraction, case["content"])
        scored_case = {**case, "entities": [e for e in case["entities"]
                                           if e["scored_target_ids"]]}
        # Triple comparison needs auxiliary endpoint IDs; entity target counts do not.
        comparison = compare_to_gold(extraction, case)
        comparison["gold_entities"] = len(scored_case["entities"])
        comparison["exact_entity_matches"] = compare_to_gold(
            extraction, {**scored_case, "relations": []}
        )["exact_entity_matches"]
        totals.update(comparison)
        totals["failed_chunks"] += row["status"] == "ERROR"
        entity_keys = {(e.type, normalize_name(e.canonical_name)) for e in extraction.entities}
        gold_by_id = {e["id"]: e for e in case["entities"]}
        predicted_by_id = {e.id: e for e in extraction.entities}
        predicted_triples = Counter()
        predicted_qualified = Counter()
        for relation in extraction.relations:
            source = predicted_by_id[relation.source_id]
            target = predicted_by_id[relation.target_id]
            key = (source.type, normalize_name(source.canonical_name), relation.type,
                   target.type, normalize_name(target.canonical_name))
            predicted_triples[key] += 1
            predicted_qualified[(*key, relation.condition_text)] += 1
            queue.append({"case_id": case["case_id"], "prediction_index": len(queue) + 1,
                          "source": source.model_dump(), "relation": relation.model_dump(),
                          "target": target.model_dump(), "evidence_support": "PENDING",
                          "allowed_verdicts": ["SUPPORT", "CONTRADICT", "INSUFFICIENT"],
                          "reviewer": None})
        for relation in case["relations"]:
            source = gold_by_id[relation["source_id"]]
            target = gold_by_id[relation["target_id"]]
            source_key = (source["type"], normalize_name(source["name"]))
            target_key = (target["type"], normalize_name(target["name"]))
            key = (*source_key, relation["type"], *target_key)
            eligible = not unavailable and source_key in entity_keys and target_key in entity_keys
            matched = predicted_triples[key] > 0
            if matched:
                predicted_triples[key] -= 1
            qualified_key = (*key, relation["condition_text"])
            qualified = predicted_qualified[qualified_key] > 0
            if qualified:
                predicted_qualified[qualified_key] -= 1
            totals["conditional_eligible"] += eligible
            totals["conditional_hits"] += eligible and matched
            totals["qualified_hits"] += qualified
            loss = ("EXTRACTION_UNAVAILABLE" if unavailable else "HIT" if matched
                    else "RELATION_WITH_ENDPOINTS_PRESENT" if eligible
                    else "MISSING_ENTITY_ENDPOINT")
            totals[loss] += 1
            diagnostics.append({"case_id": case["case_id"],
                                "origin_target_id": relation["origin_target_id"],
                                "source": source["name"], "target": target["name"],
                                "type": relation["type"], "loss": loss,
                                "source_found": source_key in entity_keys,
                                "target_found": target_key in entity_keys,
                                "condition_exact_match": qualified})
        for entity in extraction.entities:
            queue.append({"case_id": case["case_id"], "prediction_index": len(queue) + 1,
                          "entity": entity.model_dump(), "entity_verdict": "PENDING",
                          "reviewer": None})
    summary = {
        "counts": dict(totals),
        "evaluation_status": ("INCOMPLETE_API_UNAVAILABLE" if totals["unavailable_chunks"]
                              else "COMPLETE_RECALL_ONLY_HUMAN_REVIEW_PENDING"),
        "entity_recall": (None if totals["unavailable_chunks"] else
                          totals["exact_entity_matches"] / totals["gold_entities"]),
        "triple_recall": (None if totals["unavailable_chunks"] else
                          totals["exact_directed_triple_matches"] / totals["gold_relations"]),
        "conditional_recall": (totals["conditional_hits"] / totals["conditional_eligible"]
                               if totals["conditional_eligible"] else None),
        "final_precision": None, "final_f1": None, "evidence_precision": None,
        "normalization_error_rates": None,
        "human_adjudication": "pending",
        "baseline_scope": "original_two_documents_selected_targets_16/37_entities_5/20_triples",
        "attribution_limit": "model_time_and_chunk_boundaries_changed_no_paired_control",
    }
    write_json(output / "summary.json", summary)
    write_jsonl(output / "relation_diagnostics.jsonl", diagnostics)
    write_jsonl(output / "review_queue.jsonl", queue)
    (output / "README.md").write_text(
        "# 修复后四个表格 Chunk 的实际模型抽取\n\n"
        f"评估状态：{summary['evaluation_status']}。API 无返回 {totals['unavailable_chunks']}/4；"
        "若有 API 不可用，下列为已收到结果的计数，不代表完整 Recall。\n\n"
        f"Entity Recall：{totals['exact_entity_matches']}/{totals['gold_entities']}；"
        f"Relation Triple Recall：{totals['exact_directed_triple_matches']}/"
        f"{totals['gold_relations']}。\n\n"
        f"两端已抽到的 Conditional Recall：{totals['conditional_hits']}/"
        f"{totals['conditional_eligible']}；条件逐字严格匹配："
        f"{totals['qualified_hits']}/{totals['gold_relations']}。\n\n"
        f"关系损失：端点漏抽 {totals['MISSING_ENTITY_ENDPOINT']}，两端已抽到但关系未命中 "
        f"{totals['RELATION_WITH_ENDPOINTS_PRESENT']}。失败 Chunk {totals['failed_chunks']}/4。\n\n"
        "对照旧两份文档为实体 16/37、关系 5/20。目标名称、类型、方向不变；"
        "Chunk 边界与模型运行时间改变，不能将差异全部归因于列头修复。"
        "这四块不是独立留出集，也不是完整 59 Chunk 复测。\n\n"
        "Gold 是选择性迁移草稿。所有预测写入 review_queue.jsonl 等待独立人工裁决；"
        "证据逐字出现不代表语义支持。Precision、F1、Evidence Precision、"
        "Normalization 错误率暂不计算。无 PostgreSQL/Neo4j 写入。"
        f"表格提示词实验：{meta.get('table_guidance_experiment', False)}。"
        "不包含离线列抽取规则；实验提示词只在当前进程生效。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "replay", "report"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--table-guidance", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "verify":
        verify()
    elif args.output is None:
        parser.error("replay/report 需要 --output")
    elif args.command == "replay":
        asyncio.run(replay(args.output, args.table_guidance))
    else:
        report(args.output)


if __name__ == "__main__":
    main()
