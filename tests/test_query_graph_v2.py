"""可追溯映射和两条业务断言路径的完整来源、安全边界与旧接口兼容。"""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from test_query_graph import chunk as base_chunk
from test_query_graph import identity, projection, settings

from zhiqing_rag.query_retrieval.graph import OnlineGraphService
from zhiqing_rag.query_retrieval.graph_identity import identity_bridge
from zhiqing_rag.query_routing.models import GraphParams, RetrievalParams
from zhiqing_rag.query_routing.scope import AccessScope, PublishedDocument

STEP, MATERIAL, OTHER_MATERIAL, FIELD = (char * 64 for char in "abde")


def source_chunk(doc, chunk_id, content, title):
    return replace(
        base_chunk(),
        document_id=doc,
        chunk_id=chunk_id,
        generation_id=5 if doc == 4 else 11,
        revision_id=6 if doc == 4 else 7,
        document_title=title,
        document_name=title + ".md",
        content=content,
        content_sha256=sha256(content.encode()).hexdigest(),
    )


def source_projection(doc):
    return projection(
        document_id=doc,
        revision_id=6 if doc == 4 else 7,
        generation_id=5 if doc == 4 else 11,
        projection_id=20 if doc == 4 else 21,
        namespace="rebuild:first" if doc == 4 else "rebuild:second",
    )


def evidence(chunk, quote):
    return {
        **source_projection(chunk.document_id).parameters(),
        "chunk_id": chunk.chunk_id,
        "content_sha256": chunk.content_sha256,
        "quote": quote,
        "quote_start": chunk.content.index(quote),
        "quote_end": chunk.content.index(quote) + len(quote),
    }


class Repository:
    def __init__(self):
        self.projections_now = (source_projection(4), source_projection(9))
        self.chunks_now = (
            source_chunk(
                4,
                12,
                "预算已确认后，采购申请步骤需要采购申请表。\n采购申请表用于采购申请流程。",
                "采购办理说明",
            ),
            source_chunk(
                9,
                13,
                "采购申请表用于采购申请流程。\n采购申请表又称购置需求单。\n采购申请表包含用途。",
                "采购申请表",
            ),
        )
        self.exclude_final = set()
        self.reads = 0

    async def projections(self, identity, scope):
        return self.projections_now

    async def chunks(self, identity, scope, projections, evidence):
        self.reads += 1
        wanted = {row["chunk_id"] for row in evidence}
        allowed = {item.document_id for item in projections}
        return tuple(
            chunk
            for chunk in self.chunks_now
            if chunk.chunk_id in wanted
            and chunk.document_id in allowed
            and not (self.reads >= 3 and chunk.chunk_id in self.exclude_final)
        )


class Reader:
    def __init__(self, repository):
        self.repository = repository
        self.calls = []
        self.bad_second = None
        self.mention_override = None
        self.second_predicate = "HAS_FIELD"

    def rows(self):
        first, second = self.repository.chunks_now
        first_quote = first.content.splitlines()[0]
        second_quote = second.content.splitlines()[-1]
        return (
            {
                "assertion_id": "c" * 64,
                "seed_entity_uid": STEP,
                "subject_entity_uid": STEP,
                "object_entity_uid": MATERIAL,
                "subject_type": "Step",
                "object_type": "Material",
                "subject_name": "采购申请步骤",
                "object_name": "采购申请表",
                "predicate": "REQUIRES_MATERIAL",
                "condition_text": "预算已确认后",
                "evidence": evidence(first, first_quote),
            },
            {
                "assertion_id": "f" * 64,
                "seed_entity_uid": OTHER_MATERIAL,
                "subject_entity_uid": OTHER_MATERIAL,
                "object_entity_uid": FIELD,
                "subject_type": "Material",
                "object_type": "Field",
                "subject_name": "采购申请表",
                "object_name": "用途",
                "predicate": self.second_predicate,
                "condition_text": "",
                "evidence": evidence(second, second_quote),
            },
        )

    async def seeds(self, projections, *, names=(), uids=(), limit):
        self.calls.append(("seeds", tuple(item.namespace for item in projections)))
        records = (
            {"entity_uid": STEP, "entity_type": "Step", "normalized_name": "采购申请步骤"},
            {"entity_uid": MATERIAL, "entity_type": "Material", "normalized_name": "采购申请表"},
            {
                "entity_uid": OTHER_MATERIAL,
                "entity_type": "Material",
                "normalized_name": "采购申请表",
            },
        )
        return tuple(
            row for row in records if row["entity_uid"] in uids or row["normalized_name"] in names
        )

    async def relations(self, projections, uids, limit):
        self.calls.append(("relations", uids))
        allowed = {item.document_id for item in projections}
        result = []
        for row in self.rows():
            if row["evidence"]["document_id"] not in allowed:
                continue
            if row["subject_entity_uid"] in uids:
                row["seed_entity_uid"] = row["subject_entity_uid"]
            elif row["object_entity_uid"] in uids:
                row["seed_entity_uid"] = row["object_entity_uid"]
            else:
                continue
            if row["assertion_id"] == "f" * 64 and self.bad_second:
                row.update(self.bad_second)
            result.append(row)
        return tuple(result)

    async def mentions(self, projections, *, names=(), uids=(), limit):
        self.calls.append(("mentions", names, uids))
        if self.mention_override is not None:
            return self.mention_override
        first, second = self.repository.chunks_now
        allowed = {item.document_id for item in projections}
        return tuple(
            {
                "entity_uid": uid,
                "entity_type": "Material",
                "name": "采购申请表",
                "context_names": [],
                "evidence": evidence(
                    chunk, chunk.content.splitlines()[1 if uid == MATERIAL else 0]
                ),
            }
            for uid, chunk in ((MATERIAL, first), (OTHER_MATERIAL, second))
            if chunk.document_id in allowed
        )


@pytest.fixture
def service():
    repository = Repository()
    return OnlineGraphService(settings(), repository=repository, reader=Reader(repository))


def scope():
    return AccessScope(
        1, 2, (3,), (4, 9), (PublishedDocument(4, 3, 5), PublishedDocument(9, 3, 11))
    )


def params():
    return GraphParams(
        seed_entity_uids=(STEP,),
        projections=(source_projection(4).ref(), source_projection(9).ref()),
        max_hops=2,
    )


async def test_cross_namespace_two_hop_requires_material_then_fields_has_complete_sources(service):
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 1
    assert {hit.chunk.chunk_id for hit in result.hits} == {12, 13}
    path = result.hits[0].graph_paths[0]
    assert path.required_chunk_ids == ("12", "13")
    assert [hop.relation.predicate for hop in path.hops] == ["REQUIRES_MATERIAL", "HAS_FIELD"]
    assert path.hops[0].relation.condition_text == "预算已确认后"
    assert path.hops[1].relation.seed_entity_uid == OTHER_MATERIAL
    assert path.bridges[0].basis == "shared_context"
    assert len(path.bridges[0].identity_id) == 64
    assert path.bridges[0].identity_version == 1
    assert {source.namespace for source in path.bridges[0].sources} == {
        "rebuild:first",
        "rebuild:second",
    }
    for hit in result.hits:
        for hop in path.hops:
            if hop.source.chunk_id == str(hit.chunk.chunk_id):
                assert hop.relation in hit.graph_relations


async def test_readiness_selects_two_hops_only_after_actual_complete_path(service):
    ready = await service.readiness(
        identity(), scope(), "采购申请步骤需要哪些字段？", ("采购申请步骤",), RetrievalParams()
    )
    assert ready.params.max_hops == 2
    service.repository.projections_now = (source_projection(4),)
    ready = await service.readiness(
        identity(), scope(), "采购申请步骤需要哪些字段？", ("采购申请步骤",), RetrievalParams()
    )
    assert ready.params.max_hops == 1


async def test_explicit_alias_seed_is_resolved_and_reverified_on_formal_execution(service):
    ready = await service.readiness(
        identity(),
        scope(),
        "采购申请步骤使用购置需求单哪些字段？",
        ("采购申请步骤", "购置需求单"),
        RetrievalParams(),
    )
    assert ready.params.max_hops == 2
    assert OTHER_MATERIAL in ready.params.seed_entity_uids
    assert (await service.recall(identity(), scope(), ready.params, 5)).path_count == 1
    first, second = service.repository.chunks_now
    changed = second.content.replace("采购申请表又称购置需求单。\n", "")
    service.repository.chunks_now = (
        first,
        replace(second, content=changed, content_sha256=sha256(changed.encode()).hexdigest()),
    )
    assert not (await service.recall(identity(), scope(), ready.params, 5)).hits


async def test_same_title_different_explicit_process_does_not_bridge(service):
    first, second = service.repository.chunks_now
    changed = "此采购申请表仅用于报废回购流程。\n采购申请表包含用途。"
    service.repository.chunks_now = (
        first,
        replace(second, content=changed, content_sha256=sha256(changed.encode()).hexdigest()),
    )
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 0
    assert all(not hit.graph_paths for hit in result.hits)


@pytest.mark.parametrize(
    "field,value",
    [
        ("content_sha256", "0" * 64),
        ("quote_start", 999),
        ("namespace", "private"),
        ("generation_id", 99),
        ("revision_id", 99),
        ("tenant_id", 99),
        ("kb_id", 99),
    ],
)
async def test_mapping_evidence_tampering_prevents_entire_path(service, field, value):
    records = await service.reader.mentions(service.repository.projections_now, limit=300)
    records = deepcopy(records)
    records[1]["evidence"][field] = value
    service.reader.mention_override = records
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


@pytest.mark.parametrize(
    "quote",
    [
        "采购申请表不得包含用途。",
        "采购申请表不包含用途。",
        "采购申请表无需填写用途。",
        "采购申请表不需要填写用途。",
        "采购申请表与用途有关。",
        "采购申请表先核对用途。",
    ],
)
async def test_negative_or_weak_relation_cannot_form_positive_path(service, quote):
    first, second = service.repository.chunks_now
    changed = second.content.rsplit("\n", 1)[0] + "\n" + quote
    service.repository.chunks_now = (
        first,
        replace(second, content=changed, content_sha256=sha256(changed.encode()).hexdigest()),
    )
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


@pytest.mark.parametrize(
    "quote",
    [
        "采购申请步骤不得提交采购申请表。",
        "采购申请步骤不需要采购申请表。",
        "采购申请步骤无需提供采购申请表。",
        "采购申请步骤尚未提交采购申请表。",
        "采购申请步骤没有提交采购申请表。",
        "采购申请步骤不得直接提交采购申请表。",
        "采购申请步骤无需另行提供采购申请表。",
        "采购申请步骤不能重复填写采购申请表。",
        "采购申请步骤不得擅自提供采购申请表。",
        "采购申请步骤不再自行提交采购申请表。",
        "采购申请步骤不得再次提交采购申请表。",
        "采购申请步骤不得自行再次提交采购申请表。",
    ],
)
async def test_negated_step_material_action_does_not_form_positive_path(service, quote):
    _, second = service.repository.chunks_now
    first = source_chunk(4, 12, quote + "\n采购申请表用于采购申请流程。", "采购办理说明")
    service.repository.chunks_now = (first, second)
    rows = service.reader.rows()
    rows[0]["condition_text"] = ""
    service.reader.rows = lambda: deepcopy(rows)
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


@pytest.mark.parametrize("action", ["直接提交", "另行提供", "重复填写", "自行提交", "再次提交"])
async def test_negated_precondition_preserves_positive_action_with_adverb(service, action):
    _, second = service.repository.chunks_now
    first = source_chunk(
        4,
        12,
        "若尚未确认时，采购申请步骤需" + action + "采购申请表。\n采购申请表用于采购申请流程。",
        "采购办理说明",
    )
    service.repository.chunks_now = (first, second)
    rows = service.reader.rows()
    rows[0]["condition_text"] = "若尚未确认时"
    service.reader.rows = lambda: deepcopy(rows)
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 1
    assert result.hits[0].graph_paths[0].hops[0].relation.condition_text == "若尚未确认时"


@pytest.mark.parametrize("condition", ["预算", "预算已确认", "已确认后", ""])
async def test_path_rejects_truncated_or_missing_continuous_condition(service, condition):
    rows = service.reader.rows()
    rows[0]["condition_text"] = condition
    service.reader.rows = lambda: deepcopy(rows)
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


@pytest.mark.parametrize(
    "condition",
    ["只有预算已确认", "仅在预算已确认的情况下", "只要预算已确认", "除非预算不足"],
)
async def test_explicit_condition_prefix_is_required_in_full(service, condition):
    _, second = service.repository.chunks_now
    first = source_chunk(
        4,
        12,
        condition + "，采购申请步骤需要采购申请表。\n采购申请表用于采购申请流程。",
        "采购办理说明",
    )
    service.repository.chunks_now = (first, second)
    rows = service.reader.rows()
    rows[0]["condition_text"] = ""
    service.reader.rows = lambda: deepcopy(rows)
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0
    rows[0]["condition_text"] = condition
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 1


async def test_multiple_preconditions_remain_one_continuous_complete_source_span(service):
    _, second = service.repository.chunks_now
    first = source_chunk(
        4,
        12,
        "预算已确认后，申请已签字后，采购申请步骤需要采购申请表。\n采购申请表用于采购申请流程。",
        "采购办理说明",
    )
    service.repository.chunks_now = (first, second)
    rows = service.reader.rows()
    rows[0]["condition_text"] = "预算已确认后"
    service.reader.rows = lambda: deepcopy(rows)
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0
    rows[0]["condition_text"] = "预算已确认后，申请已签字后"
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 1
    assert (
        result.hits[0].graph_paths[0].hops[0].relation.condition_text == rows[0]["condition_text"]
    )


async def test_negated_preconditions_do_not_negate_positive_relationship_actions(service):
    first = source_chunk(
        4,
        12,
        "如果预算不需要调整，采购申请步骤需要采购申请表。\n采购申请表用于采购申请流程。",
        "采购办理说明",
    )
    second = source_chunk(
        9,
        13,
        "采购申请表用于采购申请流程。\n如果原记录不能复用，采购申请表包含用途。",
        "采购申请表",
    )
    service.repository.chunks_now = (first, second)
    rows = service.reader.rows()
    rows[0]["condition_text"] = "如果预算不需要调整"
    rows[1]["condition_text"] = "如果原记录不能复用"
    service.reader.rows = lambda: deepcopy(rows)
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 1
    assert [hop.relation.condition_text for hop in result.hits[0].graph_paths[0].hops] == [
        "如果预算不需要调整",
        "如果原记录不能复用",
    ]


@pytest.mark.parametrize(
    "denial",
    [
        "该采购申请表与采购办理说明中的采购申请表不是同一份材料。",
        "采购申请表与采购办理说明中的采购申请表并不相同。",
        "采购申请表与采购办理说明中的采购申请表属于不同材料。",
        "采购申请表与采购办理说明中的采购申请表不得等同。",
        "采购申请表与采购办理说明中的采购申请表不能等同。",
    ],
)
async def test_explicit_nonidentity_overrides_shared_process_and_same_title(service, denial):
    first, second = service.repository.chunks_now
    lines = second.content.splitlines()
    changed = "\n".join((lines[0], denial, *lines[1:]))
    service.repository.chunks_now = (
        first,
        replace(second, content=changed, content_sha256=sha256(changed.encode()).hexdigest()),
    )
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


async def test_different_attribute_value_does_not_imply_entity_nonidentity(service):
    first, second = service.repository.chunks_now
    lines = second.content.splitlines()
    changed = "\n".join((lines[0], "采购申请表在不同部门填写的用途可能不同。", *lines[1:]))
    service.repository.chunks_now = (
        first,
        replace(second, content=changed, content_sha256=sha256(changed.encode()).hexdigest()),
    )
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 1


@pytest.mark.parametrize(
    "context,denial",
    [
        ("采购申请表仅用于报废回购流程。", ""),
        (
            "采购申请表用于采购申请流程。",
            "该采购申请表与采购办理说明中的采购申请表不是同一份材料。",
        ),
    ],
)
async def test_same_document_same_uid_cannot_bypass_context_or_nonidentity_conflict(
    service, context, denial
):
    first, _ = service.repository.chunks_now
    second = source_chunk(4, 13, "\n".join((context, denial, "采购申请表包含用途。")), "采购申请表")
    service.repository.chunks_now = (first, second)
    service.repository.projections_now = (source_projection(4),)
    rows = service.reader.rows()
    rows[1]["subject_entity_uid"] = MATERIAL
    rows[1]["seed_entity_uid"] = MATERIAL
    service.reader.rows = lambda: deepcopy(rows)
    service.reader.mention_override = tuple(
        {
            "entity_uid": MATERIAL,
            "entity_type": "Material",
            "name": "采购申请表",
            "context_names": [],
            "evidence": evidence(chunk, quote),
        }
        for chunk, quote in ((first, first.content.splitlines()[1]), (second, context))
    )
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 0
    assert all(not hit.graph_paths for hit in result.hits)


async def test_source_revocation_during_mapping_invalidates_complete_path(service):
    service.repository.exclude_final = {12}
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


async def test_candidate_budget_never_returns_partial_path(service):
    result = await service.recall(identity(), scope(), params(), 1)
    assert result.path_count == 0
    assert len(result.hits) == 1


async def test_two_hop_toggle_keeps_legacy_single_hop_behavior(service):
    service.settings.query_graph_max_hops = 1
    with pytest.raises(ValueError, match="GRAPH_HOPS_UNSUPPORTED"):
        await service.recall(identity(), scope(), params(), 5)
    ready = await service.readiness(
        identity(), scope(), "采购申请步骤需要什么？", ("采购申请步骤",), RetrievalParams()
    )
    assert ready.params.max_hops == 1


async def test_has_step_then_responsibility_preserves_incoming_direction_and_condition(service):
    first = source_chunk(4, 12, "采购申请流程包含采购申请步骤。", "采购办理说明")
    second = source_chunk(4, 13, "审批完成后，系统管理员负责采购申请步骤。", "采购办理说明")
    service.repository.chunks_now = (first, second)
    service.repository.projections_now = (source_projection(4),)
    original = service.reader.rows()
    first_row = {
        **original[0],
        "predicate": "HAS_STEP",
        "condition_text": "",
        "subject_type": "Process",
        "object_type": "Step",
        "subject_name": "采购申请流程",
        "object_name": "采购申请步骤",
        "evidence": evidence(first, first.content),
    }
    second_row = {
        **original[1],
        "predicate": "RESPONSIBLE_FOR",
        "condition_text": "审批完成后",
        "subject_type": "Actor",
        "object_type": "Step",
        "subject_name": "系统管理员",
        "object_name": "采购申请步骤",
        "object_entity_uid": MATERIAL,
        "evidence": evidence(second, second.content),
    }
    service.reader.rows = lambda: (deepcopy(first_row), deepcopy(second_row))
    service.reader.mention_override = ()
    result = await service.recall(identity(), scope(), params(), 5)
    assert result.path_count == 1
    path = result.hits[0].graph_paths[0]
    assert [hop.traversal for hop in path.hops] == ["outgoing", "incoming"]
    assert path.hops[1].relation.seed_entity_uid == MATERIAL
    assert path.hops[1].relation.subject_entity_uid == OTHER_MATERIAL
    assert path.hops[1].relation.condition_text == "审批完成后"
    assert path.bridges == ()
    assert path.hops[1].relation in result.hits[1].graph_relations


@pytest.mark.parametrize("action,predicate", [("确认", "CONFIRMS"), ("审核", "REVIEWS")])
@pytest.mark.parametrize("negation", ["", "不", "未", "没有"])
async def test_step_name_starting_with_action_does_not_hide_action_negation(
    service, action, predicate, negation
):
    step_name = action + "采购申请步骤"
    first = source_chunk(4, 12, "采购申请流程包含" + step_name + "。", "采购办理说明")
    second = source_chunk(4, 13, "系统管理员" + negation + step_name + "。", "采购办理说明")
    service.repository.chunks_now = (first, second)
    service.repository.projections_now = (source_projection(4),)
    rows = service.reader.rows()
    rows[0].update(
        predicate="HAS_STEP",
        condition_text="",
        subject_type="Process",
        object_type="Step",
        subject_name="采购申请流程",
        object_name=step_name,
        evidence=evidence(first, first.content),
    )
    rows[1].update(
        predicate=predicate,
        condition_text="",
        subject_type="Actor",
        object_type="Step",
        subject_name="系统管理员",
        object_name=step_name,
        object_entity_uid=MATERIAL,
        evidence=evidence(second, second.content),
    )
    service.reader.rows = lambda: deepcopy(rows)
    service.reader.mention_override = ()
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == (
        0 if negation else 1
    )


async def test_step_identity_is_not_automatically_shared_across_documents(service):
    first = source_chunk(4, 12, "采购申请流程包含采购申请步骤。", "采购办理说明")
    second = source_chunk(9, 13, "系统管理员负责采购申请步骤。", "采购办理说明")
    service.repository.chunks_now = (first, second)
    original = service.reader.rows()
    first_row = {
        **original[0],
        "predicate": "HAS_STEP",
        "condition_text": "",
        "subject_type": "Process",
        "object_type": "Step",
        "subject_name": "采购申请流程",
        "object_name": "采购申请步骤",
        "evidence": evidence(first, first.content),
    }
    second_row = {
        **original[1],
        "predicate": "RESPONSIBLE_FOR",
        "condition_text": "",
        "subject_type": "Actor",
        "object_type": "Step",
        "subject_name": "系统管理员",
        "object_name": "采购申请步骤",
        "object_entity_uid": MATERIAL,
        "evidence": evidence(second, second.content),
    }
    service.reader.rows = lambda: (deepcopy(first_row), deepcopy(second_row))
    service.reader.mention_override = ()
    assert (await service.recall(identity(), scope(), params(), 5)).path_count == 0


async def test_knowledge_base_identity_is_stable_across_rebuild_namespaces_and_scoped_to_kb(
    service,
):
    left, right = await service._mentions(
        identity(), scope(), service.repository.projections_now, names=("采购申请表",)
    )
    original = identity_bridge(left, right)
    rebuilt = identity_bridge(
        replace(
            left, uid="1" * 64, source=left.source.model_copy(update={"namespace": "new:first"})
        ),
        replace(
            right, uid="2" * 64, source=right.source.model_copy(update={"namespace": "new:second"})
        ),
    )
    assert rebuilt.identity_id == original.identity_id
    alias_named = identity_bridge(left, replace(right, name="购置需求单"))
    assert alias_named.identity_id == original.identity_id
    other_kb = identity_bridge(
        replace(left, chunk=replace(left.chunk, knowledge_base_id=8)),
        replace(right, chunk=replace(right.chunk, knowledge_base_id=8)),
    )
    assert other_kb.identity_id != original.identity_id
    assert (
        identity_bridge(left, replace(right, chunk=replace(right.chunk, knowledge_base_id=8)))
        is None
    )


async def test_generic_actor_and_context_dependent_types_are_not_merged_by_name(service):
    left, right = await service._mentions(
        identity(), scope(), service.repository.projections_now, names=("采购申请表",)
    )
    assert (
        identity_bridge(
            replace(left, entity_type="Actor", contexts=frozenset()),
            replace(right, entity_type="Actor", contexts=frozenset()),
        )
        is None
    )
    for kind in ("Step", "State", "Field", "Rule"):
        assert (
            identity_bridge(replace(left, entity_type=kind), replace(right, entity_type=kind))
            is None
        )
