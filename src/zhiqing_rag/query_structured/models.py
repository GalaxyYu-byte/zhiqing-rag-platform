"""目录结果与仅存于聊天快照的统计来源；不将全量来源发送到客户端。"""

from datetime import date, datetime
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from zhiqing_rag.query_analysis.models import StrictModel
from zhiqing_rag.query_metadata.models import ResolvedFilter
from zhiqing_rag.query_routing.models import PublicId

MAX_STRUCTURED_DOCUMENTS = 10000
StructuredOperation = Literal["count_documents", "list_documents"]


class StructuredDocument(StrictModel):
    model_config = ConfigDict(str_strip_whitespace=False)
    knowledge_base_id: PublicId
    document_id: PublicId
    revision_id: PublicId
    generation_id: PublicId
    title: str
    document_name: str
    file_type: str
    business_status: str
    department_id: PublicId | None
    event_date: date | None
    uploaded_at: datetime


class StructuredSource(StrictModel):
    knowledge_base_id: PublicId
    document_id: PublicId
    revision_id: PublicId
    generation_id: PublicId
    updated_at: datetime
    metadata_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class StructuredProvenance(StrictModel):
    tenant_id: PublicId
    user_id: PublicId
    knowledge_base_ids: tuple[PublicId, ...]
    selected_document_ids: tuple[PublicId, ...] = ()
    operation: StructuredOperation
    applied_filters: tuple[ResolvedFilter, ...] = ()
    executed_at: datetime
    sources: tuple[StructuredSource, ...] = Field(max_length=MAX_STRUCTURED_DOCUMENTS)

    @model_validator(mode="after")
    def validate_sources(self):
        if len(set(self.knowledge_base_ids)) != len(self.knowledge_base_ids):
            raise ValueError("统计知识库锚点不能重复")
        if len(set(self.selected_document_ids)) != len(self.selected_document_ids):
            raise ValueError("统计文档选择锚点不能重复")
        identifiers = (
            self.tenant_id, self.user_id, *self.knowledge_base_ids, *self.selected_document_ids,
        )
        if any(int(value) > 2**63 - 1 for value in identifiers):
            raise ValueError("统计范围 ID 必须为正 bigint")
        seen = set()
        for source in self.sources:
            if any(
                int(getattr(source, name)) > 2**63 - 1
                for name in ("knowledge_base_id", "document_id", "revision_id", "generation_id")
            ):
                raise ValueError("统计来源 ID 必须为正 bigint")
            if source.knowledge_base_id not in self.knowledge_base_ids:
                raise ValueError("统计来源必须属于选定知识库")
            if self.selected_document_ids and source.document_id not in self.selected_document_ids:
                raise ValueError("统计来源必须属于选定文档范围")
            if source.document_id in seen:
                raise ValueError("统计来源不能重复计数同一文档")
            seen.add(source.document_id)
        return self


class StructuredResult(StrictModel):
    operation: StructuredOperation
    total: int = Field(ge=0)
    documents: tuple[StructuredDocument, ...] = ()
    offset: int = Field(default=0, ge=0, le=100000)
    limit: int = Field(default=20, ge=1, le=100)
    applied_filters: tuple[ResolvedFilter, ...] = ()
    executed_at: datetime
    scope: Literal["authorized_published_documents"] = "authorized_published_documents"
    provenance: StructuredProvenance | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def validate_page(self):
        if self.operation == "count_documents" and self.documents:
            raise ValueError("计数操作不返回文档清单")
        if len(self.documents) > self.limit or len(self.documents) > self.total:
            raise ValueError("目录页面不能超过分页限制或总量")
        if self.provenance is not None and (
            self.total != len(self.provenance.sources)
            or self.operation != self.provenance.operation
            or self.applied_filters != self.provenance.applied_filters
            or self.executed_at != self.provenance.executed_at
        ):
            raise ValueError("统计结果与服务端来源不一致")
        if self.provenance is not None and self.operation == "list_documents":
            names = ("knowledge_base_id", "document_id", "revision_id", "generation_id")
            expected = self.provenance.sources[self.offset:self.offset + self.limit]
            actual = tuple(tuple(getattr(doc, name) for name in names) for doc in self.documents)
            if actual != tuple(
                tuple(getattr(source, name) for name in names) for source in expected
            ):
                raise ValueError("文档页面必须来自统计范围中的对应页")
        return self
