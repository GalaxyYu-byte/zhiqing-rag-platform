"""文档上传请求参数、响应与用例错误。"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

BusinessStatus = Literal["UNKNOWN", "DRAFT", "EFFECTIVE", "ARCHIVED", "REPEALED"]


def normalize_document_code(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if any(ord(char) < 32 for char in value):
        raise ValueError("业务编号不能包含控制字符")
    return value or None


class UploadParameters(BaseModel):
    knowledge_base_id: int = Field(gt=0)
    confidentiality: int = Field(default=1, ge=0, le=2)
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=48, ge=0)
    document_code: str | None = Field(default=None, max_length=100)
    business_status: BusinessStatus = "UNKNOWN"
    event_date: date | None = None

    _normalize_code = field_validator("document_code")(normalize_document_code)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("重叠大小必须小于分块大小的一半")
        return self


class DocumentMetadata(BaseModel):
    document_id: str
    knowledge_base_id: str
    title: str
    document_code: str | None
    business_status: BusinessStatus
    event_date: date | None
    department_id: str | None
    department_name: str | None
    updated_at: datetime
    can_write: bool


class DocumentMetadataPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_updated_at: AwareDatetime
    title: str | None = Field(default=None, min_length=1, max_length=500)
    document_code: str | None = Field(default=None, max_length=100)
    business_status: BusinessStatus | None = None
    event_date: date | None = None

    _normalize_code = field_validator("document_code")(normalize_document_code)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError("文档标题不能为空或包含控制字符")
        return value

    @model_validator(mode="after")
    def validate_patch(self):
        if not (self.model_fields_set - {"expected_updated_at"}):
            raise ValueError("请提供需要修改的文档属性")
        for field in ("title", "business_status"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError("文档标题和业务状态不能清空")
        return self


class KnowledgeBaseOption(BaseModel):
    id: str
    code: str
    name: str
    description: str
    permission: Literal["READ", "WRITE", "ADMIN"] = "WRITE"


class UploadResult(BaseModel):
    # bigint 以字符串返回，避免浏览器整数精度丢失。
    document_id: str
    revision_id: str
    generation_id: str
    task_id: str
    knowledge_base_id: str
    original_filename: str
    file_size: int
    file_sha256: str
    mime_type: str
    status: Literal["STORED"] = "STORED"
    processing_status: Literal["PENDING"] = "PENDING"
    created_at: datetime


class DocumentTaskStatus(BaseModel):
    task_id: str
    document_id: str
    revision_id: str
    revision_no: int
    generation_id: str
    # 页面用这两个指针区分“正在构建的修订”和“当前已发布的索引批次”。
    desired_revision_id: str | None
    active_generation_id: str | None
    knowledge_base_id: str
    original_filename: str
    file_size: int
    status: Literal["PENDING", "RUNNING", "RETRY_WAIT", "SUCCEEDED", "FAILED", "CANCELLED"]
    stage: str | None
    progress: int
    attempt_count: int
    max_attempts: int
    error_code: str | None
    chunk_count: int
    document_status: str
    department_id: str | None = None
    confidentiality: int = 1
    can_write: bool = False
    can_manage_permissions: bool = False
    created_at: datetime
    finished_at: datetime | None


class DocumentTaskPage(BaseModel):
    items: list[DocumentTaskStatus]
    total: int
    overall_total: int
    overall_size: int
    counts: dict[str, int]
    offset: int
    limit: int


@dataclass(frozen=True)
class UploadIdentity:
    tenant_id: int
    member_id: int
    user_id: int
    department_id: int | None
    clearance: int


class UploadError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
