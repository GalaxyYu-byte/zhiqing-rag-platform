"""文档上传请求参数、响应与用例错误。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class UploadParameters(BaseModel):
    knowledge_base_id: int = Field(gt=0)
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=48, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size // 2:
            raise ValueError("重叠大小必须小于分块大小的一半")
        return self


class KnowledgeBaseOption(BaseModel):
    id: str
    code: str
    name: str
    description: str


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
    generation_id: str
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
