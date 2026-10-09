"""知识库授权、文档共享与密级管理接口契约。"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GrantRequest(BaseModel):
    user_id: int | None = Field(default=None, gt=0, le=9223372036854775807)
    department_id: int | None = Field(default=None, gt=0, le=9223372036854775807)
    permission: Literal["READ", "WRITE", "ADMIN"]

    @model_validator(mode="after")
    def exactly_one_subject(self):
        if (self.user_id is None) == (self.department_id is None):
            raise ValueError("用户或部门必须且只能选择一个")
        return self


class DocumentGrantRequest(GrantRequest):
    permission: Literal["READ", "WRITE"]


class ClassificationRequest(BaseModel):
    confidentiality: int = Field(ge=0, le=2)


class GrantView(BaseModel):
    id: str
    user_id: str | None
    department_id: str | None
    permission: Literal["READ", "WRITE", "ADMIN"]
    subject_name: str


class UserSubject(BaseModel):
    id: str
    email: str
    display_name: str
    department_id: str | None
    clearance: int


class DepartmentSubject(BaseModel):
    id: str
    name: str


class PermissionSubjects(BaseModel):
    users: list[UserSubject]
    departments: list[DepartmentSubject]


class KnowledgeBasePermissionResource(BaseModel):
    id: str
    name: str
    acl_version: int


class KnowledgeBasePermissions(BaseModel):
    knowledge_base: KnowledgeBasePermissionResource
    grants: list[GrantView]
    subjects: PermissionSubjects


class DocumentPermissionResource(BaseModel):
    id: str
    title: str
    department_id: str | None
    confidentiality: int


class DocumentPermissions(BaseModel):
    document: DocumentPermissionResource
    grants: list[GrantView]
    subjects: PermissionSubjects
