"""知识库授权、文档共享和文档密级管理 API，不依赖对象存储。"""

from collections.abc import Iterator
from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, Path, Response

from zhiqing_rag.core.config import Settings, get_settings
from zhiqing_rag.infrastructure.database.connection import database_connection
from zhiqing_rag.modules.identity.auth_schemas import AuthError, AuthIdentity
from zhiqing_rag.modules.identity.router import require_cookie_request, require_identity

from .permission_repository import PermissionRepository
from .permission_schemas import (
    ClassificationRequest,
    DocumentGrantRequest,
    DocumentPermissions,
    GrantRequest,
    KnowledgeBasePermissions,
)

router = APIRouter(prefix="/api", tags=["知识权限"])
IdentityDependency = Annotated[AuthIdentity, Depends(require_identity)]
ResourceId = Annotated[int, Path(gt=0, le=9223372036854775807)]


def get_permission_repository(
    identity: IdentityDependency, response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[PermissionRepository]:
    response.headers["Cache-Control"] = "no-store"
    try:
        with database_connection(settings) as connection:
            yield PermissionRepository(connection, settings.db_schema)
    except psycopg.Error as error:
        raise AuthError(503, "DATABASE_UNAVAILABLE", "数据库暂不可用，请稍后重试") from error


RepositoryDependency = Annotated[PermissionRepository, Depends(get_permission_repository)]
MutationGuard = Annotated[None, Depends(require_cookie_request)]


@router.get("/knowledge-bases/{kb_id}/permissions", response_model=KnowledgeBasePermissions)
def knowledge_base_permissions(kb_id: ResourceId, identity: IdentityDependency,
                               repository: RepositoryDependency):
    return repository.knowledge_base_permissions(identity, kb_id)


@router.post("/knowledge-bases/{kb_id}/permissions", response_model=KnowledgeBasePermissions)
def upsert_knowledge_base_grant(kb_id: ResourceId, body: GrantRequest, guard: MutationGuard,
                                 identity: IdentityDependency, repository: RepositoryDependency):
    return repository.upsert_knowledge_base_grant(identity, kb_id, body)


@router.delete("/knowledge-bases/{kb_id}/permissions/{grant_id}",
               response_model=KnowledgeBasePermissions)
def delete_knowledge_base_grant(kb_id: ResourceId, grant_id: ResourceId, guard: MutationGuard,
                                 identity: IdentityDependency, repository: RepositoryDependency):
    return repository.delete_knowledge_base_grant(identity, kb_id, grant_id)


@router.get("/documents/{document_id}/permissions", response_model=DocumentPermissions)
def document_permissions(document_id: ResourceId, identity: IdentityDependency,
                         repository: RepositoryDependency):
    return repository.document_permissions(identity, document_id)


@router.post("/documents/{document_id}/permissions", response_model=DocumentPermissions)
def upsert_document_grant(document_id: ResourceId, body: DocumentGrantRequest, guard: MutationGuard,
                          identity: IdentityDependency, repository: RepositoryDependency):
    return repository.upsert_document_grant(identity, document_id, body)


@router.delete("/documents/{document_id}/permissions/{grant_id}",
               response_model=DocumentPermissions)
def delete_document_grant(document_id: ResourceId, grant_id: ResourceId, guard: MutationGuard,
                          identity: IdentityDependency, repository: RepositoryDependency):
    return repository.delete_document_grant(identity, document_id, grant_id)


@router.post("/documents/{document_id}/classification", response_model=DocumentPermissions)
def classify_document(document_id: ResourceId, body: ClassificationRequest, guard: MutationGuard,
                       identity: IdentityDependency, repository: RepositoryDependency):
    return repository.classify_document(identity, document_id, body.confidentiality)
