"""知识库 ACL、文档部门范围和密级的统一授权条件。"""

import re
from typing import Literal, Protocol

from sqlalchemy import and_, exists, or_, select

Permission = Literal["READ", "WRITE", "ADMIN"]


class Identity(Protocol):
    tenant_id: int
    member_id: int
    user_id: int
    department_id: int | None
    clearance: int


def _alias(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", value) is None:
        raise ValueError("无效 SQL 别名")
    return value


def permissions_for(permission: Permission, *, document: bool = False) -> tuple[str, ...]:
    if permission not in {"READ", "WRITE", "ADMIN"}:
        raise ValueError("无效授权等级")
    if document:
        return ("READ", "WRITE") if permission == "READ" else ("WRITE",)
    return {"READ": ("READ", "WRITE", "ADMIN"),
            "WRITE": ("WRITE", "ADMIN"), "ADMIN": ("ADMIN",)}[permission]


def _quoted_permissions(permission: Permission, *, document: bool = False) -> str:
    return ", ".join(f"'{value}'" for value in permissions_for(permission, document=document))


def knowledge_base_access_sql(
    identity: Identity, *, alias: str = "kb", permission: Permission = "READ",
) -> tuple[str, tuple]:
    alias = _alias(alias)
    levels = _quoted_permissions(permission)
    return (
        f"""({alias}.tenant_id = %s AND {alias}.status = 'ACTIVE'
          AND {alias}.deleted_at IS NULL AND EXISTS (
            SELECT 1 FROM {{schema}}.knowledge_base_grant acl
            WHERE acl.tenant_id = {alias}.tenant_id AND acl.kb_id = {alias}.id
              AND acl.permission IN ({levels})
              AND (acl.user_id = %s OR acl.department_id = %s)))""",
        (identity.tenant_id, identity.user_id, identity.department_id),
    )


def document_scope_sql(
    identity: Identity, *, alias: str = "doc", permission: Permission = "READ",
) -> tuple[str, tuple]:
    alias = _alias(alias)
    levels = _quoted_permissions(permission, document=True)
    return (
        f"""({alias}.confidentiality = 0
          OR ({alias}.department_id IS NOT NULL AND {alias}.department_id = %s)
          OR ({alias}.department_id IS NULL AND {alias}.created_by = %s) OR EXISTS (
            SELECT 1 FROM {{schema}}.document_access_grant doc_acl
            WHERE doc_acl.tenant_id = {alias}.tenant_id AND doc_acl.document_id = {alias}.id
              AND doc_acl.permission IN ({levels})
              AND (doc_acl.user_id = %s OR doc_acl.department_id = %s)))""",
        (identity.department_id, identity.member_id, identity.user_id, identity.department_id),
    )


def document_access_sql(
    identity: Identity, *, document_alias: str = "doc", kb_alias: str = "kb",
    permission: Permission = "READ", include_deleted: bool = False,
    published_only: bool = False,
) -> tuple[str, tuple]:
    doc = _alias(document_alias)
    kb = _alias(kb_alias)
    acl, acl_args = knowledge_base_access_sql(identity, alias=kb, permission=permission)
    scope, scope_args = document_scope_sql(identity, alias=doc, permission=permission)
    if published_only:
        state = (f"{doc}.status = 'PUBLISHED' AND {doc}.deleted_at IS NULL "
                 f"AND {doc}.active_generation_id IS NOT NULL")
    elif include_deleted:
        state = f"{doc}.status IN ('DRAFT', 'PUBLISHED', 'DELETED')"
    else:
        state = f"{doc}.status IN ('DRAFT', 'PUBLISHED') AND {doc}.deleted_at IS NULL"
    return (
        f"({doc}.tenant_id = %s AND {kb}.tenant_id = {doc}.tenant_id "
        f"AND {kb}.id = {doc}.kb_id AND {acl} AND {doc}.confidentiality <= %s "
        f"AND {scope} AND {state})",
        (identity.tenant_id, *acl_args, identity.clearance, *scope_args),
    )


def document_access_condition(
    identity: Identity, *, document=None, kb=None, permission: Permission = "READ",
    published_only: bool = True, include_deleted: bool = False,
):
    """SQLAlchemy 查询门禁；KB 通过 EXISTS 关联，无须调用者预先 JOIN。"""
    # 延迟导入避免 ORM 注册模块的环形导入。
    from zhiqing_rag.modules.documents.document import Document
    from zhiqing_rag.modules.documents.document_access_grant import DocumentAccessGrant

    from .knowledge_base import KnowledgeBase
    from .knowledge_base_grant import KnowledgeBaseGrant

    doc = Document if document is None else document
    base = KnowledgeBase if kb is None else kb
    grant = KnowledgeBaseGrant
    sharing = DocumentAccessGrant
    kb_acl = exists(select(1).where(
        grant.tenant_id == base.tenant_id, grant.kb_id == base.id,
        grant.permission.in_(permissions_for(permission)),
        or_(grant.user_id == identity.user_id,
            grant.department_id == identity.department_id if identity.department_id is not None
            else grant.id.is_(None)),
    ).correlate(base))
    base_access = exists(select(1).select_from(base).where(
        base.tenant_id == doc.tenant_id, base.id == doc.kb_id,
        base.tenant_id == identity.tenant_id, base.status == "ACTIVE",
        base.deleted_at.is_(None), kb_acl,
    ).correlate(doc))
    explicit = exists(select(1).where(
        sharing.tenant_id == doc.tenant_id, sharing.document_id == doc.id,
        sharing.permission.in_(permissions_for(permission, document=True)),
        or_(sharing.user_id == identity.user_id,
            sharing.department_id == identity.department_id if identity.department_id is not None
            else sharing.id.is_(None)),
    ).correlate(doc))
    scope = or_(
        doc.confidentiality == 0,
        and_(doc.department_id.is_(None), doc.created_by == identity.member_id), explicit,
        and_(doc.department_id.is_not(None), doc.department_id == identity.department_id)
        if identity.department_id is not None else doc.id.is_(None),
    )
    if published_only:
        state = and_(doc.status == "PUBLISHED", doc.deleted_at.is_(None),
                     doc.active_generation_id.is_not(None))
    elif include_deleted:
        state = doc.status.in_(("DRAFT", "PUBLISHED", "DELETED"))
    else:
        state = and_(doc.status.in_(("DRAFT", "PUBLISHED")), doc.deleted_at.is_(None))
    return and_(doc.tenant_id == identity.tenant_id, base_access,
                doc.confidentiality <= identity.clearance, scope, state)
