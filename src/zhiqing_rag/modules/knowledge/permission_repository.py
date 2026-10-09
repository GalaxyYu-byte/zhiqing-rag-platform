"""权限管理事务；先锁知识库，再核验身份和文档范围，最后写授权与审计。"""

from psycopg import Connection, sql
from psycopg.types.json import Jsonb

from zhiqing_rag.modules.identity.auth_schemas import AuthError

from .permission_schemas import (
    DepartmentSubject,
    DocumentPermissionResource,
    DocumentPermissions,
    GrantRequest,
    GrantView,
    KnowledgeBasePermissionResource,
    KnowledgeBasePermissions,
    PermissionSubjects,
    UserSubject,
)
from .permissions import Identity, document_access_sql, knowledge_base_access_sql


def _id(value: int | None) -> str | None:
    return str(value) if value is not None else None


def not_found() -> None:
    raise AuthError(404, "RESOURCE_NOT_FOUND", "资源不存在或没有管理权限")


class PermissionRepository:
    def __init__(self, connection: Connection, schema: str):
        self.connection = connection
        self.schema = schema

    def query(self, statement: str, parameters=()):
        return self.connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.schema)), parameters,
        )

    def _current_identity(self, identity: Identity, *, lock: bool = False) -> None:
        parameters = [identity.tenant_id, identity.member_id, identity.user_id,
                      identity.department_id, identity.clearance]
        extra = ""
        if hasattr(identity, "auth_version"):
            extra += " AND m.auth_version = %s"
            parameters.append(identity.auth_version)
        if hasattr(identity, "tenant_auth_version"):
            extra += " AND t.auth_version = %s"
            parameters.append(identity.tenant_auth_version)
        if hasattr(identity, "session_key"):
            extra += """ AND EXISTS (SELECT 1 FROM {schema}.auth_session a
               WHERE a.tenant_id = m.tenant_id AND a.tenant_member_id = m.id
                 AND a.session_key = %s AND a.revoked_at IS NULL
                 AND a.expires_at > CURRENT_TIMESTAMP)"""
            parameters.append(identity.session_key)
        row = self.query(
            """SELECT m.id FROM {schema}.tenant_member m
               JOIN {schema}.tenant t ON t.id = m.tenant_id
               JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE m.tenant_id = %s AND m.id = %s AND m.user_id = %s
                 AND m.department_id IS NOT DISTINCT FROM %s AND m.clearance = %s
                 AND m.status = 'ACTIVE' AND t.status = 'ACTIVE' AND u.is_active
                 AND (m.department_id IS NULL OR d.is_active)""" + extra +
            (" FOR SHARE OF m, t, u" if lock else ""), tuple(parameters),
        ).fetchone()
        if row is None:
            raise AuthError(401, "AUTHORIZATION_CHANGED", "登录身份或授权已变更，请重新登录")
        if lock and identity.department_id is not None:
            department = self.query(
                "SELECT id FROM {schema}.department WHERE tenant_id = %s AND id = %s "
                "AND is_active FOR SHARE", (identity.tenant_id, identity.department_id),
            ).fetchone()
            if department is None:
                raise AuthError(401, "AUTHORIZATION_CHANGED", "所属部门已停用，请重新登录")

    def _kb(self, identity: Identity, kb_id: int, *, lock: bool = False) -> dict:
        if lock:
            row = self.query(
                "SELECT id FROM {schema}.knowledge_base WHERE tenant_id = %s AND id = %s "
                "FOR UPDATE", (identity.tenant_id, kb_id),
            ).fetchone()
            if row is None:
                not_found()
        self._current_identity(identity, lock=lock)
        predicate, args = knowledge_base_access_sql(identity, permission="ADMIN")
        row = self.query(
            "SELECT kb.id, kb.name, kb.acl_version FROM {schema}.knowledge_base kb "
            f"WHERE kb.id = %s AND {predicate}", (kb_id, *args),
        ).fetchone()
        if row is None:
            not_found()
        if lock:
            self.query(
                "SELECT id FROM {schema}.knowledge_base_grant "
                "WHERE tenant_id = %s AND kb_id = %s FOR UPDATE",
                (identity.tenant_id, kb_id),
            ).fetchall()
        return row

    def _document(self, identity: Identity, document_id: int, *, lock: bool = False) -> dict:
        if lock:
            locator = self.query(
                "SELECT kb_id FROM {schema}.document WHERE tenant_id = %s AND id = %s",
                (identity.tenant_id, document_id),
            ).fetchone()
            if locator is None:
                not_found()
            self._kb(identity, locator["kb_id"], lock=True)
        else:
            self._current_identity(identity)
        predicate, args = document_access_sql(identity, permission="ADMIN")
        row = self.query(
            """SELECT doc.id, doc.kb_id, doc.title, doc.department_id, doc.confidentiality
               FROM {schema}.document doc JOIN {schema}.knowledge_base kb
                 ON kb.tenant_id = doc.tenant_id AND kb.id = doc.kb_id
               WHERE doc.id = %s AND """ + predicate + (" FOR UPDATE OF doc" if lock else ""),
            (document_id, *args),
        ).fetchone()
        if row is None:
            not_found()
        if lock:
            self.query(
                "SELECT id FROM {schema}.document_access_grant "
                "WHERE tenant_id = %s AND document_id = %s FOR UPDATE",
                (identity.tenant_id, document_id),
            ).fetchall()
        return row

    def _subjects(self, identity: Identity) -> PermissionSubjects:
        users = self.query(
            """SELECT u.id, u.email, u.display_name, m.department_id, m.clearance
               FROM {schema}.tenant_member m JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE m.tenant_id = %s AND m.status = 'ACTIVE' AND u.is_active
                 AND (m.department_id IS NULL OR d.is_active) ORDER BY u.display_name, u.id""",
            (identity.tenant_id,),
        ).fetchall()
        departments = self.query(
            "SELECT id, name FROM {schema}.department WHERE tenant_id = %s AND is_active "
            "ORDER BY name, id", (identity.tenant_id,),
        ).fetchall()
        return PermissionSubjects(
            users=[UserSubject(id=str(row["id"]), email=row["email"],
                               display_name=row["display_name"],
                               department_id=_id(row["department_id"]),
                               clearance=row["clearance"]) for row in users],
            departments=[DepartmentSubject(id=str(row["id"]), name=row["name"])
                         for row in departments],
        )

    def _grants(self, identity: Identity, resource_id: int, *, document: bool) -> list[GrantView]:
        table = "document_access_grant" if document else "knowledge_base_grant"
        target = "document_id" if document else "kb_id"
        rows = self.query(
            f"""SELECT g.id, g.user_id, g.department_id, g.permission,
                 coalesce(u.display_name, d.name, '不可用对象') AS subject_name
               FROM {{schema}}.{table} g
               LEFT JOIN {{schema}}.user_account u ON u.id = g.user_id
               LEFT JOIN {{schema}}.department d
                 ON d.tenant_id = g.tenant_id AND d.id = g.department_id
               WHERE g.tenant_id = %s AND g.{target} = %s ORDER BY g.id""",
            (identity.tenant_id, resource_id),
        ).fetchall()
        return [GrantView(id=str(row["id"]), user_id=_id(row["user_id"]),
                          department_id=_id(row["department_id"]),
                          permission=row["permission"], subject_name=row["subject_name"])
                for row in rows]

    def _kb_view(self, identity: Identity, row: dict) -> KnowledgeBasePermissions:
        return KnowledgeBasePermissions(
            knowledge_base=KnowledgeBasePermissionResource(
                id=str(row["id"]), name=row["name"], acl_version=row["acl_version"],
            ), grants=self._grants(identity, row["id"], document=False),
            subjects=self._subjects(identity),
        )

    def _document_view(self, identity: Identity, row: dict) -> DocumentPermissions:
        return DocumentPermissions(
            document=DocumentPermissionResource(id=str(row["id"]), title=row["title"],
                                                 department_id=_id(row["department_id"]),
                                                 confidentiality=row["confidentiality"]),
            grants=self._grants(identity, row["id"], document=True),
            subjects=self._subjects(identity),
        )

    def knowledge_base_permissions(
        self, identity: Identity, kb_id: int,
    ) -> KnowledgeBasePermissions:
        return self._kb_view(identity, self._kb(identity, kb_id))

    def document_permissions(self, identity: Identity, document_id: int) -> DocumentPermissions:
        return self._document_view(identity, self._document(identity, document_id))

    def _require_subject(self, identity: Identity, grant: GrantRequest) -> None:
        if grant.user_id is not None:
            subject = self.query(
                """SELECT m.department_id FROM {schema}.tenant_member m
                   JOIN {schema}.user_account u ON u.id = m.user_id
                   LEFT JOIN {schema}.department d
                     ON d.tenant_id = m.tenant_id AND d.id = m.department_id
                   WHERE m.tenant_id = %s AND m.user_id = %s AND m.status = 'ACTIVE'
                     AND u.is_active AND (m.department_id IS NULL OR d.is_active)
                   FOR SHARE OF m, u""", (identity.tenant_id, grant.user_id),
            ).fetchone()
            department_id = subject["department_id"] if subject else None
        else:
            subject = True
            department_id = grant.department_id
        if subject and department_id is not None:
            subject = self.query(
                "SELECT id FROM {schema}.department WHERE tenant_id = %s AND id = %s "
                "AND is_active FOR SHARE", (identity.tenant_id, department_id),
            ).fetchone()
        if not subject:
            raise AuthError(422, "PERMISSION_SUBJECT_INVALID", "授权对象不存在或已停用")

    def _available_admin(self, identity: Identity, kb_id: int) -> bool:
        return self.query(
            """SELECT EXISTS (
               SELECT 1 FROM {schema}.knowledge_base_grant g
               JOIN {schema}.tenant_member m ON m.tenant_id = g.tenant_id
                 AND (m.user_id = g.user_id OR m.department_id = g.department_id)
               JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE g.tenant_id = %s AND g.kb_id = %s AND g.permission = 'ADMIN'
                 AND m.status = 'ACTIVE' AND u.is_active
                 AND (m.department_id IS NULL OR d.is_active)) AS available""",
            (identity.tenant_id, kb_id),
        ).fetchone()["available"]

    def _ensure_admin(self, identity: Identity, kb_id: int) -> None:
        if not self._available_admin(identity, kb_id):
            raise AuthError(409, "LAST_ADMIN_REQUIRED", "必须保留至少一个可用的知识库管理员")

    def _upsert(self, identity: Identity, resource_id: int, grant: GrantRequest,
                *, document: bool) -> tuple[dict, str | None]:
        self._require_subject(identity, grant)
        table = "document_access_grant" if document else "knowledge_base_grant"
        target = "document_id" if document else "kb_id"
        subject = "user_id" if grant.user_id is not None else "department_id"
        subject_id = grant.user_id if grant.user_id is not None else grant.department_id
        previous = self.query(
            f"SELECT id, permission FROM {{schema}}.{table} "
            f"WHERE tenant_id = %s AND {target} = %s AND {subject} = %s FOR UPDATE",
            (identity.tenant_id, resource_id, subject_id),
        ).fetchone()
        if previous:
            if previous["permission"] != grant.permission:
                self.query(
                    f"UPDATE {{schema}}.{table} SET permission = %s, granted_by = %s, "
                    "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = %s AND id = %s",
                    (grant.permission, identity.member_id, identity.tenant_id, previous["id"]),
                )
            return {"id": previous["id"], "user_id": grant.user_id,
                    "department_id": grant.department_id, "permission": grant.permission}, \
                previous["permission"]
        row = self.query(
            f"""INSERT INTO {{schema}}.{table}
               (tenant_id, {target}, user_id, department_id, permission, granted_by)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (identity.tenant_id, resource_id, grant.user_id, grant.department_id,
             grant.permission, identity.member_id),
        ).fetchone()
        return {"id": row["id"], "user_id": grant.user_id,
                "department_id": grant.department_id, "permission": grant.permission}, None

    def _remove(self, identity: Identity, resource_id: int, grant_id: int,
                *, document: bool) -> dict:
        table = "document_access_grant" if document else "knowledge_base_grant"
        target = "document_id" if document else "kb_id"
        row = self.query(
            f"DELETE FROM {{schema}}.{table} WHERE tenant_id = %s AND {target} = %s "
            "AND id = %s RETURNING id, user_id, department_id, permission",
            (identity.tenant_id, resource_id, grant_id),
        ).fetchone()
        if row is None:
            not_found()
        return row

    def _record_change(self, identity: Identity, kb_id: int, action: str,
                       resource_type: str, resource_id: int, details: dict) -> int:
        version = self.query(
            "UPDATE {schema}.knowledge_base SET acl_version = acl_version + 1, "
            "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = %s AND id = %s "
            "RETURNING acl_version", (identity.tenant_id, kb_id),
        ).fetchone()["acl_version"]
        self.query(
            """INSERT INTO {schema}.audit_event
               (tenant_id, actor_member_id, action, target_type, target_id, details)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (identity.tenant_id, identity.member_id, action, resource_type, str(resource_id),
             Jsonb({**details, "acl_version": version})),
        )
        return version

    @staticmethod
    def _grant_details(row: dict, previous: str | None = None) -> dict:
        return {"grant_id": str(row["id"]), "user_id": _id(row["user_id"]),
                "department_id": _id(row["department_id"]), "permission": row["permission"],
                "previous_permission": previous}

    def upsert_knowledge_base_grant(self, identity: Identity, kb_id: int,
                                    grant: GrantRequest) -> KnowledgeBasePermissions:
        with self.connection.transaction():
            kb = self._kb(identity, kb_id, lock=True)
            row, previous = self._upsert(identity, kb_id, grant, document=False)
            self._ensure_admin(identity, kb_id)
            if previous != grant.permission:
                kb["acl_version"] = self._record_change(
                    identity, kb_id, "KNOWLEDGE_BASE_PERMISSION_UPSERT", "KNOWLEDGE_BASE",
                    kb_id, self._grant_details(row, previous),
                )
            result = self._kb_view(identity, kb)
        self.connection.commit()
        return result

    def delete_knowledge_base_grant(self, identity: Identity, kb_id: int,
                                    grant_id: int) -> KnowledgeBasePermissions:
        with self.connection.transaction():
            kb = self._kb(identity, kb_id, lock=True)
            row = self._remove(identity, kb_id, grant_id, document=False)
            self._ensure_admin(identity, kb_id)
            kb["acl_version"] = self._record_change(
                identity, kb_id, "KNOWLEDGE_BASE_PERMISSION_REVOKE", "KNOWLEDGE_BASE", kb_id,
                self._grant_details(row),
            )
            result = self._kb_view(identity, kb)
        self.connection.commit()
        return result

    def upsert_document_grant(self, identity: Identity, document_id: int,
                              grant: GrantRequest) -> DocumentPermissions:
        with self.connection.transaction():
            doc = self._document(identity, document_id, lock=True)
            row, previous = self._upsert(identity, document_id, grant, document=True)
            if previous != grant.permission:
                self._record_change(identity, doc["kb_id"], "DOCUMENT_PERMISSION_UPSERT",
                                    "DOCUMENT", document_id, self._grant_details(row, previous))
            result = self._document_view(identity, doc)
        self.connection.commit()
        return result

    def delete_document_grant(self, identity: Identity, document_id: int,
                              grant_id: int) -> DocumentPermissions:
        with self.connection.transaction():
            doc = self._document(identity, document_id, lock=True)
            row = self._remove(identity, document_id, grant_id, document=True)
            self._record_change(identity, doc["kb_id"], "DOCUMENT_PERMISSION_REVOKE", "DOCUMENT",
                                document_id, self._grant_details(row))
            result = self._document_view(identity, doc)
        self.connection.commit()
        return result

    def classify_document(self, identity: Identity, document_id: int,
                           confidentiality: int) -> DocumentPermissions:
        with self.connection.transaction():
            doc = self._document(identity, document_id, lock=True)
            if confidentiality > identity.clearance:
                raise AuthError(403, "CLEARANCE_EXCEEDED", "不能设置高于自身密级的文档权限")
            previous = doc["confidentiality"]
            if previous != confidentiality:
                self.query(
                    "UPDATE {schema}.document SET confidentiality = %s, "
                    "updated_at = CURRENT_TIMESTAMP WHERE tenant_id = %s AND id = %s",
                    (confidentiality, identity.tenant_id, document_id),
                )
                doc["confidentiality"] = confidentiality
                self._record_change(
                    identity, doc["kb_id"], "DOCUMENT_CLASSIFICATION_CHANGE", "DOCUMENT",
                    document_id, {"previous_confidentiality": previous,
                                  "confidentiality": confidentiality},
                )
            result = self._document_view(identity, doc)
        self.connection.commit()
        return result
