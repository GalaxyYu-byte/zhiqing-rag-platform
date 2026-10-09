"""认证会话数据库记录；会话撤销和实时授权每次请求都会检查。"""

from datetime import datetime
from uuid import UUID

from psycopg import Connection, sql


class AuthRepository:
    def __init__(self, connection: Connection, schema: str):
        self.connection = connection
        self.schema = schema

    def query(self, statement: str, parameters=()):
        return self.connection.execute(
            sql.SQL(statement).format(schema=sql.Identifier(self.schema)), parameters
        )

    def user_by_email(self, email: str) -> dict | None:
        return self.query(
            "SELECT id, email, display_name, password_hash, is_active "
            "FROM {schema}.user_account WHERE email = %s", (email,),
        ).fetchone()

    def lock_user(self, user_id: int) -> dict | None:
        # 所有会话创建、刷新和全部退出都先锁用户，再锁会话，保持一致锁顺序。
        return self.query(
            "SELECT id, password_hash, is_active FROM {schema}.user_account "
            "WHERE id = %s FOR UPDATE", (user_id,),
        ).fetchone()

    def tenants(self, user_id: int) -> list[dict]:
        return self.query(
            """SELECT t.id, t.code, t.name, m.role FROM {schema}.tenant_member m
               JOIN {schema}.tenant t ON t.id = m.tenant_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE m.user_id = %s AND m.status = 'ACTIVE' AND t.status = 'ACTIVE'
                 AND (m.department_id IS NULL OR d.is_active)
               ORDER BY t.id""", (user_id,),
        ).fetchall()

    def revocation_stamp(self, user_id: int) -> str:
        row = self.query(
            """SELECT max(a.revoked_at) AS revoked_at,
                 max(a.id) FILTER (WHERE a.revoked_at IS NOT NULL) AS revoked_id,
                 count(*) FILTER (WHERE a.revoked_at IS NOT NULL) AS revoked_count
               FROM {schema}.auth_session a
               JOIN {schema}.tenant_member m
                 ON m.tenant_id = a.tenant_id AND m.id = a.tenant_member_id
               WHERE m.user_id = %s""", (user_id,),
        ).fetchone()
        # 主键保证同一秒连续退出仍改变代次；不能只依赖秒级时间戳。
        return (f"{row['revoked_at'].isoformat()}:{row['revoked_id']}:{row['revoked_count']}"
                if row and row["revoked_at"] else "")

    def identity(self, user_id: int, tenant_id: int, member_id: int | None = None) -> dict | None:
        return self.query(
            """SELECT u.id AS user_id, u.email, u.display_name,
                 m.tenant_id, m.id AS member_id, m.department_id, m.role, m.clearance,
                 m.auth_version, t.auth_version AS tenant_auth_version,
                 t.code AS tenant_code, t.name AS tenant_name
               FROM {schema}.tenant_member m
               JOIN {schema}.tenant t ON t.id = m.tenant_id
               JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE u.id = %s AND t.id = %s AND (%s::bigint IS NULL OR m.id = %s)
                 AND u.is_active AND t.status = 'ACTIVE' AND m.status = 'ACTIVE'
                 AND (m.department_id IS NULL OR d.is_active)""",
            (user_id, tenant_id, member_id, member_id),
        ).fetchone()

    def session(self, session_key: UUID, *, lock: bool = False) -> dict | None:
        return self.query(
            """SELECT a.*, m.user_id FROM {schema}.auth_session a
               JOIN {schema}.tenant_member m
                 ON m.tenant_id = a.tenant_id AND m.id = a.tenant_member_id
               WHERE a.session_key = %s""" + (" FOR UPDATE OF a" if lock else ""),
            (session_key,),
        ).fetchone()

    def session_identity(self, session_key: UUID) -> tuple[dict | None, dict | None]:
        """一次 SELECT 读取当前会话和身份；每次复核重新执行，不缓存结果。"""
        row = self.query(
            """SELECT m.user_id AS session_user_id, a.tenant_id AS session_tenant_id,
                 a.tenant_member_id, a.revoked_at, a.expires_at,
                 u.id AS user_id, u.email, u.display_name,
                 m.tenant_id, m.id AS member_id, m.department_id, m.role, m.clearance,
                 m.auth_version, t.auth_version AS tenant_auth_version,
                 t.code AS tenant_code, t.name AS tenant_name,
                 (u.is_active AND t.status = 'ACTIVE' AND m.status = 'ACTIVE'
                  AND (m.department_id IS NULL OR d.is_active)) AS authorization_active
               FROM {schema}.auth_session a
               JOIN {schema}.tenant_member m
                 ON m.tenant_id = a.tenant_id AND m.id = a.tenant_member_id
               LEFT JOIN {schema}.tenant t ON t.id = m.tenant_id
               LEFT JOIN {schema}.user_account u ON u.id = m.user_id
               LEFT JOIN {schema}.department d
                 ON d.tenant_id = m.tenant_id AND d.id = m.department_id
               WHERE a.session_key = %s""",
            (session_key,),
        ).fetchone()
        if row is None:
            return None, None
        # 不在 WHERE 中过滤禁用状态或令牌中的身份 ID，保留原有错误类别与校验顺序。
        session = {
            "user_id": row["session_user_id"],
            "tenant_id": row["session_tenant_id"],
            "tenant_member_id": row["tenant_member_id"],
            "revoked_at": row["revoked_at"],
            "expires_at": row["expires_at"],
        }
        return session, row if row["authorization_active"] is True else None

    def create_session(
        self, row: dict, session_key: UUID, token_hash: str, created_at: datetime,
        expires_at: datetime, user_agent: str | None, ip_address: str | None,
    ) -> None:
        self.query(
            """INSERT INTO {schema}.auth_session
               (tenant_id, tenant_member_id, session_key, refresh_token_hash, rotation,
                created_at, expires_at, last_used_at, user_agent, ip_address)
               VALUES (%s, %s, %s, %s, 0, %s, %s, %s, %s, %s)""",
            (row["tenant_id"], row["member_id"], session_key, token_hash, created_at,
             expires_at, created_at, user_agent, ip_address),
        )

    def rotate(self, session_key: UUID, token_hash: str, now: datetime) -> None:
        self.query(
            """UPDATE {schema}.auth_session SET refresh_token_hash = %s,
               rotation = rotation + 1, last_used_at = %s WHERE session_key = %s""",
            (token_hash, now, session_key),
        )

    def revoke_user(self, user_id: int, now: datetime) -> list[dict]:
        return self.query(
            """UPDATE {schema}.auth_session a SET revoked_at = %s
               FROM {schema}.tenant_member m
               WHERE a.tenant_id = m.tenant_id AND a.tenant_member_id = m.id
                 AND m.user_id = %s AND a.revoked_at IS NULL
               RETURNING a.session_key, a.refresh_token_hash""", (now, user_id),
        ).fetchall()
