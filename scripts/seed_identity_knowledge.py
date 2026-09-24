"""Generate identity and knowledge-base fixtures for development."""

import argparse
import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql

from zhiqing_rag.core.config import get_settings

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "data" / "dev-seed"
VERSION = "identity-knowledge-v1"
DOMAIN = "zhiqing.test"

TENANTS = [
    ("DEFAULT", "默认租户", "ACTIVE"),
    ("QA_ISOLATION", "测试·隔离租户", "ACTIVE"),
    ("QA_DISABLED", "测试·停用租户", "DISABLED"),
]
DEPARTMENTS = [
    ("DEFAULT", "QA_HQ", "测试·总部", None, True),
    ("DEFAULT", "QA_TECH", "测试·研发部", "QA_HQ", True),
    ("DEFAULT", "QA_BACKEND", "测试·后端组", "QA_TECH", True),
    ("DEFAULT", "QA_PRODUCT", "测试·产品部", "QA_HQ", True),
    ("DEFAULT", "QA_HR", "测试·人事部", "QA_HQ", True),
    ("DEFAULT", "QA_FINANCE", "测试·财务部", "QA_HQ", True),
    ("DEFAULT", "QA_DISABLED_DEPT", "测试·停用部门", "QA_HQ", False),
    ("QA_ISOLATION", "QA_HQ", "测试·隔离总部", None, True),
    ("QA_ISOLATION", "QA_TECH", "测试·隔离研发部", "QA_HQ", True),
    ("QA_DISABLED", "QA_HQ", "测试·停用租户总部", None, True),
]
# slug, display_name, globally active, purpose
USERS = [
    ("admin", "测试·平台管理员", True, "默认租户管理员；知识库显式 ADMIN 授权"),
    ("kb_admin", "测试·知识库管理员", True, "普通成员，仅管理公共与技术知识库"),
    ("tech_writer", "测试·研发编辑", True, "技术库用户 WRITE + 部门 READ"),
    ("tech_reader", "测试·研发只读", True, "技术库部门 READ；同时加入隔离租户"),
    ("backend_reader", "测试·后端只读", True, "子部门独立授权，不依赖父部门继承"),
    ("product_writer", "测试·产品编辑", True, "产品库部门 WRITE"),
    ("hr_manager", "测试·人事专员", True, "人事库部门 WRITE，密级 2"),
    ("finance_manager", "测试·财务专员", True, "财务库部门 WRITE，密级 2"),
    ("guest", "测试·外协访客", True, "无部门，仅公共库用户 READ"),
    ("no_access", "测试·无授权成员", True, "无部门、无任何知识库授权"),
    ("disabled_user", "测试·停用账号", False, "全局账号禁用，但成员和授权仍存在"),
    ("disabled_member", "测试·停用成员", True, "账号正常，默认租户成员禁用"),
    ("disabled_department", "测试·停用部门成员", True, "账号和成员正常，所在部门禁用"),
    ("shared", "测试·跨租户成员", True, "默认租户技术库 WRITE；隔离租户公共库 READ"),
    ("isolation_admin", "测试·隔离租户管理员", True, "只属于隔离租户"),
    ("disabled_tenant", "测试·停用租户管理员", True, "账号与成员正常，租户禁用"),
]
# tenant, user slug, department, role, clearance, member status
MEMBERS = [
    ("DEFAULT", "admin", "QA_HQ", "ADMIN", 2, "ACTIVE"),
    ("DEFAULT", "kb_admin", "QA_TECH", "MEMBER", 2, "ACTIVE"),
    ("DEFAULT", "tech_writer", "QA_TECH", "MEMBER", 1, "ACTIVE"),
    ("DEFAULT", "tech_reader", "QA_TECH", "MEMBER", 0, "ACTIVE"),
    ("DEFAULT", "backend_reader", "QA_BACKEND", "MEMBER", 1, "ACTIVE"),
    ("DEFAULT", "product_writer", "QA_PRODUCT", "MEMBER", 1, "ACTIVE"),
    ("DEFAULT", "hr_manager", "QA_HR", "MEMBER", 2, "ACTIVE"),
    ("DEFAULT", "finance_manager", "QA_FINANCE", "MEMBER", 2, "ACTIVE"),
    ("DEFAULT", "guest", None, "MEMBER", 0, "ACTIVE"),
    ("DEFAULT", "no_access", None, "MEMBER", 0, "ACTIVE"),
    ("DEFAULT", "disabled_user", "QA_TECH", "MEMBER", 1, "ACTIVE"),
    ("DEFAULT", "disabled_member", "QA_TECH", "MEMBER", 1, "DISABLED"),
    ("DEFAULT", "disabled_department", "QA_DISABLED_DEPT", "MEMBER", 1, "ACTIVE"),
    ("DEFAULT", "shared", "QA_TECH", "MEMBER", 1, "ACTIVE"),
    ("QA_ISOLATION", "isolation_admin", "QA_HQ", "ADMIN", 2, "ACTIVE"),
    ("QA_ISOLATION", "shared", "QA_TECH", "MEMBER", 0, "ACTIVE"),
    ("QA_ISOLATION", "tech_reader", "QA_TECH", "MEMBER", 0, "ACTIVE"),
    ("QA_DISABLED", "disabled_tenant", "QA_HQ", "ADMIN", 2, "ACTIVE"),
]
KBS = [
    ("DEFAULT", "QA_PUBLIC", "测试·公共知识库", "ACTIVE"),
    ("DEFAULT", "QA_TECH", "测试·技术知识库", "ACTIVE"),
    ("DEFAULT", "QA_HR", "测试·人事知识库", "ACTIVE"),
    ("DEFAULT", "QA_FINANCE", "测试·财务知识库", "ACTIVE"),
    ("DEFAULT", "QA_PRODUCT", "测试·产品知识库", "ACTIVE"),
    ("DEFAULT", "QA_ARCHIVED", "测试·归档知识库", "ARCHIVED"),
    ("DEFAULT", "QA_DELETED", "测试·已删除知识库", "DELETED"),
    ("QA_ISOLATION", "QA_PUBLIC", "测试·隔离公共知识库", "ACTIVE"),
    ("QA_DISABLED", "QA_PUBLIC", "测试·停用租户知识库", "ACTIVE"),
]
ADMINS = {"DEFAULT": "admin", "QA_ISOLATION": "isolation_admin", "QA_DISABLED": "disabled_tenant"}
# tenant, KB, user/department, subject, permission
GRANTS = [(t, k, "user", ADMINS[t], "ADMIN") for t, k, _, _ in KBS]
GRANTS += [
    ("DEFAULT", "QA_PUBLIC", "user", "kb_admin", "ADMIN"),
    ("DEFAULT", "QA_TECH", "user", "kb_admin", "ADMIN"),
    ("DEFAULT", "QA_TECH", "user", "tech_writer", "WRITE"),
    ("DEFAULT", "QA_PUBLIC", "user", "guest", "READ"),
    ("DEFAULT", "QA_TECH", "user", "shared", "WRITE"),
    ("DEFAULT", "QA_TECH", "department", "QA_TECH", "READ"),
    ("DEFAULT", "QA_TECH", "department", "QA_BACKEND", "READ"),
    ("DEFAULT", "QA_PRODUCT", "department", "QA_PRODUCT", "WRITE"),
    ("DEFAULT", "QA_HR", "department", "QA_HR", "WRITE"),
    ("DEFAULT", "QA_FINANCE", "department", "QA_FINANCE", "WRITE"),
    ("DEFAULT", "QA_ARCHIVED", "user", "kb_admin", "READ"),
    ("DEFAULT", "QA_PUBLIC", "user", "disabled_user", "READ"),
    ("DEFAULT", "QA_PUBLIC", "user", "disabled_member", "READ"),
    ("DEFAULT", "QA_PUBLIC", "department", "QA_DISABLED_DEPT", "READ"),
    ("QA_ISOLATION", "QA_PUBLIC", "user", "shared", "READ"),
    ("QA_ISOLATION", "QA_PUBLIC", "user", "tech_reader", "READ"),
]
GRANTS += [
    ("DEFAULT", "QA_PUBLIC", "department", d, "READ")
    for d in ["QA_HQ", "QA_TECH", "QA_BACKEND", "QA_PRODUCT", "QA_HR", "QA_FINANCE"]
]


def email(slug):
    return f"{slug}@{DOMAIN}"


def hash_password(password):
    # Same self-describing format as zq-rag-py1; random salt per account.
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=16384, r=8, p=1, dklen=32)
    return "$".join(
        [
            "scrypt",
            "16384",
            "8",
            "1",
            base64.urlsafe_b64encode(salt).decode(),
            base64.urlsafe_b64encode(digest).decode(),
        ]
    )


def verify_password(password, encoded):
    try:
        scheme, n, r, p, salt, digest = encoded.split("$")
        if scheme != "scrypt":
            return False
        actual = hashlib.scrypt(
            password.encode(),
            salt=base64.urlsafe_b64decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=32,
        )
        return hmac.compare_digest(actual, base64.urlsafe_b64decode(digest))
    except (ValueError, TypeError):
        return False


def credentials():
    LOCAL.mkdir(parents=True, exist_ok=True)
    path = LOCAL / "credentials.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    for slug, *_ in USERS:
        if email(slug) not in data:
            password = "ZqDev!" + secrets.token_urlsafe(15)
            data[email(slug)] = {"password": password, "password_hash": hash_password(password)}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def literal(value):
    return sql.Literal(value).as_string()


def lookup(table, **keys):
    return (
        "(SELECT id FROM zhiqing_rag."
        + table
        + " WHERE "
        + " AND ".join(f"{key}={value}" for key, value in keys.items())
        + ")"
    )


def tenant(code):
    return lookup("tenant", code=literal(code))


def user(slug):
    return lookup("user_account", email=literal(email(slug)))


def member(t, slug):
    return lookup("tenant_member", tenant_id=tenant(t), user_id=user(slug))


def department(t, code):
    return lookup("department", tenant_id=tenant(t), code=literal(code)) if code else "NULL"


def insert(table, values):
    statement = (
        f"INSERT INTO zhiqing_rag.{table} ({', '.join(values)}) "
        f"VALUES ({', '.join(values.values())}) ON CONFLICT DO NOTHING;"
    )
    return (
        statement
        + "\nGET DIAGNOSTICS affected = ROW_COUNT;\ninserted_total := inserted_total + affected;"
    )


def render_sql(creds):
    parts = [
        "-- Development identity/KB fixtu"
        "res. Contains password hashes on"
        "ly.\n"
        "-- Add missing rows; preserve ex"
        "isting accounts, passwords and p"
        "ermissions.\nBEGIN;\n"
        "DO $$ BEGIN IF current_database("
        ") <> 'zhiqing_rag_platform' THEN"
        " "
        "RAISE EXCEPTION 'Wrong database'; END IF; END $$;\n"
        "SELECT pg_advisory_xact_lock(has"
        "htext('zhiqing:identity-knowledg"
        "e-v1'));\n"
        "DO $seed$ DECLARE affected INTEG"
        "ER; inserted_total INTEGER := 0;"
        " BEGIN"
    ]
    for code, name, status in TENANTS:
        parts.append(
            insert("tenant", dict(code=literal(code), name=literal(name), status=literal(status)))
        )
    for slug, name, active, _ in USERS:
        parts.append(
            insert(
                "user_account",
                dict(
                    email=literal(email(slug)),
                    display_name=literal(name),
                    is_active=literal(active),
                    password_hash=literal(creds[email(slug)]["password_hash"]),
                ),
            )
        )
    for t, code, name, parent, active in DEPARTMENTS:
        parts.append(
            insert(
                "department",
                dict(
                    tenant_id=tenant(t),
                    code=literal(code),
                    name=literal(name),
                    parent_id=department(t, parent),
                    is_active=literal(active),
                ),
            )
        )
    for t, slug, dep, role, clearance, status in MEMBERS:
        parts.append(
            insert(
                "tenant_member",
                dict(
                    tenant_id=tenant(t),
                    user_id=user(slug),
                    department_id=department(t, dep),
                    role=literal(role),
                    clearance=literal(clearance),
                    status=literal(status),
                ),
            )
        )
    for t, code, name, status in KBS:
        parts.append(
            insert(
                "knowledge_base",
                dict(
                    tenant_id=tenant(t),
                    code=literal(code),
                    name=literal(name),
                    description=literal("开发调试空知识库；文档后续通过离线流程导入"),
                    created_by=member(t, ADMINS[t]),
                    status=literal(status),
                    deleted_at="CURRENT_TIMESTAMP" if status == "DELETED" else "NULL",
                ),
            )
        )
    for t, kb, kind, subject, permission in GRANTS:
        parts.append(
            insert(
                "knowledge_base_grant",
                dict(
                    tenant_id=tenant(t),
                    kb_id=lookup("knowledge_base", tenant_id=tenant(t), code=literal(kb)),
                    user_id=user(subject) if kind == "user" else "NULL",
                    department_id=department(t, subject) if kind == "department" else "NULL",
                    permission=literal(permission),
                    granted_by=member(t, ADMINS[t]),
                ),
            )
        )
    parts.append("IF inserted_total > 0 THEN")
    for t, _, _ in TENANTS:
        parts.append(
            "INSERT INTO zhiqing_rag.audit_ev"
            "ent (tenant_id, action, target_t"
            "ype, target_id, details) "
            f"VALUES ({tenant(t)}, 'DEV_SEED_IMPORT', 'SEED_BATCH', {literal(VERSION)}, "
            f"jsonb_build_object('fixture_version', {literal(VERSION)}, "
            "'batch_inserted_rows', inserted_total, "
            "'source', 'development seed script')); "
        )
    parts.append("END IF; END $seed$;\nCOMMIT;\n")
    return "\n\n".join(parts)


def counts(conn):
    names = conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='zhiqing_rag' ORDER BY tablename"
    ).fetchall()
    return {
        name: conn.execute(
            sql.SQL("SELECT count(*) FROM zhiqing_rag.{}").format(sql.Identifier(name))
        ).fetchone()[0]
        for (name,) in names
    }


def apply_seed(text, creds):
    s = get_settings()
    if s.db_name != "zhiqing_rag_platform":
        raise RuntimeError("Refusing to seed another database")
    with psycopg.connect(
        host=s.db_host,
        port=s.db_port,
        dbname=s.db_name,
        user=s.db_username,
        password=s.db_password.get_secret_value(),
        connect_timeout=10,
        options="-c statement_timeout=30000 -c lock_timeout=10000",
    ) as conn:
        before = counts(conn)
        # Keep the script and verification in one transaction; commit only on successful checks.
        body = text.replace("\nBEGIN;\n", "\n", 1).removesuffix("COMMIT;\n")
        conn.execute(body, prepare=False)
        after = counts(conn)
        allowed = {
            "tenant",
            "user_account",
            "department",
            "tenant_member",
            "knowledge_base",
            "knowledge_base_grant",
            "audit_event",
        }
        for name in before:
            if name not in allowed and after[name] != before[name]:
                raise RuntimeError(f"Unexpected change outside seed scope: {name}")
        matched, different = [], []
        for slug, *_ in USERS:
            stored = conn.execute(
                "SELECT password_hash FROM zhiqing_rag.user_account WHERE email=%s", (email(slug),)
            ).fetchone()
            if stored and verify_password(creds[email(slug)]["password"], stored[0]):
                matched.append(email(slug))
            else:
                different.append(email(slug))
        report = {
            "fixture_version": VERSION,
            "applied_at_utc": datetime.now(UTC).isoformat(),
            "database": s.db_name,
            "before": before,
            "after": after,
            "inserted": {name: after[name] - before[name] for name in after},
            "credentials_matched": len(matched),
            "existing_credentials_differ": different,
        }
    return report


def write_docs(creds):
    lines = [
        "# 身份与知识库测试数据",
        "",
        "数据版本：`identity-knowledge-v1`。目标库"
        "：`zhiqing_rag_platform`，schema：`"
        "zhiqing_rag`。",
        "",
        "此批数据包含 3 个租户（包含已有 DEFAULT）、16 个用"
        "户、10 个部门、18 个租户成员、9 个空知识库、31 条授权"
        "。仅新增缺失项，不覆盖已有账号、密码或你后续修改的权限。",
        "",
        "密码清单：[本地测试账号密码](../data/dev-seed"
        "/测试账号密码.md)。每个账号使用独立随机密码；数据库仅保存 "
        "scrypt 哈希，格式与旧项目一致：`scrypt$16384"
        "$8$1$base64url(salt)$base64url(d"
        "igest)`。本地 data 目录已被 .gitignore "
        "排除。",
        "",
        "当前认证及授权 API 尚未实现，以下是用于后续接口调试的数据库"
        "场景；数据录入不等于这些业务行为已经通过接口测试。",
        "",
        "## 测试账号",
        "",
        "所有邮箱后缀为 `@zhiqing.test`，是虚构测试域名。",
        "",
        "| 邮箱前缀 | 显示名 | 全局启用 | 场景 |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{slug}` | {name} | {'是' if active else '否'} | {purpose} |"
        for slug, name, active, purpose in USERS
    ]
    lines += [
        "",
        "## 租户成员与部门",
        "",
        "部门为树形组织，一名成员在一个租户内至多属于一个部门。父部门不会"
        "因这些数据自动向子部门继承权限；子部门所需授权已单独录入。",
        "",
        "| 租户 | 邮箱前缀 | 部门 | 角色 | 密级 | 成员状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{t}` | `{u}` | `{d or '无部门'}` | `{role}` | {cl} | `{st}` |"
        for t, u, d, role, cl, st in MEMBERS
    ]
    lines += ["", "| 租户 | 部门代码 | 名称 | 父部门 | 启用 |", "| --- | --- | --- | --- | --- |"]
    lines += [
        f"| `{t}` | `{code}` | {name} | `{parent or '根部门'}` | {'是' if active else '否'} |"
        for t, code, name, parent, active in DEPARTMENTS
    ]
    lines += ["", "## 空知识库", "", "| 租户 | 代码 | 名称 | 状态 |", "| --- | --- | --- | --- |"]
    lines += [f"| `{t}` | `{code}` | {name} | `{status}` |" for t, code, name, status in KBS]
    lines += [
        "",
        "## 授权矩阵",
        "",
        "表内是显式 ACL；用户授权和部门授权如何合并、停用状态如何阻断"
        "访问，需要后续权限服务实现。租户 ADMIN 也不得绕过租户边界"
        "。",
        "",
        "| 租户 | 知识库代码 | 主体类型 | 主体（邮箱前缀/部门代码） | 权限 |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{t}` | `{kb}` | {'用户' if kind == 'user' else '部门'} | `{subject}` | `{perm}` |"
        for t, kb, kind, subject, perm in GRANTS
    ]
    lines += [
        "",
        "## 建议调试场景",
        "",
        "- `admin` 与 `kb_admin`：区分租户管理员、知识库管理员。",
        "- `tech_writer`、`tech_reader`：用户 WRITE 与部门 READ 的合并、只读账号写操作拦截。",
        "- `guest`、`no_access`：无部门但有显式授权，以及登录成功但无知识库访问权限。",
        "- `shared` 切换 DEFAULT/QA_ISOLATI"
        "ON：同一用户在两个租户中角色、密级及授权独立，不能携带另一个租"
        "户权限。",
        "- `disabled_user`、`disabled_memb"
        "er`、`disabled_tenant`：分别覆盖账号、成员、"
        "租户禁用；禁用优先于存在的授权。",
        "- `disabled_department`：覆盖已停用部门仍有 ACL 的情况，验证权限服务对停用部门的处理。",
        "- `QA_ARCHIVED`、`QA_DELETED`：验证知识库生命周期处理，不能仅凭 ACL 放行。",
        "- `backend_reader`：验证多级部门和显式子部门授"
        "权。文档密级 0/1/2 的访问测试等离线文档导入后再进行。",
        "",
        "## 重复导入",
        "",
        "推荐在项目根目录执行：",
        "",
        "```powershell",
        "uv run python scripts/seed_identity_knowledge.py --apply",
        "```",
        "",
        "不带 `--apply` 时仅生成 SQL 和本地账号清单。SQ"
        "L 文件：[04_seed_identity_knowledge"
        ".sql](../sql/04_seed_identity_kn"
        "owledge.sql)，也可在数据库客户端完整执行。脚本用自然"
        "键查找关联 ID，并使用 ON CONFLICT DO NOTH"
        "ING；重复导入不会复制账号、授权或重置密码。首次执行仍要求 0"
        "1 建表脚本已完成。",
        "",
        "若已有账号密码被手动修改，本地原密码不会强行覆盖，运行报告的 e"
        "xisting_credentials_differ 会列出不匹"
        "配账号。保留本地 credentials.json，避免重新生成"
        "一套与库中已有账号不对应的密码。",
        "",
        "只有实际新增记录时才追加 DEV_SEED_IMPORT 系统审"
        "计事件。auth_session 由将来的真实登录流程产生，本次"
        "保持为空。文档、文件修订、索引批次、分块、任务、问答和评估数据均"
        "不由本脚本生成；原有 embedding_profile 配置保"
        "留。",
        "",
    ]
    (ROOT / "docs" / "身份与知识库测试数据.md").write_text("\n".join(lines), encoding="utf-8")
    private = [
        "# 本地测试账号密码",
        "",
        "仅供本项目开发调试。data 目录已被 Git 忽略；请勿发布此文件。",
        "",
        "用户状态、租户与权限见 [测试数据说明](../../docs/"
        "身份与知识库测试数据.md)。登录接口尚待实现，当前可直接用于数"
        "据库调试及后续认证开发。",
        "",
        "| 邮箱 | 密码 |",
        "| --- | --- |",
    ]
    private += [f"| `{email(slug)}` | `{creds[email(slug)]['password']}` |" for slug, *_ in USERS]
    (LOCAL / "测试账号密码.md").write_text("\n".join(private) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Commit fixtures to the configured development database",
    )
    args = parser.parse_args()
    creds = credentials()
    text = render_sql(creds)
    (ROOT / "sql" / "04_seed_identity_knowledge.sql").write_text(text, encoding="utf-8")
    write_docs(creds)
    print(f"Generated fixtures: {len(USERS)} users, {len(MEMBERS)} members, {len(GRANTS)} grants")
    if args.apply:
        report = apply_seed(text, creds)
        (LOCAL / "last_run.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
