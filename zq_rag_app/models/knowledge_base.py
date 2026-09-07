"""知识库及知识库权限实体。

本模块对应 ``schema.sql`` 中的前两张表：

* ``kb_knowledge_base``：保存知识库的基本信息和所属部门；
* ``kb_permission``：保存部门或用户对知识库的访问权限。

当前 SQL 脚本没有声明外键，所以这里保留 ID 字段而不声明 ORM relationship；
这样实体定义会与现有数据库结构保持一致，关联查询由 Service 层显式完成。
"""

from datetime import datetime

from sqlalchemy import BigInteger, Boolean, Index, String, Text, UniqueConstraint, text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class KnowledgeBase(Base):
    """知识库实体，对应 ``kb_knowledge_base`` 表。

    一个部门可以创建多个知识库。``is_deleted`` 是软删除标记，业务查询通常
    只查询值为 ``False`` 的记录；对应的部门索引和后续文档索引也都围绕这个
    多租户过滤条件设计。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_knowledge_base"
    __table_args__ = (
        # 只为未删除记录建立索引，提升按部门查询可见知识库的效率。
        Index(
            "idx_kb_department",
            "department_id",
            postgresql_where=text("is_deleted = FALSE"),
        ),
    )

    # BIGSERIAL 主键。插入时由 PostgreSQL 序列自动生成，Python 中使用 int 接收。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 知识库名称，数据库限制最大长度为 100 个字符。
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    # 知识库简介，可以为空；用于列表展示或帮助用户了解知识库用途。
    description: Mapped[str | None] = mapped_column(Text)
    # 知识库所属部门 ID。这里使用字符串以兼容组织系统中的部门编码。
    department_id: Mapped[str] = mapped_column(String(50), nullable=False)
    # 是否公开。公开知识库不需要额外的用户/部门授权即可被允许的用户检索。
    # 默认值由数据库设置为 FALSE，避免插入时因未传值而产生歧义。
    is_public: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    # 创建者用户 ID，用于审计以及后续的管理权限判断。
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 创建时间，由 PostgreSQL 在插入记录时写入当前时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
    # 最后更新时间。新增时默认为当前时间，ORM 更新时通过 onupdate 自动刷新。
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        server_default=text("NOW()"),
        onupdate=text("NOW()"),
    )
    # 软删除标记。删除知识库时通常只更新该字段，不物理删除关联数据。
    is_deleted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )


class Permission(Base):
    """知识库权限实体，对应 ``kb_permission`` 表。

    ``subject_type`` 和 ``subject_id`` 共同表示授权对象：

    * ``DEPARTMENT``：``subject_id`` 是部门 ID；
    * ``USER``：``subject_id`` 是用户 ID 的字符串形式。

    同一个知识库不能对同一个授权对象重复授予权限，数据库通过联合唯一约束
    ``(kb_id, subject_type, subject_id)`` 保证这一点。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_permission"
    __table_args__ = (
        # 防止同一知识库对同一部门/用户产生重复授权记录。
        UniqueConstraint(
            "kb_id",
            "subject_type",
            "subject_id",
            name="uq_kb_permission_subject",
        ),
        # 权限校验通常按授权对象反查其可访问的知识库。
        Index("idx_permission_subject", "subject_type", "subject_id"),
    )

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 被授权的知识库 ID。当前数据库脚本未声明物理外键。
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 授权主体类型，约定值为 DEPARTMENT 或 USER。
    subject_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # 授权主体 ID：部门编码或用户 ID 字符串。
    subject_id: Mapped[str] = mapped_column(String(50), nullable=False)
    # 权限级别，约定值为 READ、WRITE 或 ADMIN。
    permission: Mapped[str] = mapped_column(String(20), nullable=False)
    # 执行授权操作的用户 ID，用于审计。
    granted_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 授权时间，由数据库自动写入。
    granted_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
