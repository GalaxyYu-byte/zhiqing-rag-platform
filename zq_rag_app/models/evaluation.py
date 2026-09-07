"""RAG 评估数据集及评估结果实体。

本模块对应离线评估使用的两张表：数据集保存问题和期望答案，结果表保存
某个评估版本实际召回和生成后的指标。评估表中的外部 ID 与知识库、文档块
之间暂不声明 ForeignKey，与当前 schema.sql 保持一致。
"""

from datetime import datetime

from sqlalchemy import ARRAY, BigInteger, Boolean, Float, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.database import Base


class EvalDataset(Base):
    """RAG 评估数据集实体，对应 ``kb_eval_dataset`` 表。

    一条数据集记录代表一个待评估的问题。``expected_chunk_ids`` 保存期望被
    召回的文档分块 ID 数组，可用于计算命中率、MRR 等检索指标。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_eval_dataset"
    __table_args__ = ()

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 评估问题所属的知识库 ID。
    kb_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 待评估的问题文本。
    question: Mapped[str] = mapped_column(Text, nullable=False)
    # 人工或基准答案，可为空；为空时只能评估检索命中情况。
    expected_answer: Mapped[str | None] = mapped_column(Text)
    # 期望召回的文档分块 ID 数组，对应 PostgreSQL BIGINT[]。
    expected_chunk_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger))
    # 创建评估数据的用户 ID。
    created_by: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 数据集记录创建时间。
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )


class EvalResult(Base):
    """RAG 评估结果实体，对应 ``kb_eval_result`` 表。

    每条结果对应一个数据集问题在某个 ``eval_version`` 下的评估结果。指标
    是否可用取决于评估流程：检索评估至少需要 ``hit``，答案质量评估可以额外
    写入 ``faithfulness`` 和 ``answer_relevancy``。
    """

    # 数据库表名必须与 schema.sql 中的表名完全一致。
    __tablename__ = "kb_eval_result"
    __table_args__ = ()

    # BIGSERIAL 主键。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 对应 kb_eval_dataset.id。
    dataset_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 评估版本，例如 v1_chunk512_hybrid，用于比较不同参数方案。
    eval_version: Mapped[str] = mapped_column(String(50), nullable=False)
    # 期望的文档分块是否在实际召回结果中命中。
    hit: Mapped[bool] = mapped_column(Boolean, nullable=False)
    # 命中分块的排名，从评估流程约定的排名起点开始；未命中时为空。
    rank: Mapped[int | None] = mapped_column(Integer)
    # RAG 实际生成的答案文本，可为空。
    actual_answer: Mapped[str | None] = mapped_column(Text)
    # RAGAS Faithfulness 指标，评估答案是否忠实于召回上下文。
    faithfulness: Mapped[float | None] = mapped_column(Float)
    # RAGAS Answer Relevancy 指标，评估答案是否切合用户问题。
    answer_relevancy: Mapped[float | None] = mapped_column(Float)
    # 评估执行时间。
    eval_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=text("NOW()")
    )
