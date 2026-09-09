from sqlalchemy import CheckConstraint, UniqueConstraint

from zq_rag_app.models.document import DocChunk, IndexTask
from zq_rag_app.services.document_service import (
    DocumentIndexService,
    is_retryable_index_exception,
)


def test_chunk_business_key_is_unique_for_idempotent_upsert() -> None:
    constraints = [
        constraint
        for constraint in DocChunk.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]

    assert any(
        [column.name for column in constraint.columns]
        == ["doc_id", "doc_version", "chunk_index"]
        for constraint in constraints
    )


def test_task_progress_has_database_check_constraint() -> None:
    checks = [
        constraint
        for constraint in IndexTask.__table__.constraints
        if isinstance(constraint, CheckConstraint)
    ]

    assert any(
        constraint.name == "ck_task_progress_percent"
        for constraint in checks
    )


def test_retry_classification_only_accepts_transient_io() -> None:
    assert is_retryable_index_exception(OSError("network interrupted"))
    assert is_retryable_index_exception(TimeoutError("timeout"))
    assert not is_retryable_index_exception(FileNotFoundError("missing"))
    assert not is_retryable_index_exception(ValueError("bad document"))


def test_minio_location_supports_object_bucket_prefix_and_s3_uri() -> None:
    default_bucket, plain_object = DocumentIndexService._resolve_minio_location(
        "manuals/setup.pdf"
    )
    prefixed_bucket, prefixed_object = (
        DocumentIndexService._resolve_minio_location(
            f"{default_bucket}/manuals/setup.pdf"
        )
    )
    s3_bucket, s3_object = DocumentIndexService._resolve_minio_location(
        "s3://another-bucket/manuals/setup.pdf"
    )

    assert plain_object == "manuals/setup.pdf"
    assert (prefixed_bucket, prefixed_object) == (
        default_bucket,
        "manuals/setup.pdf",
    )
    assert (s3_bucket, s3_object) == (
        "another-bucket",
        "manuals/setup.pdf",
    )
