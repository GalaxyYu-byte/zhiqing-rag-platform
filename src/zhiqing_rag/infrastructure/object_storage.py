"""不可变文档对象存储，客户端与连接池按用例释放。"""

from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import urllib3
from minio import Minio

from zhiqing_rag.core.config import Settings


class DocumentObjectStorage:
    def __init__(self, client: Minio, bucket: str):
        self.client = client
        self.bucket = bucket

    def put(self, key: str, path: Path, mime_type: str) -> None:
        self.client.fput_object(self.bucket, key, str(path), content_type=mime_type)

    def remove(self, bucket: str, key: str) -> None:
        """按精确对象名清理所有版本和删除标记，不删除同前缀的其他文件。"""
        versions = list(self.client.list_objects(bucket, prefix=key, include_version=True))
        for item in versions:
            if item.object_name == key:
                self.client.remove_object(bucket, key, version_id=item.version_id)


@contextmanager
def document_object_storage(settings: Settings):
    endpoint = urlsplit(settings.minio_endpoint)
    if endpoint.scheme not in {"http", "https"} or not endpoint.netloc:
        raise ValueError("MINIO_ENDPOINT 必须包含 http:// 或 https://")
    pool = urllib3.PoolManager(
        timeout=urllib3.Timeout(connect=settings.service_timeout_seconds, read=120), retries=False
    )
    try:
        client = Minio(
            endpoint.netloc,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=endpoint.scheme == "https",
            http_client=pool,
        )
        yield DocumentObjectStorage(client, settings.minio_bucket)
    finally:
        pool.clear()
