"""MinIO 客户端及文件存储辅助方法。"""

from urllib.parse import urlsplit

from minio import Minio

from .config import settings


def _parse_endpoint(endpoint: str) -> tuple[str, bool]:
    """将 http(s)://host:port 转换为 MinIO SDK 所需的 host:port。"""

    value = endpoint.strip().rstrip("/")
    # 如果没有协议前缀，则直接返回原始值，并假设为非安全连接
    if "://" not in value:
        return value, False

# 解析 URL
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"无效的 MinIO endpoint: {endpoint}")

    return parsed.netloc, parsed.scheme == "https"

# 解析 MinIO endpoint
minio_endpoint, minio_secure = _parse_endpoint(settings.minio_endpoint)

# 初始化 MinIO 客户端
minio_client = Minio(
    minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=minio_secure,
)


def ensure_bucket(bucket_name: str | None = None) -> str:
    """确保文件桶存在，并返回桶名。"""

    bucket = bucket_name or settings.minio_bucket
    if not minio_client.bucket_exists(bucket):
        # 如果桶不存在，则创建它
        minio_client.make_bucket(bucket)
    return bucket


def ping_minio() -> bool:
    """检查 MinIO 是否可访问且目标 Bucket 存在。"""

    return bool(minio_client.bucket_exists(settings.minio_bucket))
