"""对象存储接口与实现。"""

from src.common.object_storage.base import ObjectStorage
from src.common.object_storage.local import LocalObjectStorage
from src.common.object_storage.minio import MinioObjectStorage


def create_object_storage(settings) -> ObjectStorage:
    """根据配置创建 MinIO 或本地对象存储实现。"""

    if settings.object_storage_provider == "minio":
        if not all(
            (
                settings.minio_endpoint_url,
                settings.minio_access_key,
                settings.minio_secret_key,
            )
        ):
            raise ValueError("启用 MinIO 时必须配置 endpoint、access key 和 secret key")
        return MinioObjectStorage(
            endpoint_url=settings.minio_endpoint_url,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
        )
    return LocalObjectStorage(settings.local_storage_root)


__all__ = [
    "LocalObjectStorage",
    "MinioObjectStorage",
    "ObjectStorage",
    "create_object_storage",
]
