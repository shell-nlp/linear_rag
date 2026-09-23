"""对象存储接口与实现。"""

from src.common.object_storage.base import ObjectStorage
from src.common.object_storage.minio import MinioObjectStorage

__all__ = ["MinioObjectStorage", "ObjectStorage"]
