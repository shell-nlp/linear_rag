"""对象存储适配器统一入口。"""

from src.adapters.storage.minio_storage import MinioObjectStorage

__all__ = ["MinioObjectStorage"]
