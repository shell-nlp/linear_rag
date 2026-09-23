from __future__ import annotations

from typing import Protocol


class ObjectStorage(Protocol):
    """对象存储端口，用于隔离 MinIO、S3 或本地文件系统。"""

    def get_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """读取对象内容。"""

        ...

    def put_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """写入对象内容。"""

        ...
