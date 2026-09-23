from __future__ import annotations

from typing import Protocol

from src.models import ParsedDocument


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


class DocumentParser(Protocol):
    """文档解析端口，业务层只接收统一的 ParsedDocument。"""

    def parse(
        self,
        bucket_name: str,
        file_path: str,
        file_id: str | None = None,
    ) -> list[ParsedDocument]:
        """解析文档并返回可入库的片段。"""

        ...
