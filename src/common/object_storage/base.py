from __future__ import annotations

from pathlib import PurePosixPath
import re
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

    def exists(self, bucket_name: str, object_key: str) -> bool:
        """判断对象是否存在。"""

        ...

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """删除对象，对象不存在时保持幂等。"""

        ...


def normalize_object_location(bucket_name: str, object_key: str) -> tuple[str, str]:
    """校验并规范化桶名和对象键，确保本地与远端实现使用相同语义。"""

    normalized_bucket = bucket_name.strip()
    normalized_key = object_key.strip()
    if not normalized_bucket or normalized_bucket in {".", ".."}:
        raise ValueError("bucket_name 不能为空或使用相对路径标记")
    if any(separator in normalized_bucket for separator in ("/", "\\")):
        raise ValueError("bucket_name 不能包含路径分隔符")
    if not normalized_key or "\\" in normalized_key:
        raise ValueError("object_key 不能为空且必须使用正斜杠")

    key_path = PurePosixPath(normalized_key)
    if key_path.is_absolute() or any(part in {"", ".", ".."} for part in key_path.parts):
        raise ValueError("object_key 必须是桶内的安全相对路径")
    if any(
        re.search(r'[<>:"|?*\x00-\x1f]', part)
        for part in key_path.parts
    ):
        raise ValueError("object_key 包含对象存储或本地文件系统不支持的字符")
    return normalized_bucket, key_path.as_posix()
