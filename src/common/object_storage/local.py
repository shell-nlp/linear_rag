from __future__ import annotations

import os
from pathlib import Path

from src.common.object_storage.base import normalize_object_location


class LocalObjectStorage:
    """使用本地目录模拟 bucket_name/object_key 对象存储结构。"""

    def __init__(self, root_dir: str | Path):
        """保存并创建对象存储根目录。"""

        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def get_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """从本地对象路径读取完整字节。"""

        return self._resolve_path(bucket_name, object_name).read_bytes()

    def put_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """以不覆盖方式写入对象，失败时删除不完整文件。"""

        target = self._resolve_path(bucket_name, object_name)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as file_handle:
                file_handle.write(data)
                file_handle.flush()
                os.fsync(file_handle.fileno())
        except FileExistsError as exc:
            raise FileExistsError(f"对象已存在: {bucket_name}/{object_name}") from exc
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def exists(self, bucket_name: str, object_key: str) -> bool:
        """判断本地对象路径是否存在。"""

        return self._resolve_path(bucket_name, object_key).is_file()

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """删除本地对象，并清理对象根目录下的空父目录。"""

        target = self._resolve_path(bucket_name, object_key)
        target.unlink(missing_ok=True)
        parent = target.parent
        while parent != self.root_dir:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def _resolve_path(self, bucket_name: str, object_key: str) -> Path:
        """将对象地址解析为受根目录约束的绝对路径。"""

        bucket_name, object_key = normalize_object_location(bucket_name, object_key)
        target = (self.root_dir / bucket_name / Path(*object_key.split("/"))).resolve()
        if self.root_dir not in target.parents:
            raise ValueError("对象路径超出本地存储根目录")
        return target
