from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from src.common.object_storage.base import normalize_object_location


class MinioObjectStorage:
    """基于 S3 兼容协议的 MinIO 对象存储实现。"""

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
    ):
        """使用显式配置初始化 MinIO 客户端。"""

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=False,
        )

    def get_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """读取对象内容。"""

        bucket_name, object_name = normalize_object_location(bucket_name, object_name)
        response = self.client.get_object(
            Bucket=bucket_name,
            Key=object_name,
        )
        return response["Body"].read()

    def put_bytes(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        """写入对象内容，并在存储桶不存在时自动创建。"""

        bucket_name, object_name = normalize_object_location(bucket_name, object_name)
        self._ensure_bucket(bucket_name)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        try:
            self.client.put_object(
                Bucket=bucket_name,
                Key=object_name,
                Body=data,
                IfNoneMatch="*",
                **extra_args,
            )
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise FileExistsError(
                    f"对象已存在: {bucket_name}/{object_name}"
                ) from exc
            raise

    def exists(self, bucket_name: str, object_key: str) -> bool:
        """通过对象元数据判断目标是否存在。"""

        bucket_name, object_key = normalize_object_location(bucket_name, object_key)
        try:
            self.client.head_object(Bucket=bucket_name, Key=object_key)
            return True
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def delete_object(self, bucket_name: str, object_key: str) -> None:
        """删除指定对象，S3 删除语义天然幂等。"""

        bucket_name, object_key = normalize_object_location(bucket_name, object_key)
        self.client.delete_object(Bucket=bucket_name, Key=object_key)

    def _ensure_bucket(self, bucket_name: str) -> None:
        """确保目标存储桶存在。"""

        try:
            self.client.head_bucket(Bucket=bucket_name)
        except ClientError as exc:
            error_code = str(exc.response.get("Error", {}).get("Code", ""))
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            self.client.create_bucket(Bucket=bucket_name)
