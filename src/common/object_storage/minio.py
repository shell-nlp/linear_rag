from __future__ import annotations

import boto3
from botocore.exceptions import ClientError

from src.settings import get_settings


class MinioObjectStorage:
    """基于 S3 兼容协议的 MinIO 对象存储实现。"""

    def __init__(
        self,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
    ):
        """初始化 MinIO 客户端，默认读取项目配置。"""

        settings = get_settings()
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url
            or f"http://{settings.minio_service_addresses}",
            aws_access_key_id=access_key or settings.minio_access_key,
            aws_secret_access_key=secret_key or settings.minio_secret_key,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=False,
        )

    def get_bytes(self, bucket_name: str, object_name: str) -> bytes:
        """读取对象内容。"""

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

        self._ensure_bucket(bucket_name)
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        self.client.put_object(
            Bucket=bucket_name,
            Key=object_name,
            Body=data,
            **extra_args,
        )

    def _ensure_bucket(self, bucket_name: str) -> None:
        """确保目标存储桶存在。"""

        try:
            self.client.head_bucket(Bucket=bucket_name)
        except ClientError as exc:
            error_code = int(exc.response["Error"]["Code"])
            if error_code != 404:
                raise
            self.client.create_bucket(Bucket=bucket_name)
