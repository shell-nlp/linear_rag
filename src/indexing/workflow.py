from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable

from loguru import logger

from src.common.document_processing.pdf_parser import PDFParser
from src.common.object_storage.base import ObjectStorage, normalize_object_location
from src.indexing.mappers import documents_to_passages
from src.indexing.service import IndexingService


def _resolve_uploaded_passages_worker(
    file_bytes: bytes,
    bucket_name: str,
    object_key: str,
    file_id: str,
) -> dict[str, list[Any]]:
    """在子进程中解析上传的 PDF 字节并转换为段落结构。"""

    parser = PDFParser(
        bucket_name=bucket_name,
        object_key=object_key,
        file_id=file_id,
    )
    return documents_to_passages(parser.get_chunk(file_bytes))


def resolve_uploaded_passages(
    process_pool: ProcessPoolExecutor | None,
    file_bytes: bytes,
    bucket_name: str,
    object_key: str,
    file_id: str,
) -> dict[str, list[Any]]:
    """解析上传字节，存在进程池时把 CPU 密集工作交给子进程。"""

    if process_pool is not None:
        future = process_pool.submit(
            _resolve_uploaded_passages_worker,
            file_bytes,
            bucket_name,
            object_key,
            file_id,
        )
        return future.result()
    return _resolve_uploaded_passages_worker(
        file_bytes,
        bucket_name,
        object_key,
        file_id,
    )


class FileIndexingWorkflow:
    """协调文件持久化和索引，并在单侧失败时执行补偿回滚。"""

    def __init__(
        self,
        indexing_service: IndexingService,
        object_storage: ObjectStorage,
        process_pool: ProcessPoolExecutor | None,
        storage_io_pool: ThreadPoolExecutor,
        max_upload_bytes: int,
        passage_resolver: Callable[..., dict[str, list[Any]]] = resolve_uploaded_passages,
    ):
        """注入索引服务、对象存储和并行执行资源。"""

        self.indexing_service = indexing_service
        self.object_storage = object_storage
        self.process_pool = process_pool
        self.storage_io_pool = storage_io_pool
        self.max_upload_bytes = max_upload_bytes
        self.passage_resolver = passage_resolver

    def index_uploaded_file(
        self,
        file_bytes: bytes,
        kb_name: str,
        bucket_name: str,
        file_path: str,
        file_id: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """并行持久化和解析索引，只有两侧都成功才返回成功。"""

        if not file_bytes:
            raise ValueError("上传文件不能为空")
        if len(file_bytes) > self.max_upload_bytes:
            raise ValueError(
                f"上传文件超过大小限制: {self.max_upload_bytes} bytes"
            )
        normalized_bucket, normalized_key = normalize_object_location(
            bucket_name,
            file_path,
        )
        normalized_kb_name = kb_name.strip()
        if not normalized_kb_name:
            raise ValueError("kb_name 不能为空")
        if self.object_storage.exists(normalized_bucket, normalized_key):
            raise FileExistsError(
                f"对象已存在: {normalized_bucket}/{normalized_key}"
            )

        resolved_file_id = (
            file_id.strip() if file_id and file_id.strip() else uuid.uuid4().hex
        )
        storage_future = self.storage_io_pool.submit(
            self.object_storage.put_bytes,
            normalized_bucket,
            normalized_key,
            file_bytes,
            content_type,
        )
        index_attempted = False
        storage_created = False
        node_ids: list[str] = []
        try:
            passages = self.passage_resolver(
                self.process_pool,
                file_bytes,
                normalized_bucket,
                normalized_key,
                resolved_file_id,
            )
            if not passages.get("text"):
                raise ValueError("PDF 解析后没有可索引文本")

            # 实体抽取和向量化继续与文件上传并行，ES 提交则等待对象落盘成功。
            documents, unique_entity_ids = self.indexing_service.prepare_documents(
                passages
            )
            node_ids = [document.id for document in documents]
            storage_future.result()
            storage_created = True
            index_attempted = True
            index_result = self.indexing_service.write_documents(
                kb_name=normalized_kb_name,
                documents=documents,
                unique_entity_ids=unique_entity_ids,
            )
            if int(index_result.get("failed_passages", 0)) > 0:
                raise RuntimeError("部分段落写入搜索数据库失败")
            return {
                "status": "success",
                "bucket_name": normalized_bucket,
                "file_path": normalized_key,
                "file_id": resolved_file_id,
                "chunk_count": len(passages.get("text", [])),
                **index_result,
            }
        except Exception:
            # 写入可能仍在执行，必须等待其结束后才能可靠判断是否需要删除。
            try:
                storage_future.result()
                storage_created = True
            except Exception:
                storage_created = False
            self._rollback(
                index_name=normalized_kb_name,
                node_ids=node_ids,
                bucket_name=normalized_bucket,
                object_key=normalized_key,
                index_attempted=index_attempted,
                storage_created=storage_created,
            )
            raise

    def _rollback(
        self,
        index_name: str,
        node_ids: list[str],
        bucket_name: str,
        object_key: str,
        index_attempted: bool,
        storage_created: bool,
    ) -> None:
        """按实际执行进度补偿删除索引和本次新建对象。"""

        rollback_errors = []
        if index_attempted:
            try:
                self.indexing_service.delete_nodes(index_name, node_ids)
            except Exception as exc:
                rollback_errors.append(f"索引回滚失败: {exc}")
        if storage_created:
            try:
                self.object_storage.delete_object(bucket_name, object_key)
            except Exception as exc:
                rollback_errors.append(f"文件回滚失败: {exc}")
        if rollback_errors:
            logger.error("上传索引事务补偿不完整: {}", "; ".join(rollback_errors))
