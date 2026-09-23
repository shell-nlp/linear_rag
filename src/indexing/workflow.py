from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.common.document_processing.pdf_parser import PDFParser
from src.indexing.mappers import documents_to_passages


def _resolve_index_passages_worker(
    bucket_name: str,
    file_path: str,
    file_id: str | None,
) -> dict[str, list[Any]]:
    """在子进程中解析 PDF 并转换为段落结构。"""

    parser = PDFParser(
        bucket_name=bucket_name,
        file_path=file_path,
        file_id=file_id,
    )
    return documents_to_passages(parser.get_chunk())


def resolve_index_passages(
    process_pool: ProcessPoolExecutor | None,
    bucket_name: str,
    file_path: str,
    file_id: str | None,
) -> dict[str, list[Any]]:
    """解析文件，并在存在进程池时使用进程池执行。"""

    if process_pool is not None:
        future = process_pool.submit(
            _resolve_index_passages_worker,
            bucket_name,
            file_path,
            file_id,
        )
        return future.result()
    return _resolve_index_passages_worker(bucket_name, file_path, file_id)
