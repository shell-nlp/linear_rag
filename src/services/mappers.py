from __future__ import annotations

from typing import Any, Sequence

from src.models import ParsedDocument


def documents_to_passages(documents: Sequence[Any]) -> dict[str, list[Any]]:
    """将统一文档模型转换为现有索引流程使用的平行数组结构。"""

    passages: dict[str, list[Any]] = {
        "text": [],
        "pages_number": [],
        "content_table": [],
        "content_image": [],
        "ori_text": [],
        "file_name": [],
        "file_id": [],
        "segment_id": [],
        "file_path": [],
        "bucket_name": [],
    }

    for document in documents:
        if isinstance(document, ParsedDocument):
            metadata = document.metadata
            text = document.text
        else:
            metadata = getattr(document, "metadata", {}) or {}
            text = getattr(document, "page_content", "") or metadata.get("text", "")
        passages["text"].append(text)
        passages["pages_number"].append(metadata.get("pages_number"))
        passages["content_table"].append(metadata.get("content_table") or [])
        passages["content_image"].append(metadata.get("content_image") or [])
        passages["ori_text"].append(metadata.get("ori_text") or text)
        passages["file_name"].append(metadata.get("file_name"))
        passages["file_id"].append(metadata.get("file_id"))
        passages["segment_id"].append(metadata.get("segment_id"))
        passages["file_path"].append(metadata.get("file_path"))
        passages["bucket_name"].append(metadata.get("bucket_name"))

    return passages
