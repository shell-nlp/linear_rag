from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.constants import ADMIN_API_PREFIX
from src.api.dependencies import (
    ApplicationStateDependency,
    IndexingDependency,
    KnowledgeBaseDependency,
)
from src.api.schemas import Response
from src.indexing.schemas import (
    DeleteFilesPayload,
    DeleteSinglePassagePayload,
    IndexPayload,
    SinglePassagePayload,
)
from src.indexing.workflow import resolve_index_passages

router = APIRouter(prefix=ADMIN_API_PREFIX, tags=["索引"])


@router.post("/index", response_model=Response)
def index_documents(
    payload: IndexPayload,
    indexing_service: IndexingDependency,
    state: ApplicationStateDependency,
):
    """建立 PDF 文件索引。"""

    passages = resolve_index_passages(
        process_pool=state.index_process_pool,
        bucket_name=payload.bucket_name,
        file_path=payload.file_path,
        file_id=payload.file_id,
    )
    if not passages["text"]:
        raise HTTPException(status_code=400, detail="PDFParser did not return chunks")
    result = indexing_service.index(passages=passages, kb_name=payload.kb_name)
    file_ids = sorted(
        {str(file_id) for file_id in passages.get("file_id", []) if file_id}
    )
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"Successfully indexed into {payload.kb_name}",
            "chunk_count": len(passages.get("text", [])),
            "file_ids": file_ids,
            **result,
        },
    )


@router.post("/index_single", response_model=Response)
def index_single_passage(
    payload: SinglePassagePayload,
    kb_service: KnowledgeBaseDependency,
):
    """向搜索库上传独立文本片段。"""

    hash_ids = kb_service.upsert_passages(
        index_name=payload.index_name,
        texts=payload.texts,
        keyword=payload.keyword,
    )
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": (
                f"Successfully indexed {len(payload.texts)} passages "
                f"into {payload.index_name}"
            ),
            "count": len(payload.texts),
            "hash_ids": hash_ids,
        },
    )


@router.post("/delete_single", response_model=Response)
def delete_single_passage(
    payload: DeleteSinglePassagePayload,
    kb_service: KnowledgeBaseDependency,
):
    """按文档 ID 删除独立文本片段。"""

    result = kb_service.delete_passages_by_ids(
        index_name=payload.index_name,
        ids=payload.normalized_ids(),
    )
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": (
                f"Deleted {result.get('deleted', 0)} passages "
                f"from {payload.index_name}"
            ),
            **result,
        },
    )


@router.post("/delete_files", response_model=Response)
def delete_files(
    payload: DeleteFilesPayload,
    indexing_service: IndexingDependency,
):
    """删除文件关联的搜索文档和图节点。"""

    result = indexing_service.delete_files(
        index_name=payload.index_name,
        file_ids=payload.file_ids,
    )
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"File {payload.file_ids} deleted from {payload.index_name}",
            **result,
        },
    )
