from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.api.constants import ADMIN_API_PREFIX
from src.api.dependencies import (
    FileIndexingWorkflowDependency,
    IndexingDependency,
    KnowledgeBaseDependency,
)
from src.api.schemas import Response
from src.indexing.schemas import (
    DeleteFilesPayload,
    DeleteSinglePassagePayload,
    SinglePassagePayload,
)

router = APIRouter(prefix=ADMIN_API_PREFIX, tags=["索引"])


@router.post("/index", response_model=Response)
def index_documents(
    workflow: FileIndexingWorkflowDependency,
    kb_name: Annotated[str, Form(description="知识库索引名称")],
    bucket_name: Annotated[str, Form(description="逻辑存储桶名称")],
    file_path: Annotated[str, Form(description="桶内文件路径，使用正斜杠")],
    file: Annotated[UploadFile, File(description="待持久化并索引的 PDF 文件")],
    file_id: Annotated[str | None, Form(description="文件 ID，可选")] = None,
):
    """接收 PDF 二进制，并以补偿事务并行执行持久化和索引。"""

    try:
        file_bytes = file.file.read(workflow.max_upload_bytes + 1)
        result = workflow.index_uploaded_file(
            file_bytes=file_bytes,
            kb_name=kb_name,
            bucket_name=bucket_name,
            file_path=file_path,
            file_id=file_id,
            content_type=file.content_type,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        file.file.close()
    return Response(code="0", msg="ok", data=result)


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
    """删除文件关联的搜索文档及其预计算关系元数据。"""

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
