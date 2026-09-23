from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.constants import ADMIN_API_PREFIX
from src.api.dependencies import KnowledgeBaseDependency, RetrievalDependency
from src.api.schemas import Response
from src.retrieval.schemas import RetrievePayload, SearchStorePayload
from src.common.models import SearchMode

router = APIRouter(prefix=ADMIN_API_PREFIX, tags=["检索"])


@router.post("/retrieve", response_model=Response)
def retrieve_documents(
    payload: RetrievePayload,
    retrieval_service: RetrievalDependency,
):
    """执行知识库检索。"""

    results = retrieval_service.retrieve(
        payload.questions,
        index_names=payload.index_names,
        top_k=payload.top_k,
        search_mode=payload.search_mode,
    )
    return Response(code="0", msg="ok", data=results)


@router.post("/search_es", response_model=Response)
def search_store_documents(
    payload: SearchStorePayload,
    kb_service: KnowledgeBaseDependency,
):
    """执行搜索库字段查询，支持向量、BM25 和混合模式。"""

    search_mode = payload.search_mode or (
        SearchMode.VECTOR if payload.use_vector else SearchMode.BM25
    )
    try:
        data = kb_service.search(
            index_name=payload.index_name,
            field_name=payload.field_name,
            search_key=payload.search_key,
            mode=search_mode,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    return Response(code="0", msg="ok", data=data)
