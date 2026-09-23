from __future__ import annotations

from fastapi import APIRouter
from loguru import logger

from src.api.constants import ADMIN_API_PREFIX
from src.api.dependencies import KnowledgeBaseDependency
from src.api.schemas import Response
from src.knowledge_bases.schemas import (
    CreateKnowledgeBaseRequest,
    DeleteKnowledgeBaseRequest,
)

router = APIRouter(prefix=ADMIN_API_PREFIX, tags=["知识库"])


@router.post("/create_knowledgebase", response_model=Response)
def create_knowledgebase(
    request: CreateKnowledgeBaseRequest,
    kb_service: KnowledgeBaseDependency,
):
    """创建知识库索引。"""

    logger.info("创建知识库入参：{}", request.model_dump_json(indent=2))
    kb_service.create_knowledgebase(request.index_name)
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"Knowledge base {request.index_name} created",
        },
    )


@router.post("/delete_knowledgebase", response_model=Response)
def delete_knowledgebase(
    request: DeleteKnowledgeBaseRequest,
    kb_service: KnowledgeBaseDependency,
):
    """删除知识库索引。"""

    logger.info("删除知识库入参：{}", request.model_dump_json(indent=2))
    kb_service.delete_knowledgebase(request.index_name)
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"Knowledge base {request.index_name} deleted",
        },
    )
