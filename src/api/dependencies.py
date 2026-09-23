from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from src.api.state import ApplicationState
from src.indexing.service import IndexingService
from src.indexing.workflow import FileIndexingWorkflow
from src.knowledge_bases.service import KnowledgeBaseService
from src.retrieval.service import RetrievalService


def get_application_state(request: Request) -> ApplicationState:
    """从 FastAPI app.state 中读取应用资源。"""

    return request.app.state.resources


ApplicationStateDependency = Annotated[
    ApplicationState,
    Depends(get_application_state),
]


def get_indexing_service(state: ApplicationStateDependency) -> IndexingService:
    """获取文档索引服务。"""

    if state.indexing_service is None:
        raise RuntimeError("IndexingService is not initialized")
    return state.indexing_service


def get_retrieval_service(state: ApplicationStateDependency) -> RetrievalService:
    """获取知识检索服务。"""

    if state.retrieval_service is None:
        raise RuntimeError("RetrievalService is not initialized")
    return state.retrieval_service


def get_file_indexing_workflow(
    state: ApplicationStateDependency,
) -> FileIndexingWorkflow:
    """获取文件上传与索引事务工作流。"""

    if state.file_indexing_workflow is None:
        raise RuntimeError("FileIndexingWorkflow is not initialized")
    return state.file_indexing_workflow


def get_knowledge_base_service(
    state: ApplicationStateDependency,
) -> KnowledgeBaseService:
    """获取知识库业务服务。"""

    if state.kb_service is None:
        raise RuntimeError("KnowledgeBaseService is not initialized")
    return state.kb_service


IndexingDependency = Annotated[IndexingService, Depends(get_indexing_service)]
FileIndexingWorkflowDependency = Annotated[
    FileIndexingWorkflow,
    Depends(get_file_indexing_workflow),
]
RetrievalDependency = Annotated[RetrievalService, Depends(get_retrieval_service)]
KnowledgeBaseDependency = Annotated[
    KnowledgeBaseService,
    Depends(get_knowledge_base_service),
]
