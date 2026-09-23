from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from src.common.graph_store import Neo4jGraphStore
from src.common.model_providers import EmbeddingProvider, LLMProvider
from src.indexing.service import IndexingService
from src.knowledge_bases.service import KnowledgeBaseService
from src.retrieval.service import RetrievalService


@dataclass(slots=True)
class ApplicationState:
    """应用级资源容器，由 FastAPI lifespan 创建和关闭。"""

    indexing_service: IndexingService | None = None
    retrieval_service: RetrievalService | None = None
    kb_service: KnowledgeBaseService | None = None
    embedding_provider: EmbeddingProvider | None = None
    llm_provider: LLMProvider | None = None
    index_process_pool: ProcessPoolExecutor | None = None
    graph_store: Neo4jGraphStore | None = None
