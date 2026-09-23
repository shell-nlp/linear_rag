from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

from src.api.state import ApplicationState
from src.common.document_processing.ner import SpacyNER
from src.common.graph_store import Neo4jGraphStore
from src.common.model_providers import create_model_providers
from src.common.search_store import ElasticsearchSearchStore
from src.indexing.service import IndexingService
from src.knowledge_bases.service import KnowledgeBaseService
from src.retrieval.service import RetrievalService
from src.settings import get_settings
from src.utils import get_es_client, get_redis_client, setup_logging


def build_application_state() -> ApplicationState:
    """创建 FastAPI 运行期间需要的全部应用资源。"""

    settings = get_settings()
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, "log.txt"))

    state = ApplicationState()
    state.index_process_pool = ProcessPoolExecutor(
        max_workers=settings.index_process_workers
    )
    redis_client = get_redis_client()
    redis_client.ping()

    model_providers = create_model_providers()
    state.embedding_provider = model_providers.embedding
    state.llm_provider = model_providers.llm

    search_store = ElasticsearchSearchStore(get_es_client())
    state.graph_store = Neo4jGraphStore.from_settings(redis_client=redis_client)
    state.graph_store.start()

    config = settings.runtime_config(state.embedding_provider)
    entity_extractor = SpacyNER(settings.spacy_model)
    state.indexing_service = IndexingService(
        config=config,
        search_store=search_store,
        graph_store=state.graph_store,
        embedding_provider=state.embedding_provider,
        entity_extractor=entity_extractor,
    )
    state.kb_service = KnowledgeBaseService(
        search_store=search_store,
        embedding_provider=state.embedding_provider,
        vector_dim=settings.embedding_dim,
    )
    state.retrieval_service = RetrievalService(
        config=config,
        search_store=search_store,
        graph_store=state.graph_store,
        embedding_provider=state.embedding_provider,
        entity_extractor=entity_extractor,
    )
    return state


def close_application_state(state: ApplicationState) -> None:
    """关闭应用资源。"""

    if state.index_process_pool:
        state.index_process_pool.shutdown(wait=True, cancel_futures=True)
        state.index_process_pool = None
    if state.graph_store:
        state.graph_store.close()
        state.graph_store = None
