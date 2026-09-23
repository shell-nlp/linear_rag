from __future__ import annotations

from typing import Any

from src.services.indexing_service import IndexingService
from src.services.retrieval_service import RetrievalService
from src.models import SearchMode
from src.document_processing.ner import SpacyNER
from src.interfaces.embedding import EmbeddingProvider
from src.interfaces.entity import EntityExtractor
from src.interfaces.graph import GraphStore
from src.interfaces.search import SearchStore


class LinearRAG:
    """LinearRAG 兼容门面，负责组合应用服务但不再实现具体存储逻辑。"""

    def __init__(
        self,
        global_config,
        search_store: SearchStore,
        graph_store: GraphStore,
        embedding_provider: EmbeddingProvider,
        entity_extractor: EntityExtractor | None = None,
    ):
        """初始化检索与索引门面，所有外部能力都由启动层注入。"""

        self.config = global_config
        self.embedding_model = embedding_provider
        self.search_store = search_store
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor or SpacyNER(
            global_config.spacy_model
        )
        self.indexing_service = IndexingService(
            config=global_config,
            embedding_provider=self.embedding_model,
            search_store=self.search_store,
            graph_store=self.graph_store,
            entity_extractor=self.entity_extractor,
        )
        self.retrieval_service = RetrievalService(
            config=global_config,
            embedding_provider=self.embedding_model,
            search_store=self.search_store,
            graph_store=self.graph_store,
            entity_extractor=self.entity_extractor,
        )

    def index(
        self,
        passages: dict[str, list[Any]],
        kb_name: str,
    ) -> dict[str, Any]:
        """建立段落和实体索引。"""

        return self.indexing_service.index(passages, kb_name)

    def retrieve(
        self,
        question: str,
        index_names: list[str],
        top_k: int = 5,
        search_mode: SearchMode | str = SearchMode.VECTOR,
    ) -> list[list[dict[str, Any]]]:
        """执行检索，并保持旧版返回单元素列表的兼容格式。"""

        mode = (
            search_mode
            if isinstance(search_mode, SearchMode)
            else SearchMode(search_mode)
        )
        return [
            self.retrieval_service.retrieve(
                question=question,
                index_names=index_names,
                top_k=top_k,
                search_mode=mode,
            )
        ]

    def delete_files(
        self,
        index_name: str,
        file_ids: list[str],
    ) -> dict[str, Any]:
        """删除文件关联的搜索文档和图节点。"""

        return self.indexing_service.delete_files(index_name, file_ids)
