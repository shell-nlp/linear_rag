from __future__ import annotations

from typing import Any

from src.services.vector_utils import normalize_vector
from src.common.utils import compute_mdhash_id
from src.models import SearchDocument, SearchMode, SearchQuery
from src.interfaces.embedding import EmbeddingProvider
from src.interfaces.search import SearchStore


class KnowledgeBaseService:
    """封装知识库生命周期和单片段检索操作。"""

    def __init__(
        self,
        search_store: SearchStore,
        embedding_provider: EmbeddingProvider,
        vector_dim: int,
    ):
        """注入搜索库和向量化端口。"""

        self.search_store = search_store
        self.embedding_provider = embedding_provider
        self.vector_dim = vector_dim

    def create_knowledgebase(self, index_name: str) -> None:
        """创建知识库索引。"""

        self.search_store.create_index(
            index_name=index_name,
            vector_dim=self.vector_dim,
        )

    def delete_knowledgebase(self, index_name: str) -> None:
        """删除知识库索引。"""

        self.search_store.delete_index(index_name=index_name)

    def upsert_passages(
        self,
        index_name: str,
        texts: list[str],
        keyword: str,
    ) -> list[str]:
        """向量化并写入独立文本片段。"""

        hash_ids = [
            compute_mdhash_id(text, prefix=f"{keyword}-")
            for text in texts
        ]
        embeddings = self.embedding_provider.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        documents = [
            SearchDocument(
                id=hash_id,
                text=text,
                vector=normalize_vector(embedding),
                doc_type=keyword,
            )
            for hash_id, text, embedding in zip(hash_ids, texts, embeddings)
        ]
        self.search_store.upsert_documents(
            index_name=index_name,
            documents=documents,
            refresh=True,
        )
        return hash_ids

    def delete_passages_by_ids(
        self,
        index_name: str,
        ids: list[str],
    ) -> dict[str, Any]:
        """按文档 ID 删除独立文本片段。"""

        result = self.search_store.delete_documents_by_ids(
            index_name=index_name,
            ids=ids,
            refresh=True,
        )
        return {
            "deleted": result.success,
            "failed": result.failed,
            "index_name": index_name,
            "ids": ids,
        }

    def search(
        self,
        index_name: str,
        field_name: str,
        search_key: str,
        mode: SearchMode,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """执行字段检索，支持向量、BM25 和混合模式。"""

        self.validate_search_index(index_name, field_name, mode)
        query_vector = None
        if mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
            query_vector = normalize_vector(
                self.embedding_provider.encode(
                    [search_key],
                    show_progress_bar=False,
                )[0]
            )
        hits = self.search_store.search(
            SearchQuery(
                index_names=[index_name],
                mode=mode,
                query_text=search_key,
                query_vector=query_vector,
                top_k=top_k,
                num_candidates=max(top_k * 10, 100),
                text_fields=[field_name],
            )
        )
        return [
            {
                "index": index_name,
                "id": hit.id,
                "score": hit.score,
                "source": {
                    "text": hit.document.text,
                    "hash_id": hit.id,
                    "type": hit.document.doc_type,
                    **hit.document.metadata,
                },
            }
            for hit in hits
        ]

    def validate_search_index(
        self,
        index_name: str,
        field_name: str,
        mode: SearchMode,
    ) -> None:
        """校验索引和字段是否存在。"""

        if not self.search_store.index_exists(index_name):
            raise ValueError(f"Index '{index_name}' not found")
        mapping = self.search_store.get_index_mapping(index_name)
        properties = mapping.get("properties", {})
        if not field_exists_in_mapping(properties, field_name):
            raise ValueError(
                f"Field '{field_name}' does not exist in index '{index_name}'"
            )
        if mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
            if not field_exists_in_mapping(properties, "vector"):
                raise ValueError(
                    f"Index '{index_name}' does not contain vector field"
                )


def field_exists_in_mapping(
    properties: dict[str, Any],
    field_name: str,
) -> bool:
    """递归判断点分字段是否存在于索引映射中。"""

    current = properties
    parts = field_name.split(".")
    for index, part in enumerate(parts):
        field_info = current.get(part)
        if field_info is None:
            return False
        if index == len(parts) - 1:
            return True
        if "properties" in field_info:
            current = field_info["properties"]
            continue
        if "fields" in field_info:
            current = field_info["fields"]
            continue
        return False
    return False
