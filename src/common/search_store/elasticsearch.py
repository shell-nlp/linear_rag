from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Sequence

from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import NotFoundError

from src.common.models import (
    SearchDocument,
    SearchHit,
    SearchMode,
    SearchQuery,
    WriteResult,
)


DEFAULT_INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "default": {
                "type": "ik_smart",
            }
        }
    },
    "number_of_replicas": "1",
    "number_of_shards": "1",
    "routing": {
        "allocation": {
            "include": {
                "_tier_preference": "data_content",
            }
        }
    },
}

# 领域层使用稳定字段名，适配器负责转换为 Elasticsearch 的精确检索字段。
FILTER_FIELD_ALIASES = {
    "file_id": "file_id.keyword",
    "file_name": "file_name.keyword",
    "file_path": "file_path.keyword",
    "bucket_name": "bucket_name.keyword",
}

# 关系扩展字段既写在文档顶层，也保留在 metadata 中供统一模型还原。
RELATION_FIELD_MAPPINGS = {
    "entity_id": {"type": "keyword"},
    "passage_id": {"type": "keyword"},
    "entity_ids": {"type": "keyword"},
    "entity_names": {"type": "keyword"},
    "entities": {
        "properties": {
            "id": {"type": "keyword"},
            "name": {"type": "keyword"},
            "count": {"type": "integer"},
            "importance": {"type": "float"},
        }
    },
    "previous_passage_id": {"type": "keyword"},
    "next_passage_id": {"type": "keyword"},
}


def _keyword_mapping() -> dict[str, Any]:
    """构造同时支持全文检索和精确过滤的字段映射。"""

    return {
        "type": "text",
        "fields": {
            "keyword": {
                "type": "keyword",
                "ignore_above": 256,
            }
        },
    }


def build_index_mapping(vector_dim: int) -> dict[str, Any]:
    """构造 Elasticsearch 索引映射。"""

    metadata_properties = {
        "content_image": _keyword_mapping(),
        "content_pages_number": {"type": "long"},
        "content_table": {"type": "object", "enabled": False},
        "file_id": _keyword_mapping(),
        "file_name": _keyword_mapping(),
        "file_path": _keyword_mapping(),
        "bucket_name": _keyword_mapping(),
        **RELATION_FIELD_MAPPINGS,
        "pages_number": {"type": "long"},
        "parent_text": _keyword_mapping(),
        "segment_id": {"type": "long"},
        "shared_tenant_id_list": _keyword_mapping(),
        "state": {"type": "boolean"},
        "tenant_id": _keyword_mapping(),
        "text": _keyword_mapping(),
        "type": {"type": "keyword"},
    }
    properties = {
        **metadata_properties,
        "hash_id": {"type": "keyword"},
        "metadata": {
            "properties": dict(metadata_properties),
        },
        "vector": {
            "type": "dense_vector",
            "dims": vector_dim,
            "index": True,
            "similarity": "cosine",
        },
    }
    return {
        "settings": DEFAULT_INDEX_SETTINGS,
        "mappings": {
            "properties": properties,
        },
    }


class ElasticsearchSearchStore:
    """Elasticsearch 对 SearchStore 端口的实现。"""

    def __init__(self, client: Elasticsearch):
        self.client = client

    def create_index(
        self,
        index_name: str,
        vector_dim: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """创建索引，已存在时直接复用。"""

        if self.client.indices.exists(index=index_name):
            # ES 允许为已有索引追加新字段，确保升级后的关系字段类型稳定。
            self.client.indices.put_mapping(
                index=index_name,
                properties=RELATION_FIELD_MAPPINGS,
            )
            return
        body = build_index_mapping(vector_dim)
        if settings:
            body["settings"].update(settings)
        self.client.indices.create(index=index_name, body=body)

    def delete_index(self, index_name: str, ignore_unavailable: bool = True) -> None:
        """删除索引。"""

        self.client.indices.delete(
            index=index_name,
            ignore_unavailable=ignore_unavailable,
        )

    def index_exists(self, index_name: str) -> bool:
        """判断索引是否存在。"""

        return bool(self.client.indices.exists(index=index_name))

    def get_index_mapping(self, index_name: str) -> dict[str, Any]:
        """读取 Elasticsearch 字段映射。"""

        response = self.client.indices.get_mapping(index=index_name)
        return response.get(index_name, {}).get("mappings", {})

    def upsert_documents(
        self,
        index_name: str,
        documents: Sequence[SearchDocument],
        refresh: bool = True,
    ) -> WriteResult:
        """批量写入文档，使用 index 操作实现覆盖式 upsert。"""

        if not documents:
            return WriteResult()

        vector_dim = next(
            (
                len(document.vector)
                for document in documents
                if document.vector is not None
            ),
            1024,
        )
        self.create_index(index_name=index_name, vector_dim=vector_dim)

        actions = [
            {
                "_op_type": "index",
                "_index": index_name,
                "_id": document.id,
                "_source": document.as_source(),
            }
            for document in documents
        ]
        success, failed = helpers.bulk(
            self.client,
            actions,
            stats_only=True,
            refresh=refresh,
            raise_on_error=True,
        )
        return WriteResult(success=success, failed=failed)

    def delete_documents_by_ids(
        self,
        index_name: str,
        ids: Sequence[str],
        refresh: bool = True,
    ) -> WriteResult:
        """按文档 ID 批量删除。"""

        unique_ids = list(dict.fromkeys(ids))
        if not unique_ids:
            return WriteResult()
        actions = [
            {
                "_op_type": "delete",
                "_index": index_name,
                "_id": doc_id,
            }
            for doc_id in unique_ids
        ]
        success, failed = helpers.bulk(
            self.client,
            actions,
            stats_only=True,
            refresh=refresh,
            raise_on_error=False,
        )
        return WriteResult(success=success, failed=failed)

    def delete_documents_by_filters(
        self,
        index_name: str,
        filters: dict[str, Any],
        refresh: bool = True,
    ) -> WriteResult:
        """按字段条件批量删除。"""

        if not self.index_exists(index_name):
            return WriteResult()
        try:
            response = self.client.delete_by_query(
                index=index_name,
                body={"query": self._build_filter_query(filters)},
                refresh=refresh,
            )
        except NotFoundError:
            return WriteResult()
        return WriteResult(
            success=int(response.get("deleted", 0)),
            failed=len(response.get("failures", [])),
        )

    def get_documents_by_ids(
        self,
        index_names: Sequence[str],
        ids: Sequence[str],
    ) -> dict[str, SearchDocument]:
        """按 ID 回查文档。"""

        if not ids:
            return {}
        existing_indices = self._existing_indices(index_names)
        if not existing_indices:
            return {}
        response = self.client.search(
            index=existing_indices,
            body={
                "size": len(ids),
                "query": {"ids": {"values": list(ids)}},
            },
        )
        documents: dict[str, SearchDocument] = {}
        for hit in response.get("hits", {}).get("hits", []):
            document = self._hit_to_document(hit)
            documents[document.id] = document
        return documents

    def search(self, query: SearchQuery) -> list[SearchHit]:
        """根据检索模式路由到向量、BM25 或混合检索。"""

        if query.filter_only:
            return self._filter_search(query)
        if query.mode in {SearchMode.LINEAR, SearchMode.LINEAR_LOCAL}:
            raise ValueError("linear 图算法由检索服务执行，搜索库只提供节点和候选查询")
        if query.mode == SearchMode.VECTOR:
            return self._vector_search(query)
        if query.mode == SearchMode.BM25:
            return self._bm25_search(query)
        return self._hybrid_search(query)

    def search_entity_passages(
        self,
        index_names: Sequence[str],
        entity_ids: Sequence[str],
        per_entity_limit: int,
    ) -> list[SearchHit]:
        """使用 msearch 分别约束每个实体的段落数，避免全量 scroll。"""

        indices = self._existing_indices(index_names)
        if not indices or not entity_ids:
            return []
        searches = []
        for entity_id in dict.fromkeys(entity_ids):
            searches.extend((
                {"index": indices},
                {
                    "size": per_entity_limit,
                    "query": {
                        "bool": {
                            "filter": [
                                {"term": {"type": "passage"}},
                                {"term": {"entity_ids": entity_id}},
                            ]
                        }
                    },
                    "sort": [{"hash_id": "asc"}],
                },
            ))
        response = self.client.msearch(searches=searches)
        hits = []
        seen = set()
        for item in response.get("responses", []):
            if "error" in item:
                raise RuntimeError(f"实体段落查询失败: {item['error']}")
            for hit in self._response_to_hits(item):
                if hit.id not in seen:
                    seen.add(hit.id)
                    hits.append(hit)
        return hits

    def search_graph_nodes(
        self,
        index_names: Sequence[str],
        doc_type: str,
        filters: dict[str, Any],
        limit: int,
    ) -> list[SearchDocument]:
        """使用有界查询代替 scroll，避免高频实体导致全量节点扫描。"""

        indices = self._existing_indices(index_names)
        if not indices or limit <= 0:
            return []
        query = self._build_filter_query({"type": doc_type, **filters})
        response = self.client.search(
            index=indices,
            body={
                "size": limit,
                "query": {"constant_score": {"filter": query}},
                "sort": [{"hash_id": "asc"}],
            },
        )
        return [
            hit.document for hit in self._response_to_hits(response)
        ]

    def scan_documents(
        self,
        index_names: Sequence[str],
        doc_types: Sequence[str],
        filters: dict[str, Any] | None = None,
        max_documents: int | None = None,
    ) -> list[SearchDocument]:
        """使用 ES scroll 扫描图节点，超限显式报错而非静默截断。"""

        indices = self._existing_indices(index_names)
        if not indices:
            return []
        clauses = [{"terms": {"type": list(doc_types)}}]
        extra = self._build_filter_query(filters)
        if extra:
            clauses.append(extra)
        documents = []
        for hit in helpers.scan(
            self.client,
            index=indices,
            query={"query": {"bool": {"filter": clauses}}},
            scroll="2m",
        ):
            documents.append(self._hit_to_document(hit))
            if max_documents is not None and len(documents) > max_documents:
                raise ValueError("图节点超过 LINEAR_MAX_NODES")
        return documents

    def _filter_search(self, query: SearchQuery) -> list[SearchHit]:
        """只按结构化字段召回，用于实体和相邻关系扩展。"""

        existing_indices = self._existing_indices(query.index_names)
        if not existing_indices:
            return []
        filter_query = self._build_filter_query(query.filters)
        if not filter_query:
            return []
        response = self.client.search(
            index=existing_indices,
            body={
                "size": query.top_k,
                "query": {"constant_score": {"filter": filter_query}},
                "_source": query.source_fields or True,
            },
        )
        return self._response_to_hits(response)

    def _vector_search(self, query: SearchQuery) -> list[SearchHit]:
        """执行向量 KNN 检索。"""

        if not query.query_vector:
            return []
        existing_indices = self._existing_indices(query.index_names)
        if not existing_indices:
            return []
        knn: dict[str, Any] = {
            "field": "vector",
            "query_vector": query.query_vector,
            "k": query.top_k,
            "num_candidates": max(query.num_candidates, query.top_k),
        }
        filter_query = self._build_filter_query(query.filters)
        if filter_query:
            knn["filter"] = filter_query
        response = self.client.search(
            index=existing_indices,
            knn=knn,
            source=query.source_fields or True,
            size=query.top_k,
        )
        return self._response_to_hits(response)

    def _bm25_search(self, query: SearchQuery) -> list[SearchHit]:
        """执行 BM25 全文检索。"""

        if not query.query_text:
            return []
        existing_indices = self._existing_indices(query.index_names)
        if not existing_indices:
            return []
        filter_query = self._build_filter_query(query.filters)
        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query.query_text,
                        "fields": query.text_fields,
                    }
                }
            ]
        }
        if filter_query:
            bool_query["filter"] = [filter_query]
        response = self.client.search(
            index=existing_indices,
            body={
                "size": query.top_k,
                "query": {"bool": bool_query},
                "_source": query.source_fields or True,
            },
        )
        return self._response_to_hits(response)

    def _hybrid_search(self, query: SearchQuery) -> list[SearchHit]:
        """分别执行向量和 BM25，再使用 RRF 融合排名。"""

        candidate_k = max(query.top_k * 4, 20)
        vector_query = replace(query, mode=SearchMode.VECTOR, top_k=candidate_k)
        bm25_query = replace(query, mode=SearchMode.BM25, top_k=candidate_k)
        vector_hits = self._vector_search(vector_query)
        bm25_hits = self._bm25_search(bm25_query)

        fused: dict[str, SearchHit] = {}
        scores: dict[str, float] = {}
        for hits in (vector_hits, bm25_hits):
            for rank, hit in enumerate(hits, start=1):
                fused[hit.id] = hit
                scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (60 + rank)

        result = [
            SearchHit(id=hit_id, score=scores[hit_id], document=hit.document)
            for hit_id, hit in fused.items()
        ]
        result.sort(key=lambda item: item.score, reverse=True)
        return result[: query.top_k]

    def _response_to_hits(self, response: dict[str, Any]) -> list[SearchHit]:
        """将 Elasticsearch 响应转换为统一命中模型。"""

        return [
            SearchHit(
                id=document.id,
                score=float(hit.get("_score") or 0.0),
                document=document,
            )
            for hit in response.get("hits", {}).get("hits", [])
            for document in [self._hit_to_document(hit)]
        ]

    def _hit_to_document(self, hit: dict[str, Any]) -> SearchDocument:
        """将单条 Elasticsearch 命中转换为领域文档。"""

        source = dict(hit.get("_source") or {})
        metadata = dict(source.get("metadata") or {})
        doc_type = source.get("type") or metadata.pop("type", None)
        for key, value in source.items():
            if key not in {"hash_id", "text", "type", "vector", "metadata"}:
                metadata[key] = value
        return SearchDocument(
            id=source.get("hash_id") or hit.get("_id"),
            text=source.get("text", ""),
            vector=source.get("vector"),
            doc_type=doc_type,
            metadata=metadata,
        )

    def _build_filter_query(
        self,
        filters: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """将通用过滤条件转换为 Elasticsearch 查询。"""

        if not filters:
            return None
        clauses = []
        for field, value in filters.items():
            field = FILTER_FIELD_ALIASES.get(field, field)
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                clauses.append({"terms": {field: list(value)}})
            else:
                clauses.append({"term": {field: value}})
        return {"bool": {"must": clauses}}

    def _existing_indices(self, index_names: Sequence[str]) -> list[str]:
        """过滤掉当前不存在的索引。"""

        return [
            index_name
            for index_name in index_names
            if self.index_exists(index_name)
        ]
