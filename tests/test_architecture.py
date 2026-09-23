from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.common.models import SearchDocument, SearchHit, SearchMode, SearchQuery, WriteResult
from src.indexing.service import IndexingService
from src.knowledge_bases.service import KnowledgeBaseService
from src.retrieval.service import RetrievalService
from src.api import bootstrap


class FakeEmbeddingProvider:
    """测试用向量模型，记录调用且不访问外部服务。"""

    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=32, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


class FakeEntityExtractor:
    """测试用实体识别器，可按段落文本生成重复实体。"""

    def extract_passage_entities(self, hash_id_to_passage, max_workers):
        return {
            passage_id: (["水库", "水库", "监测"] if "水库" in text else [])
            for passage_id, text in hash_id_to_passage.items()
        }


class FakeSearchStore:
    """测试用搜索库，覆盖统一检索和关系字段过滤。"""

    def __init__(self):
        self.documents: dict[str, SearchDocument] = {}
        self.queries: list[SearchQuery] = []
        self.deleted_ids: list[str] = []
        self.mapping = {
            "properties": {
                "text": {"type": "text"},
                "vector": {"type": "dense_vector"},
                "entity_ids": {"type": "keyword"},
            }
        }

    def create_index(self, index_name, vector_dim, settings=None):
        return None

    def delete_index(self, index_name, ignore_unavailable=True):
        self.documents.clear()

    def index_exists(self, index_name):
        return True

    def get_index_mapping(self, index_name):
        return self.mapping

    def upsert_documents(self, index_name, documents, refresh=True):
        for document in documents:
            self.documents[document.id] = document
        return WriteResult(success=len(documents))

    def delete_documents_by_ids(self, index_name, ids, refresh=True):
        self.deleted_ids.extend(ids)
        deleted = 0
        for doc_id in ids:
            if doc_id in self.documents:
                deleted += 1
                del self.documents[doc_id]
        return WriteResult(success=deleted)

    def delete_documents_by_filters(self, index_name, filters, refresh=True):
        file_ids = set(filters.get("file_id", []))
        deleted = 0
        for doc_id, document in list(self.documents.items()):
            if document.metadata.get("file_id") in file_ids:
                deleted += 1
                del self.documents[doc_id]
        return WriteResult(success=deleted)

    def get_documents_by_ids(self, index_names, ids):
        return {doc_id: self.documents[doc_id] for doc_id in ids if doc_id in self.documents}

    def search(self, query: SearchQuery):
        self.queries.append(query)
        hits = []
        expected_type = query.filters.get("type")
        expected_entities = set(query.filters.get("entity_ids", []))
        for document in self.documents.values():
            if expected_type and document.doc_type != expected_type:
                continue
            document_entities = set(document.metadata.get("entity_ids", []))
            if expected_entities and not expected_entities.intersection(document_entities):
                continue
            if query.mode == SearchMode.BM25 and query.query_text:
                # 实体扩展由过滤条件召回，不要求文本再次包含原始问题。
                if not expected_entities and query.query_text not in document.text:
                    continue
            hits.append(SearchHit(id=document.id, score=1.0, document=document))
        return hits[: query.top_k]


def build_config():
    """构造测试用运行时配置。"""

    return SimpleNamespace(
        batch_size=4,
        max_workers=1,
        entity_expansion_enabled=True,
        entity_expansion_max_entities=20,
        entity_expansion_top_k=50,
        neighbor_expansion_enabled=True,
    )


class ArchitectureTests(unittest.TestCase):
    """验证业务只依赖统一搜索端口，并由 ES 字段承载关系信息。"""

    def test_indexing_precomputes_entity_and_neighbor_metadata(self):
        """索引服务应保存实体频次和同文件相邻段落 ID。"""

        search_store = FakeSearchStore()
        service = IndexingService(
            config=build_config(),
            embedding_provider=FakeEmbeddingProvider(),
            search_store=search_store,
            entity_extractor=FakeEntityExtractor(),
        )
        passages = {
            "text": ["水库监测一", "水库监测二"],
            "file_id": ["file-1", "file-1"],
            "file_name": ["demo.pdf", "demo.pdf"],
            "segment_id": [1, 2],
        }

        result = service.index(passages, "kb_test")

        documents = sorted(
            search_store.documents.values(),
            key=lambda item: item.metadata["segment_id"],
        )
        entity_counts = {
            item["name"]: item["count"] for item in documents[0].metadata["entities"]
        }
        self.assertEqual(result["new_passages"], 2)
        self.assertEqual(result["new_entities"], 2)
        self.assertEqual(entity_counts["水库"], 2)
        self.assertEqual(documents[0].metadata["next_passage_id"], documents[1].id)
        self.assertEqual(documents[1].metadata["previous_passage_id"], documents[0].id)

    def test_identical_text_in_different_files_has_distinct_ids(self):
        """相同文本必须按文件隔离，并清理旧版纯文本哈希文档。"""

        search_store = FakeSearchStore()
        service = IndexingService(
            build_config(),
            FakeEmbeddingProvider(),
            search_store,
            FakeEntityExtractor(),
        )
        passages = {
            "text": ["相同段落", "相同段落"],
            "file_id": ["file-1", "file-2"],
            "segment_id": [1, 1],
        }

        service.index(passages, "kb_test")

        self.assertEqual(len(search_store.documents), 2)
        self.assertEqual(len(set(search_store.documents)), 2)
        self.assertEqual(len(set(search_store.deleted_ids)), 1)

    def test_neighbors_are_isolated_by_file_path_without_file_id(self):
        """缺少文件 ID 时也不能把不同文件的段落连接起来。"""

        service = IndexingService(
            build_config(),
            FakeEmbeddingProvider(),
            FakeSearchStore(),
            FakeEntityExtractor(),
        )
        passages = {
            "text": ["文件一", "文件二"],
            "file_path": ["a.pdf", "b.pdf"],
            "bucket_name": ["docs", "docs"],
            "segment_id": [1, 1],
        }

        documents = service._build_passage_documents(passages)

        self.assertTrue(
            all(document.metadata["previous_passage_id"] is None for document in documents)
        )
        self.assertTrue(
            all(document.metadata["next_passage_id"] is None for document in documents)
        )

    def test_bm25_retrieval_expands_entities_without_embedding(self):
        """BM25 主召回和实体扩展都应保持 BM25，且不调用向量模型。"""

        search_store = FakeSearchStore()
        primary = SearchDocument(
            id="passage-1",
            text="水库监测",
            doc_type="passage",
            metadata={
                "entity_ids": ["entity-reservoir"],
                "entities": [
                    {"id": "entity-reservoir", "name": "水库", "importance": 1.0}
                ],
            },
        )
        related = SearchDocument(
            id="passage-2",
            text="相关设施说明",
            doc_type="passage",
            metadata={"entity_ids": ["entity-reservoir"]},
        )
        search_store.documents = {primary.id: primary, related.id: related}
        embedding = FakeEmbeddingProvider()
        service = RetrievalService(build_config(), embedding, search_store)

        results = service.retrieve(
            "水库监测", ["kb_test"], top_k=3, search_mode=SearchMode.BM25
        )

        self.assertEqual({item["hash_id"] for item in results}, {"passage-1", "passage-2"})
        self.assertFalse(embedding.calls)
        self.assertTrue(search_store.queries[-1].filters["entity_ids"])
        self.assertTrue(search_store.queries[-1].filter_only)
        self.assertTrue(all(query.mode == SearchMode.BM25 for query in search_store.queries))

    def test_delete_files_only_uses_search_store(self):
        """文件删除应由搜索库一次完成，无需同步第二种存储。"""

        search_store = FakeSearchStore()
        search_store.documents["passage-1"] = SearchDocument(
            id="passage-1",
            text="水库监测",
            doc_type="passage",
            metadata={"file_id": "file-1"},
        )
        service = IndexingService(
            build_config(), FakeEmbeddingProvider(), search_store, FakeEntityExtractor()
        )

        result = service.delete_files("kb_test", ["file-1"])

        self.assertEqual(result["deleted_documents"], 1)
        self.assertFalse(search_store.documents)

    def test_knowledge_base_search_supports_hybrid_mode(self):
        """知识库服务应能把 hybrid 模式传递到搜索端口。"""

        search_store = FakeSearchStore()
        search_store.documents["passage-1"] = SearchDocument(
            id="passage-1", text="水库监测", doc_type="passage"
        )
        service = KnowledgeBaseService(search_store, FakeEmbeddingProvider(), 2)

        results = service.search("kb_test", "text", "水库", SearchMode.HYBRID, 3)

        self.assertEqual(results[0]["id"], "passage-1")
        self.assertEqual(search_store.queries[-1].mode, SearchMode.HYBRID)

    def test_bootstrap_uses_current_retrieval_service_parameters(self):
        """启动装配传参必须与检索服务构造函数保持一致。"""

        settings = SimpleNamespace(
            index_process_workers=1,
            spacy_model="test-model",
            embedding_dim=2,
            runtime_config=lambda: build_config(),
        )
        providers = SimpleNamespace(embedding=object(), llm=object())
        with (
            patch.object(bootstrap, "get_settings", return_value=settings),
            patch.object(bootstrap, "setup_logging"),
            patch.object(bootstrap.os, "makedirs"),
            patch.object(bootstrap, "ProcessPoolExecutor"),
            patch.object(bootstrap, "create_model_providers", return_value=providers),
            patch.object(bootstrap, "get_es_client", return_value=object()),
            patch.object(bootstrap, "ElasticsearchSearchStore", return_value=object()),
            patch.object(bootstrap, "SpacyNER", return_value=object()),
            patch.object(bootstrap, "IndexingService", return_value=object()),
            patch.object(bootstrap, "KnowledgeBaseService", return_value=object()),
            patch.object(bootstrap, "RetrievalService", return_value=object()) as service,
        ):
            bootstrap.build_application_state()

        self.assertEqual(
            set(service.call_args.kwargs),
            {"config", "search_store", "embedding_provider"},
        )


if __name__ == "__main__":
    unittest.main()
