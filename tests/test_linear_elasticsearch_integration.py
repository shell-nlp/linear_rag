from __future__ import annotations

import os
import unittest
import uuid
from types import SimpleNamespace

from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.indexing.service import IndexingService
from src.retrieval.linear import LinearRetriever
from src.utils import get_es_client


class FixedEmbedding:
    """集成验证只测搜索链路，避免访问外部模型服务。"""

    def encode(self, texts, **kwargs):
        return [
            [1.0, 0.2] if "水库" in text or "问题" in text else [0.2, 1.0]
            for text in texts
        ]


class FixedNER:
    """构造实体—句子—段落的稳定最小图。"""

    def extract_graph_entities(self, passages, max_workers):
        return (
            {key: ["水库", "监测"] for key in passages},
            {key: {text: ["水库", "监测"]} for key, text in passages.items()},
        )

    def extract_question_entities(self, question):
        return ["水库"]


@unittest.skipUnless(
    os.getenv("LINEAR_ES_INTEGRATION") == "1",
    "仅显式启用时连接真实 Elasticsearch",
)
class LinearElasticsearchIntegrationTests(unittest.TestCase):
    """对临时索引执行读写及两条图检索路径，完成后清理测试数据。"""

    def test_graph_index_and_retrieval(self):
        client = get_es_client()
        store = ElasticsearchSearchStore(client)
        index_name = f"linearrag-integration-{uuid.uuid4().hex[:12]}"
        config = SimpleNamespace(
            batch_size=4, max_workers=1, linear_max_nodes=100,
            linear_local_candidates=5, max_iterations=3, top_k_sentence=1,
            iteration_threshold=0.5, passage_ratio=1.5,
            passage_node_weight=0.05, damping=0.5,
            linear_embedding_batch_size=128,
            linear_seed_entities=5, linear_passages_per_entity=5,
            linear_local_max_passages=10, linear_local_max_sentences=20,
        )
        embedding, ner = FixedEmbedding(), FixedNER()
        indexing = IndexingService(config, embedding, store, ner)
        retrieval = LinearRetriever(config, embedding, ner, store)
        try:
            result = indexing.index({
                "text": ["水库监测方案", "监测数据汇总"],
                "file_id": ["test-file", "test-file"],
                "file_path": ["sample.pdf", "sample.pdf"],
                "bucket_name": ["integration", "integration"],
                "segment_id": [1, 2],
            }, index_name)
            self.assertEqual(result["new_passages"], 2)
            nodes = store.scan_documents(
                [index_name], ["passage", "entity", "sentence"]
            )
            self.assertEqual(
                {node.doc_type for node in nodes},
                {"passage", "entity", "sentence"},
            )
            for local in (False, True):
                hits = retrieval.retrieve("水库问题", [index_name], 2, local=local)
                self.assertTrue(hits)
                self.assertTrue(all(hit["type"] == "passage" for hit in hits))
        finally:
            store.delete_index(index_name)


if __name__ == "__main__":
    unittest.main()
