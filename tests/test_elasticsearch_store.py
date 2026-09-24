from __future__ import annotations

import unittest
from unittest.mock import patch

from src.common.search_store.elasticsearch import (
    ElasticsearchSearchStore,
    build_index_mapping,
)
from src.common.models import SearchMode, SearchQuery


class FakeIndicesClient:
    """测试用索引客户端，固定返回索引不存在。"""

    def exists(self, index):
        return False


class FakeElasticsearchClient:
    """测试用 Elasticsearch 客户端，确保不访问真实服务。"""

    def __init__(self):
        self.indices = FakeIndicesClient()

    def search(self, **kwargs):
        raise AssertionError("索引不存在时不应调用 search")


class ExistingIndicesClient:
    """测试用索引客户端，固定返回索引存在。"""

    def __init__(self):
        self.mapping = None

    def exists(self, index):
        return True

    def put_mapping(self, **kwargs):
        self.mapping = kwargs


class RecordingElasticsearchClient:
    """记录查询体，用于验证实体扩展走纯过滤查询。"""

    def __init__(self):
        self.indices = ExistingIndicesClient()
        self.search_kwargs = None
        self.msearch_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return {"hits": {"hits": []}}

    def msearch(self, **kwargs):
        self.msearch_kwargs = kwargs
        return {"responses": [
            {"hits": {"hits": []}}
            for _ in range(len(kwargs["searches"]) // 2)
        ]}


class ElasticsearchStoreTests(unittest.TestCase):
    """验证搜索适配器对空索引的边界处理。"""

    def test_missing_index_returns_empty_result(self):
        """首次建库前查询应返回空结果，而不是抛出异常。"""

        store = ElasticsearchSearchStore(FakeElasticsearchClient())

        self.assertEqual(store.get_documents_by_ids(["missing"], ["id-1"]), {})
        self.assertEqual(
            store.search(
                SearchQuery(
                    index_names=["missing"],
                    mode=SearchMode.VECTOR,
                    query_vector=[1.0, 2.0],
                )
            ),
            [],
        )

    def test_mapping_contains_precomputed_relation_fields(self):
        """ES 映射应支持实体倒排和相邻段落字段。"""

        properties = build_index_mapping(1024)["mappings"]["properties"]

        self.assertEqual(properties["entity_ids"]["type"], "keyword")
        self.assertEqual(
            properties["entities"]["properties"]["count"]["type"],
            "integer",
        )
        self.assertEqual(properties["previous_passage_id"]["type"], "keyword")
        self.assertEqual(properties["entity_id"]["type"], "keyword")
        self.assertEqual(properties["passage_id"]["type"], "keyword")

    def test_filter_only_search_uses_constant_score(self):
        """实体扩展应只按 entity_ids 过滤，不附加原始文本条件。"""

        client = RecordingElasticsearchClient()
        store = ElasticsearchSearchStore(client)

        store.search(
            SearchQuery(
                index_names=["kb_test"],
                mode=SearchMode.BM25,
                filters={"type": "passage", "entity_ids": ["entity-1"]},
                filter_only=True,
            )
        )

        query = client.search_kwargs["body"]["query"]
        self.assertIn("constant_score", query)
        self.assertNotIn("multi_match", str(query))

    def test_existing_index_receives_relation_mapping(self):
        """已有索引也应补充实体关系字段，不要求删除后重建。"""

        client = RecordingElasticsearchClient()
        store = ElasticsearchSearchStore(client)

        store.create_index("kb_test", 1024)

        properties = client.indices.mapping["properties"]
        self.assertEqual(properties["entity_ids"]["type"], "keyword")

    def test_graph_scan_filters_and_rejects_truncation(self):
        """扫描图节点必须带类型过滤，超过上限则报错。"""

        client = RecordingElasticsearchClient()
        store = ElasticsearchSearchStore(client)
        hits = [
            {"_id": f"entity-{index}", "_source": {
                "hash_id": f"entity-{index}", "text": "甲", "type": "entity"
            }}
            for index in range(2)
        ]
        with patch(
            "src.common.search_store.elasticsearch.helpers.scan",
            return_value=iter(hits),
        ) as scan:
            with self.assertRaisesRegex(ValueError, "LINEAR_MAX_NODES"):
                store.scan_documents(
                    ["kb"], ["entity"], {"entity_id": ["a"]}, max_documents=1
                )
        query = scan.call_args.kwargs["query"]["query"]["bool"]["filter"]
        self.assertIn({"terms": {"type": ["entity"]}}, query)
        self.assertIn("entity_id", str(query))

    def test_entity_passages_are_bounded_per_seed(self):
        """每个种子实体独立限额，不能扫描整张实体倒排表。"""

        client = RecordingElasticsearchClient()
        store = ElasticsearchSearchStore(client)

        store.search_entity_passages(["kb_test"], ["a", "b"], 5)

        searches = client.msearch_kwargs["searches"]
        self.assertEqual(len(searches), 4)
        self.assertEqual(searches[1]["size"], 5)
        self.assertIn({"term": {"entity_ids": "a"}}, searches[1]["query"]["bool"]["filter"])
        self.assertIn({"term": {"entity_ids": "b"}}, searches[3]["query"]["bool"]["filter"])

    def test_sentence_nodes_are_bounded(self):
        """句子节点查询必须按候选段落过滤并限制 size。"""

        client = RecordingElasticsearchClient()
        store = ElasticsearchSearchStore(client)

        self.assertEqual(
            store.search_graph_nodes(
                ["kb_test"], "sentence", {"passage_id": ["p1", "p2"]}, 3
            ),
            [],
        )

        query = client.search_kwargs["body"]
        self.assertEqual(query["size"], 3)
        self.assertIn("passage_id", str(query["query"]))


if __name__ == "__main__":
    unittest.main()
