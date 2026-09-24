from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.common.models import SearchDocument, SearchHit, SearchMode
from src.retrieval.linear import LinearRetriever
from src.retrieval.service import RetrievalService


class Embedding:
    """用固定向量验证种子、桥接和段落排序。"""

    vectors = {
        "问题": [1.0, 0.0],
        "其他问题": [1.0, 0.0],
        "甲": [1.0, 0.0],
        "乙": [0.9, 0.1],
    }

    def encode(self, texts, **kwargs):
        return [self.vectors[text] for text in texts]


class NER:
    """测试问题实体识别结果。"""

    def extract_question_entities(self, question):
        return ["甲"] if question == "问题" else []


class Store:
    """记录图节点扫描范围以区分全图和局部模式。"""

    def __init__(self):
        self.scans = []
        self.documents = [
            SearchDocument("p1", "甲乙", [1.0, 0.0], "passage", {
                "entity_ids": ["a", "b"],
                "entities": [
                    {"id": "a", "name": "甲"},
                    {"id": "b", "name": "乙"},
                ],
                "next_passage_id": "p2",
                "file_id": "f1",
            }),
            SearchDocument("p2", "乙", [0.9, 0.1], "passage", {
                "entity_ids": ["b"],
                "entities": [{"id": "b", "name": "乙"}],
                "file_id": "f1",
            }),
            SearchDocument("a-node", "甲", [1.0, 0.0], "entity", {"entity_id": "a"}),
            SearchDocument("b-node", "乙", [0.9, 0.1], "entity", {"entity_id": "b"}),
            SearchDocument("s1", "甲乙", [1.0, 0.0], "sentence", {
                "entity_ids": ["a", "b"], "file_id": "f1", "passage_id": "p1"
            }),
        ]

    def scan_documents(self, index_names, doc_types, filters=None, max_documents=None):
        self.scans.append((tuple(doc_types), filters))
        result = [item for item in self.documents if item.doc_type in doc_types]
        for key, values in (filters or {}).items():
            result = [
                item for item in result
                if (
                    bool(set(item.metadata.get(key, [])) & set(values))
                    if key == "entity_ids" else item.metadata.get(key) in values
                )
            ]
        if max_documents is not None and len(result) > max_documents:
            raise ValueError("too many nodes")
        return result

    def search(self, query):
        return [SearchHit("p1", 1.0, self.documents[0])]

    def get_documents_by_ids(self, index_names, ids):
        return {item.id: item for item in self.documents if item.id in ids}


def config():
    """保留与官方配置同名的计算参数。"""

    return SimpleNamespace(
        batch_size=4,
        linear_max_nodes=100,
        linear_local_candidates=1,
        max_iterations=3,
        top_k_sentence=1,
        iteration_threshold=0.5,
        passage_ratio=1.5,
        passage_node_weight=0.05,
        damping=0.5,
    )


class LinearRetrievalTests(unittest.TestCase):
    def test_full_mode_runs_ppr_and_returns_passages(self):
        """全图路径读取三类节点并用 PPR 返回段落。"""

        store = Store()
        retriever = LinearRetriever(config(), Embedding(), NER(), store)
        result = retriever.retrieve("问题", ["kb"], 2, local=False)
        self.assertEqual({item["hash_id"] for item in result}, {"p1", "p2"})
        self.assertGreater(result[0]["score"], 0)
        self.assertEqual(store.scans, [(("passage", "sentence", "entity"), None)])

    def test_local_mode_only_reads_candidate_related_nodes(self):
        """局部路径先取候选，再扫描其关联实体和句子。"""

        store = Store()
        retriever = LinearRetriever(config(), Embedding(), NER(), store)
        result = retriever.retrieve("问题", ["kb"], 2, local=True)
        self.assertEqual([item["hash_id"] for item in result], ["p1"])
        self.assertEqual(len(store.scans), 2)
        self.assertEqual(store.scans[0][1], {"entity_id": ["a", "b"]})
        self.assertEqual(store.scans[1][1]["passage_id"], ["p1"])

    def test_no_query_entity_falls_back_to_dense_passages(self):
        """没有问题实体时不运行 PPR，按段落向量回退。"""

        store = Store()
        retriever = LinearRetriever(config(), Embedding(), NER(), store)
        result = retriever.retrieve("其他问题", ["kb"], 1, local=False)
        self.assertEqual(result[0]["hash_id"], "p1")

    def test_fast_and_graph_modes_remain_separate(self):
        """快速检索路径不依赖图模式，新增模式必须显式选择。"""

        store = Store()
        retriever = LinearRetriever(config(), Embedding(), NER(), store)
        service = RetrievalService(config(), Embedding(), store, retriever)
        result = service.retrieve("问题", ["kb"], 2, SearchMode.LINEAR_LOCAL)
        self.assertEqual(result[0]["hash_id"], "p1")

    def test_graph_mode_rejects_multiple_knowledge_bases(self):
        """图节点逻辑 ID 仅在单知识库内合并。"""

        retriever = LinearRetriever(config(), Embedding(), NER(), Store())
        with self.assertRaisesRegex(ValueError, "一个知识库"):
            retriever.retrieve("问题", ["kb-a", "kb-b"], 2, local=False)

    def test_graph_mode_rejects_old_index_without_entity_nodes(self):
        """旧索引缺少实体节点时提示重建，避免悄悄返回普通检索。"""

        store = Store()
        store.documents = store.documents[:2]
        retriever = LinearRetriever(config(), Embedding(), NER(), store)
        with self.assertRaisesRegex(ValueError, "重新上传"):
            retriever.retrieve("问题", ["kb"], 2, local=False)


if __name__ == "__main__":
    unittest.main()
