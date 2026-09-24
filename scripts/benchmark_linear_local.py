from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

# 支持直接执行 scripts/benchmark_linear_local.py 时找到项目根目录。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.models import SearchDocument, SearchQuery
from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.retrieval.linear import LinearRetriever
from src.utils import get_es_client


class FixedEmbedding:
    """固定向量，避免远端 Embedding 延迟影响图检索基准。"""

    def encode(self, texts, **kwargs):
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in texts]


class FixedNER:
    """固定问题实体，保证每次都从同一逻辑实体开始传播。"""

    def extract_question_entities(self, question):
        return ["entity-0"]


class TimingStore:
    """包装 SearchStore，记录向量召回和两类图节点扫描耗时。"""

    def __init__(self, store):
        self.store = store
        self.timings = defaultdict(list)

    def search(self, query: SearchQuery):
        started = time.perf_counter()
        result = self.store.search(query)
        label = "entity_seed" if query.filters.get("type") == "entity" else "vector_search"
        self.timings[label].append(time.perf_counter() - started)
        return result

    def search_entity_passages(self, index_names, entity_ids, per_entity_limit):
        started = time.perf_counter()
        result = self.store.search_entity_passages(
            index_names, entity_ids, per_entity_limit
        )
        self.timings["entity_passages"].append(time.perf_counter() - started)
        return result

    def search_graph_nodes(self, index_names, doc_type, filters, limit):
        started = time.perf_counter()
        result = self.store.search_graph_nodes(
            index_names, doc_type, filters, limit
        )
        self.timings["sentence_nodes"].append(time.perf_counter() - started)
        return result

    def scan_documents(
        self,
        index_names,
        doc_types,
        filters=None,
        max_documents=None,
    ):
        label = "entity_scan" if doc_types == ["entity"] else "sentence_scan"
        started = time.perf_counter()
        result = self.store.scan_documents(
            index_names,
            doc_types,
            filters,
            max_documents,
        )
        self.timings[label].append(time.perf_counter() - started)
        return result

    def __getattr__(self, name):
        return getattr(self.store, name)


def passage_vector(index: int) -> list[float]:
    """给候选段落制造稳定顺序，同时保持所有向量已归一化。"""

    return [
        1.0,
        index / 1_000_000,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def build_documents(
    passage_count: int,
    sentences_per_passage: int,
    shared_entity_count: int,
) -> list[SearchDocument]:
    """构造段落、实体和句子节点，模拟有共享实体枢纽的图。"""

    entity_ids = [
        f"entity-{index}" for index in range(shared_entity_count)
    ]
    documents = [
        SearchDocument(
            id=entity_id,
            text=entity_id,
            vector=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            doc_type="entity",
            metadata={"entity_id": entity_id, "file_id": "benchmark"},
        )
        for entity_id in entity_ids
    ]
    for index in range(passage_count):
        passage_id = f"passage-{index}"
        linked_entities = [
            entity_ids[(index + offset) % shared_entity_count]
            for offset in range(min(4, shared_entity_count))
        ]
        documents.append(
            SearchDocument(
                id=passage_id,
                text=f"基准段落 {index}",
                vector=passage_vector(index),
                doc_type="passage",
                metadata={
                    "file_id": "benchmark",
                    "entity_ids": linked_entities,
                    "entities": [
                        {
                            "id": entity_id,
                            "name": entity_id,
                            "count": 1,
                            "importance": 1.0,
                        }
                        for entity_id in linked_entities
                    ],
                    "previous_passage_id": (
                        f"passage-{index - 1}" if index else None
                    ),
                    "next_passage_id": (
                        f"passage-{index + 1}"
                        if index + 1 < passage_count
                        else None
                    ),
                },
            )
        )
        for sentence_index in range(sentences_per_passage):
            documents.append(
                SearchDocument(
                    id=f"sentence-{index}-{sentence_index}",
                    text=f"基准句子 {index}-{sentence_index}",
                    vector=passage_vector(index),
                    doc_type="sentence",
                    metadata={
                        "file_id": "benchmark",
                        "passage_id": passage_id,
                        "entity_ids": linked_entities,
                    },
                )
            )
    return documents


def percentile(values: list[float], ratio: float) -> float:
    """返回简单百分位，样本很小时使用最大值。"""

    ordered = sorted(values)
    position = min(len(ordered) - 1, int(len(ordered) * ratio))
    return ordered[position]


def run_scenario(args, passage_count: int, candidate_count: int) -> dict:
    """建立临时索引，执行多次局部 PPR 查询并返回统计。"""

    client = get_es_client()
    store = ElasticsearchSearchStore(client)
    timing_store = TimingStore(store)
    index_name = f"linearrag-bench-{uuid.uuid4().hex[:12]}"
    config = SimpleNamespace(
        batch_size=128,
        linear_max_nodes=args.max_nodes,
        linear_local_candidates=candidate_count,
        linear_seed_entities=5,
        linear_passages_per_entity=5,
        linear_local_max_passages=max(candidate_count + 25, args.top_k),
        linear_local_max_sentences=min(args.max_nodes, candidate_count * args.sentences_per_passage + 75),
        max_iterations=3,
        top_k_sentence=1,
        iteration_threshold=0.5,
        passage_ratio=1.5,
        passage_node_weight=0.05,
        damping=0.5,
    )
    retriever = LinearRetriever(
        config,
        FixedEmbedding(),
        FixedNER(),
        timing_store,
    )
    try:
        documents = build_documents(
            passage_count,
            args.sentences_per_passage,
            args.shared_entities,
        )
        started = time.perf_counter()
        store.create_index(index_name, vector_dim=8)
        for start in range(0, len(documents), args.index_batch_size):
            store.upsert_documents(
                index_name,
                documents[start : start + args.index_batch_size],
                refresh=False,
            )
        client.indices.refresh(index=index_name)
        index_seconds = time.perf_counter() - started

        query_seconds = []
        result_counts = []
        for query_index in range(args.warmups + args.queries):
            started = time.perf_counter()
            hits = retriever.retrieve(
                "基准问题",
                [index_name],
                args.top_k,
                local=True,
            )
            elapsed = time.perf_counter() - started
            if query_index >= args.warmups:
                query_seconds.append(elapsed)
                result_counts.append(len(hits))
        return {
            "passages": passage_count,
            "candidates": candidate_count,
            "documents": len(documents),
            "index_seconds": index_seconds,
            "query_median": statistics.median(query_seconds),
            "query_p95": percentile(query_seconds, 0.95),
            "result_count": statistics.median(result_counts),
            "vector_search_median": statistics.median(
                timing_store.timings["vector_search"][args.warmups:]
            ),
            "entity_scan_median": statistics.median(
                timing_store.timings["entity_scan"][args.warmups:]
            ),
            "sentence_scan_median": statistics.median(
                timing_store.timings["sentence_nodes"][args.warmups:]
            ),
            "seed_median": statistics.median(
                timing_store.timings["entity_seed"][args.warmups:]
            ),
            "entity_passages_median": statistics.median(
                timing_store.timings["entity_passages"][args.warmups:]
            ),
        }
    finally:
        if store.index_exists(index_name):
            store.delete_index(index_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 linear_local 在不同索引规模和候选规模下的延迟",
    )
    parser.add_argument("--passages", type=int, nargs="+", default=[1000, 5000])
    parser.add_argument("--candidates", type=int, nargs="+", default=[200])
    parser.add_argument("--sentences-per-passage", type=int, default=3)
    parser.add_argument("--shared-entities", type=int, default=100)
    parser.add_argument("--index-batch-size", type=int, default=2000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--queries", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=200000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for passage_count in args.passages:
        for candidate_count in args.candidates:
            result = run_scenario(args, passage_count, candidate_count)
            print(
                "passages={passages} candidates={candidates} docs={documents} "
                "index={index_seconds:.2f}s query_median={query_median:.3f}s "
                "p95={query_p95:.3f}s vector={vector_search_median:.3f}s "
                "seed={seed_median:.3f}s linked={entity_passages_median:.3f}s "
                "entities={entity_scan_median:.3f}s sentences={sentence_scan_median:.3f}s "
                "hits={result_count}".format(**result),
                flush=True,
            )


if __name__ == "__main__":
    main()
