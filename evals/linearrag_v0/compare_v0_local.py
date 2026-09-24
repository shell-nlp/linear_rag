from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

# 独立脚本从项目根目录导入业务模块，不改动应用运行路径。
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.models import SearchDocument, SearchMode, SearchQuery
from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.retrieval.linear import LinearRetriever
from src.utils import get_es_client


@dataclass(frozen=True)
class Case:
    question: str
    entity: str
    target: str


class FixedEmbedding:
    """对问题和实体使用显式向量，制造可复现的召回缺口。"""

    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors

    def encode(self, texts, **kwargs):
        return [self.vectors[text] for text in texts]


class FixedNER:
    """问题实体由评测标签给出，不引入 NER 模型误差。"""

    def __init__(self, cases: list[Case]):
        self.entities = {case.question: case.entity for case in cases}

    def extract_question_entities(self, question):
        return [self.entities[question]]


def vector(index: int, size: int) -> list[float]:
    """固定维度的正交向量。"""

    values = [0.0] * size
    values[index] = 1.0
    return values


def build_cases(groups: int) -> tuple[list[Case], list[SearchDocument], dict[str, list[float]]]:
    """每组包含目标段落、两个向量干扰段落和一条关联句子。"""

    cases = []
    documents = []
    vectors = {}
    dimensions = groups + 1
    for group in range(groups):
        entity_name = f"entity-{group}"
        query = f"question-{group}"
        target_id = f"target-{group}"
        query_vector = vector(group, dimensions)
        target_vector = vector(groups, dimensions)
        cases.append(Case(query, entity_name, target_id))
        vectors[query] = query_vector
        vectors[entity_name] = query_vector
        documents.append(
            SearchDocument(
                id=f"node-{group}",
                text=entity_name,
                vector=query_vector,
                doc_type="entity",
                metadata={"entity_id": entity_name, "file_id": "eval"},
            )
        )
        documents.append(
            SearchDocument(
                id=target_id,
                text=f"{entity_name} 的具体答案",
                vector=target_vector,
                doc_type="passage",
                metadata={
                    "file_id": "eval",
                    "entity_ids": [entity_name],
                    "entities": [{"id": entity_name, "name": entity_name, "count": 1}],
                },
            )
        )
        documents.append(
            SearchDocument(
                id=f"sentence-{group}",
                text=f"{entity_name} 的具体答案",
                vector=query_vector,
                doc_type="sentence",
                metadata={
                    "file_id": "eval",
                    "passage_id": target_id,
                    "entity_ids": [entity_name],
                },
            )
        )
        for offset, similarity in enumerate((0.85, 0.75)):
            distractor_vector = query_vector.copy()
            distractor_vector[groups] = 1.0 - similarity
            documents.append(
                SearchDocument(
                    id=f"distractor-{group}-{offset}",
                    text=f"相似但无答案的段落 {group}-{offset}",
                    vector=distractor_vector,
                    doc_type="passage",
                    metadata={
                        "file_id": "eval",
                        "entity_ids": [],
                        "entities": [],
                    },
                )
            )
    return cases, documents, vectors


def recall(results: list[list[str]], cases: list[Case], top_k: int) -> float:
    """计算每个问题目标段落是否出现在前 top_k。"""

    return sum(
        case.target in ranking[:top_k]
        for case, ranking in zip(cases, results)
    ) / len(cases)


def run(groups: int, candidate_count: int) -> None:
    """在临时 ES 索引上比较候选约束和实际局部图排序。"""

    cases, documents, vectors = build_cases(groups)
    store = ElasticsearchSearchStore(get_es_client())
    index_name = f"linearrag-compare-{uuid.uuid4().hex[:12]}"
    config = type("Config", (), {
        "batch_size": 32,
        "linear_max_nodes": 500,
        "linear_local_candidates": candidate_count,
        "linear_seed_entities": 1,
        "linear_passages_per_entity": 5,
        "linear_local_max_passages": candidate_count + 5,
        "linear_local_max_sentences": 30,
        "max_iterations": 3,
        "top_k_sentence": 1,
        "iteration_threshold": 0.5,
        "passage_ratio": 1.5,
        "passage_node_weight": 0.05,
        "damping": 0.5,
    })()
    embedding = FixedEmbedding(vectors)
    retriever = LinearRetriever(config, embedding, FixedNER(cases), store)
    vector_rankings = []
    local_rankings = []
    vector_times = []
    local_times = []
    try:
        store.upsert_documents(index_name, documents)
        for case in cases:
            query_vector = vectors[case.question]
            started = time.perf_counter()
            hits = store.search(SearchQuery(
                index_names=[index_name],
                mode=SearchMode.VECTOR,
                query_vector=query_vector,
                top_k=candidate_count,
                num_candidates=max(100, candidate_count * 4),
                filters={"type": "passage"},
            ))
            vector_times.append(time.perf_counter() - started)
            vector_rankings.append([hit.id for hit in hits])
            started = time.perf_counter()
            results = retriever.retrieve(case.question, [index_name], 5, local=True)
            local_times.append(time.perf_counter() - started)
            local_rankings.append([item["hash_id"] for item in results])
        print(f"cases={len(cases)} passages={groups * 3} candidates={candidate_count}")
        for top_k in (1, 3, 5):
            print(
                f"Recall@{top_k}: vector={recall(vector_rankings, cases, top_k):.3f} "
                f"v0-candidate-ceiling={recall(vector_rankings, cases, candidate_count):.3f} "
                f"linear_local={recall(local_rankings, cases, top_k):.3f}"
            )
        print(
            f"median latency: vector={statistics.median(vector_times):.3f}s "
            f"linear_local={statistics.median(local_times):.3f}s"
        )
    finally:
        if store.index_exists(index_name):
            store.delete_index(index_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="受控语料比较 v0 候选上限与 linear_local")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=2)
    args = parser.parse_args()
    run(args.groups, args.candidates)


if __name__ == "__main__":
    main()
