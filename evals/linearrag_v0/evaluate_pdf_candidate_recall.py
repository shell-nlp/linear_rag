"""在真实 PDF 上比较向量候选、v0 风格排序与 linear_local 召回。"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from loguru import logger
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.document_processing.ner import SpacyNER
from src.common.model_providers import create_embedding_provider
from src.common.models import SearchMode, SearchQuery
from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.indexing.service import IndexingService
from src.indexing.workflow import resolve_uploaded_passages
from src.retrieval.linear import LinearRetriever
from src.settings import get_settings
from src.utils import get_es_client


def v0_style_rank(retriever, question, candidates, entity_nodes, all_passages):
    """按 v0 的候选、有限扩散和分数融合近似计算，不声称复现 Neo4j GDS。"""

    if not candidates:
        return []
    question_vector = retriever._encode([question])[0]
    names = retriever.ner.extract_question_entities(question)
    if not names:
        return [hit.id for hit in candidates]
    entities = {
        item.metadata["entity_id"]: item
        for item in entity_nodes if item.vector and item.metadata.get("entity_id")
    }
    if not entities:
        return [hit.id for hit in candidates]
    seed_scores = {}
    for vector in retriever._encode(names):
        entity_id, score = max(
            (
                (entity_id, float(np.dot(vector, node.vector)))
                for entity_id, node in entities.items()
            ),
            key=lambda item: item[1],
        )
        seed_scores[entity_id] = score
    # v0 最多两轮扩散，每实体最多访问五个关联段落。
    activated = dict(seed_scores)
    frontier = dict(seed_scores)
    used_passages = set()
    for _ in range(2):
        next_frontier = {}
        for entity_id, entity_score in frontier.items():
            if entity_score < 0.01:
                continue
            related = [
                item for item in all_passages
                if entity_id in (item.metadata.get("entity_ids") or [])
                and item.id not in used_passages
            ][:5]
            for passage in related:
                used_passages.add(passage.id)
                similarity = float(np.dot(passage.vector, question_vector))
                for next_id in passage.metadata.get("entity_ids") or []:
                    score = entity_score * similarity
                    if score >= 0.5:
                        next_frontier[next_id] = max(
                            next_frontier.get(next_id, 0.0), score
                        )
        activated.update(next_frontier)
        frontier = next_frontier

    raw_scores = [hit.score for hit in candidates]
    minimum, maximum = min(raw_scores), max(raw_scores)
    base_scores = {}
    for hit in candidates:
        dense = (
            (hit.score - minimum) / (maximum - minimum)
            if maximum != minimum else 0.5
        )
        bonus = 0.0
        for entity_id, score in activated.items():
            node = entities.get(entity_id)
            if node and entity_id in (hit.document.metadata.get("entity_ids") or []):
                count = hit.document.text.lower().count(node.text.lower())
                bonus += score * math.log1p(count)
        base_scores[hit.id] = 0.6 * dense + math.log1p(bonus)

    # v0 投影的是激活实体与 ES top-50 段落，sourceNodes 只取分数大于 0.5 的节点。
    graph = retriever._build_graph(
        [hit.document for hit in candidates],
        entities,
    )
    source_ids = {
        node_id for node_id, score in {**activated, **base_scores}.items()
        if score > 0.5
    }
    if not source_ids:
        source_ids = set(graph)
    reset = {node_id: 1.0 for node_id in source_ids}
    scorer = LinearRetriever(
        SimpleNamespace(damping=0.85),
        retriever.embedding,
        retriever.ner,
        retriever.store,
    )
    ppr = scorer._pagerank(graph, reset)
    return sorted(
        base_scores,
        key=lambda passage_id: (
            0.5 * base_scores[passage_id] + 0.5 * ppr.get(passage_id, 0.0)
        ),
        reverse=True,
    )


def cases_from_documents(documents, limit):
    """从真实 PDF 中选取稀有实体的唯一目标段落，生成可审查的弱标注。"""

    passages = [item for item in documents if item.doc_type == "passage"]
    by_entity = {}
    for passage in passages:
        for entity in passage.metadata.get("entities") or []:
            name = entity["name"].strip()
            han_count = sum("\u4e00" <= char <= "\u9fff" for char in name)
            if (
                3 <= len(name) <= 16
                and han_count >= max(3, len(name) * 0.7)
                and not re.search(r"\d", name)
                and name in passage.text
            ):
                by_entity.setdefault(name, set()).add(passage.id)
    cases = []
    used = set()
    for name, passage_ids in sorted(
        by_entity.items(), key=lambda item: (len(item[1]), item[0])
    ):
        if len(passage_ids) != 1:
            continue
        passage_id = next(iter(passage_ids))
        if passage_id in used:
            continue
        used.add(passage_id)
        cases.append((f"{name}的相关规定是什么？", passage_id, name))
        if len(cases) == limit:
            break
    return cases


def main():
    parser = argparse.ArgumentParser(description="真实 PDF 弱标注候选召回对比")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "evals" / "linearrag_v0" / "data" / "document.pdf",
    )
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings()
    store = ElasticsearchSearchStore(get_es_client())
    embedding = create_embedding_provider()
    ner = SpacyNER(settings.spacy_model)
    config = settings.runtime_config()
    config.linear_local_candidates = args.candidates
    config.linear_local_max_passages = args.candidates + 5
    suffix = uuid.uuid4().hex[:12]
    index_name = f"linearrag-pdf-compare-{suffix}"
    logger.disable("src.common.document_processing.pdf_parser")
    try:
        passages = resolve_uploaded_passages(
            None, args.pdf.read_bytes(), "compare", f"{suffix}/document.pdf", suffix
        )
        indexer = IndexingService(config, embedding, store, ner)
        documents, entity_ids = indexer.prepare_documents(passages)
        cases = cases_from_documents(documents, args.cases)
        if not cases:
            raise RuntimeError("PDF 中没有适合生成弱标注的问题")
        indexer.write_documents(index_name, documents, entity_ids)
        retriever = LinearRetriever(config, embedding, ner, store)
        entity_nodes = [item for item in documents if item.doc_type == "entity"]
        all_passages = [item for item in documents if item.doc_type == "passage"]
        results = {"vector": [], "v0_style": [], "local": []}
        v0_candidate_hits = []
        timings = {"vector": [], "local": []}
        for question, target, entity in cases:
            query_vector = retriever._encode([question])[0].tolist()
            start = time.perf_counter()
            vector_hits = store.search(SearchQuery(
                index_names=[index_name],
                mode=SearchMode.VECTOR,
                query_vector=query_vector,
                top_k=args.candidates,
                num_candidates=max(100, args.candidates * 4),
                filters={"type": "passage"},
            ))
            timings["vector"].append(time.perf_counter() - start)
            vector_ids = [hit.id for hit in vector_hits]
            v0_hits = store.search(SearchQuery(
                index_names=[index_name],
                mode=SearchMode.VECTOR,
                query_vector=query_vector,
                top_k=50,
                num_candidates=100,
                filters={"type": "passage"},
            ))
            v0_candidate_hits.append([hit.id for hit in v0_hits])
            results["v0_style"].append(
                v0_style_rank(
                    retriever, question, v0_hits, entity_nodes, all_passages
                )[:15]
            )
            seed_names = ner.extract_question_entities(question)
            candidate_passages, _, _, seeds = retriever._load_local_graph(
                retriever._encode([question])[0],
                seed_names,
                [index_name],
            )
            merged_ids = [passage.id for passage in candidate_passages]
            start = time.perf_counter()
            local_hits = retriever.retrieve(question, [index_name], 15, local=True)
            timings["local"].append(time.perf_counter() - start)
            local_ids = [hit["hash_id"] for hit in local_hits]
            results["vector"].append(vector_ids)
            results["local"].append(local_ids)
            print(
                f"entity={entity} ner={seed_names} seeds={len(seeds)} "
                f"vector_rank={vector_ids.index(target) + 1 if target in vector_ids else '-'} "
                f"v0_rank={v0_candidate_hits[-1].index(target) + 1 if target in v0_candidate_hits[-1] else '-'} "
                f"v0_style_rank={results['v0_style'][-1].index(target) + 1 if target in results['v0_style'][-1] else '-'} "
                f"merged_rank={merged_ids.index(target) + 1 if target in merged_ids else '-'} "
                f"local_rank={local_ids.index(target) + 1 if target in local_ids else '-'}",
                flush=True,
            )
        for top_k in (1, 3, 5, 8, 10, 15):
            vector_recall = sum(
                target in hits[:top_k] for (_, target, _), hits in zip(cases, results["vector"])
            ) / len(cases)
            local_recall = sum(
                target in hits[:top_k] for (_, target, _), hits in zip(cases, results["local"])
            ) / len(cases)
            v0_style_recall = sum(
                target in hits[:top_k] for (_, target, _), hits in zip(cases, results["v0_style"])
            ) / len(cases)
            print(
                f"weak Recall@{top_k}: vector={vector_recall:.3f} "
                f"v0-style={v0_style_recall:.3f} "
                f"linear_local={local_recall:.3f}"
            )
        candidate_ceiling = sum(
            target in hits for (_, target, _), hits in zip(cases, results["vector"])
        ) / len(cases)
        v0_ceiling = sum(
            target in hits for (_, target, _), hits in zip(cases, v0_candidate_hits)
        ) / len(cases)
        print(
            f"vector-candidate ceiling@{args.candidates}={candidate_ceiling:.3f} "
            f"v0-candidate ceiling@50={v0_ceiling:.3f}"
        )
        print(
            f"median latency: vector={statistics.median(timings['vector']):.3f}s "
            f"linear_local={statistics.median(timings['local']):.3f}s"
        )
    finally:
        if store.index_exists(index_name):
            store.delete_index(index_name)


if __name__ == "__main__":
    main()
