from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src.common.vector_utils import cosine_similarity, normalize_vector
from src.common.models import SearchHit, SearchMode, SearchQuery
from src.common.model_providers.base import EmbeddingProvider
from src.common.document_processing.base import EntityExtractor
from src.common.graph_store.base import GraphStore
from src.common.search_store.base import SearchStore


class RetrievalService:
    """负责向量、BM25、混合检索以及知识图谱多跳重排。"""

    def __init__(
        self,
        config,
        embedding_provider: EmbeddingProvider,
        search_store: SearchStore,
        graph_store: GraphStore,
        entity_extractor: EntityExtractor,
    ):
        """通过端口注入检索所需依赖。"""

        self.config = config
        self.embedding_provider = embedding_provider
        self.search_store = search_store
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor

    def retrieve(
        self,
        question: str,
        index_names: list[str],
        top_k: int = 5,
        search_mode: SearchMode = SearchMode.VECTOR,
    ) -> list[dict[str, Any]]:
        """执行完整检索流程。"""

        embedding_question = (
            f"下面是一个问题，从数据库中检索到问题相关的段落\n问题：{question}"
        )
        question_embedding = self._encode_question(embedding_question)
        seed_entities = self._get_seed_entities(question, index_names)
        if seed_entities:
            return self._graph_search(
                question_embedding,
                seed_entities,
                index_names,
                top_k,
            )
        return self._fallback_search(
            question=question,
            question_embedding=question_embedding,
            index_names=index_names,
            top_k=top_k,
            search_mode=search_mode,
        )

    def _get_seed_entities(
        self,
        question: str,
        index_names: list[str],
    ) -> list[dict[str, Any]]:
        """通过实体向量检索获取图扩散的种子实体。"""

        question_entities = list(
            self.entity_extractor.extract_question_entities(question)
        )
        if not question_entities:
            return []
        entity_embeddings = self._encode_texts(question_entities, is_batch=True)
        seed_entities = []
        for index, entity in enumerate(question_entities):
            query_vector = entity_embeddings[index]
            hits = self.search_store.search(
                SearchQuery(
                    index_names=index_names,
                    mode=SearchMode.VECTOR,
                    query_vector=query_vector,
                    top_k=10,
                    num_candidates=100,
                    filters={"type": "entity"},
                    source_fields=["hash_id", "text", "type"],
                )
            )
            if not hits:
                continue
            top_hit = hits[0]
            seed_entities.append(
                {
                    "hash_id": top_hit.id,
                    "text": top_hit.document.text,
                    "score": top_hit.score,
                    "embedding": query_vector,
                }
            )
        return seed_entities

    def _fallback_search(
        self,
        question: str,
        question_embedding: Sequence[float],
        index_names: list[str],
        top_k: int,
        search_mode: SearchMode,
    ) -> list[dict[str, Any]]:
        """没有实体时执行普通向量、BM25 或混合检索。"""

        hits = self.search_store.search(
            SearchQuery(
                index_names=index_names,
                mode=search_mode,
                query_text=question,
                query_vector=list(question_embedding),
                top_k=top_k,
                num_candidates=100,
                filters={"type": "passage"},
                source_fields=self._passage_source_fields(),
            )
        )
        passages = self._hits_to_passages(
            hits,
            include_extra_fields={"pagerank_score": 0, "base_score": None},
        )
        for passage in passages:
            passage["base_score"] = passage.get("score", 0)
        return passages

    def _graph_search(
        self,
        question_embedding: Sequence[float],
        seed_entities: list[dict[str, Any]],
        index_names: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """执行实体扩散、段落打分和 PageRank 重排。"""

        entity_scores = self._calculate_entity_scores(
            seed_entities,
            question_embedding,
            index_names,
        )
        passage_scores = self._calculate_passage_scores(
            question_embedding,
            entity_scores,
            index_names,
        )
        final_passages = self._run_pagerank_and_filter(
            entity_scores,
            passage_scores,
        )[:top_k]
        if not final_passages:
            return []

        documents = self.search_store.get_documents_by_ids(
            index_names,
            [passage["hash_id"] for passage in final_passages],
        )
        for passage in final_passages:
            document = documents.get(passage["hash_id"])
            if document:
                passage.update(document.as_source())
        return final_passages

    def _calculate_entity_scores(
        self,
        seed_entities: list[dict[str, Any]],
        question_embedding: Sequence[float],
        index_names: list[str],
    ) -> dict[str, float]:
        """执行有限轮实体到段落再到实体的扩散。"""

        activated_entities = {
            entity["hash_id"]: entity["score"] for entity in seed_entities
        }
        current_entities = activated_entities.copy()
        used_passages: set[str] = set()
        iteration = 1
        while current_entities and iteration < 3:
            new_entities: dict[str, float] = {}
            for entity_id, entity_score in current_entities.items():
                if entity_score < 0.01:
                    continue
                related_passages = self.graph_store.get_related_passages(
                    entity_ids=[entity_id],
                    index_names=index_names,
                    limit_per_entity=5,
                )
                for passage in related_passages:
                    if passage.id in used_passages:
                        continue
                    used_passages.add(passage.id)
                    passage_similarity = (
                        cosine_similarity(passage.vector, question_embedding)
                        if passage.vector
                        else 0.5
                    )
                    related_entities = self.graph_store.get_related_entities(
                        passage_ids=[passage.id],
                        index_names=index_names,
                    )
                    for next_entity_id in related_entities:
                        next_entity_score = entity_score * passage_similarity
                        if next_entity_score < self.config.iteration_threshold:
                            continue
                        if next_entity_id not in activated_entities:
                            new_entities[next_entity_id] = next_entity_score
                        elif new_entities.get(next_entity_id, 0) < next_entity_score:
                            new_entities[next_entity_id] = next_entity_score
            activated_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return activated_entities

    def _calculate_passage_scores(
        self,
        question_embedding: Sequence[float],
        entity_scores: Mapping[str, float],
        index_names: list[str],
    ) -> dict[str, float]:
        """结合向量召回分数和图关联实体权重计算段落分数。"""

        hits = self.search_store.search(
            SearchQuery(
                index_names=index_names,
                mode=SearchMode.VECTOR,
                query_vector=list(question_embedding),
                top_k=50,
                num_candidates=100,
                filters={"type": "passage"},
                source_fields=self._passage_source_fields(),
            )
        )
        if not hits:
            return {}

        dpr_scores = self._min_max_normalize([hit.score for hit in hits])
        passage_ids = [hit.id for hit in hits]
        links = self.graph_store.get_entity_passage_links(
            entity_ids=list(entity_scores.keys()),
            passage_ids=passage_ids,
            index_names=index_names,
        )
        links_by_passage: dict[str, dict[str, str]] = {}
        for link in links:
            links_by_passage.setdefault(link.passage_id, {})[
                link.entity_id
            ] = link.entity_name.lower()

        passage_scores: dict[str, float] = {}
        for index, hit in enumerate(hits):
            passage_text = hit.document.text
            passage_text_lower = passage_text.lower()
            total_entity_bonus = 0.0
            for entity_id, entity_score in entity_scores.items():
                entity_name = links_by_passage.get(hit.id, {}).get(entity_id)
                if entity_name and entity_name in passage_text_lower:
                    count = passage_text_lower.count(entity_name)
                    total_entity_bonus += entity_score * math.log(1 + count)
            passage_scores[hit.id] = 0.6 * dpr_scores[index] + math.log(
                1 + total_entity_bonus
            )
        return passage_scores

    def _run_pagerank_and_filter(
        self,
        entity_scores: Mapping[str, float],
        passage_scores: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        """使用图数据库端口执行个性化 PageRank 并融合分数。"""

        all_node_ids = list(
            dict.fromkeys([*entity_scores.keys(), *passage_scores.keys()])
        )
        if not all_node_ids:
            return []

        source_weights = {
            node_id: score
            for node_id, score in {
                **entity_scores,
                **passage_scores,
            }.items()
            if score > 0.5
        }
        try:
            pagerank_scores = self.graph_store.personalized_pagerank(
                node_ids=all_node_ids,
                source_weights=source_weights,
                damping=0.85,
            )
        except Exception:
            # 图算法不可用时退回基础分数，保证检索链路可用。
            pagerank_scores = {}

        final_passages = []
        for passage_id, base_score in passage_scores.items():
            pagerank_score = pagerank_scores.get(passage_id, 0.0)
            final_score = base_score * 0.5 + pagerank_score * 0.5
            final_passages.append(
                {
                    "hash_id": passage_id,
                    "score": final_score,
                    "pagerank_score": pagerank_score,
                    "base_score": base_score,
                }
            )
        final_passages.sort(key=lambda item: item["score"], reverse=True)
        return final_passages

    def _encode_question(self, text: str) -> list[float]:
        """编码检索问题。"""

        return self._encode_texts(text, is_batch=False)

    def _encode_texts(self, texts, is_batch: bool) -> list[list[float]] | list[float]:
        """统一处理单条和批量文本向量。"""

        if isinstance(texts, str):
            texts = [texts]
            is_batch = False
        embeddings = self.embedding_provider.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        normalized = [normalize_vector(embedding) for embedding in embeddings]
        if is_batch:
            return normalized
        return normalized[0]

    def _hits_to_passages(
        self,
        hits: Sequence[SearchHit],
        include_extra_fields: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """把统一命中模型转换为旧 API 返回结构。"""

        defaults = include_extra_fields or {}
        passages = []
        for hit in hits:
            passage = hit.source
            for key, default_value in defaults.items():
                passage.setdefault(key, default_value)
            passages.append(passage)
        return passages

    def _min_max_normalize(self, scores: Sequence[float]) -> list[float]:
        """执行 Min-Max 归一化。"""

        if not scores:
            return []
        array = np.asarray(scores, dtype=float)
        minimum = float(array.min())
        maximum = float(array.max())
        if maximum == minimum:
            return [0.5] * len(scores)
        return ((array - minimum) / (maximum - minimum)).tolist()

    def _passage_source_fields(self) -> list[str]:
        """返回检索段落需要回查的字段。"""

        return [
            "hash_id",
            "text",
            "file_name",
            "file_id",
            "pages_number",
            "segment_id",
            "ori_text",
            "content_table",
            "content_image",
            "file_path",
            "bucket_name",
        ]
