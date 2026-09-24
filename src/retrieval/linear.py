from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from src.common.models import SearchDocument, SearchMode, SearchQuery
from src.common.model_providers.base import EmbeddingProvider
from src.common.search_store.base import SearchStore
from src.common.vector_utils import normalize_vector
from src.common.document_processing.base import EntityExtractor


class LinearRetriever:
    """在 ES 持久化的节点上运行作者的种子激活和个性化 PageRank。"""

    def __init__(
        self,
        config,
        embedding_provider: EmbeddingProvider,
        entity_extractor: EntityExtractor,
        search_store: SearchStore,
    ):
        self.config = config
        self.embedding = embedding_provider
        self.ner = entity_extractor
        self.store = search_store

    def retrieve(
        self, question: str, index_names: list[str], top_k: int, local: bool
    ) -> list[dict]:
        """全图模式遍历知识库；局部模式先用 ES 确定候选段落。"""

        if len(index_names) != 1:
            raise ValueError("linear 图模式一次只能查询一个知识库")
        question_vector = self._encode([question])[0]
        names = self.ner.extract_question_entities(question)
        if local:
            passages, entities, sentences, seeds = self._load_local_graph(
                question_vector, names, index_names
            )
            if not passages:
                return []
        else:
            documents = self.store.scan_documents(
                index_names, ["passage", "sentence", "entity"],
                max_documents=self.config.linear_max_nodes,
            )
            passages = [item for item in documents if item.doc_type == "passage"]
            entities = [item for item in documents if item.doc_type == "entity"]
            sentences = [item for item in documents if item.doc_type == "sentence"]
            if not passages:
                return []
            seeds = None
        if not entities and any(
            passage.metadata.get("entity_ids") for passage in passages
        ):
            raise ValueError("索引缺少实体图节点，请重新上传文件建立 LinearRAG 图索引")

        # 同一知识库中同名实体可由多个文件写入，图计算按逻辑 ID 合并。
        unique_entities = {}
        for item in sorted(entities, key=lambda document: document.id):
            entity_id = item.metadata.get("entity_id")
            if entity_id and item.vector and entity_id not in unique_entities:
                unique_entities[entity_id] = item
        named_entities = list(unique_entities.values())
        if not names or not named_entities:
            return self._dense_fallback(passages, question_vector, top_k)
        if seeds is None:
            name_vectors = self._encode(names)
            entity_vectors = np.asarray([item.vector for item in named_entities])
            seeds = {}
            for vector in name_vectors:
                similarities = entity_vectors @ vector
                index = int(np.argmax(similarities))
                entity_id = named_entities[index].metadata["entity_id"]
                seeds[entity_id] = float(similarities[index])

        entity_to_sentences: dict[str, list[SearchDocument]] = defaultdict(list)
        for sentence in sentences:
            for entity_id in sentence.metadata.get("entity_ids") or []:
                if entity_id in unique_entities:
                    entity_to_sentences[entity_id].append(sentence)
        weights = dict(seeds)
        active = {entity_id: (score, 1) for entity_id, score in seeds.items()}
        activated = dict(active)
        used_sentences: set[str] = set()
        iteration = 1
        while active and iteration < self.config.max_iterations:
            next_active: dict[str, tuple[float, int]] = {}
            for entity_id, (score, _) in active.items():
                if score < self.config.iteration_threshold:
                    continue
                candidates = [
                    sentence for sentence in entity_to_sentences[entity_id]
                    if sentence.id not in used_sentences and sentence.vector
                ]
                ranked = sorted(
                    candidates,
                    key=lambda sentence: float(np.dot(sentence.vector, question_vector)),
                    reverse=True,
                )
                for sentence in ranked[:self.config.top_k_sentence]:
                    used_sentences.add(sentence.id)
                    similarity = float(np.dot(sentence.vector, question_vector))
                    for next_id in sentence.metadata.get("entity_ids") or []:
                        next_score = score * similarity
                        if next_id not in unique_entities or next_score < self.config.iteration_threshold:
                            continue
                        weights[next_id] = weights.get(next_id, 0.0) + next_score
                        next_active[next_id] = (next_score, iteration + 1)
            activated.update(next_active)
            active = next_active
            iteration += 1

        graph = self._build_graph(passages, unique_entities)
        dense_scores = self._dense_scores(passages, question_vector)
        reset = {entity_id: max(0.0, score) for entity_id, score in weights.items()}
        for passage in passages:
            bonus = 0.0
            for entity_id, (score, tier) in activated.items():
                entity = unique_entities.get(entity_id)
                if entity:
                    count = passage.text.lower().count(entity.text.lower())
                    if count:
                        bonus += score * math.log1p(count) / max(1, tier)
            score = (
                self.config.passage_ratio * dense_scores[passage.id]
                + math.log1p(max(0.0, bonus))
            ) * self.config.passage_node_weight
            reset[passage.id] = max(0.0, score)
        scores = self._pagerank(graph, reset)
        ordered = sorted(passages, key=lambda item: scores.get(item.id, 0.0), reverse=True)
        return [
            {**item.as_source(), "score": scores.get(item.id, 0.0)}
            for item in ordered[:top_k]
        ]

    def _load_local_graph(
        self,
        question_vector: np.ndarray,
        names: list[str],
        index_names: list[str],
    ) -> tuple[
        list[SearchDocument], list[SearchDocument], list[SearchDocument], dict[str, float]
    ]:
        """合并向量与问题实体两路候选，再按预算读取局部图节点。"""

        max_passages = self.config.linear_local_max_passages
        vector_hits = self.store.search(
            SearchQuery(
                index_names=index_names,
                mode=SearchMode.VECTOR,
                query_vector=question_vector.tolist(),
                top_k=min(self.config.linear_local_candidates, max_passages),
                num_candidates=max(
                    100, self.config.linear_local_candidates * 4
                ),
                filters={"type": "passage"},
            )
        )
        seeds: dict[str, float] = {}
        if names:
            # 和 v0 一样先用实体向量找种子，不依赖段落向量能否召回目标。
            for name_vector in self._encode(names[:self.config.linear_seed_entities]):
                entity_hits = self.store.search(
                    SearchQuery(
                        index_names=index_names,
                        mode=SearchMode.VECTOR,
                        query_vector=name_vector.tolist(),
                        top_k=1,
                        num_candidates=100,
                        filters={"type": "entity"},
                    )
                )
                if entity_hits:
                    entity = entity_hits[0].document
                    entity_id = entity.metadata.get("entity_id")
                    if entity_id:
                        seeds[entity_id] = max(
                            seeds.get(entity_id, 0.0), entity_hits[0].score
                        )

        entity_hits = self.store.search_entity_passages(
            index_names,
            sorted(seeds),
            self.config.linear_passages_per_entity,
        ) if seeds else []
        # 种子关联段落优先保留；向量段落补足预算，保证稀有关系不会被截掉。
        passage_map: dict[str, SearchDocument] = {}
        for hit in [*entity_hits, *vector_hits]:
            if hit.document.doc_type == "passage" and len(passage_map) < max_passages:
                passage_map.setdefault(hit.id, hit.document)
        passages = list(passage_map.values())
        selected_ids = {
            entity_id
            for passage in passages
            for entity_id in passage.metadata.get("entity_ids") or []
        } | set(seeds)
        if not selected_ids:
            return passages, [], [], seeds

        remaining = self.config.linear_max_nodes - len(passages)
        if remaining < 0:
            raise ValueError("图节点超过 LINEAR_MAX_NODES")
        entities = self.store.scan_documents(
            index_names, ["entity"],
            {"entity_id": sorted(selected_ids)},
            remaining,
        )
        remaining -= len(entities)
        sentence_budget = min(
            self.config.linear_local_max_sentences, remaining
        )
        sentences = self.store.search_graph_nodes(
            index_names,
            "sentence",
            {"passage_id": sorted(passage_map)},
            sentence_budget,
        ) if sentence_budget else []
        return passages, entities, sentences, seeds

    def _build_graph(
        self, passages: Sequence[SearchDocument], entities: dict[str, SearchDocument]
    ) -> dict[str, dict[str, float]]:
        """复用作者实体—段落和相邻段落边，保持无向加权语义。"""

        graph: dict[str, dict[str, float]] = defaultdict(dict)
        passage_ids = {item.id for item in passages}
        for passage in passages:
            graph[passage.id]
            counts = {
                entity["id"]: passage.text.count(entity["name"])
                for entity in passage.metadata.get("entities") or []
                if entity["id"] in entities
            }
            total = sum(counts.values())
            if total:
                for entity_id, count in counts.items():
                    self._edge(graph, passage.id, entity_id, count / total)
            neighbor = passage.metadata.get("next_passage_id")
            if neighbor in passage_ids:
                self._edge(graph, passage.id, neighbor, 1.0)
        return graph

    @staticmethod
    def _edge(graph: dict, source: str, target: str, weight: float) -> None:
        graph[source][target] = weight
        graph[target][source] = weight

    def _pagerank(self, graph: dict, reset: dict[str, float]) -> dict[str, float]:
        """在应用进程迭代求解 PPR，收敛阈值与迭代上限控制耗时。"""

        nodes = list(graph)
        if not nodes:
            return {}
        teleport = np.asarray([reset.get(node, 0.0) for node in nodes], dtype=float)
        if not np.any(teleport):
            teleport[:] = 1.0
        teleport /= teleport.sum()
        index = {node: pos for pos, node in enumerate(nodes)}
        scores = teleport.copy()
        for _ in range(200):
            next_scores = (1 - self.config.damping) * teleport.copy()
            for node, neighbors in graph.items():
                pos = index[node]
                total = sum(neighbors.values())
                if total:
                    for target, weight in neighbors.items():
                        next_scores[index[target]] += (
                            self.config.damping * scores[pos] * weight / total
                        )
                else:
                    next_scores += self.config.damping * scores[pos] * teleport
            if np.abs(next_scores - scores).sum() < 1e-10:
                scores = next_scores
                break
            scores = next_scores
        return dict(zip(nodes, scores.tolist()))

    def _dense_scores(
        self, passages: Sequence[SearchDocument], vector: np.ndarray
    ) -> dict[str, float]:
        raw = {
            item.id: float(np.dot(item.vector, vector)) if item.vector else 0.0
            for item in passages
        }
        if not raw:
            return {item.id: 0.0 for item in passages}
        low, high = min(raw.values()), max(raw.values())
        return {
            item.id: (raw[item.id] - low) / (high - low) if high != low else 0.0
            for item in passages
        }

    def _dense_fallback(
        self, passages: Sequence[SearchDocument], vector: np.ndarray, top_k: int
    ) -> list[dict]:
        ranked = sorted(
            (item for item in passages if item.vector),
            key=lambda item: float(np.dot(item.vector, vector)),
            reverse=True,
        )
        return [
            {**item.as_source(), "score": float(np.dot(item.vector, vector))}
            for item in ranked[:top_k]
        ]

    def _encode(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [
            np.asarray(normalize_vector(vector))
            for vector in self.embedding.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        ]
