from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from src.common.models import SearchHit, SearchMode, SearchQuery
from src.common.model_providers.base import EmbeddingProvider
from src.common.search_store.base import SearchStore
from src.common.vector_utils import normalize_vector


class RetrievalService:
    """执行向量、BM25、混合检索以及基于 ES 字段的实体扩展。"""

    def __init__(self, config, embedding_provider, search_store: SearchStore):
        """注入统一搜索端口和向量模型。"""

        self.config = config
        self.embedding_provider: EmbeddingProvider = embedding_provider
        self.search_store = search_store

    def retrieve(
        self,
        question: str,
        index_names: list[str],
        top_k: int = 5,
        search_mode: SearchMode = SearchMode.VECTOR,
    ) -> list[dict[str, Any]]:
        """按指定模式主召回，再融合实体关联段落和相邻段落。"""

        query_vector = self._encode_question(question, search_mode)
        primary_hits = self._search_passages(
            question,
            query_vector,
            index_names,
            search_mode,
            max(top_k * 4, 20),
        )
        if not primary_hits:
            return []

        entity_hits: list[SearchHit] = []
        if getattr(self.config, "entity_expansion_enabled", True):
            entity_ids = self._select_entity_ids(primary_hits)
            if entity_ids:
                entity_hits = self._search_related_entities(
                    index_names=index_names,
                    entity_ids=entity_ids,
                    search_mode=search_mode,
                )

        neighbor_hits: list[SearchHit] = []
        if getattr(self.config, "neighbor_expansion_enabled", True):
            neighbor_hits = self._load_neighbor_hits(
                index_names,
                [*primary_hits, *entity_hits],
            )
        return self._fuse_hits(primary_hits, entity_hits, neighbor_hits)[:top_k]

    def _search_passages(
        self,
        question: str,
        query_vector: list[float] | None,
        index_names: list[str],
        search_mode: SearchMode,
        top_k: int,
    ) -> list[SearchHit]:
        """通过统一搜索端口查询段落，确保调用方选择的模式始终生效。"""

        return self.search_store.search(
            SearchQuery(
                index_names=index_names,
                mode=search_mode,
                query_text=question,
                query_vector=query_vector,
                top_k=top_k,
                num_candidates=max(top_k * 4, 100),
                filters={"type": "passage"},
                source_fields=self._passage_source_fields(),
            )
        )

    def _search_related_entities(
        self,
        index_names: list[str],
        entity_ids: list[str],
        search_mode: SearchMode,
    ) -> list[SearchHit]:
        """使用实体 ID 倒排索引召回关联段落，不重复约束原问题文本。"""

        return self.search_store.search(
            SearchQuery(
                index_names=index_names,
                mode=search_mode,
                top_k=getattr(self.config, "entity_expansion_top_k", 50),
                filters={"type": "passage", "entity_ids": entity_ids},
                source_fields=self._passage_source_fields(),
                filter_only=True,
            )
        )

    def _select_entity_ids(self, hits: Sequence[SearchHit]) -> list[str]:
        """按主召回排名和段落内重要度选择有限数量的扩展实体。"""

        scores: dict[str, float] = defaultdict(float)
        for rank, hit in enumerate(hits, start=1):
            rank_weight = 1.0 / rank
            entities = hit.document.metadata.get("entities") or []
            if entities:
                for entity in entities:
                    entity_id = entity.get("id")
                    if entity_id:
                        scores[entity_id] += rank_weight * float(
                            entity.get("importance") or 1.0
                        )
                continue
            for entity_id in hit.document.metadata.get("entity_ids") or []:
                scores[entity_id] += rank_weight
        max_entities = getattr(self.config, "entity_expansion_max_entities", 20)
        return [
            entity_id
            for entity_id, _ in sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            )[:max_entities]
        ]

    def _load_neighbor_hits(
        self,
        index_names: list[str],
        hits: Sequence[SearchHit],
    ) -> list[SearchHit]:
        """批量回查候选段落的前后段落，避免逐条访问搜索数据库。"""

        source_ids = {hit.id for hit in hits}
        neighbor_ids = []
        for hit in hits:
            metadata = hit.document.metadata
            neighbor_ids.extend(
                passage_id
                for passage_id in (
                    metadata.get("previous_passage_id"),
                    metadata.get("next_passage_id"),
                )
                if passage_id and passage_id not in source_ids
            )
        unique_ids = list(dict.fromkeys(neighbor_ids))
        documents = self.search_store.get_documents_by_ids(index_names, unique_ids)
        return [
            SearchHit(id=passage_id, score=0.0, document=documents[passage_id])
            for passage_id in unique_ids
            if passage_id in documents
        ]

    @staticmethod
    def _fuse_hits(
        primary_hits: Sequence[SearchHit],
        entity_hits: Sequence[SearchHit],
        neighbor_hits: Sequence[SearchHit],
    ) -> list[dict[str, Any]]:
        """使用加权 RRF 融合主召回、实体扩展和相邻段落候选。"""

        documents = {}
        fused_scores: dict[str, float] = defaultdict(float)
        base_scores: dict[str, float] = {}
        entity_scores: dict[str, float] = {}
        for hits, weight, raw_scores in (
            (primary_hits, 1.0, base_scores),
            (entity_hits, 0.7, entity_scores),
            (neighbor_hits, 0.25, {}),
        ):
            for rank, hit in enumerate(hits, start=1):
                documents[hit.id] = hit.document
                fused_scores[hit.id] += weight / (60 + rank)
                raw_scores[hit.id] = hit.score

        results = []
        for passage_id, score in fused_scores.items():
            source = documents[passage_id].as_source()
            source.update(
                {
                    "score": score,
                    "base_score": base_scores.get(passage_id),
                    "entity_expansion_score": entity_scores.get(passage_id),
                }
            )
            results.append(source)
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def _encode_question(
        self,
        question: str,
        search_mode: SearchMode,
    ) -> list[float] | None:
        """仅在向量或混合检索时生成问题向量。"""

        if search_mode == SearchMode.BM25:
            return None
        prompt = f"下面是一个问题，从数据库中检索到问题相关的段落\n问题：{question}"
        embedding = self.embedding_provider.encode(
            [prompt],
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return normalize_vector(embedding)

    @staticmethod
    def _passage_source_fields() -> list[str]:
        """返回检索和实体扩展所需的完整段落字段。"""

        return [
            "hash_id", "text", "file_name", "file_id", "pages_number",
            "segment_id", "ori_text", "content_table", "content_image",
            "file_path", "bucket_name", "entities", "entity_ids",
            "entity_names", "previous_passage_id", "next_passage_id",
        ]
