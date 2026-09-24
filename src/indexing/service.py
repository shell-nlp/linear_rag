from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping

from src.common.document_processing.base import EntityExtractor
from src.common.models import SearchDocument
from src.common.model_providers.base import EmbeddingProvider
from src.common.search_store.base import SearchStore
from src.common.vector_utils import normalize_vector
from src.utils import compute_mdhash_id


class IndexingService:
    """负责预计算段落关系元数据，并把完整检索文档写入搜索库。"""

    _META_FIELDS = [
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

    def __init__(
        self,
        config,
        embedding_provider: EmbeddingProvider,
        search_store: SearchStore,
        entity_extractor: EntityExtractor,
    ):
        """注入模型、搜索数据库和实体抽取能力，不依赖具体数据库实现。"""

        self.config = config
        self.embedding_provider = embedding_provider
        self.search_store = search_store
        self.entity_extractor = entity_extractor

    def index(self, passages: Mapping[str, list[Any]], kb_name: str) -> dict[str, Any]:
        """预计算实体和相邻段落元数据，并覆盖写入搜索库。"""

        documents, unique_entity_ids = self.prepare_documents(passages)
        return self.write_documents(kb_name, documents, unique_entity_ids)

    def prepare_documents(
        self,
        passages: Mapping[str, list[Any]],
    ) -> tuple[list[SearchDocument], set[str]]:
        """完成实体抽取和向量化，但不向搜索数据库提交数据。"""

        documents = self._build_passage_documents(passages)
        if not documents:
            return [], set()

        passage_texts = {document.id: document.text for document in documents}
        passage_entities, sentence_entities = (
            self.entity_extractor.extract_graph_entities(
                passage_texts, self.config.max_workers
            )
        )
        unique_entity_ids = self._attach_entity_metadata(
            documents,
            passage_entities,
        )
        self._attach_embeddings(documents)
        documents.extend(self._build_graph_documents(documents, sentence_entities))
        return documents, unique_entity_ids

    def _build_graph_documents(
        self,
        passages: list[SearchDocument],
        sentence_entities: Mapping[str, Mapping[str, list[str]]],
    ) -> list[SearchDocument]:
        """实体和句子节点随所属文件一起写入，删除时无需全局引用计数。"""

        if not sentence_entities:
            return []
        grouped: dict[str, list[SearchDocument]] = defaultdict(list)
        for passage in passages:
            grouped[self._source_key(passage.metadata)].append(passage)
        graph_documents = []
        for file_key, file_passages in grouped.items():
            source = file_passages[0].metadata
            names = sorted({
                item["name"]
                for passage in file_passages
                for item in passage.metadata["entities"]
            })
            # 句子按所属段落建立 ID；相同句子在不同文件中不会相互删除。
            sentence_items = []
            for passage in file_passages:
                for sentence, entities in sentence_entities.get(passage.id, {}).items():
                    sentence_items.append((passage, sentence, entities))
                    names.extend(entities)
            names = sorted(set(names))
            texts = [*names, *(text for _, text, _ in sentence_items)]
            if not texts:
                continue
            vectors = self.embedding_provider.encode(
                texts,
                # 图节点批次独立限制，避免实体和句子过多时单次请求过大。
                batch_size=min(
                    self.config.batch_size,
                    self.config.linear_embedding_batch_size,
                ),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            shared = {
                "file_id": source.get("file_id"),
                "file_path": source.get("file_path"),
                "bucket_name": source.get("bucket_name"),
            }
            for index, name in enumerate(names):
                graph_documents.append(SearchDocument(
                    id=compute_mdhash_id(f"{file_key}:{name}", prefix="entity-node-"),
                    text=name,
                    vector=normalize_vector(vectors[index]),
                    doc_type="entity",
                    metadata={
                        **shared,
                        "entity_id": compute_mdhash_id(name.casefold(), prefix="entity-"),
                    },
                ))
            for index, (passage, text, entities) in enumerate(sentence_items):
                graph_documents.append(SearchDocument(
                    id=compute_mdhash_id(
                        f"{passage.id}:{text}", prefix="sentence-"
                    ),
                    text=text,
                    vector=normalize_vector(vectors[len(names) + index]),
                    doc_type="sentence",
                    metadata={
                        **shared,
                        "passage_id": passage.id,
                        "entity_ids": [
                            compute_mdhash_id(name.casefold(), prefix="entity-")
                            for name in entities
                        ],
                    },
                ))
        return graph_documents

    def write_documents(
        self,
        kb_name: str,
        documents: list[SearchDocument],
        unique_entity_ids: set[str],
    ) -> dict[str, Any]:
        """提交准备好的段落文档，返回统一写入统计。"""

        if not documents:
            return {"status": "skipped", "new_passages": 0, "new_entities": 0}
        write_result = self.search_store.upsert_documents(
            index_name=kb_name,
            documents=documents,
            refresh=True,
        )
        return {
            "status": "success",
            "new_passages": sum(document.doc_type == "passage" for document in documents),
            "new_entities": len(unique_entity_ids),
            "failed_passages": write_result.failed,
        }

    def delete_files(self, index_name: str, file_ids: list[str]) -> dict[str, Any]:
        """按文件 ID 删除搜索文档，关系元数据随文档一起删除。"""

        normalized_ids = [str(file_id) for file_id in file_ids]
        result = self.search_store.delete_documents_by_filters(
            index_name=index_name,
            filters={"file_id": normalized_ids},
            refresh=True,
        )
        return {
            "deleted_documents": result.success,
            "failed_documents": result.failed,
        }

    def _build_passage_documents(
        self,
        passages: Mapping[str, list[Any]],
    ) -> list[SearchDocument]:
        """构造段落文档，并按同一文件内的顺序计算前后段落 ID。"""

        texts = passages.get("text", [])
        documents: list[SearchDocument] = []
        for index, text in enumerate(texts):
            metadata = {
                field: self._value_at(passages.get(field, []), index)
                for field in self._META_FIELDS
            }
            source_key = self._source_key(metadata)
            segment_id = metadata.get("segment_id")
            passage_key = f"{source_key}:{segment_id if segment_id is not None else index}:{text}"
            documents.append(
                SearchDocument(
                    # 文件来源和段落序号用于隔离不同文件中的相同文本。
                    id=compute_mdhash_id(passage_key, prefix="passage-"),
                    text=text,
                    doc_type="passage",
                    metadata=metadata,
                )
            )

        grouped_documents: dict[str, list[SearchDocument]] = defaultdict(list)
        for document in documents:
            grouped_documents[self._source_key(document.metadata)].append(document)
        for file_documents in grouped_documents.values():
            file_documents.sort(key=self._segment_sort_key)
            for index, document in enumerate(file_documents):
                document.metadata["previous_passage_id"] = (
                    file_documents[index - 1].id if index > 0 else None
                )
                document.metadata["next_passage_id"] = (
                    file_documents[index + 1].id
                    if index + 1 < len(file_documents)
                    else None
                )
        return documents

    def _attach_entity_metadata(
        self,
        documents: list[SearchDocument],
        passage_entities: Mapping[str, list[str]],
    ) -> set[str]:
        """保存实体列表、段落内频次和局部重要度，供 ES 倒排扩展使用。"""

        unique_entity_ids: set[str] = set()
        for document in documents:
            raw_entities = [
                entity.strip()
                for entity in passage_entities.get(document.id, [])
                if entity and entity.strip()
            ]
            entity_counts = Counter(raw_entities)
            total_mentions = sum(entity_counts.values())
            entities = []
            for entity_name, count in sorted(entity_counts.items()):
                entity_id = compute_mdhash_id(
                    entity_name.casefold(),
                    prefix="entity-",
                )
                # 使用对数频次抑制重复提及，结果只表示当前段落内的重要度。
                importance = (
                    math.log1p(count) / math.log1p(total_mentions)
                    if total_mentions
                    else 0.0
                )
                entities.append(
                    {
                        "id": entity_id,
                        "name": entity_name,
                        "count": count,
                        "importance": importance,
                    }
                )
                unique_entity_ids.add(entity_id)
            document.metadata["entities"] = entities
            document.metadata["entity_ids"] = [item["id"] for item in entities]
            document.metadata["entity_names"] = [item["name"] for item in entities]
        return unique_entity_ids

    def delete_nodes(self, index_name: str, node_ids: list[str]) -> None:
        """按本次生成的段落、句子和实体节点 ID 精确回滚。"""

        if node_ids:
            self.search_store.delete_documents_by_ids(
                index_name=index_name,
                ids=node_ids,
                refresh=True,
            )

    def _attach_embeddings(self, documents: list[SearchDocument]) -> None:
        """批量生成段落向量并归一化。"""

        embeddings = self.embedding_provider.encode(
            [document.text for document in documents],
            batch_size=min(
                self.config.batch_size,
                self.config.linear_embedding_batch_size,
            ),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for document, embedding in zip(documents, embeddings):
            document.vector = normalize_vector(embedding)

    @staticmethod
    def _value_at(values: list[Any], index: int) -> Any:
        """安全读取平行数组，缺失字段统一返回 None。"""

        return values[index] if index < len(values) else None

    @staticmethod
    def _source_key(metadata: Mapping[str, Any]) -> str:
        """组合对象地址和文件 ID，保证段落 ID 稳定且跨文件隔离。"""

        bucket_name = metadata.get("bucket_name") or ""
        file_path = metadata.get("file_path") or metadata.get("file_name") or "unknown"
        file_id = metadata.get("file_id") or ""
        return f"{bucket_name}:{file_path}:{file_id}"

    @staticmethod
    def _segment_sort_key(document: SearchDocument) -> tuple[int, str]:
        """优先按数字段落序号排序，无法转换时保持稳定字符串顺序。"""

        segment_id = document.metadata.get("segment_id")
        try:
            return int(segment_id), document.id
        except (TypeError, ValueError):
            return 0, f"{segment_id}:{document.id}"
