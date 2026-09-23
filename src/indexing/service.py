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

        documents = self._build_passage_documents(passages)
        if not documents:
            return {"status": "skipped", "new_passages": 0, "new_entities": 0}

        passage_texts = {document.id: document.text for document in documents}
        passage_entities = self.entity_extractor.extract_passage_entities(
            passage_texts,
            self.config.max_workers,
        )
        unique_entity_ids = self._attach_entity_metadata(
            documents,
            passage_entities,
        )
        self._attach_embeddings(documents)
        self._delete_legacy_passage_ids(kb_name, documents)
        write_result = self.search_store.upsert_documents(
            index_name=kb_name,
            documents=documents,
            refresh=True,
        )
        return {
            "status": "success",
            "new_passages": write_result.success,
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

    def _delete_legacy_passage_ids(
        self,
        index_name: str,
        documents: list[SearchDocument],
    ) -> None:
        """删除旧版纯文本哈希文档，避免重新索引后产生重复段落。"""

        legacy_ids = list(
            dict.fromkeys(
                legacy_id
                for document in documents
                if (legacy_id := compute_mdhash_id(document.text, prefix="passage-"))
                != document.id
            )
        )
        if legacy_ids:
            self.search_store.delete_documents_by_ids(
                index_name=index_name,
                ids=legacy_ids,
                refresh=False,
            )

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

    def _attach_embeddings(self, documents: list[SearchDocument]) -> None:
        """批量生成段落向量并归一化。"""

        embeddings = self.embedding_provider.encode(
            [document.text for document in documents],
            batch_size=self.config.batch_size,
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
        """按可靠性选择文件标识，保证段落 ID 稳定且跨文件隔离。"""

        file_id = metadata.get("file_id")
        if file_id:
            return str(file_id)
        bucket_name = metadata.get("bucket_name") or ""
        file_path = metadata.get("file_path") or metadata.get("file_name") or "unknown"
        return f"{bucket_name}:{file_path}"

    @staticmethod
    def _segment_sort_key(document: SearchDocument) -> tuple[int, str]:
        """优先按数字段落序号排序，无法转换时保持稳定字符串顺序。"""

        segment_id = document.metadata.get("segment_id")
        try:
            return int(segment_id), document.id
        except (TypeError, ValueError):
            return 0, f"{segment_id}:{document.id}"
