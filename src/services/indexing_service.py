from __future__ import annotations

from typing import Any, Mapping

from src.services.graph_builder import GraphBuilder
from src.services.vector_utils import normalize_vector
from src.common.utils import compute_mdhash_id
from src.models import SearchDocument
from src.interfaces.embedding import EmbeddingProvider
from src.interfaces.entity import EntityExtractor
from src.interfaces.graph import GraphStore
from src.interfaces.search import SearchStore


class IndexingService:
    """负责把段落和实体写入搜索库，并构建、写入知识图谱。"""

    def __init__(
        self,
        config,
        embedding_provider: EmbeddingProvider,
        search_store: SearchStore,
        graph_store: GraphStore,
        entity_extractor: EntityExtractor,
        graph_builder: GraphBuilder | None = None,
    ):
        """通过端口注入所有外部依赖，便于替换搜索库、图库和模型。"""

        self.config = config
        self.embedding_provider = embedding_provider
        self.search_store = search_store
        self.graph_store = graph_store
        self.entity_extractor = entity_extractor
        self.graph_builder = graph_builder or GraphBuilder()

    def index(self, passages: Mapping[str, list[Any]], kb_name: str) -> dict[str, Any]:
        """执行完整的增量索引流程。"""

        new_passages, all_passages = self._upsert_passages(
            passages=passages,
            doc_type="passage",
            index_name=kb_name,
        )
        if not new_passages:
            return {
                "status": "skipped",
                "new_passages": 0,
                "new_entities": 0,
            }

        passage_entities = self.entity_extractor.extract_passage_entities(
            new_passages,
            self.config.max_workers,
        )
        entity_node_info = self._extract_entity_info(
            passage_entities,
            new_passages,
            passages,
        )
        new_entities, all_entities = self._upsert_passages(
            passages=entity_node_info,
            doc_type="entity",
            index_name=kb_name,
        )
        graph_batch = self.graph_builder.build(
            kb_name=kb_name,
            new_hash_id_to_passage=new_passages,
            all_hash_id_to_passage=all_passages,
            new_passage_hash_id_to_entities=passage_entities,
            all_hash_id_to_entity=all_entities,
            entity_node_info=entity_node_info,
            passages=passages,
        )
        graph_result = self.graph_store.save_graph(graph_batch)
        return {
            "status": "success",
            "new_passages": len(new_passages),
            "new_entities": len(new_entities),
            "graph_nodes": graph_result.node_count,
            "graph_edges": graph_result.edge_count,
        }

    def delete_files(self, index_name: str, file_ids: list[str]) -> dict[str, Any]:
        """同时删除搜索文档和图节点。"""

        normalized_ids = [str(file_id) for file_id in file_ids]
        search_result = self.search_store.delete_documents_by_filters(
            index_name=index_name,
            filters={"file_id": normalized_ids},
            refresh=True,
        )
        graph_result = self.graph_store.delete_file_nodes(
            index_name=index_name,
            file_ids=normalized_ids,
        )
        return {
            "deleted_documents": search_result.success,
            "failed_documents": search_result.failed,
            "total_graph_nodes": graph_result.total_nodes,
            "deleted_graph_nodes": graph_result.deleted_nodes,
        }

    def _upsert_passages(
        self,
        passages: Mapping[str, list[Any]],
        doc_type: str,
        index_name: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """对文本去重、向量化并批量写入搜索库。"""

        if not passages or not passages.get("text"):
            return {}, {}

        meta_fields = [
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
        all_documents: dict[str, SearchDocument] = {}
        text_only_map: dict[str, str] = {}
        for index, text in enumerate(passages["text"]):
            hash_id = compute_mdhash_id(text, prefix=f"{doc_type}-")
            metadata = {}
            for field in meta_fields:
                values = passages.get(field, [])
                metadata[field] = values[index] if index < len(values) else None
            all_documents[hash_id] = SearchDocument(
                id=hash_id,
                text=text,
                doc_type=doc_type,
                metadata=metadata,
            )
            text_only_map[hash_id] = text

        existing_ids = self.search_store.get_existing_ids(
            index_name,
            list(all_documents.keys()),
        )
        missing_ids = [
            hash_id
            for hash_id in all_documents
            if hash_id not in existing_ids
        ]
        if not missing_ids:
            return {}, text_only_map

        texts_to_encode = [all_documents[hash_id].text for hash_id in missing_ids]
        embeddings = self.embedding_provider.encode(
            texts_to_encode,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        documents_to_write = []
        for hash_id, embedding in zip(missing_ids, embeddings):
            document = all_documents[hash_id]
            document.vector = normalize_vector(embedding)
            documents_to_write.append(document)
        self.search_store.upsert_documents(
            index_name=index_name,
            documents=documents_to_write,
            refresh=True,
        )
        inserted_map = {
            hash_id: all_documents[hash_id].text
            for hash_id in missing_ids
        }
        return inserted_map, text_only_map

    def _extract_entity_info(
        self,
        new_passage_hash_id_to_entities: Mapping[str, list[str]],
        new_hash_id_to_passage: Mapping[str, str],
        passages: Mapping[str, list[Any]],
    ) -> dict[str, list[Any]]:
        """把实体映射为可写入搜索库的平行数组结构。"""

        result: dict[str, list[Any]] = {
            "text": [],
            "file_name": [],
            "file_id": [],
        }
        text_to_index = {
            text: index for index, text in enumerate(passages.get("text", []))
        }
        file_names = passages.get("file_name", [])
        file_ids = passages.get("file_id", [])

        for hash_id, entities in new_passage_hash_id_to_entities.items():
            passage_text = new_hash_id_to_passage.get(hash_id)
            if passage_text is None:
                continue
            source_index = text_to_index.get(passage_text)
            if source_index is None:
                continue
            file_name = (
                file_names[source_index]
                if source_index < len(file_names)
                else "Unknown"
            )
            file_id = (
                file_ids[source_index]
                if source_index < len(file_ids)
                else "Unknown"
            )
            for entity in entities:
                result["text"].append(entity)
                result["file_name"].append(file_name)
                result["file_id"].append(file_id)
        return result
