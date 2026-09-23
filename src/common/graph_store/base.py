from __future__ import annotations

from typing import Protocol, Sequence

from src.common.models import (
    EntityPassageLink,
    GraphBatch,
    GraphDeleteResult,
    GraphWriteResult,
    RelatedPassage,
)


class GraphStore(Protocol):
    """图数据库端口，只暴露语义级操作，不暴露 Cypher 或 SDK 对象。"""

    def save_graph(self, batch: GraphBatch) -> GraphWriteResult:
        """写入增量图数据。"""

        ...

    def delete_file_nodes(
        self,
        index_name: str,
        file_ids: Sequence[str],
    ) -> GraphDeleteResult:
        """删除指定文件关联的图节点。"""

        ...

    def get_related_passages(
        self,
        entity_ids: Sequence[str],
        index_names: Sequence[str],
        limit_per_entity: int = 5,
    ) -> list[RelatedPassage]:
        """查询实体一跳关联的段落。"""

        ...

    def get_related_entities(
        self,
        passage_ids: Sequence[str],
        index_names: Sequence[str],
    ) -> list[str]:
        """查询段落关联的实体 ID。"""

        ...

    def get_entity_passage_links(
        self,
        entity_ids: Sequence[str],
        passage_ids: Sequence[str],
        index_names: Sequence[str],
    ) -> list[EntityPassageLink]:
        """查询实体与段落的关联关系。"""

        ...

    def personalized_pagerank(
        self,
        node_ids: Sequence[str],
        source_weights: dict[str, float],
        damping: float = 0.85,
    ) -> dict[str, float]:
        """执行个性化 PageRank，返回节点分数。"""

        ...
