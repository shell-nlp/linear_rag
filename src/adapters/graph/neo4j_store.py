from __future__ import annotations

import uuid
from typing import Any, Sequence

from src.models import (
    EntityPassageLink,
    GraphBatch,
    GraphDeleteResult,
    GraphWriteResult,
    RelatedPassage,
)
from src.adapters.graph.neo4j_write_operations import (
    delete_file_nodes,
    save_graph_batches,
)
from src.interfaces.graph import GraphWriteQueue


class Neo4jGraphStore:
    """Neo4j 对 GraphStore 端口的语义化实现。"""

    def __init__(self, graph_driver, write_queue: GraphWriteQueue | None = None):
        self.graph_driver = graph_driver
        self.write_queue = write_queue

    def save_graph(self, batch: GraphBatch) -> GraphWriteResult:
        """通过写队列或直连方式保存增量图。"""

        payload = {
            "kb_name": batch.index_name,
            "node_batch": [
                {
                    "orig_id": node.orig_id,
                    "type": node.node_type,
                    "name": node.name,
                    "new_file_ids": node.file_ids,
                }
                for node in batch.nodes
            ],
            "edge_batch": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                }
                for edge in batch.edges
            ],
            "anchor_label": batch.anchor_label,
            "edge_label": batch.edge_label,
        }
        if self.write_queue is not None:
            result = self.write_queue.submit("save_graph", payload)
        else:
            result = save_graph_batches(
                neo4j_driver=self.graph_driver,
                kb_name=batch.index_name,
                node_batch=payload["node_batch"],
                edge_batch=payload["edge_batch"],
                anchor_label=batch.anchor_label,
                edge_label=batch.edge_label,
            )
        return GraphWriteResult(
            index_name=batch.index_name,
            node_count=int(result.get("node_count", len(batch.nodes))),
            edge_count=int(result.get("edge_count", len(batch.edges))),
        )

    def delete_file_nodes(
        self,
        index_name: str,
        file_ids: Sequence[str],
    ) -> GraphDeleteResult:
        """删除文件关联节点，并返回处理统计。"""

        normalized_file_ids = [str(file_id) for file_id in file_ids]
        if self.write_queue is not None:
            result = self.write_queue.submit(
                "delete_file_nodes",
                {
                    "index_name": index_name,
                    "file_ids": normalized_file_ids,
                },
            )
        else:
            result = delete_file_nodes(
                neo4j_driver=self.graph_driver,
                index_name=index_name,
                file_ids=normalized_file_ids,
            )
        return GraphDeleteResult(
            index_name=index_name,
            file_ids=normalized_file_ids,
            total_nodes=int(result.get("total_nodes", 0)),
            deleted_nodes=int(result.get("deleted_nodes", 0)),
        )

    def get_related_passages(
        self,
        entity_ids: Sequence[str],
        index_names: Sequence[str],
        limit_per_entity: int = 5,
    ) -> list[RelatedPassage]:
        """查询实体一跳可达的段落。"""

        if not entity_ids or not index_names:
            return []
        query = """
        MATCH (e {orig_id: $entity_id})-[r]-(p)
        WHERE p.type = 'passage'
        AND ANY(label IN labels(p) WHERE label IN $index_names)
        RETURN p.orig_id AS passage_id,
               p.name AS passage_name,
               p.vector AS passage_vector
        LIMIT $top_k
        """
        passages: list[RelatedPassage] = []
        with self.graph_driver.get_session() as session:
            for entity_id in entity_ids:
                records = session.run(
                    query,
                    entity_id=entity_id,
                    index_names=list(index_names),
                    top_k=limit_per_entity,
                ).data()
                for record in records:
                    passages.append(
                        RelatedPassage(
                            id=record["passage_id"],
                            text=record.get("passage_name") or "",
                            vector=record.get("passage_vector"),
                        )
                    )
        return passages

    def get_related_entities(
        self,
        passage_ids: Sequence[str],
        index_names: Sequence[str],
    ) -> list[str]:
        """查询段落关联的实体 ID。"""

        if not passage_ids or not index_names:
            return []
        query = """
        MATCH (p)-[r]-(e)
        WHERE p.orig_id IN $passage_ids
        AND e.type = 'entity'
        AND ANY(label IN labels(e) WHERE label IN $index_names)
        RETURN DISTINCT e.orig_id AS entity_id
        """
        with self.graph_driver.get_session() as session:
            records = session.run(
                query,
                passage_ids=list(passage_ids),
                index_names=list(index_names),
            ).data()
        return [
            record["entity_id"]
            for record in records
            if record.get("entity_id")
        ]

    def get_entity_passage_links(
        self,
        entity_ids: Sequence[str],
        passage_ids: Sequence[str],
        index_names: Sequence[str],
    ) -> list[EntityPassageLink]:
        """批量查询实体和段落之间的关联事实。"""

        if not entity_ids or not passage_ids or not index_names:
            return []
        query = """
        UNWIND $entity_ids AS entity_id
        UNWIND $passage_ids AS passage_id
        MATCH (e {orig_id: entity_id})-[r]-(p {orig_id: passage_id})
        WHERE e.type = 'entity'
        AND p.type = 'passage'
        AND ANY(label IN labels(e) WHERE label IN $index_names)
        RETURN e.orig_id AS entity_id,
               p.orig_id AS passage_id,
               e.name AS entity_name
        """
        with self.graph_driver.get_session() as session:
            records = session.run(
                query,
                entity_ids=list(entity_ids),
                passage_ids=list(passage_ids),
                index_names=list(index_names),
            ).data()
        return [
            EntityPassageLink(
                entity_id=record["entity_id"],
                passage_id=record["passage_id"],
                entity_name=record.get("entity_name") or "",
            )
            for record in records
        ]

    def personalized_pagerank(
        self,
        node_ids: Sequence[str],
        source_weights: dict[str, float],
        damping: float = 0.85,
    ) -> dict[str, float]:
        """使用 Neo4j GDS 执行个性化 PageRank。"""

        unique_node_ids = list(dict.fromkeys(node_ids))
        if not unique_node_ids:
            return {}

        graph_name = f"rag_gds_ppr_{uuid.uuid4().hex[:8]}"
        # 投影当前候选节点组成的子图，避免对全图执行算法。
        node_query = """
        MATCH (n)
        WHERE n.orig_id IN $node_ids
        RETURN id(n) AS id
        """
        edge_query = """
        MATCH (n)-[r]-(m)
        WHERE n.orig_id IN $node_ids AND m.orig_id IN $node_ids
        RETURN id(n) AS source, id(m) AS target
        """
        source_ids = [
            node_id
            for node_id in source_weights
            if node_id in unique_node_ids
        ]

        with self.graph_driver.get_session() as session:
            session.run(
                "CALL gds.graph.drop($graph_name, false)",
                graph_name=graph_name,
            )
            try:
                # source_ids 为空时使用全图 PageRank，否则使用个性化 PageRank。
                session.run(
                    """
                    CALL gds.graph.project.cypher(
                        $graph_name,
                        $node_query,
                        $edge_query,
                        { parameters: { node_ids: $node_ids } }
                    )
                    """,
                    graph_name=graph_name,
                    node_query=node_query,
                    edge_query=edge_query,
                    node_ids=unique_node_ids,
                )

                if source_ids:
                    internal_records = session.run(
                        """
                        MATCH (n)
                        WHERE n.orig_id IN $source_ids
                        RETURN id(n) AS internal_id
                        """,
                        source_ids=source_ids,
                    ).data()
                    internal_ids = [
                        record["internal_id"]
                        for record in internal_records
                    ]
                    pagerank_records = session.run(
                        """
                        CALL gds.pageRank.stream($graph_name, {
                            sourceNodes: $source_nodes,
                            dampingFactor: $damping
                        })
                        YIELD nodeId, score
                        RETURN gds.util.asNode(nodeId).orig_id AS id, score
                        """,
                        graph_name=graph_name,
                        source_nodes=internal_ids,
                        damping=damping,
                    ).data()
                else:
                    pagerank_records = session.run(
                        """
                        CALL gds.pageRank.stream($graph_name, {
                            dampingFactor: $damping
                        })
                        YIELD nodeId, score
                        RETURN gds.util.asNode(nodeId).orig_id AS id, score
                        """,
                        graph_name=graph_name,
                        damping=damping,
                    ).data()
            finally:
                session.run(
                    "CALL gds.graph.drop($graph_name, false)",
                    graph_name=graph_name,
                )

        return {
            record["id"]: float(record["score"])
            for record in pagerank_records
            if record.get("id")
        }
