from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping

import igraph as ig

from src.common.models import GraphBatch, GraphEdge, GraphNode


class GraphBuilder:
    """在内存中构建段落和实体子图，并输出与图数据库无关的批次模型。"""

    def build(
        self,
        kb_name: str,
        new_hash_id_to_passage: Mapping[str, str],
        all_hash_id_to_passage: Mapping[str, str],
        new_passage_hash_id_to_entities: Mapping[str, list[str]],
        all_hash_id_to_entity: Mapping[str, str],
        entity_node_info: Mapping[str, list[Any]],
        passages: Mapping[str, list[Any]],
    ) -> GraphBatch:
        """构建增量图并转换为统一的 GraphBatch。"""

        graph = ig.Graph(directed=False)
        node_to_node_stats: dict[str, dict[str, float]] = defaultdict(dict)
        node_to_node_stats = self._add_entity_to_passage_edges(
            new_passage_hash_id_to_entities,
            new_hash_id_to_passage,
            all_hash_id_to_entity,
            node_to_node_stats,
        )
        node_to_node_stats = self._add_adjacent_passage_edges(
            all_hash_id_to_passage,
            node_to_node_stats,
        )

        active_nodes = set(all_hash_id_to_passage.keys())
        for source, targets in node_to_node_stats.items():
            active_nodes.add(source)
            active_nodes.update(targets.keys())

        graph = self._augment_graph(
            active_nodes,
            all_hash_id_to_entity,
            all_hash_id_to_passage,
            graph,
            node_to_node_stats,
        )
        return self._to_graph_batch(
            kb_name=kb_name,
            graph=graph,
            entity_node_info=entity_node_info,
            passages=passages,
        )

    def _augment_graph(
        self,
        restrict_to_nodes: set[str],
        entity_hash_id_to_text: Mapping[str, str],
        passage_hash_id_to_text: Mapping[str, str],
        graph: ig.Graph,
        node_to_node_stats: Mapping[str, Mapping[str, float]],
    ) -> ig.Graph:
        """向内存图补充节点和关系。"""

        graph = self._add_nodes(
            restrict_to_nodes,
            entity_hash_id_to_text,
            passage_hash_id_to_text,
            graph,
        )
        return self._add_edges(graph, node_to_node_stats)

    def _add_nodes(
        self,
        restrict_to_nodes: set[str],
        entity_hash_id_to_text: Mapping[str, str],
        passage_hash_id_to_text: Mapping[str, str],
        graph: ig.Graph,
    ) -> ig.Graph:
        """只添加本次活跃集合中存在的节点。"""

        existing_nodes = {
            vertex["name"]
            for vertex in graph.vs
            if "name" in vertex.attributes()
        }
        all_hash_id_to_text = {
            **entity_hash_id_to_text,
            **passage_hash_id_to_text,
        }
        target_ids = restrict_to_nodes.intersection(all_hash_id_to_text.keys())
        for hash_id in target_ids:
            if hash_id not in existing_nodes:
                graph.add_vertex(
                    name=hash_id,
                    content=all_hash_id_to_text[hash_id],
                )
        return graph

    def _add_edges(
        self,
        graph: ig.Graph,
        node_to_node_stats: Mapping[str, Mapping[str, float]],
    ) -> ig.Graph:
        """将关系统计写入 igraph。"""

        vertex_names = {
            vertex["name"]
            for vertex in graph.vs
            if "name" in vertex.attributes()
        }
        edges = []
        weights = []
        for source, targets in node_to_node_stats.items():
            for target, weight in targets.items():
                if source == target:
                    continue
                if source in vertex_names and target in vertex_names:
                    edges.append((source, target))
                    weights.append(weight)
        if edges:
            graph.add_edges(edges)
            graph.es["weight"] = weights
        return graph

    def _add_adjacent_passage_edges(
        self,
        all_hash_id_to_passage: Mapping[str, str],
        node_to_node_stats: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """按照段落输入顺序建立相邻段落边。"""

        node_ids = list(all_hash_id_to_passage.keys())
        for index in range(len(node_ids) - 1):
            current_node = node_ids[index]
            next_node = node_ids[index + 1]
            node_to_node_stats.setdefault(current_node, {})[next_node] = 1.0
        return node_to_node_stats

    def _add_entity_to_passage_edges(
        self,
        new_passage_hash_id_to_entities: Mapping[str, list[str]],
        new_hash_id_to_passage: Mapping[str, str],
        all_hash_id_to_entity: Mapping[str, str],
        node_to_node_stats: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """根据实体在段落中的出现次数计算段落和实体之间的权重。"""

        entity_text_to_hash_id = {
            value: key for key, value in all_hash_id_to_entity.items()
        }
        passage_to_entity_count: dict[tuple[str, str], int] = {}
        passage_to_all_score: dict[str, int] = defaultdict(int)

        for passage_hash_id, entities in new_passage_hash_id_to_entities.items():
            passage = new_hash_id_to_passage.get(passage_hash_id)
            if passage is None:
                continue
            for entity_text in set(entities):
                entity_hash_id = entity_text_to_hash_id.get(entity_text)
                if not entity_hash_id:
                    continue
                count = passage.count(entity_text)
                if count <= 0:
                    continue
                passage_to_entity_count[(passage_hash_id, entity_hash_id)] = count
                passage_to_all_score[passage_hash_id] += count

        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total_count = passage_to_all_score[passage_hash_id]
            if total_count > 0:
                node_to_node_stats.setdefault(passage_hash_id, {})[
                    entity_hash_id
                ] = count / total_count
        return node_to_node_stats

    def _to_graph_batch(
        self,
        kb_name: str,
        graph: ig.Graph,
        entity_node_info: Mapping[str, list[Any]],
        passages: Mapping[str, list[Any]],
    ) -> GraphBatch:
        """把 igraph 对象转换为图数据库无关的批次模型。"""

        text_to_file_ids: dict[str, set[str]] = {}
        if entity_node_info and "text" in entity_node_info:
            for text, file_id in zip(
                entity_node_info["text"],
                entity_node_info["file_id"],
            ):
                text_to_file_ids.setdefault(text, set()).add(str(file_id))

        default_file_ids = passages.get("file_id", []) if passages else []
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        if graph.vcount() == 0 or "name" not in graph.vertex_attributes():
            return GraphBatch(index_name=kb_name, nodes=nodes, edges=edges)

        nodes_list, edges_list = graph.to_dict_list(use_vids=False)
        for node_item in nodes_list:
            unique_id = node_item.get("name")
            if not unique_id:
                continue
            clean_id = str(unique_id).strip()
            match = re.match(r"^([^-]+)", clean_id)
            node_type = match.group(1) if match else "Unknown"
            content = node_item.get("content", "")
            target_file_ids = text_to_file_ids.get(content, set())
            if not target_file_ids and default_file_ids:
                target_file_ids = {str(file_id) for file_id in default_file_ids}
            nodes.append(
                GraphNode(
                    orig_id=clean_id,
                    node_type=node_type,
                    name=content,
                    file_ids=sorted(target_file_ids),
                )
            )

        for edge_item in edges_list:
            source = edge_item.get("source")
            target = edge_item.get("target")
            if not source or not target:
                continue
            edges.append(
                GraphEdge(
                    source=str(source).strip(),
                    target=str(target).strip(),
                    weight=float(edge_item.get("weight") or 0.0),
                )
            )

        # 固定写入顺序，降低图数据库批量写入时的死锁概率。
        nodes.sort(key=lambda node: node.orig_id)
        edges.sort(key=lambda edge: (edge.source, edge.target))
        return GraphBatch(index_name=kb_name, nodes=nodes, edges=edges)
