import logging
import math
import re
import time
import uuid
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple

import igraph as ig
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions
from neo4j import exceptions as neo4j_exceptions
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.es import Customize_Elastic
from src.ner import SpacyNER
from src.utils import compute_mdhash_id

# 忽略所有警告
warnings.filterwarnings("ignore")
for logger_name in [
    "elasticsearch",
    "elastic_transport",
    "neo4j",
    "urllib3",
    "spacy",
    "transformers",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class LinearRAG:
    def __init__(
        self,
        global_config,
        es_client: Elasticsearch,
        neo4j_driver,
        es_write_queue=None,
    ):
        self.config = global_config
        logger.info(f"Initializing LinearRAG with config: {self.config}")
        self.spacy_ner = SpacyNER(self.config.spacy_model)
        self.es_client = es_client
        self.neo4j_driver = neo4j_driver
        self.embedding_model = self.config.embedding_model
        self.es_write_queue = es_write_queue

    def retrieve(self, question: str, index_names: List[str], top_k: int = 5):
        """
        主检索入口
        :param questions: 问题列表
        :param index_names: 知识库/索引名称列表
        :param top_k: 返回数量
        """
        retrieval_results = []

        embedding_question = (
            f"下面是一个问题，从数据库中检索到问题相关的段落\n问题：{question}"
        )
        question_embedding = self._encode_question(embedding_question)

        # 1. 尝试获取种子实体
        seed_entities = self._get_seed_entities_from_es(question, index_names)

        if len(seed_entities) > 0:
            # 2. 如果有实体，走图搜索
            final_passages = self._graph_search_with_neo4j(
                question_embedding, seed_entities, index_names, top_k=top_k
            )
        else:
            # 3. 如果没有实体，退回到纯向量搜索
            final_passages = self._fallback_vector_search(
                question_embedding, index_names, top_k=top_k
            )
        retrieval_results.append(final_passages)
        return retrieval_results

    def _build_knn_query(
        self, query_vector, k, num_candidates=100, filter_type="passage"
    ):
        """构建ES KNN查询的通用方法"""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.flatten().tolist()

        return {
            "field": "vector",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates,
            "filter": {"bool": {"must": [{"term": {"type": filter_type}}]}},
        }

    def _execute_es_search(self, index_names: List[str], knn_query: Dict, size: int):
        """执行ES搜索，支持多索引"""
        es_tool = Customize_Elastic(self.es_client)
        try:
            res = es_tool.es_search(
                index_name=index_names,
                knn=knn_query,
                size=size,
                _source=[
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
                ],
            )
            return res
        except Exception as e:
            print(f"ES search failed on indices {index_names}: {e}")
            return {"hits": {"hits": []}}

    def _extract_passages_from_es_hits(self, es_hits, include_extra_fields=None):
        """从ES结果中提取passage信息的通用方法"""
        if include_extra_fields is None:
            include_extra_fields = {}

        passages = []
        for hit in es_hits:
            # 1. 直接获取 ES 中的所有原始字段
            passage = hit.get("_source", {}).copy()

            # 2. 注入元数据字段（如 score）
            passage["score"] = hit.get("_score", 0)

            # 3. 补充默认值（如果 ES 结果中不存在这些字段）
            for key, default_val in include_extra_fields.items():
                if key not in passage:
                    passage[key] = default_val

            passages.append(passage)
        return passages

    def _encode_texts(self, texts, is_batch=True):
        """统一的文本编码方法"""
        if isinstance(texts, str):
            texts = [texts]
            is_batch = False

        embedding = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.config.batch_size,
        )
        embedding = np.asarray(embedding, dtype=float)

        # 统一处理向量维度
        if len(embedding.shape) == 2:
            if embedding.shape[0] > 1 and not is_batch:
                embedding = np.mean(embedding, axis=0, keepdims=True)
            if embedding.shape[1] > 1024:
                embedding = embedding[:, :1024]
        elif len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)

        return embedding if is_batch else embedding[0]

    def _safe_execute(self, func, default_return=None, error_prefix=""):
        """统一的异常处理包装"""
        try:
            return func()
        except Exception as e:
            error_msg = f"{error_prefix}: {e}" if error_prefix else str(e)
            print(error_msg)
            return default_return

    def _fallback_vector_search(
        self, question_embedding, index_names: List[str], top_k=5
    ):
        """当没有实体时，在多个索引中进行向量检索"""
        knn_query = self._build_knn_query(
            question_embedding, k=top_k, num_candidates=100, filter_type="passage"
        )

        es_results = self._execute_es_search(index_names, knn_query, top_k)

        passages = self._extract_passages_from_es_hits(
            es_results["hits"]["hits"],
            include_extra_fields={"pagerank_score": 0, "base_score": None},
        )

        for p in passages:
            p["base_score"] = p.get("score", 0)

        return passages

    def _encode_question(self, text: str) -> np.ndarray:
        """编码问题文本"""
        return self._encode_texts(text, is_batch=True)

    def _get_seed_entities_from_es(
        self, question: str, index_names: List[str]
    ) -> List[Dict]:
        """从多个索引中检索种子实体"""
        question_entities = list(self.spacy_ner.question_ner(question))
        if len(question_entities) == 0:
            return []

        question_entity_embeddings = self._encode_texts(
            question_entities, is_batch=True
        )
        seed_entities = []

        for query_entity_idx, query_entity in enumerate(question_entities):
            query_embedding = question_entity_embeddings[query_entity_idx]
            knn_query = self._build_knn_query(
                query_embedding, k=10, num_candidates=100, filter_type="entity"
            )

            # 在 index_names 列表覆盖的所有索引中查询实体
            def search_entity():
                return self.es_client.search(
                    index=index_names, knn=knn_query, source=["hash_id", "text", "type"]
                )

            es_results = self._safe_execute(
                search_entity,
                default_return={"hits": {"hits": []}},
                error_prefix=f"Error searching for entity {query_entity} in {index_names}",
            )

            if es_results["hits"]["hits"]:
                top_hit = es_results["hits"]["hits"][0]
                seed_entity = {
                    "hash_id": top_hit["_source"]["hash_id"],
                    "text": top_hit["_source"]["text"],
                    "score": top_hit["_score"],
                    "embedding": query_embedding,
                }
                seed_entities.append(seed_entity)

        return seed_entities

    def _calculate_entity_scores_neo4j(
        self,
        session,
        seed_entities: List[Dict],
        question_embedding: np.ndarray,
        index_names: List[str],
    ) -> Dict[str, float]:
        """Neo4j多跳扩散，增加索引过滤"""
        activated_entities = {e["hash_id"]: e["score"] for e in seed_entities}
        used_passages = set()
        iteration = 1
        current_entities = activated_entities.copy()

        while len(current_entities) > 0 and iteration < 3:
            new_entities = {}
            for entity_id, entity_score in current_entities.items():
                if entity_score < 0.01:
                    continue

                # Cypher 说明: labels(p) 获取节点标签，ANY (...) IN $index_names 确保节点属于目标知识库
                cypher_query = """
                MATCH (e {orig_id: $entity_id})-[r]-(p)
                WHERE p.type = 'passage'
                AND NOT (p.orig_id IN $used_passages)
                AND ANY(label in labels(p) WHERE label IN $index_names)
                RETURN p.orig_id as passage_id, p.name as passage_name, p.vector as passage_vector
                LIMIT $top_k
                """

                passage_results = self._safe_execute(
                    lambda: session.run(
                        cypher_query,
                        entity_id=entity_id,
                        used_passages=list(used_passages),
                        index_names=index_names,
                        top_k=5,
                    ).data(),
                    default_return=[],
                    error_prefix=f"Error querying passages for entity {entity_id}",
                )

                for passage_result in passage_results:
                    passage_id = passage_result["passage_id"]
                    used_passages.add(passage_id)

                    # 向量相似度计算 (省略部分逻辑以保持简洁，同原代码)
                    passage_similarity = 0.5
                    try:
                        vec_data = passage_result.get("passage_vector")
                        if vec_data:
                            passage_vector = np.array(vec_data)
                            passage_similarity = np.dot(
                                passage_vector, question_embedding.T
                            ).flatten()[0]
                    except:
                        pass

                    # Passage -> Entity 扩散
                    cypher_query_entity = """
                    MATCH (p {orig_id: $passage_id})-[r]-(e)
                    WHERE e.type = 'entity'
                    AND ANY(label in labels(e) WHERE label IN $index_names)
                    RETURN e.orig_id as entity_id
                    """

                    entity_results = self._safe_execute(
                        lambda: session.run(
                            cypher_query_entity,
                            passage_id=passage_id,
                            index_names=index_names,
                        ).data(),
                        default_return=[],
                        error_prefix=f"Error querying entities for passage {passage_id}",
                    )

                    for entity_result in entity_results:
                        next_entity_id = entity_result["entity_id"]
                        if not next_entity_id:
                            continue
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

    def _calculate_passage_scores_neo4j(
        self,
        session,
        question_embedding: np.ndarray,
        entity_scores: Dict[str, float],
        index_names: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """结合ES检索结果与实体权重，支持多索引"""
        passage_scores = {}
        passage_cache = {}

        knn_query = self._build_knn_query(
            question_embedding, k=50, num_candidates=100, filter_type="passage"
        )
        es_results = self._execute_es_search(index_names, knn_query, 50)

        hits = es_results["hits"]["hits"]
        if not hits:
            return {}, {}

        dpr_scores = self._min_max_normalize([hit["_score"] for hit in hits])

        for idx, hit in enumerate(hits):
            source = hit["_source"]
            passage_id = source["hash_id"]
            passage_text = source.get("text", "")
            passage_cache[passage_id] = {
                "text": passage_text,
                "file_id": source.get("file_id", ""),
            }

            total_entity_bonus = 0
            passage_text_lower = passage_text.lower()

            for entity_id, entity_score in entity_scores.items():
                # 检查这些实体是否与当前段落在 Neo4j 中有关联，且在指定索引内
                cypher_query = """
                MATCH (e {orig_id: $entity_id})-[:CONTAINS|:MENTIONED_IN]-(p {orig_id: $passage_id})
                WHERE ANY(label in labels(e) WHERE label IN $index_names)
                RETURN e.name as entity_name
                """
                result = self._safe_execute(
                    lambda: session.run(
                        cypher_query,
                        entity_id=entity_id,
                        passage_id=passage_id,
                        index_names=index_names,
                    ).data(),
                    default_return=[],
                )

                if result:
                    entity_name = str(result[0]["entity_name"]).lower()
                    if entity_name in passage_text_lower:
                        count = passage_text_lower.count(entity_name)
                        total_entity_bonus += entity_score * math.log(1 + count)

            passage_scores[passage_id] = 0.6 * dpr_scores[idx] + math.log(
                1 + total_entity_bonus
            )

        return passage_scores, passage_cache

    def _run_pagerank_and_filter(
        self,
        session,
        entity_scores: Dict[str, float],
        passage_scores: Dict[str, float],
    ) -> List[Dict]:
        """
        使用 orig_id 进行匹配
        """
        # 1. 收集所有涉及的节点 ID (来自 ES 的 hash_id，对应 Neo4j 的 orig_id)
        all_node_ids = list(
            set(list(entity_scores.keys()) + list(passage_scores.keys()))
        )
        if not all_node_ids:
            return []

        graph_name = f"rag_gds_ppr_{uuid.uuid4().hex[:8]}"

        # 清理旧图
        self._safe_execute(
            lambda: session.run(f"CALL gds.graph.drop('{graph_name}', false)"),
            default_return=None,
        )

        try:
            # --- 2. 投影子图 ---
            node_query = """
            MATCH (n)
            WHERE n.orig_id IN $node_ids
            RETURN id(n) as id
            """
            edge_query = """
            MATCH (n)-[r]-(m)
            WHERE n.orig_id IN $node_ids AND m.orig_id IN $node_ids
            RETURN id(n) as source, id(m) as target
            """

            self._safe_execute(
                lambda: session.run(
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
                    node_ids=all_node_ids,
                ),
                default_return=None,
                error_prefix="Error projecting graph for GDS",
            )

            # --- 3. 准备 Personalized PageRank 的源点 ---
            high_score_ids = [k for k, v in entity_scores.items() if v > 0.5]
            high_score_ids += [k for k, v in passage_scores.items() if v > 0.5]
            high_score_ids = list(set(high_score_ids))

            if not high_score_ids:
                high_score_ids = all_node_ids

            # 查内部 ID 时也用 orig_id
            id_map_query = """
            MATCH (n) WHERE n.orig_id IN $high_ids
            RETURN id(n) as internal_id
            """

            res_data = self._safe_execute(
                lambda: session.run(id_map_query, high_ids=high_score_ids).data(),
                default_return=[],
                error_prefix="Error mapping node IDs",
            )

            source_node_internal_ids = [r["internal_id"] for r in res_data]

            # --- 4. 运行 PageRank ---
            if not source_node_internal_ids:
                pr_query = f"""
                CALL gds.pageRank.stream($graph_name, {{
                    dampingFactor: 0.85
                }})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).orig_id as id, score
                ORDER BY score DESC
                """
                pr_result = self._safe_execute(
                    lambda: session.run(pr_query, graph_name=graph_name).data(),
                    default_return=[],
                    error_prefix="Error running PageRank without source nodes",
                )
            else:
                pr_query = f"""
                CALL gds.pageRank.stream($graph_name, {{
                    sourceNodes: $source_nodes,
                    dampingFactor: 0.85
                }})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).orig_id as id, score
                ORDER BY score DESC
                """
                pr_result = self._safe_execute(
                    lambda: session.run(
                        pr_query,
                        graph_name=graph_name,
                        source_nodes=source_node_internal_ids,
                    ).data(),
                    default_return=[],
                    error_prefix="Error running PageRank with source nodes",
                )

            pagerank_scores = {record["id"]: record["score"] for record in pr_result}

            # 5. 结果融合
            final_passages = []
            for pid, base_score in passage_scores.items():
                pr_score = pagerank_scores.get(pid, 0.0)
                final_score = base_score * 0.5 + pr_score * 0.5
                final_passages.append(
                    {
                        "hash_id": pid,
                        "score": final_score,
                        "pagerank_score": pr_score,
                        "base_score": base_score,
                    }
                )

            final_passages.sort(key=lambda x: x["score"], reverse=True)
            return final_passages

        except Exception as e:
            print(f"GDS/PPR execution failed: {e}")
            fallback = [
                {"hash_id": k, "score": v, "pagerank_score": 0, "base_score": v}
                for k, v in passage_scores.items()
            ]
            fallback.sort(key=lambda x: x["score"], reverse=True)
            return fallback

        finally:
            self._safe_execute(
                lambda: session.run(f"CALL gds.graph.drop('{graph_name}', false)"),
                default_return=None,
            )

    def _graph_search_with_neo4j(
        self,
        question_embedding: np.ndarray,
        seed_entities: List[Dict],
        index_names: List[str],
        top_k: int = 5,
    ) -> List[Dict]:
        """完整的图检索流程"""
        with self.neo4j_driver.get_session() as session:
            # 1. 计算实体扩散分数
            entity_scores = self._calculate_entity_scores_neo4j(
                session, seed_entities, question_embedding, index_names
            )

            # 2. 计算段落初步分数
            passage_scores, _ = self._calculate_passage_scores_neo4j(
                session, question_embedding, entity_scores, index_names
            )

            # 3. PageRank 重新排序
            final_passages = self._run_pagerank_and_filter(
                session, entity_scores, passage_scores
            )
            final_passages = final_passages[:top_k]

            # 4. 批量回查 ES 获取详细字段 (支持多索引回查)
            all_top_ids = [p["hash_id"] for p in final_passages]
            if all_top_ids:
                es_tool = Customize_Elastic(self.es_client)
                # 注意：_batch_fetch_passages 内部也需要支持传入 index_names 列表进行 ids 查询
                full_data_map = es_tool._batch_fetch_passages(all_top_ids, index_names)

                for p in final_passages:
                    info = full_data_map.get(p["hash_id"])
                    if info:
                        p.update(info)

        return final_passages

    def _min_max_normalize(self, scores: List[float]) -> List[float]:
        """Min-Max归一化"""
        if len(scores) == 0:
            return []
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        if max_score == min_score:
            return [0.5] * len(scores)
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

    def index(self, passages: list, kb_name):
        graph = ig.Graph(directed=False)
        node_to_node_stats = defaultdict(dict)

        new_hash_id_to_passage, all_hash_id_to_passage = self.insert_text(
            passages,
            self.embedding_model,
            self.config.batch_size,
            "passage",
            kb_name,
            self.es_client,
        )

        if not new_hash_id_to_passage:
            print("没有新文档，跳过处理。")
            return

        print(f"检测到 {len(new_hash_id_to_passage)} 个新段落，准备处理子图...")

        # NER (仅新文档)
        new_passage_hash_id_to_entities = self.spacy_ner.batch_ner(
            new_hash_id_to_passage, self.config.max_workers
        )

        # 获取entity_node信息
        entity_node_info = self.extract_entity_info(
            new_passage_hash_id_to_entities, new_hash_id_to_passage, passages
        )

        # 向量入库
        _, all_hash_id_to_entity = self.insert_text(
            entity_node_info,
            self.embedding_model,
            self.config.batch_size,
            "entity",
            kb_name,
            self.es_client,
        )

        # 图构建
        node_to_node_stats = self.add_entity_to_passage_edges(
            new_passage_hash_id_to_entities,
            new_hash_id_to_passage,
            all_hash_id_to_entity,
            node_to_node_stats,
        )
        node_to_node_stats = self.add_adjacent_passage_edges(
            all_hash_id_to_passage, node_to_node_stats
        )

        active_nodes = set()
        active_nodes.update(all_hash_id_to_passage.keys())
        for src, targets in node_to_node_stats.items():
            active_nodes.add(src)
            for tgt in targets.keys():
                active_nodes.add(tgt)

        # 这里传入不带序号的passage
        graph = self.augment_graph(
            active_nodes,
            all_hash_id_to_entity,
            all_hash_id_to_passage,
            graph,
            node_to_node_stats,
        )
        print(f"正在将增量子图写入 Neo4j... (节点数: {len(active_nodes)})")
        self.save_igraph(kb_name, graph, entity_node_info, passages)
        print("增量更新完成。")

    def save_igraph(self, kb_name, graph, entity_node_info, passages):
        """
        将图文件数据批量保存到Neo4j。
        包含：全局锚点标签、ID排序防死锁、以及针对关系的死锁重试机制。
        """

        # --- 1. 预处理 (保持不变) ---
        text_to_file_ids = {}
        if entity_node_info and "text" in entity_node_info:
            for text, f_id in zip(
                entity_node_info["text"], entity_node_info["file_id"]
            ):
                if text not in text_to_file_ids:
                    text_to_file_ids[text] = set()
                text_to_file_ids[text].add(str(f_id))

        default_passages_fids = passages.get("file_id", []) if passages else []

        if graph.vcount() == 0:
            print("警告：当前内存图为空，跳过保存。")
            return

        if "name" not in graph.vertex_attributes():
            print("警告：节点缺少 'name' 属性，无法作为唯一标识。")
            return

        edge_label = "LINK"
        anchor_label = "BaseNode"  # 全局锚点标签

        # --- 2. 约束与索引 (保持不变) ---
        try:
            cypher_kb = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{kb_name}`) REQUIRE n.orig_id IS UNIQUE"
            self.neo4j_driver.execute_query(cypher_kb)
            cypher_base = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{anchor_label}`) REQUIRE n.orig_id IS UNIQUE"
            self.neo4j_driver.execute_query(cypher_base)
        except Exception as e:
            print(f"索引创建提示: {e}")

        # 获取数据
        graph_tuple = graph.to_dict_list(use_vids=False)
        nodes_list, edges_list = graph_tuple

        # --- 3. 准备节点数据 ---
        node_batch = []
        for node_item in nodes_list:
            unique_id = node_item.get("name")
            content = node_item.get("content", "")
            if not unique_id:
                continue

            clean_id = str(unique_id).strip()
            pattern = r"^([^-]+)"
            match = re.match(pattern, clean_id)
            node_type = match.group(1) if match else "Unknown"

            target_fids = text_to_file_ids.get(content, set())
            if not target_fids and default_passages_fids:
                target_fids = set(map(str, default_passages_fids))

            node_batch.append(
                {
                    "orig_id": clean_id,
                    "type": node_type,
                    "name": content,
                    "new_file_ids": list(target_fids),
                }
            )

        # 【防死锁】节点排序
        node_batch.sort(key=lambda x: x["orig_id"])

        # 节点 Cypher (使用 Anchor Label)
        create_nodes_cypher = f"""
        UNWIND $batch_data AS row
        MERGE (n:`{anchor_label}` {{orig_id: row.orig_id}})
        SET n:`{kb_name}`, n.name = row.name, n.type = row.type
        WITH n, row
        WHERE row.new_file_ids IS NOT NULL
        SET n.file_id = REDUCE(s = coalesce(n.file_id, []), fid IN row.new_file_ids | 
            CASE WHEN fid IN s THEN s ELSE s + fid END
        )
        """

        node_batch_size = 2000
        total_nodes = len(node_batch)
        print(f"正在写入 {total_nodes} 个节点...")

        # 节点写入循环
        for i in range(0, total_nodes, node_batch_size):
            batch = node_batch[i : i + node_batch_size]
            if batch:
                # 简单的重试封装
                for attempt in range(3):
                    try:
                        self.neo4j_driver.execute_query(
                            create_nodes_cypher, {"batch_data": batch}
                        )
                        break
                    except TransientError:
                        time.sleep(0.2 * (attempt + 1))
                        if attempt == 2:
                            raise

        # --- 4. 准备关系数据 (重点修改部分) ---
        edge_batch = []
        for edge_item in edges_list:
            source_key = edge_item.get("source")
            target_key = edge_item.get("target")
            weight = edge_item.get("weight")

            if source_key and target_key:
                edge_batch.append(
                    {
                        "source": str(source_key).strip(),
                        "target": str(target_key).strip(),
                        "weight": weight if weight is not None else 0,
                    }
                )

        # 【防死锁】关系排序
        edge_batch.sort(key=lambda x: (x["source"], x["target"]))

        create_edges_cypher = f"""
        UNWIND $batch_data AS row
        MATCH (s:`{anchor_label}` {{orig_id: row.source}})
        MATCH (t:`{anchor_label}` {{orig_id: row.target}})
        WITH s, t, row
        ORDER BY id(s), id(t)  // 这里的排序有助于减少单条语句内的死锁，但不能完全避免跨事务死锁
        MERGE (s)-[r:`{edge_label}`]->(t)
        SET r.weight = row.weight
        """

        # 【优化】关系写入的 Batch Size 建议调小，减少锁持有时间
        edge_batch_size = 1000
        total_edges = len(edge_batch)
        print(f"开始写入 {total_edges} 条关系...")

        # --- 关系写入循环 (增加重试机制) ---
        for i in range(0, total_edges, edge_batch_size):
            batch = edge_batch[i : i + edge_batch_size]
            if not batch:
                continue

            # === 重试逻辑开始 ===
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    self.neo4j_driver.execute_query(
                        create_edges_cypher, {"batch_data": batch}
                    )
                    # print(f" -> 关系批次 {i//edge_batch_size + 1} 成功")
                    break
                except (TransientError, ServiceUnavailable) as e:
                    # 如果是死锁或连接忙，进行等待和重试
                    if attempt < max_retries - 1:
                        sleep_time = 0.5 * (2**attempt)  # 指数退避: 0.5s, 1s, 2s, 4s...
                        print(
                            f" [警告] 关系写入发生死锁，正在重试 (第 {attempt+1} 次)... Error: {str(e)[:50]}"
                        )
                        time.sleep(sleep_time)
                    else:
                        print(f" [错误] 关系写入重试耗尽，放弃该批次。")
                        raise e

    # 添加“相邻段落”之间的边。
    # 这个函数假设输入的文本是有格式要求的（必须以 序号: 开头）。如果文本没有序号，这个函数将不会添加任何边。
    def add_adjacent_passage_edges(self, all_hash_id_to_passage, node_to_node_stats):
        """
        不再依赖文本内的数字，直接按照传入字典的顺序自动建立相邻连接。
        """
        # 1. 直接获取所有的 ID 列表（Python 3.7+ 保证了字典顺序就是插入顺序）
        node_ids = list(all_hash_id_to_passage.keys())

        # 定义限制范围（如果有特定需求可以从外部传入，这里默认处理当前所有 ID）
        restrict_to_ids = set(node_ids)

        # 2. 遍历 ID 列表，将相邻的两个 ID 连起来
        for i in range(len(node_ids) - 1):
            current_node = node_ids[i]
            next_node = node_ids[i + 1]

            # 3. 只要其中一个节点在本次处理的 ID 集中，就建立边
            if (current_node in restrict_to_ids) or (next_node in restrict_to_ids):
                # 初始化字典结构，防止 KeyError
                if current_node not in node_to_node_stats:
                    node_to_node_stats[current_node] = {}

                # 建立连接：current_node -> next_node
                node_to_node_stats[current_node][next_node] = 1.0

        return node_to_node_stats

    def augment_graph(
        self,
        restrict_to_nodes,
        new_hash_id_to_entity,
        all_hash_id_to_passage,
        graph,
        node_to_node_stats,
    ):
        # 将过滤集合传递给 add_nodes
        graph = self.add_nodes(
            restrict_to_nodes, new_hash_id_to_entity, all_hash_id_to_passage, graph
        )
        graph = self.add_edges(graph, node_to_node_stats)
        return graph

    # 将实体和段落作为“顶点（Vertex）”添加到图对象中。
    def add_nodes(
        self, restrict_to_nodes, new_hash_id_to_entity, all_hash_id_to_passage, graph
    ):
        """
        修改后的 add_nodes:
        如果不传 restrict_to_nodes，默认行为是添加全量（兼容旧逻辑）。
        如果传了集合，只添加集合在 Embedding Store 中存在的节点。
        """
        # 检查图中已有的节点
        existing_nodes_in_graph = {
            v["name"]: v for v in graph.vs if "name" in v.attributes()
        }

        entity_hash_id_to_text = new_hash_id_to_entity
        passage_hash_id_to_text = all_hash_id_to_passage
        all_hash_id_to_text = {**entity_hash_id_to_text, **passage_hash_id_to_text}

        # 确定要遍历的目标 ID 列表
        if restrict_to_nodes is not None:
            # 只关心计算出来的 active_nodes
            # 使用 set intersection 确保 ID 确实存在于 store 中
            target_ids = restrict_to_nodes.intersection(all_hash_id_to_text.keys())
        else:
            target_ids = all_hash_id_to_text.keys()

        # 循环添加节点
        for hash_id in target_ids:
            if hash_id not in existing_nodes_in_graph:
                text = all_hash_id_to_text[hash_id]
                graph.add_vertex(name=hash_id, content=text)

        # 5. 重建索引
        self.node_name_to_vertex_idx = {
            v["name"]: v.index for v in graph.vs if "name" in v.attributes()
        }

        # 缓存段落节点索引
        passage_hash_ids = set(passage_hash_id_to_text.keys())
        self.passage_node_indices = [
            self.node_name_to_vertex_idx[passage_id]
            for passage_id in passage_hash_ids
            if passage_id in self.node_name_to_vertex_idx
        ]
        return graph

    def add_edges(self, graph, node_to_node_stats):
        edges = []
        weights = []

        # 遍历 node_to_node_stats
        for node_hash_id, node_to_node_stats in node_to_node_stats.items():
            for neighbor_hash_id, weight in node_to_node_stats.items():
                if node_hash_id == neighbor_hash_id:
                    continue

                if (
                    node_hash_id in self.node_name_to_vertex_idx
                    and neighbor_hash_id in self.node_name_to_vertex_idx
                ):
                    edges.append((node_hash_id, neighbor_hash_id))
                    weights.append(weight)
        if edges:
            graph.add_edges(edges)
            graph.es["weight"] = weights
        return graph

    # 计算并添加“段落”指向“实体”的边，并计算边的权重。确定哪个实体对哪个段落更重要。
    def add_entity_to_passage_edges(
        self,
        new_passage_hash_id_to_entities,
        new_hash_id_to_passage,
        new_hash_id_to_entity,
        node_to_node_stats,
    ):
        if isinstance(new_hash_id_to_entity, tuple):
            if len(new_hash_id_to_entity) > 0:
                new_hash_id_to_entity = new_hash_id_to_entity[0]
            else:
                raise ValueError("new_hash_id_to_entity 是一个空元组")

        # 构建反向映射表：从 '实体文本' -> 'Hash ID'
        # 因为 new_hash_id_to_entity 是 {ID: Name}，我们需要 {Name: ID}
        entity_text_to_hash_id = {v: k for k, v in new_hash_id_to_entity.items()}

        passage_to_entity_count = {}
        passage_to_all_score = defaultdict(int)

        for passage_hash_id, entities in new_passage_hash_id_to_entities.items():
            # 1. 确保段落存在
            if passage_hash_id not in new_hash_id_to_passage:
                continue

            passage = new_hash_id_to_passage[passage_hash_id]

            # 2. 如果 list 中可能存在重复实体，建议用 set(entities) 去重，
            # 这样可以保持逻辑与之前一致，避免同一个实体在同一个段落里被多次累加 score
            unique_entities = set(entities)

            for entity_text in unique_entities:
                if entity_text in entity_text_to_hash_id:
                    entity_hash_id = entity_text_to_hash_id[entity_text]

                    # 3. 计算权重
                    # 注意：count = passage.count(entity_text) 是子串匹配
                    count = passage.count(entity_text)

                    if count > 0:
                        # 记录该段落中该实体的出现次数
                        passage_to_entity_count[(passage_hash_id, entity_hash_id)] = (
                            count
                        )
                        # 累加该段落的总权重得分
                        passage_to_all_score[passage_hash_id] += count
                else:
                    # 实体不在字典中则跳过
                    continue

        # 计算最终分数并存入 node_to_node_stats
        for (passage_hash_id, entity_hash_id), count in passage_to_entity_count.items():
            total_count = passage_to_all_score[passage_hash_id]
            if total_count > 0:
                score = count / total_count
                # 确保字典初始化
                if passage_hash_id not in node_to_node_stats:
                    node_to_node_stats[passage_hash_id] = {}
                node_to_node_stats[passage_hash_id][entity_hash_id] = score

        return node_to_node_stats

    def extract_nodes_and_edges(self, new_passage_hash_id_to_entities):
        # 1. 初始化容器
        entity_nodes = set()
        passage_hash_id_to_entities = defaultdict(set)

        # 2. 处理“段落-实体”数据 (只遍历新数据)
        for passage_hash_id, entities in new_passage_hash_id_to_entities.items():
            for entity in entities:
                entity_nodes.add(entity)
                passage_hash_id_to_entities[passage_hash_id].add(entity)

        return entity_nodes, passage_hash_id_to_entities

    def insert_text(
        self, passages_dict, embedding_model, batch_size, type, index_name, es_client
    ):
        es = Customize_Elastic(es_client, self.es_write_queue)

        # 1. 基础校验
        if (
            not passages_dict
            or "text" not in passages_dict
            or not passages_dict["text"]
        ):
            return {}, {}

        text_list = passages_dict["text"]

        # 定义两个字典：
        # full_docs_map: 用于传给 ES 保存，包含所有元数据
        full_docs_map = {}
        # text_only_map: 用于函数返回，只包含纯文本，符合你要求的 return 格式
        text_only_map = {}

        # 需要提取的元数据字段列表
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

        # 2. 遍历文本，组装数据
        for i, text in enumerate(text_list):
            # 计算 ID
            h_id = compute_mdhash_id(text, prefix=type + "-")

            # A. 构建用于保存的完整对象 (Rich Object)
            doc_info = {"text": text, "hash_id": h_id}
            # 动态提取元数据（安全获取，没有则为 None）
            for field in meta_fields:
                field_list = passages_dict.get(field, [])
                if field_list and i < len(field_list):
                    doc_info[field] = field_list[i]
                else:
                    doc_info[field] = None

            full_docs_map[h_id] = doc_info

            # B. 构建用于返回的简单对象 (String Only)
            text_only_map[h_id] = text

        # 3. 查重逻辑（使用 text_only_map 的 keys 即可）
        all_hash_ids = list(text_only_map.keys())

        check_query = {
            "size": len(all_hash_ids),
            "query": {"terms": {"hash_id": all_hash_ids}},
            "_source": ["hash_id"],
        }

        existing_records = es.es_search(index_name=index_name, query_body=check_query)
        hits_list = existing_records.get("hits", {}).get("hits", [])

        existing_ids = {
            item.get("_source", {}).get("hash_id")
            for item in hits_list
            if item.get("_source", {}).get("hash_id")
        }

        # 计算缺失的 ID
        missing_ids = [h for h in all_hash_ids if h not in existing_ids]

        # 如果没有缺失，直接返回 text_only_map
        if not missing_ids:
            return {}, text_only_map

        # 4. 向量化
        # 只需要对文本进行 Embedding
        texts_to_encode = [text_only_map[hash_id] for hash_id in missing_ids]

        all_embeddings = embedding_model.encode(
            texts_to_encode,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # 5. 保存数据 (关键点)
        # 传给 save_batch 的是 full_docs_map 中的完整对象（包含 meta），而不是纯文本
        docs_to_save = [full_docs_map[hash_id] for hash_id in missing_ids]

        if missing_ids:
            es.save_batch(missing_ids, docs_to_save, all_embeddings, index_name)

        # 6. 返回结果 (关键点)
        # 构造 inserted_dict，值使用 text_only_map 中的纯文本
        inserted_dict = {h_id: text_only_map[h_id] for h_id in missing_ids}

        # 返回格式：{hash_id: string}, {hash_id: string}
        return inserted_dict, text_only_map

    def extract_entity_info(
        self, new_passage_hash_id_to_entities, new_hash_id_to_passage, passages
    ):
        """
        将实体映射转换为包含元数据（文件名、文件ID）的平铺字典结构。

        Args:
            new_passage_hash_id_to_entities (dict): hash_id 到 实体列表 的映射
            new_hash_id_to_passage (dict): hash_id 到 段落文本 的映射
            passages (dict): 包含 'text', 'file_name', 'file_id' 等列表的原始数据字典

        Returns:
            dict: 包含 'text' (实体名), 'file_name', 'file_id' 的字典
        """

        # 初始化输出结构
        new_entity_nodes = {
            "text": [],  # 存放实体名称
            "file_name": [],  # 存放对应的文件名
            "file_id": [],  # 存放对应的文件ID
        }

        text_to_index_map = {text: i for i, text in enumerate(passages["text"])}

        # 遍历每一个 hash_id 及其对应的实体列表
        for hash_id, entities in new_passage_hash_id_to_entities.items():
            passage_text = new_hash_id_to_passage.get(hash_id)

            if passage_text is None:
                continue

            if passage_text in text_to_index_map:
                idx = text_to_index_map[passage_text]
                current_file_name = (
                    passages["file_name"][idx]
                    if idx < len(passages["file_name"])
                    else "Unknown"
                )
                current_file_id = (
                    passages["file_id"][idx]
                    if idx < len(passages["file_id"])
                    else "Unknown"
                )

                for entity in entities:
                    new_entity_nodes["text"].append(entity)
                    new_entity_nodes["file_name"].append(current_file_name)
                    new_entity_nodes["file_id"].append(current_file_id)
            else:
                print(
                    f"Warning: Passage text for hash {hash_id} not found in passages source."
                )
        return new_entity_nodes

    def delete_files(self, index_name, file_ids):
        """
        任务描述：批量删除指定文件 ID 列表相关的所有节点信息。
        策略：适配 BaseNode 标签，先排序 ID 防止死锁，再清理无用节点。

        参数：
        - index_name: 对应 ES 的索引名，也对应 Neo4j 的 Label。
        - file_ids: 要删除的文件 ID 列表 (list)。
        """
        if not isinstance(file_ids, list):
            file_ids = [file_ids]

        try:
            es_query = {"query": {"terms": {"file_id.keyword": file_ids}}}
            es_tool = Customize_Elastic(self.es_client, self.es_write_queue)
            res = es_tool.delete_by_query(
                index_name=index_name,
                body=es_query,
                refresh=True,
            )
            print(f"[ES] 已从索引 {index_name} 中删除 {res.get('deleted')} 条文档。")
        except es_exceptions.NotFoundError:
            print(f"[ES] 索引 {index_name} 未找到，跳过删除。")
        except Exception as e:
            print(f"[ES] 删除出错: {e}")

        # 查找涉及的节点 ID 并按升序排序
        find_ids_query = f"""
            MATCH (n:`{index_name}`)
            WHERE any(fid IN $file_ids WHERE fid IN n.file_id)
            RETURN id(n) as node_id
            ORDER BY node_id ASC
            """

        update_query = f"""
            MATCH (n)
            WHERE id(n) IN $batch_node_ids
            REMOVE n:`{index_name}`
            SET n.file_id = [x IN n.file_id WHERE NOT x IN $file_ids]
            WITH n
            WHERE size(n.file_id) = 0 OR size(labels(n)) <= 1
            DETACH DELETE n
            """

        try:
            with self.neo4j_driver.get_session() as session:
                # 获取排序后的 ID 列表
                result = session.run(find_ids_query, file_ids=file_ids)
                sorted_node_ids = [record["node_id"] for record in result]

                total_nodes = len(sorted_node_ids)
                if total_nodes == 0:
                    print(f"[Neo4j] 没有找到与 {file_ids} 相关的节点，无需操作。")
                    return

                print(
                    f"[Neo4j] 找到 {total_nodes} 个相关节点，开始按 ID 顺序分批处理..."
                )

                # 分批执行
                batch_size = 1000
                deleted_count = 0

                for i in range(0, total_nodes, batch_size):
                    batch_ids = sorted_node_ids[i : i + batch_size]

                    # 执行删除/更新
                    # 如果这里频繁出现死锁，也可以像 save_igraph 一样加上 try-catch 重试机制
                    update_result = session.run(
                        update_query, batch_node_ids=batch_ids, file_ids=file_ids
                    )
                    summary = update_result.consume()
                    deleted_count += summary.counters.nodes_deleted

                print(
                    f"[Neo4j] 操作完成。共处理节点 {total_nodes} 个，实际物理删除节点 {deleted_count} 个。"
                )

        except neo4j_exceptions.Neo4jError as e:
            print(f"[Neo4j] Cypher 执行错误: {e}")
        except Exception as e:
            print(f"[Neo4j] 未知错误: {e}")
