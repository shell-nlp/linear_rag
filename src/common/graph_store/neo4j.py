from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Sequence

from loguru import logger
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, TransientError

from src.common.models import (
    EntityPassageLink,
    GraphBatch,
    GraphDeleteResult,
    GraphWriteResult,
    RelatedPassage,
)
from src.settings import get_settings


class RedisNeo4jWriteQueue:
    """Neo4j 单写者队列，使用 Redis List 和分布式锁保证串行写入。"""

    def __init__(
        self,
        graph_store: "Neo4jGraphStore",
        redis_client,
        queue_key: str = "rag:neo4j:write:queue",
        lock_key: str = "rag:neo4j:write:leader",
        result_prefix: str = "rag:neo4j:write:result",
        lock_ttl_seconds: int = 60,
        result_ttl_seconds: int = 86400,
        submit_timeout_seconds: int = 3600,
    ):
        """初始化队列，但不在构造阶段启动后台线程。"""

        self.graph_store = graph_store
        self.redis = redis_client
        self.queue_key = queue_key
        self.lock_key = lock_key
        self.result_prefix = result_prefix
        self.lock_ttl_seconds = lock_ttl_seconds
        self.result_ttl_seconds = result_ttl_seconds
        self.submit_timeout_seconds = submit_timeout_seconds
        self.instance_id = uuid.uuid4().hex
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._leader_lock = threading.Lock()
        self._is_leader_flag = False

    def start(self) -> None:
        """启动队列工作线程。"""

        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="redis-neo4j-write-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def close(self) -> None:
        """停止队列并释放领导锁。"""

        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        self._release_leader()

    def submit(
        self,
        operation: str,
        payload: dict,
        timeout_seconds: int | None = None,
    ):
        """提交写任务并等待执行结果。"""

        job_id = uuid.uuid4().hex
        result_key = self._result_key(job_id)
        job = {
            "job_id": job_id,
            "operation": operation,
            "payload": payload,
            "created_at": time.time(),
        }
        self.redis.rpush(self.queue_key, json.dumps(job, ensure_ascii=False))

        deadline = time.time() + (timeout_seconds or self.submit_timeout_seconds)
        while time.time() < deadline:
            raw_result = self.redis.get(result_key)
            if raw_result:
                self.redis.delete(result_key)
                result = json.loads(raw_result)
                if result.get("ok"):
                    return result.get("data")
                raise RuntimeError(result.get("error", "Neo4j queue job failed"))
            time.sleep(0.2)
        raise TimeoutError(
            f"Timed out waiting for Neo4j queue job {job_id} ({operation})"
        )

    def _worker_loop(self) -> None:
        """单写者主循环。"""

        while not self._stop_event.is_set():
            if not self._try_acquire_leader():
                time.sleep(1)
                continue
            logger.info("Acquired Redis Neo4j write leadership: {}", self.instance_id)
            self._start_heartbeat()
            try:
                while not self._stop_event.is_set() and self._is_leader():
                    item = self.redis.blpop(self.queue_key, timeout=2)
                    if item is None:
                        continue
                    _, raw_job = item
                    try:
                        job = json.loads(raw_job)
                    except json.JSONDecodeError:
                        logger.exception("Invalid Neo4j queue job payload: {}", raw_job)
                        continue
                    self._process_job(job)
            finally:
                self._stop_heartbeat()
                self._release_leader()

    def _process_job(self, job: dict) -> None:
        """执行单个写任务并保存结果。"""

        job_id = job.get("job_id", "")
        operation = job.get("operation")
        payload = job.get("payload", {})
        result_key = self._result_key(job_id)
        try:
            data = self._dispatch(operation, payload)
            result = {"ok": True, "data": data}
        except Exception as exc:
            logger.exception("Neo4j queue job failed: {}", operation)
            result = {"ok": False, "error": str(exc)}
        self.redis.set(
            result_key,
            json.dumps(result, ensure_ascii=False),
            ex=self.result_ttl_seconds,
        )

    def _dispatch(self, operation: str, payload: dict):
        """将队列操作分发给 Neo4jGraphStore 的完整业务实现。"""

        if operation == "save_graph":
            return self.graph_store._save_graph_payload(payload)
        if operation == "delete_file_nodes":
            return self.graph_store._delete_file_payload(
                index_name=payload["index_name"],
                file_ids=payload["file_ids"],
            )
        raise ValueError(f"Unsupported Neo4j queue operation: {operation}")

    def _start_heartbeat(self) -> None:
        """启动领导锁续约线程。"""

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="redis-neo4j-write-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """停止领导锁续约线程。"""

        self._set_is_leader(False)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1)
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """定期刷新 Redis 领导锁。"""

        interval = max(1, self.lock_ttl_seconds // 3)
        while not self._stop_event.is_set() and self._is_leader():
            time.sleep(interval)
            try:
                if self.redis.get(self.lock_key) != self.instance_id:
                    self._set_is_leader(False)
                    break
                self.redis.expire(self.lock_key, self.lock_ttl_seconds)
            except Exception:
                logger.exception("Failed to refresh Redis Neo4j write lock")
                self._set_is_leader(False)
                break

    def _try_acquire_leader(self) -> bool:
        """尝试获取 Redis 领导锁。"""

        acquired = self.redis.set(
            self.lock_key,
            self.instance_id,
            nx=True,
            ex=self.lock_ttl_seconds,
        )
        self._set_is_leader(bool(acquired))
        return bool(acquired)

    def _release_leader(self) -> None:
        """释放当前实例持有的领导锁。"""

        try:
            if self.redis.get(self.lock_key) == self.instance_id:
                self.redis.delete(self.lock_key)
        except Exception:
            logger.exception("Failed to release Redis Neo4j write lock")
        finally:
            self._set_is_leader(False)

    def _is_leader(self) -> bool:
        """返回当前实例是否是写队列领导。"""

        with self._leader_lock:
            return self._is_leader_flag

    def _set_is_leader(self, value: bool) -> None:
        """更新领导状态。"""

        with self._leader_lock:
            self._is_leader_flag = value

    def _result_key(self, job_id: str) -> str:
        """生成写任务结果键。"""

        return f"{self.result_prefix}:{job_id}"


class Neo4jGraphStore:
    """Neo4j 的完整 GraphStore 实现，包含连接、查询、写入和删除。"""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
        redis_client=None,
    ):
        """创建 Neo4j 驱动，并在传入 Redis 时启用单写者队列。"""

        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.write_queue = (
            RedisNeo4jWriteQueue(self, redis_client)
            if redis_client is not None
            else None
        )

    @classmethod
    def from_settings(cls, redis_client=None) -> "Neo4jGraphStore":
        """从 pydantic-settings 配置创建 Neo4j 实现。"""

        settings = get_settings()
        return cls(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            redis_client=redis_client,
        )

    def start(self) -> None:
        """启动可选的写队列。"""

        if self.write_queue is not None:
            self.write_queue.start()

    def close(self) -> None:
        """关闭写队列和 Neo4j 驱动。"""

        if self.write_queue is not None:
            self.write_queue.close()
        self.driver.close()

    def save_graph(self, batch: GraphBatch) -> GraphWriteResult:
        """保存增量图，队列启用时通过单写者执行。"""

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
            result = self._save_graph_payload(payload)
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
        """删除文件关联节点，队列启用时通过单写者执行。"""

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
            result = self._delete_file_payload(index_name, normalized_file_ids)
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
        with self.driver.session(database=self.database) as session:
            for entity_id in entity_ids:
                records = session.run(
                    query,
                    entity_id=entity_id,
                    index_names=list(index_names),
                    top_k=limit_per_entity,
                ).data()
                passages.extend(
                    RelatedPassage(
                        id=record["passage_id"],
                        text=record.get("passage_name") or "",
                        vector=record.get("passage_vector"),
                    )
                    for record in records
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
        records = self._run_query(
            """
            MATCH (p)-[r]-(e)
            WHERE p.orig_id IN $passage_ids
            AND e.type = 'entity'
            AND ANY(label IN labels(e) WHERE label IN $index_names)
            RETURN DISTINCT e.orig_id AS entity_id
            """,
            passage_ids=list(passage_ids),
            index_names=list(index_names),
        )
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
        """查询实体与段落之间的关联事实。"""

        if not entity_ids or not passage_ids or not index_names:
            return []
        records = self._run_query(
            """
            UNWIND $entity_ids AS entity_id
            UNWIND $passage_ids AS passage_id
            MATCH (e {orig_id: entity_id})-[r]-(p {orig_id: passage_id})
            WHERE e.type = 'entity'
            AND p.type = 'passage'
            AND ANY(label IN labels(e) WHERE label IN $index_names)
            RETURN e.orig_id AS entity_id,
                   p.orig_id AS passage_id,
                   e.name AS entity_name
            """,
            entity_ids=list(entity_ids),
            passage_ids=list(passage_ids),
            index_names=list(index_names),
        )
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
        source_ids = [
            node_id
            for node_id in source_weights
            if node_id in unique_node_ids
        ]
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

        with self.driver.session(database=self.database) as session:
            session.run("CALL gds.graph.drop($graph_name, false)", graph_name=graph_name)
            try:
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
                    internal_ids = [
                        record["internal_id"]
                        for record in session.run(
                            """
                            MATCH (n)
                            WHERE n.orig_id IN $source_ids
                            RETURN id(n) AS internal_id
                            """,
                            source_ids=source_ids,
                        ).data()
                    ]
                    records = session.run(
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
                    records = session.run(
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
            for record in records
            if record.get("id")
        }

    def _save_graph_payload(self, payload: dict) -> dict[str, Any]:
        """执行 Neo4j 节点和关系批量写入。"""

        kb_name = payload["kb_name"]
        node_batch = payload["node_batch"]
        edge_batch = payload["edge_batch"]
        anchor_label = payload.get("anchor_label", "BaseNode")
        edge_label = payload.get("edge_label", "LINK")
        self._execute_query(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{kb_name}`) "
            "REQUIRE n.orig_id IS UNIQUE"
        )
        self._execute_query(
            f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{anchor_label}`) "
            "REQUIRE n.orig_id IS UNIQUE"
        )

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
        for batch in self._chunk(node_batch, 2000):
            self._execute_with_retry(create_nodes_cypher, {"batch_data": batch})

        create_edges_cypher = f"""
        UNWIND $batch_data AS row
        MATCH (s:`{anchor_label}` {{orig_id: row.source}})
        MATCH (t:`{anchor_label}` {{orig_id: row.target}})
        WITH s, t, row
        ORDER BY id(s), id(t)
        MERGE (s)-[r:`{edge_label}`]->(t)
        SET r.weight = row.weight
        """
        for batch in self._chunk(edge_batch, 1000):
            self._execute_with_retry(create_edges_cypher, {"batch_data": batch})
        return {
            "kb_name": kb_name,
            "node_count": len(node_batch),
            "edge_count": len(edge_batch),
        }

    def _delete_file_payload(
        self,
        index_name: str,
        file_ids: Sequence[str],
    ) -> dict[str, Any]:
        """执行按文件删除图节点和标签。"""

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
        with self.driver.session(database=self.database) as session:
            sorted_node_ids = [
                record["node_id"]
                for record in session.run(find_ids_query, file_ids=list(file_ids))
            ]
            deleted_count = 0
            for batch_ids in self._chunk(sorted_node_ids, 1000):
                result = session.run(
                    update_query,
                    batch_node_ids=batch_ids,
                    file_ids=list(file_ids),
                )
                deleted_count += result.consume().counters.nodes_deleted
        return {
            "index_name": index_name,
            "file_ids": list(file_ids),
            "total_nodes": len(sorted_node_ids),
            "deleted_nodes": deleted_count,
        }

    def _execute_query(self, query: str, parameters: dict | None = None):
        """执行不需要返回记录的 Cypher。"""

        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters or {}).consume()

    def _run_query(self, query: str, **parameters) -> list[dict[str, Any]]:
        """执行查询并返回字典记录。"""

        with self.driver.session(database=self.database) as session:
            return session.run(query, parameters).data()

    def _execute_with_retry(
        self,
        query: str,
        parameters: dict,
        attempts: int = 5,
    ) -> None:
        """对临时 Neo4j 故障执行退避重试。"""

        for attempt in range(attempts):
            try:
                self._execute_query(query, parameters)
                return
            except (TransientError, ServiceUnavailable):
                if attempt == attempts - 1:
                    raise
                time.sleep(0.5 * (2**attempt))

    @staticmethod
    def _chunk(items: Sequence[Any], size: int) -> list[Sequence[Any]]:
        """按固定大小切分批量数据。"""

        return [items[index : index + size] for index in range(0, len(items), size)]
