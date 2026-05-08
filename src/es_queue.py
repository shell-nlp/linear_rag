import json
import logging
import threading
import time
import uuid

from src.es import Customize_Elastic

logger = logging.getLogger(__name__)


class RedisESWriteQueue:
    """
    使用 Redis List + 分布式锁实现的 ES 单写者队列。

    - 所有实例都可以入队。
    - 所有实例都会启动一个后台消费者线程，但只有拿到锁的实例会真正消费。
    - 同一时刻全局只有一个实例顺序执行 ES 写操作。
    """

    def __init__(
        self,
        es_client,
        redis_client,
        queue_key: str = "rag:es:write:queue",
        lock_key: str = "rag:es:write:leader",
        result_prefix: str = "rag:es:write:result",
        lock_ttl_seconds: int = 60,
        result_ttl_seconds: int = 86400,
        submit_timeout_seconds: int = 3600,
    ):
        self.es_client = es_client
        self.redis = redis_client
        self.queue_key = queue_key
        self.lock_key = lock_key
        self.result_prefix = result_prefix
        self.lock_ttl_seconds = lock_ttl_seconds
        self.result_ttl_seconds = result_ttl_seconds
        self.submit_timeout_seconds = submit_timeout_seconds
        self.instance_id = uuid.uuid4().hex

        self._stop_event = threading.Event()
        self._worker_thread = None
        self._heartbeat_thread = None
        self._leader_lock = threading.Lock()
        self._is_leader_flag = False

    def start(self):
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="redis-es-write-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def close(self):
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        self._release_leader()

    def submit(self, operation: str, payload: dict, timeout_seconds: int | None = None):
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
                raise RuntimeError(result.get("error", "ES queue job failed"))
            time.sleep(0.2)

        raise TimeoutError(
            f"Timed out waiting for ES queue job {job_id} ({operation}) to complete"
        )

    def _worker_loop(self):
        while not self._stop_event.is_set():
            if not self._try_acquire_leader():
                time.sleep(1)
                continue

            logger.info("Acquired Redis ES write leadership: %s", self.instance_id)
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
                        logger.exception("Invalid ES queue job payload: %s", raw_job)
                        continue

                    self._process_job(job)
            finally:
                self._stop_heartbeat()
                self._release_leader()

    def _process_job(self, job: dict):
        job_id = job.get("job_id", "")
        operation = job.get("operation")
        payload = job.get("payload", {})
        result_key = self._result_key(job_id)

        try:
            data = self._dispatch(operation, payload)
            result = {"ok": True, "data": data}
        except Exception as exc:
            logger.exception("ES queue job failed: %s", operation)
            result = {"ok": False, "error": str(exc)}

        self.redis.set(
            result_key,
            json.dumps(result, ensure_ascii=False),
            ex=self.result_ttl_seconds,
        )

    def _dispatch(self, operation: str, payload: dict):
        es_tool = Customize_Elastic(self.es_client)

        if operation == "create_index":
            return es_tool._create_index_local(
                index_name=payload["index_name"],
                body=payload.get("body"),
                only_if_missing=payload.get("only_if_missing", False),
            )

        if operation == "delete_index":
            return es_tool._delete_index_local(
                index_name=payload["index_name"],
                ignore_unavailable=payload.get("ignore_unavailable", True),
            )

        if operation == "save_batch":
            return es_tool._save_batch_local(
                hash_ids=payload["hash_ids"],
                doc_infos=payload["doc_infos"],
                embeddings=payload["embeddings"],
                index_name=payload["index_name"],
            )

        if operation == "delete_by_query":
            return es_tool._delete_by_query_local(
                index_name=payload["index_name"],
                body=payload["body"],
                refresh=payload.get("refresh", True),
            )

        raise ValueError(f"Unsupported ES queue operation: {operation}")

    def _start_heartbeat(self):
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="redis-es-write-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self):
        self._set_is_leader(False)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1)
        self._heartbeat_thread = None

    def _heartbeat_loop(self):
        interval = max(1, self.lock_ttl_seconds // 3)
        while not self._stop_event.is_set() and self._is_leader():
            time.sleep(interval)
            try:
                if self.redis.get(self.lock_key) != self.instance_id:
                    self._set_is_leader(False)
                    break
                self.redis.expire(self.lock_key, self.lock_ttl_seconds)
            except Exception:
                logger.exception("Failed to refresh Redis ES write lock")
                self._set_is_leader(False)
                break

    def _try_acquire_leader(self) -> bool:
        acquired = self.redis.set(
            self.lock_key,
            self.instance_id,
            nx=True,
            ex=self.lock_ttl_seconds,
        )
        self._set_is_leader(bool(acquired))
        return bool(acquired)

    def _release_leader(self):
        try:
            if self.redis.get(self.lock_key) == self.instance_id:
                self.redis.delete(self.lock_key)
        except Exception:
            logger.exception("Failed to release Redis ES write lock")
        finally:
            self._set_is_leader(False)

    def _is_leader(self) -> bool:
        with self._leader_lock:
            return self._is_leader_flag

    def _set_is_leader(self, value: bool):
        with self._leader_lock:
            self._is_leader_flag = value

    def _result_key(self, job_id: str) -> str:
        return f"{self.result_prefix}:{job_id}"
