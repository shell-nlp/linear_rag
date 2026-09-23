import logging
import os
from hashlib import md5

from elasticsearch import Elasticsearch
from redis import Redis
from redis.sentinel import Sentinel

from src.common.settings import get_settings


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def setup_logging(log_file):
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, format=log_format, handlers=handlers, force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_es_client():
    """
    获取 Elasticsearch 客户端
    """
    settings = get_settings()
    basic_auth = None
    if settings.es_user and settings.es_password:
        basic_auth = (settings.es_user, settings.es_password)
    return Elasticsearch(settings.es_url, basic_auth=basic_auth)


def get_redis_client():
    """
    获取 Redis 客户端
    """
    settings = get_settings()
    if settings.redis_sentinel_master and settings.redis_sentinel_nodes:
        sentinel_nodes = []
        for raw_node in settings.redis_sentinel_nodes.split(","):
            node = raw_node.strip()
            if not node:
                continue
            host, port = node.split(":", 1)
            sentinel_nodes.append((host.strip(), int(port.strip())))

        sentinel = Sentinel(
            sentinel_nodes,
            password=settings.redis_password or None,
            decode_responses=True,
        )
        return sentinel.master_for(
            settings.redis_sentinel_master,
            password=settings.redis_password or None,
            db=settings.redis_db,
            decode_responses=True,
        )

    return Redis.from_url(settings.redis_url, decode_responses=True)
