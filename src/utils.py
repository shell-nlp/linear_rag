import os
import sys
from hashlib import md5

from elasticsearch import Elasticsearch
from loguru import logger
from redis import Redis
from redis.sentinel import Sentinel

from src.settings import get_settings


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def setup_logging(log_file):
    """配置 loguru 的控制台和文件日志。"""

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=log_format,
        colorize=True,
    )
    logger.add(
        log_file,
        level="INFO",
        format=log_format,
        encoding="utf-8",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
    )


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
