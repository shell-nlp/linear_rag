import os
import sys
from hashlib import md5

from elasticsearch import Elasticsearch
from loguru import logger

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
