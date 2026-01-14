from hashlib import md5
import httpx
from openai import OpenAI
import re
import string
import logging
import numpy as np
import os

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def setup_logging(log_file):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8'))
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )
    # Suppress noisy HTTP request logs (e.g., 401 Unauthorized) from httpx/openai
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

