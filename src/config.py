import os
from dataclasses import dataclass


@dataclass
class LinearRAGConfig:
    embedding_model: str = "all-mpnet-base-v2"
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "zh_core_web_md"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5


LLM_API_KEY = os.getenv("LLM_API_KEY", "sk")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://192.168.102.19:8082/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3")
EMBEDDING_API_URL = os.getenv(
    "EMBEDDING_API_URL", "http://192.168.102.19:8082/v1/embeddings"
)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding")
SPACY_MODEL = os.getenv("SPACY_MODEL", "zh_core_web_md")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "16"))

# Neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j@2025")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
