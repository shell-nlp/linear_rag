from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class LinearRAGConfig(BaseModel):
    """索引和检索运行时配置。"""

    embedding_model: Any = None
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


class Settings(BaseSettings):
    """项目环境配置，统一从环境变量或 .env 读取。"""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 大模型与向量模型
    llm_api_key: str = Field("sk", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(
        "http://192.168.102.19:8082/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_model_name: str = Field("qwen3", validation_alias="LLM_MODEL_NAME")
    embedding_api_url: str = Field(
        "http://192.168.102.19:8082/v1/embeddings",
        validation_alias="EMBEDDING_API_URL",
    )
    embedding_model_name: str = Field(
        "qwen3-embedding",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dim: int = Field(
        1024,
        validation_alias=AliasChoices("EMBEDDING_DIM", "embdding_dim"),
    )
    spacy_model: str = Field("zh_core_web_md", validation_alias="SPACY_MODEL")
    max_workers: int = Field(16, validation_alias="MAX_WORKERS")
    chunk_token_size: int = Field(1000, validation_alias="CHUNK_TOKEN_SIZE")
    chunk_overlap_token_size: int = Field(
        100,
        validation_alias="CHUNK_OVERLAP_TOKEN_SIZE",
    )
    batch_size: int = Field(128, validation_alias="BATCH_SIZE")
    retrieval_top_k: int = Field(5, validation_alias="RETRIEVAL_TOP_K")
    max_iterations: int = Field(3, validation_alias="MAX_ITERATIONS")
    top_k_sentence: int = Field(1, validation_alias="TOP_K_SENTENCE")
    passage_ratio: float = Field(1.5, validation_alias="PASSAGE_RATIO")
    passage_node_weight: float = Field(
        0.05,
        validation_alias="PASSAGE_NODE_WEIGHT",
    )
    damping: float = Field(0.5, validation_alias="DAMPING")
    iteration_threshold: float = Field(
        0.5,
        validation_alias="ITERATION_THRESHOLD",
    )
    working_dir: str = Field("./import_qwen_new", validation_alias="WORKING_DIR")

    # Neo4j
    neo4j_uri: str = Field(
        "bolt://192.168.102.19:17687",
        validation_alias="NEO4J_URI",
    )
    neo4j_user: str = Field("neo4j", validation_alias="NEO4J_USER")
    neo4j_password: str = Field("neo4j@2025", validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field("neo4j", validation_alias="NEO4J_DATABASE")

    # MinIO
    minio_service_addresses: str = Field(
        "192.168.102.19:9001",
        validation_alias="minio_service_addresses",
    )
    minio_access_key: str = Field(
        "minioadmin",
        validation_alias="minio_access_key",
    )
    minio_secret_key: str = Field(
        "minioadmin",
        validation_alias="minio_secret_key",
    )

    # Elasticsearch
    es_url: str = Field(
        "http://192.168.102.19:9200",
        validation_alias="es_url",
    )
    es_user: str = Field("elastic", validation_alias="es_user")
    es_password: str = Field("elastic@2024", validation_alias="es_password")

    # Redis
    redis_url: str = Field("redis://127.0.0.1:6379/0", validation_alias="REDIS_URL")
    redis_db: int = Field(0, validation_alias="REDIS_DB")
    redis_password: str = Field("", validation_alias="REDIS_PASSWORD")
    redis_sentinel_master: str = Field(
        "",
        validation_alias="REDIS_SENTINEL_MASTER",
    )
    redis_sentinel_nodes: str = Field(
        "",
        validation_alias="REDIS_SENTINEL_NODES",
    )

    def runtime_config(self, embedding_model: Any = None) -> LinearRAGConfig:
        """构造索引和检索运行时配置。"""

        return LinearRAGConfig(
            embedding_model=embedding_model,
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap_token_size,
            spacy_model=self.spacy_model,
            working_dir=self.working_dir,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            retrieval_top_k=self.retrieval_top_k,
            max_iterations=self.max_iterations,
            top_k_sentence=self.top_k_sentence,
            passage_ratio=self.passage_ratio,
            passage_node_weight=self.passage_node_weight,
            damping=self.damping,
            iteration_threshold=self.iteration_threshold,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内唯一的配置实例。"""

    return Settings()
