from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import find_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 自动从当前源码目录向上查找 .env；Docker 仅注入环境变量时允许文件不存在。
ENV_FILE = find_dotenv(filename=".env", raise_error_if_not_found=False)


class LinearRAGConfig(BaseModel):
    """索引和检索运行时配置。"""

    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100
    spacy_model: str = "zh_core_web_md"
    working_dir: str = "./import"
    batch_size: int = 128
    max_workers: int = 16
    entity_expansion_enabled: bool = True
    entity_expansion_max_entities: int = 20
    entity_expansion_top_k: int = 50
    neighbor_expansion_enabled: bool = True
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    linear_local_candidates: int = 200
    linear_max_nodes: int = 100000
    linear_embedding_batch_size: int = 128


class Settings(BaseSettings):
    """项目环境配置，统一从环境变量或 .env 读取。"""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE or None,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # 大模型与向量模型
    llm_api_key: str = Field("sk", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(
        "http://localhost:8082/v1",
        validation_alias="LLM_BASE_URL",
    )
    llm_model_name: str = Field("qwen3", validation_alias="LLM_MODEL_NAME")
    embedding_api_url: str = Field(
        "http://localhost:8082/v1/embeddings",
        validation_alias="EMBEDDING_API_URL",
    )
    embedding_model_name: str = Field(
        "qwen3-embedding",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    embedding_dim: int = Field(
        1024,
        validation_alias="EMBEDDING_DIM",
    )
    spacy_model: str = Field("zh_core_web_md", validation_alias="SPACY_MODEL")
    max_workers: int = Field(16, validation_alias="MAX_WORKERS")
    api_port: int = Field(12125, validation_alias="API_PORT")
    index_process_workers: int = Field(
        10,
        validation_alias="INDEX_PROCESS_WORKERS",
    )
    chunk_token_size: int = Field(1000, validation_alias="CHUNK_TOKEN_SIZE")
    chunk_overlap_token_size: int = Field(
        100,
        validation_alias="CHUNK_OVERLAP_TOKEN_SIZE",
    )
    batch_size: int = Field(128, validation_alias="BATCH_SIZE")
    entity_expansion_enabled: bool = Field(
        True,
        validation_alias="ENTITY_EXPANSION_ENABLED",
    )
    entity_expansion_max_entities: int = Field(
        20,
        ge=1,
        validation_alias="ENTITY_EXPANSION_MAX_ENTITIES",
    )
    entity_expansion_top_k: int = Field(
        50,
        ge=1,
        validation_alias="ENTITY_EXPANSION_TOP_K",
    )
    neighbor_expansion_enabled: bool = Field(
        True,
        validation_alias="NEIGHBOR_EXPANSION_ENABLED",
    )
    max_iterations: int = Field(3, ge=1, validation_alias="LINEAR_MAX_ITERATIONS")
    top_k_sentence: int = Field(1, ge=1, validation_alias="LINEAR_TOP_K_SENTENCE")
    passage_ratio: float = Field(1.5, ge=0, validation_alias="LINEAR_PASSAGE_RATIO")
    passage_node_weight: float = Field(
        0.05, ge=0, validation_alias="LINEAR_PASSAGE_NODE_WEIGHT"
    )
    damping: float = Field(0.5, gt=0, lt=1, validation_alias="LINEAR_DAMPING")
    iteration_threshold: float = Field(
        0.5, ge=0, validation_alias="LINEAR_ITERATION_THRESHOLD"
    )
    linear_local_candidates: int = Field(
        200, ge=1, validation_alias="LINEAR_LOCAL_CANDIDATES"
    )
    linear_max_nodes: int = Field(
        100000, ge=1, validation_alias="LINEAR_MAX_NODES"
    )
    linear_embedding_batch_size: int = Field(
        128, ge=1, validation_alias="LINEAR_EMBEDDING_BATCH_SIZE"
    )
    max_upload_bytes: int = Field(
        100 * 1024 * 1024,
        ge=1,
        validation_alias="MAX_UPLOAD_BYTES",
    )
    working_dir: str = Field("./import_qwen_new", validation_alias="WORKING_DIR")

    # 对象存储：默认使用本地实现，显式选择 minio 时才连接远端。
    object_storage_provider: Literal["local", "minio"] = Field(
        "local",
        validation_alias="OBJECT_STORAGE_PROVIDER",
    )
    local_storage_root: str = Field(
        "./.data/object-storage",
        validation_alias="LOCAL_STORAGE_ROOT",
    )
    minio_endpoint_url: str = Field(
        "",
        validation_alias="MINIO_ENDPOINT_URL",
    )
    minio_access_key: str = Field(
        "",
        validation_alias="MINIO_ACCESS_KEY",
    )
    minio_secret_key: str = Field(
        "",
        validation_alias="MINIO_SECRET_KEY",
    )

    # Elasticsearch
    es_url: str = Field(
        "http://localhost:9200",
        validation_alias="es_url",
    )
    es_user: str = Field("elastic", validation_alias="es_user")
    es_password: str = Field("elastic@2024", validation_alias="es_password")

    def runtime_config(self) -> LinearRAGConfig:
        """构造索引和检索运行时配置。"""

        return LinearRAGConfig(
            chunk_token_size=self.chunk_token_size,
            chunk_overlap_token_size=self.chunk_overlap_token_size,
            spacy_model=self.spacy_model,
            working_dir=self.working_dir,
            batch_size=self.batch_size,
            max_workers=self.max_workers,
            entity_expansion_enabled=self.entity_expansion_enabled,
            entity_expansion_max_entities=self.entity_expansion_max_entities,
            entity_expansion_top_k=self.entity_expansion_top_k,
            neighbor_expansion_enabled=self.neighbor_expansion_enabled,
            max_iterations=self.max_iterations,
            top_k_sentence=self.top_k_sentence,
            passage_ratio=self.passage_ratio,
            passage_node_weight=self.passage_node_weight,
            damping=self.damping,
            iteration_threshold=self.iteration_threshold,
            linear_local_candidates=self.linear_local_candidates,
            linear_max_nodes=self.linear_max_nodes,
            linear_embedding_batch_size=self.linear_embedding_batch_size,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内唯一的配置实例。"""

    return Settings()
