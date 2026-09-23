"""大模型与向量模型统一入口。

业务代码统一从本模块导入模型端口、具体实现和工厂方法，
后续切换模型供应商时只需要修改这里或替换工厂实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.settings import get_settings
from src.adapters.ai.embedding import OpenAIEmbeddingProvider
from src.adapters.ai.llm import OpenAICompatibleLLM
from src.interfaces.embedding import EmbeddingProvider
from src.interfaces.llm import LLMProvider


@dataclass(slots=True)
class ModelProviders:
    """应用启动时统一持有的大模型和向量模型实例。"""

    embedding: EmbeddingProvider
    llm: LLMProvider


def create_embedding_provider(
    api_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> EmbeddingProvider:
    """创建向量模型实例。"""

    settings = get_settings()
    return OpenAIEmbeddingProvider(
        base_url=(api_url or settings.embedding_api_url).removesuffix("/embeddings"),
        model_name=model_name or settings.embedding_model_name,
        api_key=api_key or settings.llm_api_key,
    )


def create_llm_provider(
    base_url: str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """创建大模型实例。"""

    settings = get_settings()
    return OpenAICompatibleLLM(
        base_url=base_url or settings.llm_base_url,
        model_name=model_name or settings.llm_model_name,
        api_key=api_key or settings.llm_api_key,
    )


def create_model_providers() -> ModelProviders:
    """按统一配置一次性创建模型实例。"""

    return ModelProviders(
        embedding=create_embedding_provider(),
        llm=create_llm_provider(),
    )


__all__ = [
    "EmbeddingProvider",
    "LLMProvider",
    "ModelProviders",
    "OpenAICompatibleLLM",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
    "create_llm_provider",
    "create_model_providers",
]
