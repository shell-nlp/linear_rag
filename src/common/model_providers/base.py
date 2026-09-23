from __future__ import annotations

from typing import Any, Protocol, Sequence


class EmbeddingProvider(Protocol):
    """向量化端口，业务层不关心 OpenAI、本地模型或其它实现。"""

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        **kwargs,
    ) -> list[list[float]]:
        """将单条或多条文本转换为向量。"""

        ...


class LLMProvider(Protocol):
    """大模型调用端口，用于隔离 OpenAI 兼容 API 或其它模型服务。"""

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs,
    ) -> str:
        """执行对话补全并返回文本内容。"""

        ...
