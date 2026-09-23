from __future__ import annotations

from typing import Protocol, Sequence


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
