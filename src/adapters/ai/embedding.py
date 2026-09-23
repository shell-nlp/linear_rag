from __future__ import annotations

from typing import Sequence

from openai import OpenAI


class OpenAIEmbeddingProvider:
    """基于官方 openai SDK 的 OpenAI 兼容向量模型客户端。"""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float | None = None,
    ):
        """初始化向量客户端，不依赖 LangChain。"""

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def encode(
        self,
        texts: str | Sequence[str],
        batch_size: int = 32,
        **kwargs,
    ) -> list[list[float]]:
        """批量生成文本向量，并保持输入顺序。"""

        if isinstance(texts, str):
            texts = [texts]
        inputs = list(texts)
        if not inputs:
            return []

        # 官方接口支持批量输入，batch_size 仅用于限制单次请求规模。
        result: list[list[float]] = []
        api_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in {"dimensions", "encoding_format", "user"}
        }
        for start in range(0, len(inputs), max(1, batch_size)):
            batch = inputs[start : start + max(1, batch_size)]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch,
                **api_kwargs,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            result.extend([list(item.embedding) for item in ordered])
        return result
