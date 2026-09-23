from __future__ import annotations

from typing import Any, Sequence

from openai import OpenAI


class OpenAICompatibleLLM:
    """基于官方 openai SDK 的 OpenAI 兼容大模型客户端。"""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str,
        timeout: float | None = None,
    ):
        """初始化客户端，不依赖 LangChain。"""

        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs,
    ) -> str:
        """执行对话补全，并返回第一条文本结果。"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=list(messages),
            **kwargs,
        )
        if not response.choices:
            return ""
        return response.choices[0].message.content or ""
