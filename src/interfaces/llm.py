from __future__ import annotations

from typing import Any, Protocol, Sequence


class LLMProvider(Protocol):
    """大模型调用端口，用于隔离 OpenAI 兼容 API 或其它模型服务。"""

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        **kwargs,
    ) -> str:
        """执行对话补全并返回文本内容。"""

        ...
