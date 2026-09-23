from __future__ import annotations

from typing import Mapping, Protocol, Sequence


class EntityExtractor(Protocol):
    """实体识别端口，用于替换 spaCy、LLM 或其它 NER 实现。"""

    def extract_question_entities(self, question: str) -> set[str]:
        """提取问题中的实体。"""

        ...

    def extract_passage_entities(
        self,
        hash_id_to_passage: Mapping[str, str],
        max_workers: int,
    ) -> dict[str, list[str]]:
        """批量提取段落中的实体。"""

        ...
