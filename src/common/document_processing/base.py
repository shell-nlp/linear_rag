from __future__ import annotations

from typing import Mapping, Protocol

from src.common.models import ParsedDocument


class DocumentParser(Protocol):
    """文档解析端口，业务层只接收统一的 ParsedDocument。"""

    def parse(
        self,
        file_bytes: bytes,
        bucket_name: str,
        object_key: str,
        file_id: str,
    ) -> list[ParsedDocument]:
        """解析上传字节，并附加稳定的对象地址和文件标识。"""

        ...


class EntityExtractor(Protocol):
    """段落和问题的实体识别端口，用于替换具体 NER 实现。"""

    def extract_passage_entities(
        self,
        hash_id_to_passage: Mapping[str, str],
        max_workers: int,
    ) -> dict[str, list[str]]:
        """批量提取段落中的实体。"""

        ...

    def extract_graph_entities(
        self,
        hash_id_to_passage: Mapping[str, str],
        max_workers: int,
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
        """返回段落实体及按段落 ID 分组的句子实体关联。"""

        ...

    def extract_question_entities(self, question: str) -> list[str]:
        """提取问题中的种子实体。"""

        ...
