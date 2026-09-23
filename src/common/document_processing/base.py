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
    """段落实体识别端口，用于替换 spaCy、LLM 或其它 NER 实现。"""

    def extract_passage_entities(
        self,
        hash_id_to_passage: Mapping[str, str],
        max_workers: int,
    ) -> dict[str, list[str]]:
        """批量提取段落中的实体。"""

        ...
