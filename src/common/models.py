from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchMode(str, Enum):
    """检索模式：官方图算法与工程快速检索。"""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    LINEAR = "linear"
    LINEAR_LOCAL = "linear_local"


@dataclass(slots=True)
class ParsedDocument:
    """文档解析后的统一结果，避免业务层依赖具体 PDF 解析库。"""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchDocument:
    """可写入任意搜索实现的文档模型。"""

    id: str
    text: str
    vector: list[float] | None = None
    doc_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_source(self) -> dict[str, Any]:
        """转换为基础设施适配器可写入的原始文档结构。"""

        source = {
            "hash_id": self.id,
            "text": self.text,
            "type": self.doc_type,
            "vector": self.vector,
        }
        source.update(self.metadata)
        source["metadata"] = {
            "type": self.doc_type,
            **self.metadata,
        }
        return source


@dataclass(slots=True)
class SearchHit:
    """检索命中的统一返回模型。"""

    id: str
    score: float
    document: SearchDocument

    @property
    def source(self) -> dict[str, Any]:
        """返回兼容旧代码的扁平化命中数据。"""

        source = self.document.as_source()
        source["score"] = self.score
        return source


@dataclass(slots=True)
class SearchQuery:
    """统一检索请求，屏蔽不同搜索数据库的查询 DSL。"""

    index_names: list[str]
    mode: SearchMode
    query_text: str | None = None
    query_vector: list[float] | None = None
    top_k: int = 10
    num_candidates: int = 100
    filters: dict[str, Any] = field(default_factory=dict)
    source_fields: list[str] | None = None
    text_fields: list[str] = field(default_factory=lambda: ["text"])
    filter_only: bool = False


@dataclass(slots=True)
class WriteResult:
    """批量写入结果。"""

    success: int = 0
    failed: int = 0
