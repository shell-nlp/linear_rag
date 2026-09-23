from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SearchMode(str, Enum):
    """检索模式：向量、BM25 或两者融合。"""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


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


@dataclass(slots=True)
class WriteResult:
    """批量写入结果。"""

    success: int = 0
    failed: int = 0


@dataclass(slots=True)
class GraphNode:
    """图节点领域模型。"""

    orig_id: str
    node_type: str
    name: str
    file_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphEdge:
    """图关系领域模型。"""

    source: str
    target: str
    weight: float = 0.0
    label: str = "LINK"


@dataclass(slots=True)
class GraphBatch:
    """一次增量图写入的完整批次。"""

    index_name: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    anchor_label: str = "BaseNode"
    edge_label: str = "LINK"


@dataclass(slots=True)
class GraphWriteResult:
    """图写入统计结果。"""

    index_name: str
    node_count: int
    edge_count: int


@dataclass(slots=True)
class GraphDeleteResult:
    """按文件删除图节点后的统计结果。"""

    index_name: str
    file_ids: list[str]
    total_nodes: int
    deleted_nodes: int


@dataclass(slots=True)
class RelatedPassage:
    """图数据库返回的关联段落。"""

    id: str
    text: str
    vector: list[float] | None = None


@dataclass(slots=True)
class EntityPassageLink:
    """实体与段落之间的关联事实。"""

    entity_id: str
    passage_id: str
    entity_name: str
