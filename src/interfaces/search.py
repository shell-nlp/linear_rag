from __future__ import annotations

from typing import Any, Protocol, Sequence

from src.models import (
    SearchDocument,
    SearchHit,
    SearchQuery,
    WriteResult,
)


class SearchStore(Protocol):
    """统一向量检索、BM25 检索和混合检索的搜索数据库端口。"""

    def create_index(
        self,
        index_name: str,
        vector_dim: int,
        settings: dict[str, Any] | None = None,
    ) -> None:
        """创建索引，具体数据库自行决定映射结构。"""

        ...

    def delete_index(self, index_name: str, ignore_unavailable: bool = True) -> None:
        """删除索引。"""

        ...

    def index_exists(self, index_name: str) -> bool:
        """判断索引是否存在。"""

        ...

    def get_index_mapping(self, index_name: str) -> dict[str, Any]:
        """返回统一的索引字段结构。"""

        ...

    def get_existing_ids(
        self,
        index_name: str,
        ids: Sequence[str],
    ) -> set[str]:
        """查询已存在的文档 ID，用于增量去重。"""

        ...

    def upsert_documents(
        self,
        index_name: str,
        documents: Sequence[SearchDocument],
        refresh: bool = True,
    ) -> WriteResult:
        """批量新增或覆盖文档。"""

        ...

    def delete_documents_by_ids(
        self,
        index_name: str,
        ids: Sequence[str],
        refresh: bool = True,
    ) -> WriteResult:
        """按文档 ID 批量删除。"""

        ...

    def delete_documents_by_filters(
        self,
        index_name: str,
        filters: dict[str, Any],
        refresh: bool = True,
    ) -> WriteResult:
        """按字段条件批量删除。"""

        ...

    def get_documents_by_ids(
        self,
        index_names: Sequence[str],
        ids: Sequence[str],
    ) -> dict[str, SearchDocument]:
        """按文档 ID 批量回查完整数据。"""

        ...

    def search(self, query: SearchQuery) -> list[SearchHit]:
        """执行统一检索请求。"""

        ...
