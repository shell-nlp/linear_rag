from __future__ import annotations

from pydantic import BaseModel, Field

from src.common.models import SearchMode


class RetrievePayload(BaseModel):
    """知识库检索请求。"""

    questions: str
    index_names: list[str]
    top_k: int = 3
    search_mode: SearchMode = Field(
        default=SearchMode.VECTOR,
        description="检索模式：vector、bm25 或 hybrid",
    )


class SearchStorePayload(BaseModel):
    """搜索库字段查询请求。"""

    index_name: str = Field(description="要查询的搜索索引名称")
    field_name: str = Field(description="要查询的字段名")
    search_key: str = Field(description="查询关键词")
    use_vector: bool = Field(default=False, description="是否使用向量查询")
    search_mode: SearchMode | None = Field(
        default=None,
        description="显式指定检索模式，未填写时根据 use_vector 兼容旧请求",
    )
    top_k: int = Field(default=10, ge=1, description="返回结果数量")
