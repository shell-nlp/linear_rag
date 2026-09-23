from __future__ import annotations

from pydantic import BaseModel, Field


class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求。"""

    index_name: str = Field(description="知识库的索引名称")


class DeleteKnowledgeBaseRequest(BaseModel):
    """删除知识库请求。"""

    index_name: str = Field(description="知识库的索引名称")
