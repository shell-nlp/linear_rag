from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SinglePassagePayload(BaseModel):
    """独立文本片段索引请求。"""

    index_name: str = Field(description="知识库索引名称")
    texts: list[str] = Field(description="要索引的文本片段列表")
    keyword: str = Field(description="关键词")


class DeleteSinglePassagePayload(BaseModel):
    """独立文本片段删除请求。"""

    model_config = ConfigDict(populate_by_name=True)

    index_name: str = Field(description="知识库索引名称")
    doc_id: str | list[str] | None = Field(
        default=None,
        alias="_id",
        description="ES 文档 _id 或 _id 列表",
    )
    doc_ids: list[str] | None = Field(
        default=None,
        alias="_ids",
        description="ES 文档 _id 列表",
    )
    ids: list[str] | None = Field(default=None, description="ES 文档 _id 列表")

    @model_validator(mode="after")
    def require_ids(self):
        """至少提供一个文档 ID。"""

        if not self.normalized_ids():
            raise ValueError("At least one of _id, _ids, or ids is required")
        return self

    def normalized_ids(self) -> list[str]:
        """合并并去重所有文档 ID。"""

        ids: list[str] = []
        if self.doc_id:
            ids.extend(self.doc_id if isinstance(self.doc_id, list) else [self.doc_id])
        if self.doc_ids:
            ids.extend(self.doc_ids)
        if self.ids:
            ids.extend(self.ids)
        return list(dict.fromkeys(doc_id for doc_id in ids if doc_id))


class DeleteFilesPayload(BaseModel):
    """按文件删除索引请求。"""

    index_name: str
    file_ids: list[str]
