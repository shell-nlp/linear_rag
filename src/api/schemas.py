from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Response(BaseModel):
    """统一 API 响应结构。"""

    code: str = Field(default="0", description="状态码")
    msg: str = Field(default="ok", description="状态描述")
    data: Any = Field(description="响应数据")
