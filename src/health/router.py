from __future__ import annotations

from fastapi import APIRouter

from src.api.constants import ADMIN_API_PREFIX
from src.api.schemas import Response

router = APIRouter(prefix=ADMIN_API_PREFIX, tags=["健康检查"])


@router.get("/health", response_model=Response)
def health_check():
    """服务存活检查。"""

    return Response(code="0", msg="ok", data={"status": "alive"})
