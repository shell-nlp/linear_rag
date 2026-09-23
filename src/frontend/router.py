from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter(tags=["前端"])


@router.get("/")
def serve_frontend():
    """返回项目根目录的前端页面。"""

    project_root = Path(__file__).resolve().parents[3]
    return FileResponse(project_root / "index.html")
