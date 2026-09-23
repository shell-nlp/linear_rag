from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.api.constants import ADMIN_API_PREFIX
from src.api.bootstrap import build_application_state, close_application_state
from src.api.schemas import Response
from src.indexing import router as indexing_router
from src.knowledge_bases import router as knowledge_base_router
from src.retrieval import router as retrieval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """管理 FastAPI 应用资源生命周期。"""

    app.state.resources = build_application_state()
    try:
        yield
    finally:
        close_application_state(app.state.resources)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""

    app = FastAPI(
        title="LinearRAG API Service",
        lifespan=lifespan,
    )
    app.include_router(indexing_router.router)
    app.include_router(retrieval_router.router)
    app.include_router(knowledge_base_router.router)

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        """返回项目根目录的前端页面。"""

        project_root = Path(__file__).resolve().parents[2]
        return FileResponse(project_root / "index.html")

    @app.get(f"{ADMIN_API_PREFIX}/health", response_model=Response, tags=["健康检查"])
    def health_check():
        """服务存活检查。"""

        return Response(code="0", msg="ok", data={"status": "alive"})

    return app
