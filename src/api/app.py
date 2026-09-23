from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.bootstrap import build_application_state, close_application_state
from src.frontend import router as frontend_router
from src.health import router as health_router
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
    app.include_router(health_router.router)
    app.include_router(frontend_router.router)
    return app
