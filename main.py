import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.model_providers import (
    EmbeddingProvider,
    LLMProvider,
    create_model_providers,
)
from src.common.settings import get_settings
from src.services.knowledge_base_service import KnowledgeBaseService
from src.services.mappers import documents_to_passages
from src.common.utils import get_es_client, get_redis_client, setup_logging
from src.models import SearchMode
from src.adapters.graph import Neo4jDriver, Neo4jGraphStore
from src.adapters.graph.neo4j_write_queue import RedisNeo4jWriteQueue
from src.adapters.search import ElasticsearchSearchStore
from src.document_processing.pdf_parser import PDFParser
from src.services.linear_rag import LinearRAG

settings = get_settings()


warnings.filterwarnings("ignore")


INDEX_PROCESS_WORKERS = int(os.getenv("INDEX_PROCESS_WORKERS", 10))
logger.info(f"索引处理进程数设置为 {INDEX_PROCESS_WORKERS}")


class IndexPayload(BaseModel):
    kb_name: str = Field(description="知识库索引名称")
    bucket_name: str = Field(description="MinIO 桶名称")
    file_path: str = Field(description="MinIO 文件路径")
    file_id: str | None = Field(default=None, description="文件 ID，可选")


class SinglePassagePayload(BaseModel):
    index_name: str = Field(description="知识库索引名称")
    texts: List[str] = Field(description="要索引的文本片段列表")
    keyword: str = Field(description="关键词")


class DeleteSinglePassagePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    index_name: str = Field(description="知识库索引名称")
    doc_id: str | List[str] | None = Field(
        default=None, alias="_id", description="ES 文档 _id 或 _id 列表"
    )
    doc_ids: List[str] | None = Field(
        default=None, alias="_ids", description="ES 文档 _id 列表"
    )
    ids: List[str] | None = Field(default=None, description="ES 文档 _id 列表")

    @model_validator(mode="after")
    def require_ids(self):
        if not self.normalized_ids():
            raise ValueError("At least one of _id, _ids, or ids is required")
        return self

    def normalized_ids(self) -> List[str]:
        ids: List[str] = []
        if self.doc_id:
            if isinstance(self.doc_id, list):
                ids.extend(self.doc_id)
            else:
                ids.append(self.doc_id)
        if self.doc_ids:
            ids.extend(self.doc_ids)
        if self.ids:
            ids.extend(self.ids)
        return list(dict.fromkeys(doc_id for doc_id in ids if doc_id))


class RetrievePayload(BaseModel):
    questions: str
    index_names: List[str]
    top_k: int = 3
    search_mode: SearchMode = Field(
        default=SearchMode.VECTOR,
        description="检索模式：vector、bm25 或 hybrid",
    )


class DeletePayload(BaseModel):
    index_name: str
    file_ids: List[str]


class ESSearchPayload(BaseModel):
    index_name: str = Field(description="要查询的 ES 索引名称")
    field_name: str = Field(description="要查询的 ES 字段名")
    search_key: str = Field(description="查询关键词")
    use_vector: bool = Field(default=False, description="是否使用向量查询")
    search_mode: SearchMode | None = Field(
        default=None,
        description="显式指定检索模式，未填写时根据 use_vector 兼容旧请求",
    )
    top_k: int = Field(default=10, ge=1, description="返回结果数量")


class Response(BaseModel):
    code: str = Field(default="0", description="状态码")
    msg: str = Field(default="ok", description="状态描述")
    data: Any = Field(description="数据")


class AppState:
    rag_model: LinearRAG | None = None
    kb_service: KnowledgeBaseService | None = None
    embedding_provider: EmbeddingProvider | None = None
    llm_provider: LLMProvider | None = None
    index_process_pool: ProcessPoolExecutor | None = None
    neo4j_write_queue: RedisNeo4jWriteQueue | None = None


state = AppState()


def describe_redis_connection() -> str:
    if settings.redis_sentinel_master and settings.redis_sentinel_nodes:
        return (
            f"sentinel master={settings.redis_sentinel_master}, "
            f"nodes={settings.redis_sentinel_nodes}"
        )
    return settings.redis_url


def _resolve_index_passages_worker(
    bucket_name: str, file_path: str, file_id: str | None
) -> Dict[str, List[Any]]:
    parser = PDFParser(
        bucket_name=bucket_name,
        file_path=file_path,
        file_id=file_id,
    )
    documents = parser.get_chunk()
    return documents_to_passages(documents)


def resolve_index_passages(payload: IndexPayload) -> Dict[str, List[Any]]:
    if state.index_process_pool:
        future = state.index_process_pool.submit(
            _resolve_index_passages_worker,
            payload.bucket_name,
            payload.file_path,
            payload.file_id,
        )
        passages = future.result()
    else:
        passages = _resolve_index_passages_worker(
            payload.bucket_name,
            payload.file_path,
            payload.file_id,
        )

    if not passages["text"]:
        raise HTTPException(status_code=400, detail="PDFParser did not return chunks")
    return passages


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 初始化所有连接
    print("正在初始化系统资源...")
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, "log.txt"))
    state.index_process_pool = ProcessPoolExecutor(max_workers=INDEX_PROCESS_WORKERS)
    print(f"PDF解析进程池初始化完成，workers={INDEX_PROCESS_WORKERS}")
    redis_client = get_redis_client()
    redis_client.ping()

    model_providers = create_model_providers()
    embedding_model = model_providers.embedding
    state.llm_provider = model_providers.llm

    es_client = get_es_client()
    search_store = ElasticsearchSearchStore(es_client)

    neo4j_driver = Neo4jDriver(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    state.neo4j_write_queue = RedisNeo4jWriteQueue(
        neo4j_driver=neo4j_driver,
        redis_client=redis_client,
    )
    state.neo4j_write_queue.start()
    print(f"Redis Neo4j 写入队列已启动: {describe_redis_connection()}")
    graph_store = Neo4jGraphStore(
        graph_driver=neo4j_driver,
        write_queue=state.neo4j_write_queue,
    )

    config = settings.runtime_config(embedding_model)

    state.rag_model = LinearRAG(
        global_config=config,
        search_store=search_store,
        graph_store=graph_store,
        embedding_provider=embedding_model,
    )
    state.embedding_provider = embedding_model
    state.kb_service = KnowledgeBaseService(
        search_store=search_store,
        embedding_provider=embedding_model,
        vector_dim=settings.embedding_dim,
    )
    print("系统初始化完成，准备就绪。")
    yield
    print("正在关闭系统...")
    if state.index_process_pool:
        state.index_process_pool.shutdown(wait=True, cancel_futures=True)
        state.index_process_pool = None
    if state.neo4j_write_queue:
        state.neo4j_write_queue.close()
        state.neo4j_write_queue = None


app = FastAPI(title="LinearRAG API Service", lifespan=lifespan)


@app.post("/admin_api/python-knowledge-management/index", response_model=Response)
def index_documents(payload: IndexPayload):
    """
    建立索引接口
    """
    try:
        passages = resolve_index_passages(payload)
        state.rag_model.index(passages=passages, kb_name=payload.kb_name)
        file_ids = sorted(
            {str(file_id) for file_id in passages.get("file_id", []) if file_id}
        )
        return Response(
            code="0",
            msg="ok",
            data={
                "status": "success",
                "message": f"Successfully indexed into {payload.kb_name}",
                "chunk_count": len(passages.get("text", [])),
                "file_ids": file_ids,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/admin_api/python-knowledge-management/index_single", response_model=Response
)
def index_single_passage(payload: SinglePassagePayload):
    """
    向搜索库上传单独片段接口，文本会自动向量化
    """
    try:
        hash_ids = state.kb_service.upsert_passages(
            index_name=payload.index_name,
            texts=payload.texts,
            keyword=payload.keyword,
        )

        return Response(
            code="0",
            msg="ok",
            data={
                "status": "success",
                "message": f"Successfully indexed {len(payload.texts)} passages into {payload.index_name}",
                "count": len(payload.texts),
                "hash_ids": hash_ids,
            },
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/admin_api/python-knowledge-management/delete_single", response_model=Response
)
def delete_single_passage(payload: DeleteSinglePassagePayload):
    """
    按文档 ID 删除 index_single 上传的片段
    """
    try:
        ids = payload.normalized_ids()
        result = state.kb_service.delete_passages_by_ids(
            index_name=payload.index_name,
            ids=ids,
        )

        return Response(
            code="0",
            msg="ok",
            data={
                "status": "success",
                "message": f"Deleted {result.get('deleted', 0)} passages from {payload.index_name}",
                **result,
            },
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin_api/python-knowledge-management/retrieve", response_model=Response)
def retrieve_documents(payload: RetrievePayload):
    """
    检索接口
    """
    try:
        questions = payload.questions
        results = state.rag_model.retrieve(
            questions,
            index_names=payload.index_names,
            top_k=payload.top_k,
            search_mode=payload.search_mode,
        )
        return Response(code="0", msg="ok", data=results[0])
    except Exception as e:
        logger.exception("retrieve_documents failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/admin_api/python-knowledge-management/delete_files", response_model=Response
)
def delete_files(payload: DeletePayload):
    """
    删除文件索引接口
    """
    try:
        state.rag_model.delete_files(
            index_name=payload.index_name, file_ids=payload.file_ids
        )
        return Response(
            code="0",
            msg="ok",
            data={
                "status": "success",
                "message": f"File {payload.file_ids} deleted from {payload.index_name}",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteKBRequest(BaseModel):
    index_name: str = Field(description="知识库的索引名称")


# 删除知识库索引
@app.post(
    "/admin_api/python-knowledge-management/delete_knowledgebase",
    response_model=Response,
)
def delete_knowledgebase(request: DeleteKBRequest):
    """
    删除知识库接口
    """
    logger.info(f"入参：\n{request.model_dump_json(indent=2)}")
    state.kb_service.delete_knowledgebase(request.index_name)
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"Knowledge base {request.index_name} deleted",
        },
    )


class CreateKBRequest(BaseModel):
    index_name: str = Field(description="知识库的索引名称")


@app.post(
    "/admin_api/python-knowledge-management/create_knowledgebase",
    response_model=Response,
)
def create_knowledgebase(request: CreateKBRequest):
    """创建知识库"""
    logger.info(f"入参：\n{request.model_dump_json(indent=2)}")
    state.kb_service.create_knowledgebase(request.index_name)
    return Response(
        code="0",
        msg="ok",
        data={
            "status": "success",
            "message": f"Knowledge base {request.index_name} created",
        },
    )


@app.post("/admin_api/python-knowledge-management/search_es", response_model=Response)
def search_es_documents(payload: ESSearchPayload):
    """
    搜索库查询接口，支持向量、BM25 和混合检索
    """
    try:
        search_mode = payload.search_mode or (
            SearchMode.VECTOR if payload.use_vector else SearchMode.BM25
        )
        data = state.kb_service.search(
            index_name=payload.index_name,
            field_name=payload.field_name,
            search_key=payload.search_key,
            mode=search_mode,
            top_k=payload.top_k,
        )
        return Response(code="0", msg="ok", data=data)
    except ValueError as exc:
        detail = str(exc)
        status_code = 404 if "not found" in detail else 400
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/admin_api/python-knowledge-management/health", response_model=Response)
def health_check():
    return Response(code="0", msg="ok", data={"status": "alive"})


@app.get("/")
def serve_frontend():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return FileResponse(os.path.join(current_dir, "index.html"))


service_port = 12125
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=service_port)
