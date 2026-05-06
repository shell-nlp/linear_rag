import asyncio
import json
import os
import warnings
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import nacos
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, Field

from src.config import (
    EMBEDDING_API_URL,
    EMBEDDING_MODEL_NAME,
    LLM_API_KEY,
    LLM_BASE_URL,
    LOCAL_IP,
    MAX_WORKERS,
    NACOS_NAMESPACE,
    NACOS_SERVER_ADDRESSES,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    SERVICE_NAME,
    SPACY_MODEL,
    LinearRAGConfig,
    embdding_dim,
)
from src.embedding import LocalOpenAIEmbeddingModel
from src.es import Customize_Elastic
from src.graphs_utils.neo4j_db import Neo4jGraph
from src.LinearRAG import LinearRAG
from src.text_splitter import PDFParser
from src.utils import compute_mdhash_id, get_es_client, setup_logging

scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
client = nacos.NacosClient(NACOS_SERVER_ADDRESSES, namespace=NACOS_NAMESPACE)
local_ip = LOCAL_IP


@scheduler.scheduled_job("interval", seconds=6)
async def beat():
    await asyncio.to_thread(
        client.add_naming_instance,
        SERVICE_NAME,
        local_ip,
        service_port,
        group_name="DEFAULT_GROUP",
    )


default_settings = """{"settings": {"index.analysis.analyzer.default.type": "ik_smart", "index.number_of_replicas": "1", "index.number_of_shards": "1", "index.routing.allocation.include._tier_preference": "data_content"}, 
"mappings": {"properties": 
{
"content_image": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"content_pages_number": {"type": "long"}, "file_id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"file_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"metadata": {"properties": {"content_image": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"content_pages_number": {"type": "long"}, 
"file_id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"file_name": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"parent_text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"segment_id": {"type": "long"}, 
"shared_tenant_id_list": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"state": {"type": "boolean"}, "tenant_id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}}}, 
"parent_text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"segment_id": {"type": "long"}, 
"shared_tenant_id_list": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"state": {"type": "boolean"}, "tenant_id": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"text": {"type": "text", "fields": {"keyword": {"type": "keyword", "ignore_above": 256}}}, 
"vector": {"type": "dense_vector", "dims": 768, "index": true, "similarity": "cosine"}}}}"""

default_settings = json.loads(default_settings)
default_settings["mappings"]["properties"]["vector"]["dims"] = embdding_dim

warnings.filterwarnings("ignore")


es_client = get_es_client()


class IndexPayload(BaseModel):
    kb_name: str = Field(description="知识库索引名称")
    bucket_name: str = Field(description="MinIO 桶名称")
    file_path: str = Field(description="MinIO 文件路径")
    file_id: str | None = Field(default=None, description="文件 ID，可选")


class SinglePassagePayload(BaseModel):
    index_name: str = Field(description="知识库索引名称")
    texts: List[str] = Field(description="要索引的文本片段列表")
    keyword: str = Field(description="关键词")


class RetrievePayload(BaseModel):
    questions: str
    index_names: List[str]
    top_k: int = 3


class DeletePayload(BaseModel):
    index_name: str
    file_ids: List[str]


class ESSearchPayload(BaseModel):
    index_name: str = Field(description="要查询的 ES 索引名称")
    field_name: str = Field(description="要查询的 ES 字段名")
    search_key: str = Field(description="查询关键词")
    use_vector: bool = Field(default=False, description="是否使用向量查询")
    top_k: int = Field(default=10, ge=1, description="返回结果数量")


class Response(BaseModel):
    code: str = Field(default="0", description="状态码")
    msg: str = Field(default="ok", description="状态描述")
    data: Any = Field(description="数据")


class AppState:
    rag_model: LinearRAG = None


state = AppState()


def field_exists_in_mapping(properties: Dict[str, Any], field_name: str) -> bool:
    current = properties
    parts = field_name.split(".")
    for idx, part in enumerate(parts):
        field_info = current.get(part)
        if field_info is None:
            return False
        if idx == len(parts) - 1:
            return True
        if "properties" in field_info:
            current = field_info["properties"]
            continue
        if "fields" in field_info:
            current = field_info["fields"]
            continue
        return False
    return False


def validate_search_index(index_name: str, field_name: str, use_vector: bool) -> None:
    try:
        index_info = es_client.indices.get(index=index_name)
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Index '{index_name}' not found"
        ) from exc

    properties = (
        index_info.get(index_name, {}).get("mappings", {}).get("properties", {})
    )
    if not field_exists_in_mapping(properties, field_name):
        raise HTTPException(
            status_code=400,
            detail=f"Field '{field_name}' does not exist in index '{index_name}'",
        )
    if use_vector and not field_exists_in_mapping(properties, "vector"):
        raise HTTPException(
            status_code=400,
            detail=f"Index '{index_name}' does not contain vector field",
        )


def get_embedding_model() -> LocalOpenAIEmbeddingModel:
    if state.rag_model and state.rag_model.embedding_model:
        return state.rag_model.embedding_model
    return LocalOpenAIEmbeddingModel(EMBEDDING_API_URL, EMBEDDING_MODEL_NAME)


def documents_to_passages(documents: List[Any]) -> Dict[str, List[Any]]:
    passages: Dict[str, List[Any]] = {
        "text": [],
        "pages_number": [],
        "content_table": [],
        "content_image": [],
        "ori_text": [],
        "file_name": [],
        "file_id": [],
        "segment_id": [],
        "file_path": [],
        "bucket_name": [],
    }

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        text = getattr(doc, "page_content", "") or metadata.get("text", "")
        passages["text"].append(text)
        passages["pages_number"].append(metadata.get("pages_number"))
        passages["content_table"].append(metadata.get("content_table") or [])
        passages["content_image"].append(metadata.get("content_image") or [])
        passages["ori_text"].append(metadata.get("ori_text") or text)
        passages["file_name"].append(metadata.get("file_name"))
        passages["file_id"].append(metadata.get("file_id"))
        passages["segment_id"].append(metadata.get("segment_id"))
        passages["file_path"].append(metadata.get("file_path"))
        passages["bucket_name"].append(metadata.get("bucket_name"))

    return passages


def resolve_index_passages(payload: IndexPayload) -> Dict[str, List[Any]]:
    parser = PDFParser(
        bucket_name=payload.bucket_name,
        file_path=payload.file_path,
        file_id=payload.file_id,
    )
    documents = parser.get_chunk()
    passages = documents_to_passages(documents)
    if not passages["text"]:
        raise HTTPException(status_code=400, detail="PDFParser did not return chunks")
    return passages


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 初始化所有连接
    print("正在初始化系统资源...")
    scheduler.start()
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL

    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, "log.txt"))

    embedding_model = LocalOpenAIEmbeddingModel(EMBEDDING_API_URL, EMBEDDING_MODEL_NAME)

    es_client = get_es_client()

    neo4j_driver = Neo4jGraph(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    config = LinearRAGConfig(
        embedding_model=embedding_model,
        spacy_model=SPACY_MODEL,
        max_workers=MAX_WORKERS,
        working_dir="./import_qwen_new",
    )

    state.rag_model = LinearRAG(
        global_config=config, es_client=es_client, neo4j_driver=neo4j_driver
    )
    print("系统初始化完成，准备就绪。")
    yield
    print("正在关闭系统...")


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


@app.post("/admin_api/python-knowledge-management/index_single", response_model=Response)
def index_single_passage(payload: SinglePassagePayload):
    """
    向 ES 上传单独片段接口，文本会自动向量化
    """
    try:
        embedding_model = LocalOpenAIEmbeddingModel(LLM_BASE_URL, EMBEDDING_MODEL_NAME)
        vectors = embedding_model.encode(payload.texts)

        hash_ids = [
            compute_mdhash_id(text, prefix=f"{payload.keyword}-")
            for text in payload.texts
        ]

        es_tool = Customize_Elastic(es_client)
        es_tool.save_batch(
            hash_ids=hash_ids,
            doc_infos=[{"text": text} for text in payload.texts],
            embeddings=vectors,
            index_name=payload.index_name,
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


@app.post("/admin_api/python-knowledge-management/retrieve", response_model=Response)
def retrieve_documents(payload: RetrievePayload):
    """
    检索接口
    """
    try:
        questions = payload.questions
        results = state.rag_model.retrieve(
            questions, index_names=payload.index_names, top_k=payload.top_k
        )
        return Response(code="0", msg="ok", data=results[0])
    except Exception as e:
        logger.exception("retrieve_documents failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin_api/python-knowledge-management/delete_files", response_model=Response)
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
    es_client.indices.delete(index=request.index_name, ignore_unavailable=True)
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
    es_client.indices.create(index=request.index_name, body=default_settings)
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
    ES 查询接口
    """
    try:
        validate_search_index(
            index_name=payload.index_name,
            field_name=payload.field_name,
            use_vector=payload.use_vector,
        )

        es_tool = Customize_Elastic(es_client)
        if payload.use_vector:
            query_vector = get_embedding_model().encode([payload.search_key])[0]
            response = es_tool.es_search(
                index_name=payload.index_name,
                knn={
                    "field": "vector",
                    "query_vector": query_vector,
                    "k": payload.top_k,
                    "num_candidates": max(payload.top_k * 10, 100),
                    "filter": {
                        "bool": {"must": [{"exists": {"field": payload.field_name}}]}
                    },
                },
            )
        else:
            response = es_tool.es_search(
                index_name=payload.index_name,
                query_body={
                    "size": payload.top_k,
                    "query": {"match": {payload.field_name: payload.search_key}},
                },
            )

        hits = response.get("hits", {}).get("hits", [])
        data = [
            {
                "index": hit.get("_index"),
                "id": hit.get("_id"),
                "score": hit.get("_score"),
                "source": {
                    "text": hit.get("_source", {}).get("text"),
                    "hash_id": hit.get("_source", {}).get("hash_id"),
                    "type": hit.get("_source", {}).get("type"),
                },
            }
            for hit in hits
        ]

        return Response(code="0", msg="ok", data=data)
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
