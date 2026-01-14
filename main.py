import os
import warnings
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import logging
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.utils import setup_logging
from elasticsearch import Elasticsearch
from src.graphs_utils.neo4j_db import Neo4jGraph
from src.embedding import LocalOpenAIEmbeddingModel

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

LLM_API_KEY = "sk"
LLM_BASE_URL = "http://192.168.102.19:8082/v1"
LLM_MODEL_NAME = "qwen3"
EMBEDDING_API_URL = "http://192.168.102.19:8082/v1/embeddings"
EMBEDDING_MODEL_NAME = "qwen3-embedding"
SPACY_MODEL = "zh_core_web_md"
MAX_WORKERS = 16

class IndexPayload(BaseModel):
    kb_name: str
    passages: Dict[str, List[Any]]

class RetrievePayload(BaseModel):
    questions: str
    index_names: List[str]
    top_k: int = 3

class DeletePayload(BaseModel):
    index_name: str
    file_ids: List[str]

class AppState:
    rag_model: LinearRAG = None

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: 初始化所有连接
    print("正在初始化系统资源...")
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    os.environ["OPENAI_BASE_URL"] = LLM_BASE_URL
    
    log_dir = f"logs/"
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(os.path.join(log_dir, "log.txt"))

    embedding_model = LocalOpenAIEmbeddingModel(EMBEDDING_API_URL, EMBEDDING_MODEL_NAME)
    
    es_client = Elasticsearch(
        ["http://localhost:9200"],
        basic_auth=("elastic", "elastic@2024")
    )
    
    neo4j_driver = Neo4jGraph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="neo4j@2025",
        database="neo4j",
    )
    
    config = LinearRAGConfig(
        embedding_model=embedding_model,
        spacy_model=SPACY_MODEL,
        max_workers=MAX_WORKERS,
        working_dir="./import_qwen_new"
    )
    
    state.rag_model = LinearRAG(global_config=config, es_client=es_client, neo4j_driver=neo4j_driver)
    print("系统初始化完成，准备就绪。")
    yield
    print("正在关闭系统...")

app = FastAPI(title="LinearRAG API Service", lifespan=lifespan)


@app.post("/index")
def index_documents(payload: IndexPayload):
    """
    建立索引接口
    """
    try:
        state.rag_model.index(passages=payload.passages, kb_name=payload.kb_name)
        return {"status": "success", "message": f"Successfully indexed into {payload.kb_name}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/retrieve")
def retrieve_documents(payload: RetrievePayload):
    """
    检索接口
    """
    try:
        questions = payload.questions
        results = state.rag_model.retrieve(
            questions, 
            index_names=payload.index_names, 
            top_k=payload.top_k
        )
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_files")  
def delete_files(payload: DeletePayload):
    """
    删除文件索引接口
    """
    try:
        state.rag_model.delete_files(index_name=payload.index_name, file_ids=payload.file_ids)
        return {"status": "success", "message": f"File {payload.file_ids} deleted from {payload.index_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "alive"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=12124)