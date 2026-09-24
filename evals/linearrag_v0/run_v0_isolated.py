from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser(description="隔离运行 v0 tag 的真实 Neo4j/GDS 路径")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / ".v0-source",
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--index-name", required=True)
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="v0 每题返回的段落数，需覆盖对比脚本要求的最大 Recall@K",
    )
    args = parser.parse_args()

    # 在临时环境运行时只导入 tag v0，不让当前工作树的 src 遮蔽旧源码。
    sys.path.insert(0, str(args.source.resolve()))
    from elasticsearch import Elasticsearch
    from src.LinearRAG import LinearRAG
    from src.config import LinearRAGConfig
    from src.embedding import LocalOpenAIEmbeddingModel
    from src.graphs_utils.neo4j_db import Neo4jGraph

    values = dotenv_values(Path(__file__).resolve().parents[2] / ".env")
    es = Elasticsearch(
        values.get("es_url") or "http://localhost:9200",
        basic_auth=(values.get("es_user"), values.get("es_password")),
    )
    graph = Neo4jGraph(
        "bolt://localhost:7687",
        os.getenv("NEO4J_USER", "neo4j"),
        args.neo4j_password,
        "neo4j",
    )
    # 原客户端只接收 URL 和模型名；两版使用同一个真实 Embedding 服务。
    embedding = LocalOpenAIEmbeddingModel(
        values["EMBEDDING_API_URL"], values["EMBEDDING_MODEL_NAME"]
    )
    embedding.headers["Authorization"] = f"Bearer {values['LLM_API_KEY']}"
    rag = LinearRAG(
        LinearRAGConfig(
            embedding_model=embedding,
            spacy_model=values.get("SPACY_MODEL") or "zh_core_web_md",
            max_workers=1,
        ),
        es,
        graph,
    )
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    try:
        started = time.perf_counter()
        rag.index(payload["passages"], args.index_name)
        print(f"v0_index_seconds={time.perf_counter() - started:.3f}", flush=True)
        count = es.count(
            index=args.index_name, query={"term": {"type": "passage"}}
        )["count"]
        if count == 0:
            raise RuntimeError("v0 索引没有写入段落，拒绝报告空结果")
        cases = payload.get("cases", [])
        if args.max_cases is not None:
            cases = cases[: args.max_cases]
        questions = payload.get("questions") or [
            case["question"] for case in cases
        ]
        if args.max_cases is not None and payload.get("questions"):
            questions = questions[: args.max_cases]
        for question in questions:
            started = time.perf_counter()
            # 返回深度必须与当前版本一致，否则 Recall@5 会被 v0 的 top_k=3 人为截断。
            result = rag.retrieve(question, [args.index_name], top_k=args.top_k)
            print(
                json.dumps({
                    "question": question,
                    "seconds": round(time.perf_counter() - started, 3),
                    "ids": [item.get("hash_id") for item in result[0]],
                    "texts": [item.get("text") for item in result[0]],
                    "scores": [item.get("score") for item in result[0]],
                }, ensure_ascii=False),
                flush=True,
            )
    finally:
        # 只清理本次唯一标签，不触碰默认库的其它节点或投影。
        try:
            with graph.get_session() as session:
                session.run(
                    f"MATCH (n:`{args.index_name}`) DETACH DELETE n"
                ).consume()
                constraints = session.run(
                    "SHOW CONSTRAINTS YIELD name, labelsOrTypes"
                ).data()
                for constraint in constraints:
                    labels = constraint.get("labelsOrTypes") or []
                    if args.index_name in labels or "BaseNode" in labels:
                        session.run(
                            f"DROP CONSTRAINT `{constraint['name']}`"
                        ).consume()
        finally:
            es.indices.delete(index=args.index_name, ignore_unavailable=True)
        graph.close()


if __name__ == "__main__":
    main()
