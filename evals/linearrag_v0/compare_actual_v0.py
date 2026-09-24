from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.common.document_processing.ner import SpacyNER
from src.common.model_providers import create_embedding_provider
from src.common.models import SearchMode, SearchQuery
from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.indexing.service import IndexingService
from src.retrieval.linear import LinearRetriever
from src.settings import get_settings
from src.utils import get_es_client


def recall(rankings: list[list[str]], cases: list[dict], top_k: int) -> float:
    """按目标段落文本计算 Recall@K。"""

    return sum(
        case["target_text"] in ranking[:top_k]
        for case, ranking in zip(cases, rankings)
    ) / len(cases)


def run_current(payload: dict, cases: list[dict], top_k: int) -> dict:
    """用当前版本建立一次索引，并比较纯向量、Hybrid-RRF 和 linear_local。"""

    settings = get_settings()
    store = ElasticsearchSearchStore(get_es_client())
    index_name = f"linearrag-current-eval-{uuid.uuid4().hex[:10]}"
    embedding = create_embedding_provider()
    ner = SpacyNER(settings.spacy_model)
    config = settings.runtime_config()
    # 三种路径共享同一索引和同一问题向量，避免把索引差异混入检索效果比较。
    rankings = {
        SearchMode.VECTOR.value: [],
        SearchMode.HYBRID.value: [],
        SearchMode.LINEAR_LOCAL.value: [],
    }
    timings = {mode: [] for mode in rankings}
    try:
        service = IndexingService(config, embedding, store, ner)
        started = time.perf_counter()
        documents, entity_ids = service.prepare_documents(payload["passages"])
        service.write_documents(index_name, documents, entity_ids)
        index_seconds = time.perf_counter() - started
        retriever = LinearRetriever(config, embedding, ner, store)
        for case in cases:
            # 纯向量和 Hybrid 都是主召回基线，不追加实体扩展或相邻段落融合。
            question_vector = retriever._encode([case["question"]])[0].tolist()
            for search_mode in (SearchMode.VECTOR, SearchMode.HYBRID):
                started = time.perf_counter()
                hits = store.search(
                    SearchQuery(
                        index_names=[index_name],
                        mode=search_mode,
                        query_text=case["question"],
                        query_vector=question_vector,
                        top_k=top_k,
                        num_candidates=max(top_k * 4, 100),
                        filters={"type": "passage"},
                    )
                )
                timings[search_mode.value].append(time.perf_counter() - started)
                rankings[search_mode.value].append(
                    [hit.document.text for hit in hits]
                )

            started = time.perf_counter()
            hits = retriever.retrieve(
                case["question"], [index_name], top_k, local=True
            )
            timings[SearchMode.LINEAR_LOCAL.value].append(
                time.perf_counter() - started
            )
            rankings[SearchMode.LINEAR_LOCAL.value].append(
                [hit.get("text", "") for hit in hits]
            )
        return {
            "index_seconds": index_seconds,
            "rankings": rankings,
            "timings": timings,
        }
    finally:
        if store.index_exists(index_name):
            store.delete_index(index_name)


def run_v0(payload: dict, args, top_k: int) -> dict:
    """在隔离 Python 环境运行 v0 原始源码，解析其 JSON 结果。"""

    index_name = f"linearrag_v0_eval_{uuid.uuid4().hex[:10]}"
    command = [
        str(args.v0_python.resolve()),
        "-B",
        str(Path(__file__).with_name("run_v0_isolated.py")),
        "--source",
        str(args.v0_source.resolve()),
        "--payload",
        str(args.payload.resolve()),
        "--index-name",
        index_name,
        "--neo4j-password",
        args.neo4j_password,
        "--max-cases",
        str(args.cases),
        "--top-k",
        str(top_k),
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "NEO4J_USER": args.neo4j_user,
    })
    result = subprocess.run(
        command,
        cwd=str(args.v0_source.resolve()),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=args.timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"v0 运行失败，退出码 {result.returncode}\n"
            f"stdout:\n{result.stdout[-3000:]}\nstderr:\n{result.stderr[-3000:]}"
        )
    outputs = []
    index_seconds = None
    for line in result.stdout.splitlines():
        if line.startswith("v0_index_seconds="):
            index_seconds = float(line.split("=", 1)[1])
        if line.startswith("{"):
            outputs.append(json.loads(line))
    if index_seconds is None or len(outputs) != len(payload["cases"]):
        raise RuntimeError(
            "v0 输出不完整，未生成可比较结果\n"
            f"stdout tail:\n{result.stdout[-3000:]}"
        )
    return {
        "index_seconds": index_seconds,
        "rankings": [item["texts"] for item in outputs],
        "timings": [item["seconds"] for item in outputs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="纯向量、Hybrid-RRF、当前 linear_local 与实际 v0 同口径对比"
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--v0-python", type=Path, required=True)
    parser.add_argument(
        "--v0-source",
        type=Path,
        default=Path(__file__).resolve().parent / ".v0-source",
    )
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD", ""))
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="当前版本和 v0 的统一返回深度；默认 5 以覆盖 Recall@5",
    )
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    payload["cases"] = payload["cases"][: args.cases]
    cases = payload["cases"]
    if not cases:
        raise RuntimeError("评测载荷没有 cases")
    if args.top_k < 5:
        raise ValueError("--top-k 至少为 5，否则无法计算 Recall@5")
    logger.disable("src.common.document_processing.pdf_parser")
    current = run_current(payload, cases, args.top_k)
    v0 = run_v0(payload, args, args.top_k)
    print(
        f"cases={len(cases)} current_index={current['index_seconds']:.2f}s "
        f"v0_index={v0['index_seconds']:.2f}s"
    )
    for top_k in (1, 3, 5):
        print(
            f"Recall@{top_k}: "
            f"vector={recall(current['rankings']['vector'], cases, top_k):.3f} "
            f"hybrid_rrf={recall(current['rankings']['hybrid'], cases, top_k):.3f} "
            f"current_linear_local={recall(current['rankings']['linear_local'], cases, top_k):.3f} "
            f"v0={recall(v0['rankings'], cases, top_k):.3f}"
        )
    print(
        f"median query: vector={statistics.median(current['timings']['vector']):.3f}s "
        f"hybrid_rrf={statistics.median(current['timings']['hybrid']):.3f}s "
        f"current_linear_local={statistics.median(current['timings']['linear_local']):.3f}s "
        f"v0={statistics.median(v0['timings']):.3f}s"
    )
    for index, case in enumerate(cases):
        vector_rank = (
            current["rankings"]["vector"][index].index(case["target_text"]) + 1
            if case["target_text"] in current["rankings"]["vector"][index] else "-"
        )
        hybrid_rank = (
            current["rankings"]["hybrid"][index].index(case["target_text"]) + 1
            if case["target_text"] in current["rankings"]["hybrid"][index] else "-"
        )
        current_rank = (
            current["rankings"]["linear_local"][index].index(case["target_text"]) + 1
            if case["target_text"] in current["rankings"]["linear_local"][index] else "-"
        )
        v0_rank = (
            v0["rankings"][index].index(case["target_text"]) + 1
            if case["target_text"] in v0["rankings"][index] else "-"
        )
        print(
            f"case={index} entity={case['entity']} "
            f"vector_rank={vector_rank} hybrid_rank={hybrid_rank} "
            f"current_rank={current_rank} v0_rank={v0_rank}"
        )


if __name__ == "__main__":
    main()
