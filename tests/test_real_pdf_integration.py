from __future__ import annotations

import os
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from src.common.document_processing.ner import SpacyNER
from src.common.model_providers import create_embedding_provider
from src.common.object_storage import create_object_storage
from src.common.search_store.elasticsearch import ElasticsearchSearchStore
from src.indexing.service import IndexingService
from src.indexing.workflow import FileIndexingWorkflow
from src.retrieval.linear import LinearRetriever
from src.settings import get_settings
from src.utils import get_es_client


@unittest.skipUnless(
    os.getenv("LINEAR_REAL_PDF_INTEGRATION") == "1",
    "仅显式启用时连接真实模型、对象存储和 Elasticsearch",
)
class RealPdfIntegrationTests(unittest.TestCase):
    """验证真实 PDF 经上传事务、图索引和两种图检索后清理测试资源。"""

    def test_index_and_retrieve_pdf(self):
        pdf_path = Path(
            os.getenv("LINEAR_TEST_PDF", str(Path.home() / "Downloads" / "document.pdf"))
        )
        settings = get_settings()
        embedding = create_embedding_provider()
        ner = SpacyNER(settings.spacy_model)
        search_store = ElasticsearchSearchStore(get_es_client())
        object_storage = create_object_storage(settings)
        config = settings.runtime_config()
        suffix = uuid.uuid4().hex[:12]
        index_name = f"linearrag-pdf-verify-{suffix}"
        bucket = "linearrag-verify"
        object_key = f"{suffix}/document.pdf"
        logger.disable("src.common.document_processing.pdf_parser")
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                workflow = FileIndexingWorkflow(
                    IndexingService(config, embedding, search_store, ner),
                    object_storage,
                    None,
                    pool,
                    settings.max_upload_bytes,
                )
                started = time.perf_counter()
                result = workflow.index_uploaded_file(
                    pdf_path.read_bytes(),
                    index_name,
                    bucket,
                    object_key,
                    file_id=suffix,
                    content_type="application/pdf",
                )
                print(
                    f"PDF index: {result['new_passages']} passages, "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )
            self.assertGreater(result["new_passages"], 0)
            self.assertTrue(object_storage.exists(bucket, object_key))
            retriever = LinearRetriever(config, embedding, ner, search_store)
            for local in (True, False):
                started = time.perf_counter()
                hits = retriever.retrieve(
                    "公司的信息安全管理有哪些要求？", [index_name], 3, local=local
                )
                print(
                    f"{'linear_local' if local else 'linear'}: "
                    f"{len(hits)} hits, {time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                self.assertTrue(hits)
                self.assertTrue(all(hit["type"] == "passage" for hit in hits))
        finally:
            if search_store.index_exists(index_name):
                search_store.delete_index(index_name)
            if object_storage.exists(bucket, object_key):
                object_storage.delete_object(bucket, object_key)


if __name__ == "__main__":
    unittest.main()
