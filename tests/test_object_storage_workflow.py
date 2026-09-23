from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from src.common.object_storage import create_object_storage
from src.common.object_storage.base import normalize_object_location
from src.common.object_storage.local import LocalObjectStorage
from src.indexing.workflow import FileIndexingWorkflow


class FakeObjectStorage:
    """事务测试用对象存储，可模拟写入失败。"""

    def __init__(self, fail_put: bool = False):
        self.fail_put = fail_put
        self.objects = {}
        self.deleted = []
        self.put_started = Event()

    def exists(self, bucket_name, object_key):
        return (bucket_name, object_key) in self.objects

    def put_bytes(self, bucket_name, object_key, data, content_type=None):
        self.put_started.set()
        if self.fail_put:
            raise OSError("storage failed")
        self.objects[(bucket_name, object_key)] = data

    def get_bytes(self, bucket_name, object_key):
        return self.objects[(bucket_name, object_key)]

    def delete_object(self, bucket_name, object_key):
        self.deleted.append((bucket_name, object_key))
        self.objects.pop((bucket_name, object_key), None)


class FakeIndexingService:
    """事务测试用索引服务，可模拟搜索数据库写入失败。"""

    def __init__(self, fail_index: bool = False):
        self.fail_index = fail_index
        self.deleted_passages = []
        self.write_calls = 0

    def prepare_documents(self, passages):
        return [SimpleNamespace(id="passage-1")], set()

    def write_documents(self, kb_name, documents, unique_entity_ids):
        self.write_calls += 1
        if self.fail_index:
            raise OSError("index failed")
        return {"new_passages": 1, "new_entities": 0, "failed_passages": 0}

    def delete_passages(self, index_name, passage_ids):
        self.deleted_passages.append((index_name, list(passage_ids)))


def build_resolver(storage: FakeObjectStorage):
    """构造断言存储线程已启动的解析函数。"""

    def resolver(process_pool, file_bytes, bucket_name, object_key, file_id):
        if not storage.put_started.wait(timeout=1):
            raise AssertionError("文件持久化和 PDF 解析没有并行启动")
        return {
            "text": ["测试段落"],
            "file_id": [file_id],
            "bucket_name": [bucket_name],
            "file_path": [object_key],
            "segment_id": [1],
        }

    return resolver


class ObjectStorageWorkflowTests(unittest.TestCase):
    """验证本地对象存储和上传索引补偿事务。"""

    def test_local_storage_uses_bucket_and_file_path(self):
        """本地实现应按 root/bucket/file_path 保存并支持删除。"""

        with tempfile.TemporaryDirectory() as directory:
            storage = LocalObjectStorage(directory)
            storage.put_bytes("docs", "reports/a.pdf", b"pdf")

            target = Path(directory) / "docs" / "reports" / "a.pdf"
            self.assertEqual(target.read_bytes(), b"pdf")
            self.assertEqual(storage.get_bytes("docs", "reports/a.pdf"), b"pdf")
            self.assertTrue(storage.exists("docs", "reports/a.pdf"))
            with self.assertRaises(FileExistsError):
                storage.put_bytes("docs", "reports/a.pdf", b"replacement")
            self.assertEqual(target.read_bytes(), b"pdf")
            storage.delete_object("docs", "reports/a.pdf")
            self.assertFalse(target.exists())

    def test_storage_factory_defaults_to_local_implementation(self):
        """未启用 MinIO 时工厂应返回本地对象存储。"""

        with tempfile.TemporaryDirectory() as directory:
            settings = SimpleNamespace(
                object_storage_provider="local",
                local_storage_root=directory,
            )

            storage = create_object_storage(settings)

            self.assertIsInstance(storage, LocalObjectStorage)

    def test_minio_configuration_must_be_complete(self):
        """显式启用 MinIO 时缺少连接参数应立即失败。"""

        settings = SimpleNamespace(
            object_storage_provider="minio",
            minio_endpoint_url="",
            minio_access_key="",
            minio_secret_key="",
        )

        with self.assertRaises(ValueError):
            create_object_storage(settings)

    def test_object_location_rejects_path_traversal(self):
        """对象地址不得逃逸本地存储根目录。"""

        with self.assertRaises(ValueError):
            normalize_object_location("docs", "../secret.pdf")
        with self.assertRaises(ValueError):
            normalize_object_location("../docs", "a.pdf")
        with self.assertRaises(ValueError):
            normalize_object_location("docs", "reports/a:b.pdf")

    def test_success_requires_storage_and_index(self):
        """文件写入和索引都成功后才返回事务成功。"""

        storage = FakeObjectStorage()
        indexing = FakeIndexingService()
        with ThreadPoolExecutor(max_workers=2) as pool:
            workflow = FileIndexingWorkflow(
                indexing,
                storage,
                process_pool=None,
                storage_io_pool=pool,
                max_upload_bytes=1024,
                passage_resolver=build_resolver(storage),
            )
            result = workflow.index_uploaded_file(
                b"pdf",
                "kb_test",
                "docs",
                "reports/a.pdf",
                file_id="file-1",
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["file_path"], "reports/a.pdf")
        self.assertIn(("docs", "reports/a.pdf"), storage.objects)
        self.assertEqual(indexing.write_calls, 1)
        self.assertFalse(indexing.deleted_passages)

    def test_index_failure_rolls_back_new_object_and_passages(self):
        """索引失败时应删除本次对象和可能已写入的段落。"""

        storage = FakeObjectStorage()
        indexing = FakeIndexingService(fail_index=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            workflow = FileIndexingWorkflow(
                indexing,
                storage,
                process_pool=None,
                storage_io_pool=pool,
                max_upload_bytes=1024,
                passage_resolver=build_resolver(storage),
            )
            with self.assertRaises(OSError):
                workflow.index_uploaded_file(
                    b"pdf", "kb_test", "docs", "reports/a.pdf", "file-1"
                )

        self.assertEqual(storage.deleted, [("docs", "reports/a.pdf")])
        self.assertEqual(indexing.deleted_passages, [("kb_test", ["passage-1"])])

    def test_storage_failure_does_not_start_search_write(self):
        """对象写入失败时不应提交 ES 写入。"""

        storage = FakeObjectStorage(fail_put=True)
        indexing = FakeIndexingService()
        with ThreadPoolExecutor(max_workers=2) as pool:
            workflow = FileIndexingWorkflow(
                indexing,
                storage,
                process_pool=None,
                storage_io_pool=pool,
                max_upload_bytes=1024,
                passage_resolver=build_resolver(storage),
            )
            with self.assertRaises(OSError):
                workflow.index_uploaded_file(
                    b"pdf", "kb_test", "docs", "reports/a.pdf", "file-1"
                )

        self.assertFalse(indexing.deleted_passages)
        self.assertFalse(storage.deleted)
        self.assertEqual(indexing.write_calls, 0)

    def test_oversized_upload_stops_before_storage_and_parsing(self):
        """超限文件应在启动并行任务前拒绝。"""

        storage = FakeObjectStorage()
        indexing = FakeIndexingService()
        with ThreadPoolExecutor(max_workers=2) as pool:
            workflow = FileIndexingWorkflow(
                indexing,
                storage,
                process_pool=None,
                storage_io_pool=pool,
                max_upload_bytes=2,
                passage_resolver=build_resolver(storage),
            )
            with self.assertRaises(ValueError):
                workflow.index_uploaded_file(
                    b"pdf", "kb_test", "docs", "reports/a.pdf", "file-1"
                )

        self.assertFalse(storage.put_started.is_set())
        self.assertEqual(indexing.write_calls, 0)


if __name__ == "__main__":
    unittest.main()
