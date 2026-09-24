from __future__ import annotations

import json
import unittest
from pathlib import Path


class EvaluationAssetTests(unittest.TestCase):
    """确保离线评测代码依赖的数据集不会在重构中丢失。"""

    def test_v0_evaluation_assets_are_present(self):
        """固定 PDF、预切片载荷和 12 个弱标注问题应保留在 evals。"""

        root = Path(__file__).resolve().parents[1]
        data_dir = root / "evals" / "linearrag_v0" / "data"
        self.assertTrue((data_dir / "document.pdf").is_file())
        payload = json.loads(
            (data_dir / "document_eval.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["passages"]["text"]), 90)
        self.assertEqual(len(payload["cases"]), 12)


if __name__ == "__main__":
    unittest.main()
