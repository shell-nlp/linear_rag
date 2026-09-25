from __future__ import annotations

import json
import unittest
from pathlib import Path


class EvaluationAssetTests(unittest.TestCase):
    """确保离线评测代码依赖的数据集不会在重构中丢失。"""

    def test_v0_evaluation_assets_are_present(self):
        """固定 PDF、小样本载荷和 152 条多问法载荷应保留在 evals。"""

        root = Path(__file__).resolve().parents[1]
        data_dir = root / "evals" / "linearrag_v0" / "data"
        self.assertTrue((data_dir / "document.pdf").is_file())
        payload = json.loads(
            (data_dir / "document_eval.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(payload["passages"]["text"]), 90)
        self.assertEqual(len(payload["cases"]), 12)
        large_payload = json.loads(
            (data_dir / "document_eval_large.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(large_payload["passages"]["text"]), 90)
        self.assertEqual(len(large_payload["cases"]), 152)
        self.assertEqual(
            len({case["question"] for case in large_payload["cases"]}), 152
        )
        multihop_payload = json.loads(
            (data_dir / "document_eval_multihop.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(multihop_payload["cases"]), 20)
        self.assertTrue(
            all(len(case["target_texts"]) == 2 for case in multihop_payload["cases"])
        )


if __name__ == "__main__":
    unittest.main()
