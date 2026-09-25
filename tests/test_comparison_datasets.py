from __future__ import annotations

import unittest

from evals.linearrag_v0.evaluate_controlled_retrieval import build_cases, recall
from evals.linearrag_v0.evaluate_pdf_candidate_recall import cases_from_documents
from evals.linearrag_v0.build_pdf_eval_dataset import build_multihop_cases
from evals.linearrag_v0.evaluate_retrieval_paths import (
    all_recall,
    any_recall,
    passage_recall,
)
from src.common.models import SearchDocument


class ComparisonDatasetTests(unittest.TestCase):
    """校验受控语料和 PDF 弱标注的统计口径。"""

    def test_synthetic_targets_are_outside_vector_text(self):
        """每组只有目标段落包含答案实体，干扰段落不能共享实体。"""

        cases, documents, vectors = build_cases(4)

        self.assertEqual(len(cases), 4)
        self.assertEqual(len(vectors[cases[0].question]), 5)
        for case in cases:
            target = next(item for item in documents if item.id == case.target)
            self.assertIn(case.entity, target.metadata["entity_ids"])
            self.assertEqual(
                sum(
                    case.entity in item.metadata.get("entity_ids", [])
                    for item in documents
                    if item.doc_type == "passage"
                ),
                1,
            )

    def test_recall_uses_target_membership_at_k(self):
        """候选上限与最终排名必须按同一目标段落计算。"""

        cases, _, _ = build_cases(2)
        rankings = [
            ["noise", "noise", "noise", "noise", cases[0].target],
            [cases[1].target, "noise"],
        ]
        self.assertEqual(recall(rankings, cases, 1), 0.5)
        self.assertEqual(recall(rankings, cases, 2), 0.5)
        self.assertEqual(recall(rankings, cases, 5), 1.0)

    def test_weak_labels_exclude_numbers_and_repeated_entities(self):
        """弱标注仅接受足够长的中文实体与唯一目标段落。"""

        documents = [
            SearchDocument(
                "p1", "公司签订保密合同", doc_type="passage",
                metadata={"entities": [
                    {"name": "保密合同"},
                    {"name": "1000元"},
                ]},
            ),
            SearchDocument(
                "p2", "公司另有保密合同和劳动协议", doc_type="passage",
                metadata={"entities": [
                    {"name": "保密合同"},
                    {"name": "劳动协议"},
                ]},
            ),
        ]

        self.assertEqual(
            cases_from_documents(documents, 5),
            [("劳动协议的相关规定是什么？", "p2", "劳动协议")],
        )

    def test_multihop_metrics_require_all_targets_for_all_recall(self):
        """多跳评测必须同时区分任一命中、覆盖率与全部命中。"""

        cases = [
            {"target_texts": ["hop-a", "hop-b"]},
            {"target_texts": ["hop-c", "hop-d"]},
        ]
        rankings = [
            ["hop-a", "noise"],
            ["hop-c", "hop-d"],
        ]
        self.assertEqual(any_recall(rankings, cases, 2), 1.0)
        self.assertEqual(passage_recall(rankings, cases, 2), 0.75)
        self.assertEqual(all_recall(rankings, cases, 2), 0.5)

    def test_multihop_cases_require_two_distinct_targets(self):
        """桥接实体连接两个段落时，应生成两个必需目标。"""

        documents = [
            SearchDocument(
                "p1",
                "保密合同需要审批",
                doc_type="passage",
                metadata={"entities": [
                    {"name": "保密合同"},
                    {"name": "劳动合同"},
                ]},
            ),
            SearchDocument(
                "p2",
                "考勤制度需要打卡",
                doc_type="passage",
                metadata={"entities": [
                    {"name": "考勤制度"},
                    {"name": "劳动合同"},
                ]},
            ),
        ]
        cases = build_multihop_cases(documents, 5)
        self.assertEqual(len(cases), 1)
        self.assertEqual(len(cases[0]["target_texts"]), 2)
        self.assertEqual(cases[0]["bridge_entity"], "劳动合同")


if __name__ == "__main__":
    unittest.main()
