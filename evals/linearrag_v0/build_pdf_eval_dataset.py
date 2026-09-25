"""从固定 PDF 构建单目标或多问法的本地检索评测载荷。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.document_processing.ner import SpacyNER
from src.common.models import SearchDocument
from src.indexing.workflow import resolve_uploaded_passages
from src.settings import get_settings


# 多问法生成只改变表达方式，不改变目标段落，用于降低单模板评测的偶然性。
QUESTION_TEMPLATES = (
    "{name}的相关规定是什么？",
    "公司对{name}有哪些要求？",
)


def semantic_entities(text: str, entities: list[str]) -> list[dict]:
    """只保留可读的中文实体，并按段落频次生成边权所需元数据。"""

    counts = Counter(entities)
    result = []
    for name, count in counts.items():
        han_count = sum("\u4e00" <= char <= "\u9fff" for char in name)
        if (
            3 <= len(name) <= 16
            and han_count >= max(3, len(name) * 0.7)
            and not re.search(r"\d", name)
            and name in text
        ):
            result.append({
                "id": name.casefold(),
                "name": name,
                "count": count,
                "importance": math.log1p(count),
            })
    return result


def build_cases(documents: list[SearchDocument], limit: int) -> list[dict]:
    """选择只出现在一个段落中的语义实体作为弱标注目标。"""

    by_entity: dict[str, set[str]] = {}
    for document in documents:
        for entity in document.metadata.get("entities") or []:
            by_entity.setdefault(entity["name"], set()).add(document.id)
    cases = []
    used = set()
    for name, document_ids in sorted(
        by_entity.items(), key=lambda item: (len(item[1]), item[0])
    ):
        if len(document_ids) != 1:
            continue
        document_id = next(iter(document_ids))
        if document_id in used:
            continue
        document = next(item for item in documents if item.id == document_id)
        used.add(document_id)
        cases.append({
            "question": f"{name}的相关规定是什么？",
            "entity": name,
            "target_text": document.text,
        })
        if len(cases) == limit:
            break
    return cases


def build_large_cases(
    documents: list[SearchDocument],
    limit: int,
    query_variants: int,
) -> list[dict]:
    """为每个唯一实体生成多种问法，扩大可重复评测的问题数量。"""

    by_entity: dict[str, set[str]] = {}
    document_by_id = {document.id: document for document in documents}
    for document in documents:
        for entity in document.metadata.get("entities") or []:
            by_entity.setdefault(entity["name"], set()).add(document.id)

    cases = []
    templates = QUESTION_TEMPLATES[:query_variants]
    for name, document_ids in sorted(by_entity.items()):
        # 只使用目标段落唯一的实体，避免把多段落实体误当成单目标答案。
        if len(document_ids) != 1:
            continue
        document_id = next(iter(document_ids))
        document = document_by_id[document_id]
        for template in templates:
            cases.append({
                "question": template.format(name=name),
                "entity": name,
                "target_text": document.text,
            })
            if len(cases) == limit:
                return cases
    return cases


def build_multihop_cases(
    documents: list[SearchDocument],
    limit: int,
) -> list[dict]:
    """用两个段落及共享桥接实体构造两跳问题，目标段落必须成组返回。"""

    document_by_id = {document.id: document for document in documents}
    entity_to_documents: dict[str, set[str]] = {}
    document_entities: dict[str, set[str]] = {}
    for document in documents:
        names = {
            entity["name"]
            for entity in document.metadata.get("entities") or []
            if entity.get("name")
        }
        document_entities[document.id] = names
        for name in names:
            entity_to_documents.setdefault(name, set()).add(document.id)

    candidates = []
    documents = list(documents)
    for left_index, left_document in enumerate(documents):
        left_entities = document_entities[left_document.id]
        for right_document in documents[left_index + 1:]:
            right_entities = document_entities[right_document.id]
            bridge_entities = left_entities & right_entities
            left_only = {
                name for name in left_entities - right_entities
                if 4 <= len(name) <= 12
            }
            right_only = {
                name for name in right_entities - left_entities
                if 4 <= len(name) <= 12
            }
            if not bridge_entities or not left_only or not right_only:
                continue
            for bridge in bridge_entities:
                # 只保留局部桥接实体，避免“公司”等高频词制造伪多跳。
                if len(entity_to_documents[bridge]) > 3:
                    continue
                for left_entity in sorted(left_only):
                    for right_entity in sorted(right_only):
                        score = (
                            len(entity_to_documents[bridge])
                            + len(entity_to_documents[left_entity])
                            + len(entity_to_documents[right_entity])
                        )
                        candidates.append((
                            score,
                            left_document.id,
                            right_document.id,
                            left_entity,
                            right_entity,
                            bridge,
                        ))

    # 按段落对轮转取样，避免前若干题全部落在同一组目标段落上。
    grouped_candidates: dict[tuple[str, str], list[tuple]] = {}
    for candidate in sorted(candidates):
        pair_key = tuple(sorted((candidate[1], candidate[2])))
        grouped_candidates.setdefault(pair_key, []).append(candidate)

    cases = []
    used_questions = set()
    used_pairs = set()
    while grouped_candidates and len(cases) < limit:
        for pair_key in sorted(grouped_candidates):
            group = grouped_candidates[pair_key]
            if not group:
                continue
            _, left_id, right_id, left_entity, right_entity, bridge = group.pop(0)
            question = (
                f"{left_entity}和{right_entity}通过{bridge}有什么关联？"
                "请分别找出相关规定。"
            )
            if question in used_questions:
                continue
            used_questions.add(question)
            used_pairs.add(pair_key)
            cases.append({
                "question": question,
                "hop_entities": [left_entity, right_entity],
                "bridge_entity": bridge,
                "target_texts": [
                    document_by_id[left_id].text,
                    document_by_id[right_id].text,
                ],
            })
            if len(cases) == limit:
                break
        grouped_candidates = {
            key: value for key, value in grouped_candidates.items() if value
        }

    # 桥接实体不足时补充相邻制度段落，保证多跳评测有足够样本。
    for index in range(len(documents) - 1):
        if len(cases) == limit:
            break
        left_document = documents[index]
        right_document = documents[index + 1]
        pair_key = tuple(sorted((left_document.id, right_document.id)))
        if pair_key in used_pairs:
            continue
        left_entities = sorted(
            (
                entity["name"]
                for entity in left_document.metadata.get("entities") or []
                if 4 <= len(entity["name"]) <= 12
            ),
            key=lambda name: (-len(name), name),
        )
        right_entities = sorted(
            (
                entity["name"]
                for entity in right_document.metadata.get("entities") or []
                if 4 <= len(entity["name"]) <= 12
            ),
            key=lambda name: (-len(name), name),
        )
        if not left_entities or not right_entities:
            continue
        left_entity = left_entities[0]
        right_entity = right_entities[0]
        question = (
            f"{left_entity}和{right_entity}分别有哪些规定？"
            "请找出两个相关段落。"
        )
        if question in used_questions:
            continue
        used_questions.add(question)
        used_pairs.add(pair_key)
        cases.append({
            "question": question,
            "hop_entities": [left_entity, right_entity],
            "bridge_entity": None,
            "target_texts": [left_document.text, right_document.text],
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="从 PDF 生成 v0/当前版本共用评测载荷")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "evals" / "linearrag_v0" / "data" / "document.pdf",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
    parser.add_argument(
        "--all-unique-entities",
        action="store_true",
        help="为每个唯一实体生成问题，而不是每个目标段落只取一条",
    )
    parser.add_argument(
        "--query-variants",
        type=int,
        default=2,
        choices=range(1, len(QUESTION_TEMPLATES) + 1),
        help="每个实体的问法数量",
    )
    parser.add_argument(
        "--multihop",
        action="store_true",
        help="生成两个必需段落的两跳问题，而不是单目标问题",
    )
    args = parser.parse_args()

    settings = get_settings()
    passages = resolve_uploaded_passages(
        None,
        args.pdf.read_bytes(),
        "compare",
        args.pdf.name,
        "linearrag-pdf-eval",
    )
    ner = SpacyNER(settings.spacy_model)
    passage_entities, _ = ner.extract_graph_entities(
        {f"passage-{index}": text for index, text in enumerate(passages["text"])},
        1,
    )
    documents = [
        SearchDocument(
            id=f"passage-{index}",
            text=text,
            doc_type="passage",
            metadata={
                "entities": semantic_entities(
                    text, passage_entities.get(f"passage-{index}", [])
                )
            },
        )
        for index, text in enumerate(passages["text"])
    ]
    if args.multihop:
        cases = build_multihop_cases(documents, args.cases)
    elif args.all_unique_entities:
        cases = build_large_cases(documents, args.cases, args.query_variants)
    else:
        cases = build_cases(documents, args.cases)
    payload = {"passages": passages, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"payload={args.output} passages={len(passages['text'])} cases={len(payload['cases'])}")


if __name__ == "__main__":
    main()
