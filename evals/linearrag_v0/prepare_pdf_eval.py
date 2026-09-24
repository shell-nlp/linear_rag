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


def main() -> None:
    parser = argparse.ArgumentParser(description="从 PDF 生成 v0/当前版本共用评测载荷")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "evals" / "linearrag_v0" / "data" / "document.pdf",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=12)
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
    payload = {
        "passages": passages,
        "cases": build_cases(documents, args.cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"payload={args.output} passages={len(passages['text'])} cases={len(payload['cases'])}")


if __name__ == "__main__":
    main()
