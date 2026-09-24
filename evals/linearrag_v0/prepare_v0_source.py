from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# 从当前仓库的 v0 tag 抽取旧实现，生成可运行但不应提交的本地源码目录。
ROOT = Path(__file__).resolve().parents[2]
V0_FILES = [
    "src/LinearRAG.py",
    "src/config.py",
    "src/embedding.py",
    "src/es.py",
    "src/ner.py",
    "src/utils.py",
    "src/graphs_utils/base.py",
    "src/graphs_utils/neo4j_db.py",
    "src/graphs_utils/__init__.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="从 v0 tag 抽取隔离源码")
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / "evals" / "linearrag_v0" / ".v0-source",
    )
    args = parser.parse_args()
    target = args.target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    for relative_path in V0_FILES:
        result = subprocess.run(
            ["git", "show", f"v0:{relative_path}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.stdout, encoding="utf-8")
    print(f"v0_source={target}")


if __name__ == "__main__":
    main()
