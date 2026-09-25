# v0 真实对照

本目录保存实际 `v0` tag 的 Neo4j/GDS 与当前 `linear_local` 的同口径评测资产。

## 数据

- `data/document.pdf`：用户提供的源 PDF。
- `data/document_eval.json`：固定预切片和 12 个弱标注问题。
- `data/document_eval_large.json`：固定预切片和 152 条多问法弱标注问题。
- `data/document_eval_multihop.json`：固定预切片和 20 条两目标多跳问题。
- `fixtures/v0_smoke.json`：两段文本的 v0 烟雾测试。

## 脚本

- `build_pdf_eval_dataset.py`：从固定 PDF 生成单目标或多问法评测载荷。
- `evaluate_retrieval_paths.py`：统一比较纯向量、Hybrid-RRF、全图 `linear`、
  `linear_local` 和 v0。
- `run_v0_retrieval_eval.py`：在隔离 Python 环境运行 v0 的真实 ES/Neo4j/GDS 路径。
- `extract_v0_source.py`：从当前仓库的 `v0` tag 抽取隔离运行源码。
- `evaluate_controlled_retrieval.py`：在可控语料上验证候选上限与局部图排序。
- `evaluate_pdf_candidate_recall.py`：在真实 PDF 上分析候选召回和中间排名。

## 环境

评测需要项目当前环境、一个隔离的 `v0` Python 环境和运行中的 Neo4j/GDS。
先抽取 `v0` 源码：

```powershell
uv run python -B evals/linearrag_v0/extract_v0_source.py `
  --target C:\Users\n0378\AppData\Local\Temp\linearrag-v0-source
```

再按 `docs/comparison-v0-local.md` 的命令运行。`v0` 环境需要
`neo4j`、`python-igraph`、`elasticsearch`、`spacy` 和中文模型；
这些依赖只用于旧版本对照，不进入当前项目依赖。

评测运行器只使用 `linearrag-*` ES 索引和 `linearrag_*` Neo4j 标签，
`finally` 会清理本次节点、索引和 GDS 投影。运行前确认 Neo4j 默认库中没有
需要保留的同名标签。
