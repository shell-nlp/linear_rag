# v0 真实对照

本目录保存实际 `v0` tag 的 Neo4j/GDS 与当前 `linear_local` 的同口径评测资产。

## 数据

- `data/document.pdf`：用户提供的源 PDF。
- `data/document_eval.json`：固定预切片和 12 个弱标注问题。
- `fixtures/v0_smoke.json`：两段文本的 v0 烟雾测试。

## 环境

评测需要项目当前环境、一个隔离的 `v0` Python 环境和运行中的 Neo4j/GDS。
先抽取 `v0` 源码：

```powershell
uv run python -B evals/linearrag_v0/prepare_v0_source.py `
  --target C:\Users\n0378\AppData\Local\Temp\linearrag-v0-source
```

再按 `docs/comparison-v0-local.md` 的命令运行。`v0` 环境需要
`neo4j`、`python-igraph`、`elasticsearch`、`spacy` 和中文模型；
这些依赖只用于旧版本对照，不进入当前项目依赖。

评测运行器只使用 `linearrag-*` ES 索引和 `linearrag_*` Neo4j 标签，
`finally` 会清理本次节点、索引和 GDS 投影。运行前确认 Neo4j 默认库中没有
需要保留的同名标签。
