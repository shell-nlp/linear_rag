# 纯向量、Hybrid-RRF、linear_local 与 v0 检索对照

本报告是一次**本地同口径实验**，不是生产流量 A/B 测试。快速基线为 ES 纯向量和
向量 + BM25 + RRF；图算法对照对象为本仓库 `v0` tag 中的原始 `src/LinearRAG.py`，
已实际运行其 Neo4j/GDS 路径。

## 测试环境

- 数据：`evals/linearrag_v0/data/document.pdf`，解析为 90 个段落。
- 问题：`evals/linearrag_v0/data/document_eval_large.json` 中的 152 条多问法弱标注问题，
  来自 76 个唯一实体，每个实体两种问法。
- Embedding：项目 `.env` 配置的 `gpu-bge-m3`，1024 维。
- ES：8.14.2；Neo4j：5.26.27；GDS：2.13.10。
- `v0` 源码通过 `v0` tag 抽取，使用独立临时 Python 环境运行。
- 当前版本：本工作树中的 `linear_local`，向量候选 5，实体关联段落每实体 5 条。
- 纯向量和 Hybrid-RRF 都使用与 `linear_local` 相同的问题向量，只测主召回，
  不追加实体扩展和相邻段落融合。

## 检索路径差异

纯向量直接从 ES 取段落 KNN 结果。Hybrid-RRF 分别取向量和 BM25 候选，再用
`1 / (60 + rank)` 融合两路排名。两者都只返回主召回段落，用于衡量不依赖图扩展的
基础检索能力。

`v0` 先从 ES 找问题实体和段落向量候选，段落候选固定最多 50；随后在 Neo4j
做至多两轮实体—段落—实体扩展，每实体最多取 5 条段落，再投影候选子图到 GDS
PageRank，最终只从 **ES top-50 段落**中返回结果。

当前 `linear_local` 从 ES 取向量候选，同时用问题实体找关联段落；两路合并后
读取实体和句子，在应用进程进行候选子图 PPR。实体关联段落即使不在向量 top-k
内，也可以进入最终候选。

另一个实际差异是：`v0` 用纯文本哈希做 ID，本次数据中 90 个切片被去重为 88 个
段落；当前版本保留文件来源和切片 ID，因此索引 90 个段落。

## 152 条弱标注问题结果

目标段落为问题实体唯一出现的段落；问题形如“实体的相关规定是什么？”。
这不是人工相关性标注，不能替代生产评测。

| 路径 | Recall@1 | Recall@3 | Recall@5 | Recall@8 | Recall@10 | Recall@15 | 查询中位耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES 纯向量 | 51/152 | 88/152 | 104/152 | 116/152 | 121/152 | 130/152 | 0.062 秒 |
| ES 向量 + BM25 + RRF | 88/152 | 110/152 | 120/152 | 130/152 | 132/152 | 141/152 | 0.161 秒 |
| 当前 `linear_local` | 107/152 | 118/152 | 121/152 | 129/152 | 131/152 | 135/152 | 1.936 秒 |
| 实际 `v0` | 75/152 | 105/152 | 110/152 | 115/152 | 119/152 | 119/152 | 2.611 秒 |

同次运行中，当前版本临时索引约 11.72 秒，`v0` 约 12.42 秒。索引时间包含真实
Embedding、ES 写入以及 `v0` 的 Neo4j 节点/关系写入，但不包含 PDF 解析和对象存储。

主要差异：

- `linear_local` 在 Recall@1/3/5 分别比 Hybrid-RRF 多 19、8、1 条命中，说明局部图
  排序对前几名更有效。
- Hybrid-RRF 从 Recall@8 开始反超，并在 Recall@15 达到 141/152，比
  `linear_local` 多 6 条；BM25 与向量 RRF 的候选覆盖更适合较大 K。
- `v0` 在 Recall@10 和 Recall@15 都是 119/152，说明它的候选上限已经饱和。
- 部分“保密协议”“考勤制度”等问题只有 Hybrid-RRF 进入前 15，而
  `linear_local` 未命中；也有“长沙总部”等问题局部图把向量排名从第 1 位降到第 8 位。
  这些差异说明快速路径与图路径具备互补空间，后续可评估多路融合或按查询特征路由。

## 20 条多跳问题结果

多跳载荷要求每题返回两个必要段落。`Any@K` 只表示至少命中一个目标；
`PassageRecall@K` 是目标段落平均覆盖率；`All@K` 要求两个目标都进入前 K，
是多跳检索最严格的主指标。

| 路径 | Any@15 | PassageRecall@5 | All@3 | All@5 | All@10 | All@15 | 查询中位耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES 纯向量 | 95% | 52.5% | 15% | 30% | 60% | 80% | 0.061 秒 |
| ES 向量 + BM25 + RRF | 100% | 62.5% | 20% | 35% | 70% | 75% | 0.160 秒 |
| 当前 `linear_local` | 100% | 65.0% | 35% | 50% | 85% | 90% | 1.934 秒 |
| 实际 `v0` | 90% | 50.0% | 15% | 20% | 40% | 40% | 2.661 秒 |

主要差异：

- Hybrid-RRF 的 `Any@15` 达到 100%，说明 BM25 补充很容易找到至少一跳。
- `linear_local` 的 `All@10` 和 `All@15` 分别为 85% 和 90%，说明实体图扩展对补齐
  第二跳更有价值；它比 Hybrid-RRF 的 `All@15` 高 15 个百分点。
- 纯向量的 `All@15` 为 80%，高于 Hybrid-RRF 的 75%，但低于 `linear_local`。
- `v0` 的 `All@15` 只有 40%，候选和扩展上限在多跳问题上更明显。

数据集由 8 条共享桥接实体的问题和 12 条相邻制度段落问题组成。它只评价必要段落
是否被检索到，不评价答案生成阶段是否完成了关系推理，也不替代人工多跳标注。

## 结论与限制

- 在这份 90 段 PDF 和 152 条多问法问题上，当前 `linear_local` 的 Recall@1/3/5
  高于纯向量、Hybrid-RRF 和实际 `v0`，但查询中位耗时也明显更高。
- Hybrid-RRF 在 Recall@8/10/15 均高于 `linear_local`，说明图排序适合精排，而
  BM25 与向量的 RRF 候选更适合深层召回。生产上不应把两者简单视为替代关系。
- 纯向量耗时最低，但各 K 的召回均落后；`v0` 在 Recall@10 后不再提升，说明当前
  实现的候选和扩展上限比现有 Hybrid-RRF 更早饱和。
- 该结论只代表本次数据、模型、硬件和参数，不能外推到生产规模。
- 152 条问题来自同一 PDF，问题显式包含目标实体，弱标注偏向实体通道；它比
  12 条小样本更稳定，但仍不能替代人工标注的自然问题和公开中文基准。
- 当前实现和 `v0` 的图 ID、去重、边权、PageRank 参数、分数融合都不同。
- 百万/上亿节点仍需分片、离线预计算和有界子图，不能用本报告的小数据结论替代。

## 复跑

从 `v0` tag 抽取源码到临时目录，然后运行：

```powershell
$v0Source = 'C:\Users\n0378\AppData\Local\Temp\linearrag-v0-source'
$v0Python = 'C:\Users\n0378\AppData\Local\Temp\linearrag-v0-compare-venv\Scripts\python.exe'

uv run python -B evals/linearrag_v0/extract_v0_source.py --target $v0Source

uv run python -B evals/linearrag_v0/build_pdf_eval_dataset.py `
  --pdf evals/linearrag_v0/data/document.pdf `
  --output evals/linearrag_v0/data/document_eval_large.json `
  --all-unique-entities `
  --query-variants 2 `
  --cases 500

uv run python -B evals/linearrag_v0/evaluate_retrieval_paths.py `
  --payload evals/linearrag_v0/data/document_eval_large.json `
  --v0-python $v0Python `
  --v0-source $v0Source `
  --neo4j-user neo4j `
  --neo4j-password neo4j@2025 `
  --cases 152 `
  --top-k 15

uv run python -B evals/linearrag_v0/build_pdf_eval_dataset.py `
  --pdf evals/linearrag_v0/data/document.pdf `
  --output evals/linearrag_v0/data/document_eval_multihop.json `
  --multihop `
  --cases 20

uv run python -B evals/linearrag_v0/evaluate_retrieval_paths.py `
  --payload evals/linearrag_v0/data/document_eval_multihop.json `
  --v0-python $v0Python `
  --v0-source $v0Source `
  --neo4j-user neo4j `
  --neo4j-password neo4j@2025 `
  --cases 20 `
  --top-k 15
```

评测脚本使用 `linearrag-*` 临时 ES 索引和 `linearrag_*` Neo4j 标签，并在 `finally`
中清理节点和索引。`v0` 源码环境仍需要 `neo4j`、`python-igraph` 和中文 spaCy 模型。
