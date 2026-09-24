# 纯向量、Hybrid-RRF、linear_local 与 v0 检索对照

本报告是一次**本地同口径实验**，不是生产流量 A/B 测试。快速基线为 ES 纯向量和
向量 + BM25 + RRF；图算法对照对象为本仓库 `v0` tag 中的原始 `src/LinearRAG.py`，
已实际运行其 Neo4j/GDS 路径。

## 测试环境

- 数据：`evals/linearrag_v0/data/document.pdf`，解析为 90 个段落。
- 问题：`evals/linearrag_v0/data/document_eval.json` 中的 12 个弱标注问题。
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

## 12 个弱标注问题结果

目标段落为问题实体唯一出现的段落；问题形如“实体的相关规定是什么？”。
这不是人工相关性标注，不能替代生产评测。

| 路径 | Recall@1 | Recall@3 | Recall@5 | 查询中位耗时 |
| --- | ---: | ---: | ---: | ---: |
| ES 纯向量 | 4/12 | 8/12 | 8/12 | 0.060 秒 |
| ES 向量 + BM25 + RRF | 6/12 | 7/12 | 8/12 | 0.136 秒 |
| 当前 `linear_local` | 8/12 | 10/12 | 10/12 | 1.983 秒 |
| 实际 `v0` | 6/12 | 8/12 | 9/12 | 2.566 秒 |

同次运行中，当前版本临时索引约 11.32 秒，`v0` 约 12.74 秒。索引时间包含真实
Embedding、ES 写入以及 `v0` 的 Neo4j 节点/关系写入，但不包含 PDF 解析和对象存储。

Recall@5 让 `v0` 额外命中第 8 题“内蒙古”（目标排名第 4）；当前版本未命中的
第 5、6 题仍不在前 5，因此当前版本的 Recall@5 与 Recall@3 相同。Hybrid-RRF
在 Recall@1 上明显高于纯向量，但 Recall@3 略低，说明 BM25 的加入改变了排序权衡；
它单独把第 6 题“考勤制度”召回到第 4 位。

逐题目标排名：

| 题号 | 实体 | 纯向量 | Hybrid-RRF | 当前 `linear_local` | 实际 `v0` |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 一个月 | 3 | 未命中 | 3 | 2 |
| 1 | 一年内 | 2 | 未命中 | 1 | 2 |
| 2 | 上海XX信息科技有限公司 | 1 | 1 | 1 | 1 |
| 3 | 保密协议 | 未命中 | 未命中 | 1 | 1 |
| 4 | 中华人民共和国劳动法 | 1 | 1 | 1 | 1 |
| 5 | 保密合同 | 未命中 | 未命中 | 未命中 | 未命中 |
| 6 | 考勤制度 | 未命中 | 4 | 未命中 | 未命中 |
| 7 | 奖励 | 2 | 1 | 1 | 1 |
| 8 | 内蒙古 | 未命中 | 3 | 1 | 4 |
| 9 | 加班 | 1 | 1 | 1 | 1 |
| 10 | 劳动合同变更协议书 | 1 | 1 | 1 | 1 |
| 11 | 参考项 | 2 | 1 | 2 | 未命中 |

本次样本中，当前 `linear_local` 唯一召回第 11 题，并把第 8 题排到第 1；
`v0` 在第 0 题上排名更好；Hybrid-RRF 唯一召回第 6 题。第 5 题四条路径都未进入
前 5，说明实体与查询表达仍是共同边界。

## 结论与限制

- 在这份 90 段 PDF 和 12 个弱标注问题上，当前 `linear_local` 的 Recall@1、
  Recall@3、Recall@5 均高于纯向量、Hybrid-RRF 和实际 `v0`，但查询中位耗时
  也明显更高。
- Recall@5 对 v0 的提升大于当前版本：v0 从 8/12 提升到 9/12，当前版本保持
  10/12；这说明更深候选对 v0 的候选约束有帮助，但当前版本的主要瓶颈仍在
  “保密合同”和“考勤制度”的实体与候选召回。
- Hybrid-RRF 的 Recall@1 高于纯向量，但 Recall@3 略低，且两类快速路径的
  Recall@5 都为 8/12。局部图路径总体更强，但 Hybrid-RRF 能命中图路径漏掉的
  “考勤制度”，后续可评估多路候选融合或按查询特征路由。
- 该结论只代表本次数据、模型、硬件和参数，不能外推到生产规模。
- 问题显式包含目标实体，弱标注偏向实体通道；还需人工标注的自然问题。
- 当前实现和 `v0` 的图 ID、去重、边权、PageRank 参数、分数融合都不同。
- 百万/上亿节点仍需分片、离线预计算和有界子图，不能用本报告的小数据结论替代。

## 复跑

从 `v0` tag 抽取源码到临时目录，然后运行：

```powershell
$v0Source = 'C:\Users\n0378\AppData\Local\Temp\linearrag-v0-source'
$v0Python = 'C:\Users\n0378\AppData\Local\Temp\linearrag-v0-compare-venv\Scripts\python.exe'

uv run python -B evals/linearrag_v0/prepare_v0_source.py --target $v0Source

uv run python -B evals/linearrag_v0/prepare_pdf_eval.py `
  --pdf evals/linearrag_v0/data/document.pdf `
  --output evals/linearrag_v0/data/document_eval.json `
  --cases 12

uv run python -B evals/linearrag_v0/compare_actual_v0.py `
  --payload evals/linearrag_v0/data/document_eval.json `
  --v0-python $v0Python `
  --v0-source $v0Source `
  --neo4j-user neo4j `
  --neo4j-password neo4j@2025 `
  --cases 12
```

评测脚本使用 `linearrag-*` 临时 ES 索引和 `linearrag_*` Neo4j 标签，并在 `finally`
中清理节点和索引。`v0` 源码环境仍需要 `neo4j`、`python-igraph` 和中文 spaCy 模型。
