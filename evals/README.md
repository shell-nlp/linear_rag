# 离线评测资产

本目录保存可复跑的检索评测代码和固定数据，避免依赖临时目录。

## 目录

- `linearrag_v0/`：纯向量、Hybrid-RRF、当前 `linear_local` 与实际 `v0`
  Neo4j/GDS 路径的同口径对比。
- `linearrag_local/`：当前版本在不同索引规模和候选规模下的局部图性能基准。

## 数据集

- `linearrag_v0/data/document.pdf`：源 PDF，SHA256
  `F3A70A0C672BFD8EC4BBBE5DC0A6F0BBE381E62050C4A2A2508D6373FEF2F295`。
- `linearrag_v0/data/document_eval.json`：固定预切片和 12 个弱标注问题，SHA256
  `02729EC330047F138DF6AA3E5200894E0ED80C1D70D5060C0615165EDDCAAE51`。
- `linearrag_v0/data/document_eval_large.json`：同一预切片和 152 条多问法弱标注问题，
  SHA256 `2924023824854978A64E3F2529038FA94141ED00BDC847613EB155625C0731FE`。
- `linearrag_v0/data/document_eval_multihop.json`：同一预切片和 20 条两目标多跳问题，
  SHA256 `828B43772DC4E827C0276AB40848C4CB2E50D544AADCDF2606F3F4FFCE580216`。

评测数据来自用户提供的内部文档，仅用于本地优化验证；不要上传到外部服务或公开仓库。
`v0` 运行环境位于临时目录，仓库只保存运行器和数据，不复制旧版本源码。
使用 `evals/linearrag_v0/extract_v0_source.py` 可从当前仓库的 `v0` tag
重新抽取旧源码；生成目录属于本地缓存，不应提交。
