# LinearRAG

LinearRAG 是一个面向 PDF 和文本片段的知识库检索服务。项目基于 FastAPI、
Elasticsearch 和 OpenAI 兼容模型接口，提供知识库管理、文件索引、文本片段管理，
以及向量检索、BM25 检索和混合检索能力。

本项目借鉴 [作者开源实现](https://github.com/DEEP-PolyU/LinearRAG)
（对应 [LinearRAG 论文](https://arxiv.org/abs/2510.10114)）的
“以实体连接原文段落、避免显式关系三元组抽取”思路，但当前实现是面向
Elasticsearch 的工程改写，**不是作者算法的完整复现**。

## 核心能力

- 管理 Elasticsearch 知识库索引的生命周期。
- 上传 PDF，并行执行对象持久化与解析、实体抽取、向量化准备；存储成功后提交索引。
- 直接索引和删除独立文本片段。
- 支持 `vector`、`bm25`、`hybrid` 快速模式及 `linear`、`linear_local` 图计算模式。
- 通过 `entity_ids` 扩展关联段落，并回查同一文件中的相邻段落。
- 使用加权 RRF 融合主召回、实体扩展和相邻段落结果。
- 通过统一端口切换本地文件系统、MinIO、搜索数据库、Embedding 和 NER 实现。
- 内置 `index.html` 接口实验台，可直接调试全部业务接口。

## 实现思路与作者代码对照

主要对照对象是 [DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)
的 `main` 分支，核对版本 `bcc94e66`（2026-07-05），重点看
[`src/LinearRAG.py`](https://github.com/DEEP-PolyU/LinearRAG/blob/main/src/LinearRAG.py)。
论文 *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale
Corpora* 可用于理解设计动机；以下对照以仓库代码实际运行路径为准。

作者代码在索引时提取段落及句子中的实体，分别保存段落、实体、句子的
Embedding；在 `igraph` 中建立实体—段落加权边和相邻段落边，并保留
实体—句子关联用于查询阶段的语义桥接。查询先从问题中识别实体、用向量相似度
匹配种子实体，再沿实体—句子—实体多轮激活；结合段落语义分数构造重启权重，
通过 `igraph.personalized_pagerank` 排序段落。没有种子实体时退回稠密段落检索。
`qa()` 会把检索段落交给 LLM 生成答案。代码提供 BFS 式迭代和可选的 PyTorch
稀疏矩阵检索路径。

本项目把段落、句子、实体节点及关联信息保存在同一个搜索库中，不另设图数据库。
`linear` 从 ES 读取知识库图，运行问题 NER、种子匹配、句子桥接、多轮激活与应用层
PPR；`linear_local` 合并 ES 向量候选和问题实体关联段落，再读取关联节点，在候选子图运行同一套
计算。原来的 `vector`、`bm25`、`hybrid` 仍保持低延迟快速路径：主召回后只做
一次实体与邻接扩展，再按 RRF 融合。`SearchStore` 负责数据库访问，不承担算法计算。

| 维度 | 作者仓库 `src/LinearRAG.py` | 本项目当前实现 |
| --- | --- | --- |
| 输入与索引 | 批量段落；提取句子和实体，保存三类 Embedding | PDF 上传、切片；ES 保存段落/实体/句子向量及关联字段 |
| 关系表示 | `igraph` 的实体—段落加权边、相邻段落边；实体—句子关联用于桥接 | 段落的实体与前后 ID、句子的段落与实体 ID、实体节点按逻辑 ID 合并 |
| 查询种子 | 问题 NER 后以实体向量匹配种子；无种子则退回稠密段落检索 | `linear`/`linear_local` 同样进行种子匹配与回退；快速模式以段落主召回 |
| 扩展与排序 | 句子相似度引导多轮实体激活，段落权重作重启向量，PPR 排序 | 两种 `linear` 模式在应用层运行桥接与 PPR；快速模式按实体/邻接扩展和 RRF |
| 存储与交付 | Parquet/JSON 与 GraphML，运行时 `igraph`；实验脚本 `run.py` | 本地或 MinIO 保存原文件，Elasticsearch 保存索引；FastAPI 服务 |
| 答案生成 | `qa()` 将检索结果输入 LLM | `/retrieve` 只返回检索段落，不生成答案 |

这里的 `entities[].importance` 只是当前段落内对数频次归一化后的权重，
不是 PPR 分数。`linear_local` 的 ES ANN top-k 候选可能丢失全图可达段落，
因此它是加速近似，**不能称为全图 PPR 的数值等价实现**。`linear` 虽保留官方
主要计算步骤，仍因中文 NER/Embedding、图节点去重、迭代求解和数据来源不同而不
保证逐项复现作者实验结果。快速模式更不是 PPR 的近似公式。
作者仓库和论文的实验指标不能直接套用到本项目，检索效果需用相同语料与
评估口径单独验证。

### 同口径检索结果

使用固定 PDF 解析出的 90 个段落和 152 条多问法弱标注问题，在同一真实 Embedding、
Elasticsearch、Neo4j/GDS 环境中对比纯向量、Hybrid-RRF、当前 `linear_local`
与实际 `v0`：

| 路径 | Recall@1 | Recall@3 | Recall@5 | Recall@8 | Recall@10 | Recall@15 | 查询中位耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES 纯向量 | 51/152 | 88/152 | 104/152 | 116/152 | 121/152 | 130/152 | 0.062 秒 |
| ES 向量 + BM25 + RRF | 88/152 | 110/152 | 120/152 | 130/152 | 132/152 | 141/152 | 0.161 秒 |
| 当前 `linear_local` | 107/152 | 118/152 | 121/152 | 129/152 | 131/152 | 135/152 | 1.936 秒 |
| 实际 `v0` | 75/152 | 105/152 | 110/152 | 115/152 | 119/152 | 119/152 | 2.611 秒 |

这里的纯向量和 Hybrid-RRF 都是主召回基线，不追加实体扩展和相邻段落融合；
Hybrid-RRF 指 ES 分别执行向量与 BM25 后，再按排名执行 RRF。同次运行中，当前版本
临时索引约 11.72 秒，`v0` 约 12.42 秒。152 条问题来自 76 个唯一实体，每个实体
使用两种问法。`linear_local` 在 Recall@1/3/5 最高，但 Hybrid-RRF 从 Recall@8
开始反超，并在 Recall@15 达到 141/152；说明局部图排序更适合前列精排，BM25
补充更有利于扩大候选召回。该实验仍是内部 PDF 上的弱标注评测，不能替代人工相关性
标注、公开中文基准或生产规模测试。完整环境、结论和复跑命令见
[评测报告](docs/comparison-v0-local.md)。

### 多跳检索结果

使用 20 条两跳问题，每题要求两个目标段落同时进入结果。`All@K` 表示两个必需段落
都进入前 K；`PassageRecall@K` 表示目标段落的平均覆盖率：

| 路径 | Any@15 | PassageRecall@5 | All@3 | All@5 | All@10 | All@15 | 中位耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ES 纯向量 | 95% | 52.5% | 15% | 30% | 60% | 80% | 0.061 秒 |
| ES 向量 + BM25 + RRF | 100% | 62.5% | 20% | 35% | 70% | 75% | 0.160 秒 |
| 全图 `linear` | 100% | 65.0% | 35% | 50% | 85% | 90% | 1.711 秒 |
| 当前 `linear_local` | 100% | 65.0% | 35% | 50% | 85% | 90% | 1.934 秒 |
| 实际 `v0` | 90% | 50.0% | 15% | 20% | 40% | 40% | 2.661 秒 |

Hybrid-RRF 最容易找到至少一个目标段落，但 `linear_local` 的两跳补齐能力明显更强。
在这份 90 段小图上，全图 `linear` 与 `linear_local` 的召回指标相同，耗时还略低；
这是小图特例，不能说明全图 PPR 能扩展到百万级数据。
该集合由 8 条共享桥接实体的问题和 12 条相邻制度段落问题组成，属于本地派生弱标注，
只评价“必要段落是否被检索到”，不评价最终答案的多跳推理正确性。

## 架构

```text
+--------------------------------------------------------------------------+
|                 REST 客户端 / index.html 接口实验台                      |
+------------------------------------+-------------------------------------+
                                     |
                                     | HTTP JSON / multipart/form-data
                                     v
+--------------------------------------------------------------------------+
| FastAPI 应用  src/api/app.py                                             |
| lifespan 装配 -> 索引路由 / 检索路由 / 知识库路由                       |
+----------------+----------------------+----------------+-----------------+
                 |                      |                |
                 v                      v                v
+------------------------+  +------------------------+  +-------------------+
| FileIndexingWorkflow   |  | IndexingService        |  | RetrievalService  |
| 上传、持久化、补偿事务 |  | 实体、邻接关系、向量   |  | 召回、扩展、融合  |
+------------+-----------+  +------------+-----------+  +---------+---------+
             |                           |                        |
             v                           v                        v
+------------------------+  +------------------------+  +-------------------+
| ObjectStorage          |  | ModelProviders         |  | SearchStore       |
| Local / MinIO          |  | Embedding / LLM        |  | 统一搜索端口      |
+------------------------+  +------------------------+  +---------+---------+
                                                                 |
                                                                 v
                                                       +-------------------+
                                                       | Elasticsearch     |
                                                       +-------------------+

                     +------------------------------+
                     | Settings + .env              |
                     | src/settings.py              |
                     +---------------+--------------+
                                     |
                                     | 启动时注入
                                     v
                     +------------------------------+
                     | src/api/bootstrap.py         |
                     | 应用状态与具体实现装配       |
                     +------------------------------+
```

### 索引流程

`POST /index` 接收 PDF 二进制后，由 `FileIndexingWorkflow` 协调两类工作：

1. 将文件写入 `ObjectStorage`。
2. 在进程池中完成 CPU 密集的 PDF 解析和切片；随后由索引服务执行实体抽取与向量化。
3. 对象落盘成功后，将完整段落文档批量写入 Elasticsearch。
4. 任一侧失败时执行补偿：删除本次新建对象，并按本次生成的段落、句子和实体节点 ID 回滚索引。

索引同时保存段落、实体与句子节点及来源信息。相同文本在不同文件或不同对象路径
下会生成不同段落 ID，避免跨文件关系串联。原有索引不包含句子/实体节点，
使用 `linear` 模式前须重新上传并索引旧文件。

### 检索流程

1. 按请求指定的模式执行主召回，候选数取 `top_k * 4` 和 20 的较大值；混合模式先将向量和 BM25 排名用 RRF 融合。
2. 从主召回结果中选择有限数量的实体，通过结构化过滤扩展关联段落。
3. 批量回查候选段落的前后段落。
4. 按 `1.0 / 0.7 / 0.25` 的权重使用 RRF 融合主召回、实体扩展和相邻段落，最后返回 `top_k`。该排序不是 PPR。

### 图计算模式

- `linear`：读取知识库全图（上限 `LINEAR_MAX_NODES`），从问题 NER 匹配实体种子，
  经句子相似度做多轮实体激活，结合段落相似度与实体提及次数构造重启权重，
  在应用进程运行 PPR；无种子时回退段落余弦排序。
- `linear_local`：ES 向量召回段落，同时通过问题 NER 和实体向量检索种子，再按实体 ID
  各取有限数量的关联段落；关联段落优先与向量候选去重合并，按总预算读取实体和句子节点，
  在候选子图运行 PPR。它相对全图减少图读取和 PPR 规模，但候选裁剪会改变结果；
  因多做 NER、句子读取与图计算，不能保证比原有 top-k/RRF 快速模式更快。
- 目前每次 `linear` 请求都从 ES 读取图；大知识库会有内存和请求耗时开销。
  全图超限会报错，不会静默截断。生产优化应考虑按知识库版本缓存图快照，
  并与作者代码及局部模式做 Recall@K、排序和延迟对照。
- 两种图模式当前一次只查询一个知识库；多知识库联合检索可使用快速模式。
- `LINEAR_SEED_ENTITIES` 限制问题实体数，`LINEAR_PASSAGES_PER_ENTITY` 限制每实体段落数，
  `LINEAR_LOCAL_MAX_PASSAGES` 限制合并后的段落数，`LINEAR_LOCAL_MAX_SENTENCES` 限制句子节点数；
  命中高频实体时不会全量扫描其关联段落。当前实体节点仍按选中实体 ID 扫描，
  须通过 `LINEAR_MAX_NODES` 限制整个局部图规模。

### 规模边界

当前 `linear` 会把知识库图节点和向量读取到应用进程，只适合中小规模知识库和官方算法对照。
百万级节点不能按请求扫描全图；上亿节点仅 1024 维 float32 向量就接近 400 GB，尚未计算
文本、边和对象开销。生产规模应采用以下路径：

1. 用 ES 或独立向量库完成段落、实体和句子的候选召回。
2. 限制种子实体数、扩展跳数和子图节点数，在候选子图上做近似 PPR 或有界随机游走。
3. 在离线任务中预计算实体重要度、实体关联段落、社区、邻接摘要和常用 PPR 信号。
4. 按知识库或租户分片，使用 Spark、GraphX、GraphScope、cuGraph 等分布式图计算处理全图。
5. 只有在线不定深度多跳成为核心需求时，才评估独立图数据库；不要把单机 `igraph` 或 NumPy 全图排序当作上亿节点的解决方案。

## 快速开始

### 环境要求

- Python 3.12 或更高版本。
- `uv`。
- Elasticsearch 8.x，并安装 `ik_smart` 分词插件。
- 提供 OpenAI 兼容接口的大模型和 Embedding 服务。
- 可选的 MinIO 服务；未配置时默认使用本地目录模拟对象存储。

### 本地运行

1. 参考 `.env.example` 准备项目 `.env`。配置也可通过环境变量注入；
   `src/settings.py` 使用 `find_dotenv` 查找 `.env`。

2. 按实际环境修改模型、Elasticsearch 和对象存储配置。

3. 安装依赖：

   ```bash
   uv sync
   ```

   `zh_core_web_md` 已通过 `pyproject.toml` 指向 `asset/` 中的本地 wheel，
   正常执行 `uv sync` 即可完成安装。

4. 启动服务：

   ```bash
   uv run python main.py
   ```

   也可以显式指定监听地址和端口：

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 12125
   ```

5. 打开以下页面：

   - 接口实验台：<http://localhost:12125/>
   - OpenAPI 文档：<http://localhost:12125/docs>
   - 健康检查：<http://localhost:12125/admin_api/python-knowledge-management/health>

`run_api.py` 是演示索引流程的示例脚本，不是 API 服务入口。

## 配置

所有配置统一由 `pydantic-settings` 从环境变量或 `find_dotenv` 找到的 `.env`
文件读取，入口位于 `src/settings.py`。修改配置后需重启服务以刷新缓存的配置。
完整配置样例见 `.env.example`。

| 分类 | 主要变量 | 说明 |
| --- | --- | --- |
| 大模型 | `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME` | OpenAI 兼容对话模型配置 |
| Embedding | `EMBEDDING_API_URL`、`EMBEDDING_MODEL_NAME`、`EMBEDDING_DIM` | 向量服务地址、模型名和向量维度 |
| 索引 | `CHUNK_TOKEN_SIZE`、`CHUNK_OVERLAP_TOKEN_SIZE`、`BATCH_SIZE` | 文本切片与向量化参数 |
| 并发 | `MAX_WORKERS`、`INDEX_PROCESS_WORKERS` | 实体抽取和上传处理并发度 |
| 关系扩展 | `ENTITY_EXPANSION_*`、`NEIGHBOR_EXPANSION_ENABLED` | 控制实体和相邻段落扩展 |
| 图算法 | `LINEAR_MAX_ITERATIONS`、`LINEAR_TOP_K_SENTENCE`、`LINEAR_PASSAGE_RATIO`、`LINEAR_PASSAGE_NODE_WEIGHT`、`LINEAR_DAMPING`、`LINEAR_ITERATION_THRESHOLD` | 官方计算路径参数 |
| 图规模 | `LINEAR_LOCAL_CANDIDATES`、`LINEAR_MAX_NODES`、`LINEAR_EMBEDDING_BATCH_SIZE` | 局部候选数、图读取上限与索引向量批量 |
| 局部预算 | `LINEAR_SEED_ENTITIES`、`LINEAR_PASSAGES_PER_ENTITY`、`LINEAR_LOCAL_MAX_PASSAGES`、`LINEAR_LOCAL_MAX_SENTENCES` | 种子数、每实体关联段落、总段落和句子上限 |
| 对象存储 | `OBJECT_STORAGE_PROVIDER`、`LOCAL_STORAGE_ROOT` | 使用 `local` 或 `minio` 实现 |
| MinIO | `MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` | `OBJECT_STORAGE_PROVIDER=minio` 时必填 |
| Elasticsearch | `es_url`、`es_user`、`es_password` | 搜索数据库连接配置 |
| 服务 | `API_PORT`、`MAX_UPLOAD_BYTES` | 服务端口和单文件上传上限 |

本地对象存储与 MinIO 使用相同的逻辑地址 `bucket_name + file_path`：
本地模式下实际路径为 `LOCAL_STORAGE_ROOT/bucket_name/file_path`；
MinIO 模式下 `bucket_name` 是桶名，`file_path` 是桶内对象键。

## API

业务接口统一使用前缀：

```text
/admin_api/python-knowledge-management
```

| 方法 | 路径 | 请求方式 | 说明 |
| --- | --- | --- | --- |
| `GET` | `/health` | 无 | 服务存活检查 |
| `POST` | `/create_knowledgebase` | `application/json` | 创建知识库索引 |
| `POST` | `/delete_knowledgebase` | `application/json` | 删除知识库索引及全部文档 |
| `POST` | `/index` | `multipart/form-data` | 上传 PDF 并建立索引 |
| `POST` | `/index_single` | `application/json` | 索引独立文本片段 |
| `POST` | `/delete_single` | `application/json` | 按文档 ID 删除文本片段 |
| `POST` | `/delete_files` | `application/json` | 按文件 ID 删除索引文档 |
| `POST` | `/retrieve` | `application/json` | 执行知识库检索 |
| `POST` | `/search_es` | `application/json` | 在指定索引和字段上执行查询 |

### 上传 PDF

```bash
curl -X POST http://localhost:12125/admin_api/python-knowledge-management/index \
  -F "kb_name=demo" \
  -F "bucket_name=documents" \
  -F "file_path=reports/example.pdf" \
  -F "file_id=file-001" \
  -F "file=@example.pdf;type=application/pdf"
```

`kb_name`、`bucket_name`、`file_path` 和 `file` 必填，`file_id` 可选。目标对象已存在
时返回 HTTP 409，不会覆盖原文件。

### 知识库检索

```bash
curl -X POST http://localhost:12125/admin_api/python-knowledge-management/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "questions": "水库开展了哪些监测项目？",
    "index_names": ["demo"],
    "top_k": 5,
    "search_mode": "hybrid"
  }'
```

`search_mode` 支持 `vector`、`bm25`、`hybrid`、`linear`、`linear_local`。
`hybrid` 会分别执行向量和 BM25 检索并融合排名；后两者分别运行全图和局部图 PPR。
`/search_es` 是底层搜索字段接口，仅支持前三种搜索模式。

## 项目结构

```text
.
├── main.py                         FastAPI 服务入口
├── run_api.py                      本地演示脚本
├── index.html                      接口实验台
├── src/
│   ├── settings.py                 全项目配置入口
│   ├── utils.py                    通用工具
│   ├── api/                        FastAPI 装配、依赖注入和共享响应模型
│   ├── indexing/                   文档索引功能模块
│   ├── retrieval/                  知识检索功能模块（含 linear.py 图算法）
│   ├── knowledge_bases/            知识库管理功能模块
│   └── common/
│       ├── models.py               业务数据模型
│       ├── search_store/           搜索接口和 Elasticsearch 实现
│       ├── object_storage/         对象存储接口、Local 和 MinIO 实现
│       ├── model_providers/        LLM 和 Embedding 统一入口
│       └── document_processing/    PDF 解析、切片和实体识别
└── tests/                          单元测试
```

## 扩展约束

- 搜索数据库实现 `src/common/search_store/base.py` 的 `SearchStore`。
- 文件持久化实现 `src/common/object_storage/base.py` 的 `ObjectStorage`。
- 大模型和 Embedding 统一从 `src/common/model_providers` 导入。
- 快速模式的实体扩展通过 `SearchStore` 结构化过滤完成；`linear` 和 `linear_local`
  由 `src/retrieval/linear.py` 在应用进程运行 PPR，不引入 Neo4j 或 Redis。
- 业务服务不直接导入具体实现；具体实现只在 `src/api/bootstrap.py` 中装配。
- 功能模块按 `router.py / schemas.py / service.py` 组织。
- 新增或修改接口时，必须同步更新根目录 `index.html` 接口实验台。

## 测试

单元测试不依赖真实 Elasticsearch 或模型服务。

```bash
uv run python -B -m compileall -q main.py run_api.py src tests
uv run python -B -m unittest discover -s tests -v
```

真实 Elasticsearch 与真实模型链路默认跳过，显式启用后执行：

```powershell
$env:LINEAR_ES_INTEGRATION='1'
uv run python -B -m unittest discover -s tests -p test_linear_elasticsearch_integration.py -v

$env:LINEAR_REAL_PDF_INTEGRATION='1'
$env:LINEAR_TEST_PDF='C:\path\to\document.pdf'
uv run python -B -m unittest discover -s tests -p test_real_pdf_integration.py -v
```

真实集成测试会使用独立的 `linearrag-*` 测试索引和本地测试对象，结束后清理。

局部图检索压测可直接运行：

```powershell
uv run python -B evals/linearrag_local/benchmark_linear_local.py `
  --passages 1000 5000 20000 50000 `
  --candidates 200 1000 2000 `
  --queries 5
```

脚本使用固定向量和固定 NER，只测量 ES 候选召回、实体/句子扫描和局部 PPR，
避免远端模型延迟干扰；每个场景使用临时索引并在结束时删除。

与 `v0` 的真实 Neo4j/GDS 同口径对比见 [评测报告](docs/comparison-v0-local.md)，
主入口为 `evals/linearrag_v0/evaluate_retrieval_paths.py`。评测代码和固定数据集
保留在 `evals/`；该结果仍不能替代生产规模与人工标注评测。

## Docker

```bash
docker compose up -d --build --force-recreate
```

默认映射端口为 `12125`。生产环境应通过环境变量或 Compose 配置覆盖样例中的模型、
Elasticsearch 和对象存储连接信息。
