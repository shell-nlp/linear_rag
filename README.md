# LinearRAG

LinearRAG 是一个面向 PDF 和文本片段的知识库检索服务。项目基于 FastAPI、
Elasticsearch 和 OpenAI 兼容模型接口，提供知识库管理、文件索引、文本片段管理，
以及向量检索、BM25 检索和混合检索能力。

## 核心能力

- 管理 Elasticsearch 知识库索引的生命周期。
- 上传 PDF，并行执行对象持久化、PDF 解析、实体抽取、向量化和索引写入。
- 直接索引和删除独立文本片段。
- 支持 `vector`、`bm25`、`hybrid` 三种检索模式。
- 通过 `entity_ids` 扩展关联段落，并回查同一文件中的相邻段落。
- 使用加权 RRF 融合主召回、实体扩展和相邻段落结果。
- 通过统一端口切换本地文件系统、MinIO、搜索数据库、Embedding 和 NER 实现。
- 内置 `index.html` 接口实验台，可直接调试全部业务接口。

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
2. 在进程池中解析 PDF 和切片，再执行实体抽取与向量化。
3. 对象落盘成功后，将完整段落文档批量写入 Elasticsearch。
4. 任一侧失败时执行补偿：删除本次新建对象，并按本次生成的段落 ID 回滚索引。

索引文档同时保存文本、向量、来源信息、实体信息和相邻段落关系。相同文本在不同
文件或不同对象路径下会生成不同段落 ID，避免跨文件关系串联。

### 检索流程

1. 按请求指定的模式执行主召回，候选数取 `top_k * 4` 和 20 的较大值。
2. 从主召回结果中选择有限数量的实体，通过结构化过滤扩展关联段落。
3. 批量回查候选段落的前后段落。
4. 按 `1.0 / 0.7 / 0.25` 的权重使用 RRF 融合三路结果，最后返回 `top_k`。

## 快速开始

### 环境要求

- Python 3.12 或更高版本。
- `uv`。
- Elasticsearch 8.x，并安装 `ik_smart` 分词插件。
- 提供 OpenAI 兼容接口的大模型和 Embedding 服务。
- 可选的 MinIO 服务；未配置时默认使用本地目录模拟对象存储。

### 本地运行

1. 参考 `.env.example` 准备配置。配置可通过环境变量注入，也可以放入
   `src/settings.py` 中 `ENV_FILE` 指向的 `.env` 文件。

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

所有配置统一由 `pydantic-settings` 从环境变量或 `ENV_FILE` 指向的 `.env`
文件读取，入口位于 `src/settings.py`。完整配置样例见 `.env.example`。

| 分类 | 主要变量 | 说明 |
| --- | --- | --- |
| 大模型 | `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME` | OpenAI 兼容对话模型配置 |
| Embedding | `EMBEDDING_API_URL`、`EMBEDDING_MODEL_NAME`、`EMBEDDING_DIM` | 向量服务地址、模型名和向量维度 |
| 索引 | `CHUNK_TOKEN_SIZE`、`CHUNK_OVERLAP_TOKEN_SIZE`、`BATCH_SIZE` | 文本切片与向量化参数 |
| 并发 | `MAX_WORKERS`、`INDEX_PROCESS_WORKERS` | 实体抽取和上传处理并发度 |
| 关系扩展 | `ENTITY_EXPANSION_*`、`NEIGHBOR_EXPANSION_ENABLED` | 控制实体和相邻段落扩展 |
| 对象存储 | `OBJECT_STORAGE_PROVIDER`、`LOCAL_STORAGE_ROOT` | 使用 `local` 或 `minio` 实现 |
| MinIO | `MINIO_ENDPOINT_URL`、`MINIO_ACCESS_KEY`、`MINIO_SECRET_KEY` | `OBJECT_STORAGE_PROVIDER=minio` 时必填 |
| Elasticsearch | `es_url`、`es_user`、`es_password` | 搜索数据库连接配置 |
| 服务 | `API_PORT`、`MAX_UPLOAD_BYTES` | 服务端口和单文件上传上限 |

本地对象存储与 MinIO 使用相同地址语义：
`存储根目录或桶/bucket_name/file_path`。本地模式下实际路径为
`LOCAL_STORAGE_ROOT/bucket_name/file_path`。

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

`search_mode` 支持 `vector`、`bm25` 和 `hybrid`。`hybrid` 会分别执行向量检索和
BM25 检索，再进行排名融合。

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
│   ├── retrieval/                  知识检索功能模块
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
- 实体扩展通过 `SearchStore` 的结构化过滤完成，不引入 Neo4j、Redis 写队列或实时
  PageRank。
- 业务服务不直接导入具体实现；具体实现只在 `src/api/bootstrap.py` 中装配。
- 功能模块按 `router.py / schemas.py / service.py` 组织。
- 新增或修改接口时，必须同步更新根目录 `index.html` 接口实验台。

## 测试

单元测试不依赖真实 Elasticsearch 或模型服务。

```bash
uv run python -B -m compileall -q main.py run_api.py src tests
uv run python -B -m unittest discover -s tests -v
```

## Docker

```bash
docker compose up -d --build --force-recreate
```

默认映射端口为 `12125`。生产环境应通过环境变量或 Compose 配置覆盖样例中的模型、
Elasticsearch 和对象存储连接信息。
