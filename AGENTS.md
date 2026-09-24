# LinearRAG 项目开发约束

## 目录职责

- `src/settings.py`：全项目配置入口。
- `src/utils.py`：全项目通用工具。
- `src/api/`：FastAPI 应用装配、依赖注入、共享响应模型和根路由。
- `src/common/`：通用能力包和共享模型。
- `src/common/models.py`：业务数据模型。
- `src/common/search_store/`：搜索接口和具体搜索实现。
- `src/common/object_storage/`：对象存储接口和具体实现。
- `src/common/model_providers/`：大模型和 Embedding 统一入口及实现。
- `src/common/document_processing/`：PDF 解析、文本切片和实体识别。
- `src/indexing/`：文档索引功能模块。
- `src/retrieval/`：知识检索功能模块。
- `src/knowledge_bases/`：知识库管理功能模块。

功能模块内部按 `router.py / schemas.py / service.py` 组织。
根路径、健康检查这类应用级路由直接放在 `src/api/app.py`，不要单独建模块。

旧目录 `core / domain / ports / application / infra / nlp / adapters / interfaces / services` 已废弃，不要重新引入。

## 可替换能力

- 搜索数据库统一实现 `src/common/search_store/base.py` 的 `SearchStore`。
- 搜索能力必须同时考虑向量检索、BM25 检索和混合检索。
- 关系元数据保存在同一搜索索引：段落的 `entity_ids`、`entities`、`previous_passage_id`、`next_passage_id`；句子的 `passage_id` 与 `entity_ids`；实体的 `entity_id` 和向量。
- 文件持久化统一实现 `src/common/object_storage/base.py` 的 `ObjectStorage`；业务层不得直接依赖 MinIO 或本地文件 API。
- `/index` 接收文件二进制，并行执行对象持久化和索引；任一侧失败必须补偿删除另一侧已写数据。
- 未配置远端对象存储时使用本地实现，目录语义保持为 `存储根目录/bucket_name/file_path`。
- 快速模式用 `SearchStore` 的结构化过滤与 RRF；`linear`/`linear_local` 在应用进程运行 PPR，不新增 Neo4j 或 Redis。
- 只有出现高频不定深度路径查询后，才重新评估独立关系存储；不得为假设需求保留兼容空包。
- 大模型和 Embedding 统一从 `src/common/model_providers` 导入。
- 大模型和 Embedding 不得依赖 LangChain，使用官方 `openai` SDK 或其它独立 SDK。
- `langchain-text-splitters` 目前只允许用于文本切片，不得用于模型调用。
- 业务服务不得直接导入具体实现；只有启动层负责组装具体实现。
- 功能模块的 `service.py` 不应包含 FastAPI 路由代码。
- `router.py` 只负责 HTTP 参数、依赖注入和响应转换。
- 新增、删除或修改任何 FastAPI 接口，以及请求字段、请求类型或响应结构时，必须同步更新根目录 `index.html` 的接口实验台。
- `index.html` 必须覆盖当前全部业务接口，并保持请求路径、参数名称、`application/json` 与 `multipart/form-data` 的请求方式与后端一致。

## 配置

- 配置统一使用 `pydantic-settings`，入口为 `src/settings.py`。
- 禁止恢复模块级常量式配置。
- 本地配置写入 `.env`，`.env` 不提交。
- 配置样例维护在 `.env.example`，新增配置时必须同步更新样例。

## LinearRAG 算法与优化约束

- 作者参考实现是 `DEEP-PolyU/LinearRAG` 的 `src/LinearRAG.py`；`linear` 是全图计算路径，`linear_local` 是 ES 候选子图上的近似路径，`vector/bm25/hybrid` 是原有低延迟快速路径，三者不可混称等价。
- 索引阶段 `IndexingService` 按文件生成段落、实体、句子节点与向量，使用文件来源隔离物理节点，同名实体以逻辑 `entity_id` 在查询图中合并。`FileIndexingWorkflow` 将这三类节点作为同一批提交和回滚；按 `file_id` 删除所有节点。
- `SearchStore.scan_documents` 负责读取全图或局部关联节点并拒绝超限，不负责 PPR；`src/retrieval/linear.py` 负责种子实体匹配、句子桥接、多轮激活、实体—段落/相邻段落边、重启权重及应用层 PPR。
- 图计算参数通过 `src/settings.py` 传入，样例见 `.env.example`。旧索引没有句子/实体节点，必须重新索引才能使用图模式。
- 图节点数量可能远大于段落数；索引 Embedding 批量不得超过 `LINEAR_EMBEDDING_BATCH_SIZE`，避免远端批量请求长时间无响应。
- PDF 解析和文本切片属于 CPU 密集型任务，生产路径必须经 `FileIndexingWorkflow` 的 `ProcessPoolExecutor` 执行；不得为简化调用把 `PDFParser.get_chunk` 移回 FastAPI 主进程或普通线程池。
- 优化图读取或缓存时不得把局部 top-k 搜索说成全图 PPR 的数值等价；先以相同 NER、Embedding、语料与参数验证中间激活、排序、Recall@K 和延迟。快照缓存要有知识库版本及索引增删后的失效策略。
- 图模式当前仅支持单知识库，避免不同索引的逻辑实体 ID 被隐式合并；多索引查询继续使用快速模式。不要绕过该约束而不设计跨索引身份与来源语义。

## 注释和测试

- 新增或修改的代码必须补中文注释，重点说明职责、参数语义和实现差异。
- 单元测试使用 `unittest`，默认执行不得依赖真实 Elasticsearch 或模型服务；真实集成测试必须通过显式环境变量启用并自行清理测试数据。
- 修改后至少运行：

```bash
uv run python -B -m compileall -q main.py run_api.py src tests
uv run python -B -m unittest discover -s tests -v
```

## CodeGraph

- 仓库已建立 `.codegraph/`。
- 理解或定位代码时优先使用：

```bash
codegraph explore "<symbol or question>"
codegraph sync .
```
