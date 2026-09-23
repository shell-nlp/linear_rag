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
- 当前关系检索统一由搜索文档中的 `entity_ids`、`entities`、`previous_passage_id` 和 `next_passage_id` 承载。
- 实体扩展必须通过 `SearchStore` 的结构化过滤执行，不新增 Neo4j、Redis 写队列或实时 PageRank。
- 只有出现高频不定深度路径查询后，才重新评估独立关系存储；不得为假设需求保留兼容空包。
- 大模型和 Embedding 统一从 `src/common/model_providers` 导入。
- 大模型和 Embedding 不得依赖 LangChain，使用官方 `openai` SDK 或其它独立 SDK。
- `langchain-text-splitters` 目前只允许用于文本切片，不得用于模型调用。
- 业务服务不得直接导入具体实现；只有启动层负责组装具体实现。
- 功能模块的 `service.py` 不应包含 FastAPI 路由代码。
- `router.py` 只负责 HTTP 参数、依赖注入和响应转换。

## 配置

- 配置统一使用 `pydantic-settings`，入口为 `src/settings.py`。
- 禁止恢复模块级常量式配置。
- 本地配置写入 `.env`，`.env` 不提交。
- 配置样例维护在 `.env.example`，新增配置时必须同步更新样例。

## 注释和测试

- 新增或修改的代码必须补中文注释，重点说明职责、参数语义和实现差异。
- 单元测试使用 `unittest`，不得依赖真实 Elasticsearch 或模型服务。
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
