# LinearRAG 项目开发约束

## 目录职责

- `src/settings.py`：全项目配置入口。
- `src/utils.py`：全项目通用工具。
- `src/common/`：通用能力包和共享模型。
- `src/common/models.py`：业务数据模型。
- `src/common/search_store/`：搜索接口和具体搜索实现。
- `src/common/graph_store/`：图接口和具体图实现。
- `src/common/object_storage/`：对象存储接口和具体实现。
- `src/common/model_providers/`：大模型和 Embedding 统一入口及实现。
- `src/common/document_processing/`：PDF 解析、文本切片和实体识别。
- `src/services/`：索引、检索、知识库等业务用例。

旧目录 `core / domain / ports / application / infra / nlp / adapters / interfaces` 已废弃，不要重新引入。

## 可替换能力

- 搜索数据库统一实现 `src/common/search_store/base.py` 的 `SearchStore`。
- 搜索能力必须同时考虑向量检索、BM25 检索和混合检索。
- 图数据库统一实现 `src/common/graph_store/base.py` 的 `GraphStore`。
- 同一个图数据库实现只保留一个文件，Neo4j 的驱动、查询、写入、删除和队列统一放在 `src/common/graph_store/neo4j.py`。
- `GraphStore` 只暴露语义级图操作，业务层不得直接使用 Cypher、GDS 或 Neo4j SDK。
- 大模型和 Embedding 统一从 `src/common/model_providers` 导入。
- 大模型和 Embedding 不得依赖 LangChain，使用官方 `openai` SDK 或其它独立 SDK。
- `langchain-text-splitters` 目前只允许用于文本切片，不得用于模型调用。
- 业务服务不得直接导入具体实现；只有启动层负责组装具体实现。

## 配置

- 配置统一使用 `pydantic-settings`，入口为 `src/settings.py`。
- 禁止恢复模块级常量式配置。
- 本地配置写入 `.env`，`.env` 不提交。
- 配置样例维护在 `.env.example`，新增配置时必须同步更新样例。

## 注释和测试

- 新增或修改的代码必须补中文注释，重点说明职责、参数语义和实现差异。
- 单元测试使用 `unittest`，不得依赖真实 Elasticsearch、Neo4j、Redis 或模型服务。
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
