# LinearRAG 项目开发约束

## 目录职责

- `src/common/`：通用配置和工具，配置统一放在 `settings.py`。
- `src/models/`：业务数据模型。
- `src/interfaces/`：可替换能力的接口。
- `src/services/`：索引、检索、知识库等业务用例。
- `src/model_providers/`：大模型和 Embedding 的统一入口。
- `src/document_processing/`：PDF 解析、文本切片和实体识别。
- `src/adapters/`：外部系统实现，按 `search / graph / storage / ai` 分类。

旧目录 `core / domain / ports / application / infra / nlp` 已废弃，不要重新引入。

## 可替换能力

- 搜索数据库统一实现 `src/interfaces/search.py` 的 `SearchStore`。
- 搜索能力必须同时考虑向量检索、BM25 检索和混合检索。
- 图数据库统一实现 `src/interfaces/graph.py` 的 `GraphStore`。
- `GraphStore` 只暴露语义级图操作，业务层不得直接使用 Cypher、GDS 或 Neo4j SDK。
- 大模型和 Embedding 统一从 `src/model_providers` 导入。
- 大模型和 Embedding 不得依赖 LangChain，使用官方 `openai` SDK 或其它独立 SDK。
- `langchain-text-splitters` 目前只允许用于文本切片，不得用于模型调用。
- 业务服务不得直接导入 `src/adapters`；只有启动层负责组装具体实现。

## 配置

- 配置统一使用 `pydantic-settings`，入口为 `src/common/settings.py`。
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
