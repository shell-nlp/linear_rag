
```bash
pip install -r requirements.txt
```

**Step 2: 下载 Spacy language model**

```bash
python -m spacy download en_core_web_trf/zh_core_web_md

```
或者
```bash
pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_md-3.7.0/zh_core_web_md-3.7.0-py3-none-any.whl
```

模型保存地址：/home/dev/huangbinghan/LinearRAG-main/.venv/lib/python3.12/site-packages/en_core_web_trf

python -m spacy download zh_core_web_md

# docker 部署
docker build -t hubeirs-rag-base:v1 .
docker-compose up -d --build --force-recreate

## 目录结构

配置统一由 `.env` 和 `pydantic-settings` 读取，配置入口为
`src/common/settings.py`。

首次运行前复制样例文件：

```bash
cp .env.example .env
```

```text
src/
  common/                 通用配置和工具
    settings.py           pydantic-settings 配置入口
  models/                 业务数据模型
  interfaces/             可替换能力接口
  services/               索引、检索、知识库等业务用例
  model_providers/        大模型和 Embedding 统一入口
  document_processing/    PDF 解析、文本切片和实体识别
  adapters/               外部系统适配器
    search/               搜索数据库实现，例如 Elasticsearch
    graph/                图数据库实现，例如 Neo4j
    storage/              对象存储实现，例如 MinIO
    ai/                   大模型和 Embedding 具体实现
```

切换向量数据库时实现 `src/interfaces/search.py` 的 `SearchStore`；
切换图数据库时实现 `src/interfaces/graph.py` 的 `GraphStore`。
