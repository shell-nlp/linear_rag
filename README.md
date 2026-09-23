
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
  settings.py             pydantic-settings 配置入口
  utils.py                全项目通用工具
  common/                 通用能力包
    models.py             业务数据模型
    search_store/         搜索接口和 Elasticsearch 实现
    graph_store/          图接口和 Neo4j 实现
      base.py             GraphStore 接口
      neo4j.py            Neo4j 的完整实现
    object_storage/       对象存储接口和 MinIO 实现
    model_providers/      大模型和 Embedding 统一入口
    document_processing/  PDF、切片和实体识别
  services/               索引、检索、知识库等业务用例
```

切换向量数据库时实现 `src/common/search_store/base.py` 的 `SearchStore`；
切换图数据库时实现 `src/common/graph_store/base.py` 的 `GraphStore`。
