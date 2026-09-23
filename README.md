
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

python -m spacy download zh_core_web_md

# docker 部署
docker build -t linear-rag:latest .
docker compose up -d --build --force-recreate

## 目录结构

配置统一由 `.env` 和 `pydantic-settings` 读取，配置入口为
`src/settings.py`。

首次运行前复制样例文件：

```bash
cp .env.example .env
```

```text
src/
  settings.py             pydantic-settings 配置入口
  utils.py                全项目通用工具
  api/                    FastAPI 应用装配、依赖注入和根路由
  indexing/               文档索引功能模块
  retrieval/              知识检索功能模块
  knowledge_bases/        知识库管理功能模块
  common/                 通用能力包
    models.py             业务数据模型
    search_store/         搜索接口和 Elasticsearch 实现
    object_storage/       对象存储接口和 MinIO 实现
    model_providers/      大模型和 Embedding 统一入口
    document_processing/  PDF、切片和实体识别
```

切换搜索数据库时实现 `src/common/search_store/base.py` 的 `SearchStore`。
实现必须覆盖向量检索、BM25、混合检索、结构化过滤、批量写入和删除。

## 检索架构

项目只使用搜索数据库保存检索数据，不依赖 Neo4j、Redis 或实时图算法。
索引阶段会把以下预计算信息写入段落文档：

- 实体 ID、实体名称、段落内出现次数和局部重要度；
- 同一文件内的前一段和后一段 ID；
- 文档来源、段落序号以及向量。

查询阶段先执行用户指定的向量、BM25 或混合检索，再通过 `entity_ids`
倒排字段扩展关联段落，并批量回查相邻段落。三路结果使用加权 RRF 融合，
不进行实时构图或 PageRank。

已有知识库需要重新执行一次文档索引，才能补齐实体和相邻段落字段。
重新索引会清理旧版纯文本哈希 ID，并按文件来源和段落序号生成隔离后的稳定 ID。
