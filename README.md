
```bash
uv sync
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
    object_storage/       对象存储接口及本地、MinIO 实现
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

已有知识库需要单独迁移或重建，才能补齐实体和相邻段落字段。新上传流程不覆盖
已有对象，也不会在上传事务中修改历史索引；新段落 ID 按文件来源和段落序号隔离。

## 文件上传与索引

`POST /admin_api/python-knowledge-management/index` 使用 `multipart/form-data`：

```bash
curl -X POST http://localhost:12125/admin_api/python-knowledge-management/index \
  -F "kb_name=demo" \
  -F "bucket_name=documents" \
  -F "file_path=reports/example.pdf" \
  -F "file_id=file-001" \
  -F "file=@example.pdf;type=application/pdf"
```

接口收到二进制后，并行执行文件持久化、PDF 解析、实体抽取和向量化。对象写入成功
后再提交搜索数据库；
只有文件与索引都成功才返回成功。解析或索引失败时删除本次新建对象，索引写入失败
时还会按本次段落 ID 精确回滚，避免误删同一 `file_id` 下的历史数据。已存在的
`bucket_name/file_path` 返回 HTTP 409，不会覆盖原文件。

对象存储统一通过 `ObjectStorage` 抽象访问：

```dotenv
# 默认：本地目录模拟对象存储
OBJECT_STORAGE_PROVIDER=local
LOCAL_STORAGE_ROOT=./.data/object-storage

# 使用 MinIO 时改为 minio，并填写以下配置
OBJECT_STORAGE_PROVIDER=minio
MINIO_ENDPOINT_URL=http://minio:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
```

本地模式和 MinIO 模式使用相同地址语义：`bucket_name + file_path`。本地实际路径为
`LOCAL_STORAGE_ROOT/bucket_name/file_path`，因此切换存储实现不影响上传接口和索引元数据。
