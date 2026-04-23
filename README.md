
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