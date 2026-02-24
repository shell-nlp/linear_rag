# 使用你提供的带有 uv 和 cuda 的基础镜像
FROM docker.1ms.run/506610466/cuda:12.2.2-runtime-ubuntu20.04-uv

# 1. 安装系统基础依赖
# 根据你的项目需求保留了 git, vim 等工具，移除了 nodejs (除非你的 RAG 前端在同一个容器)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 环境变量设置
# 强制 uv 使用 Python 3.12 (因为你的旧环境是 3.12)
ENV UV_PYTHON=3.12
# 确保虚拟环境在 PATH 中
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. 初始化 Python 3.12 环境
# uv 会自动下载并安装 python 3.12
RUN uv venv

# 4. 复制并安装依赖
COPY requirements.txt .
# 安装依赖
RUN uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 注意：构建命令必须在 /home/dev/huangbinghan/LinearRAG-main 目录下执行
COPY /home/dev/huangbinghan/LinearRAG-main/.venv/lib/python3.12/site-packages/zh_core_web_md /app/.venv/lib/python3.12/site-packages/zh_core_web_md
COPY /home/dev/huangbinghan/LinearRAG-main/.venv/lib/python3.12/site-packages/zh_core_web_md-3.7.0.dist-info /app/.venv/lib/python3.12/site-packages/zh_core_web_md-3.7.0.dist-info

# 6. 复制项目代码
COPY . .

# 7. 启动命令
CMD ["sw-python", "run", "python", "main.py"]