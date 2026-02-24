# 使用你提供的带有 uv 和 cuda 的基础镜像
FROM docker.1ms.run/506610466/cuda:12.2.2-runtime-ubuntu20.04-uv

# 1. 安装系统基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 环境变量设置
ENV UV_PYTHON=3.12
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. 初始化 Python 3.12 环境
RUN uv venv

# 4. 复制并安装依赖 (从 requirements.txt)
# Jenkins 会从 Git 拿到这个文件
COPY requirements.txt .
RUN uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 复制项目代码
# 注意：这里不再 COPY 模型，模型通过 docker-compose 挂载进来
COPY . .

# 6. 启动命令
# 保持你原来的 sw-python 调用
CMD ["sw-python", "run", "python", "main.py"]