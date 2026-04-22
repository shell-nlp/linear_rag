
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
SHELL ["/bin/bash", "-c"]

# 设置 UV 镜像源为清华大学源以加速依赖安装
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
# 1. 安装系统基础依赖
# RUN apt update && apt install -y \
#     vim \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./ /app

RUN uv sync -v && source .venv/bin/activate && \
    uv cache clean && \
    echo '[[ -f .venv/bin/activate ]] && source .venv/bin/activate' >> ~/.bashrc

ENV PATH="/app/.venv/bin:$PATH"
CMD ["/bin/bash"]