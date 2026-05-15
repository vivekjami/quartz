# docker/Dockerfile
FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

RUN pip install --no-cache-dir maturin uv

WORKDIR /app
COPY Cargo.toml ./
COPY quartz-core/ ./quartz-core/
COPY quartz/ ./quartz/
COPY pyproject.toml ./

RUN uv venv && . .venv/bin/activate && \
    uv pip install -e "." && \
    maturin develop --release

EXPOSE 8000
CMD [".venv/bin/uvicorn", "quartz.serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
