# syntax=docker/dockerfile:1.7

# CurBench GPU-ready Docker image using uv_build + the official uv image.
#
# Build:
#   docker build -t curbench:latest .
#
# Run:
#   docker run --gpus all -it --rm \
#     -v /path/to/data:/workspace/data \
#     -v /path/to/runs:/workspace/runs \
#     curbench:latest

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_FIND_LINKS=https://data.pyg.org/whl/torch-2.4.0+cu121.html \
    HF_HOME=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/.cache

WORKDIR /workspace/CurBench

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY curbench ./curbench
COPY examples ./examples
COPY run.py ./
COPY docs ./docs

# Install the core Python stack into /opt/venv.
# torch/torchvision/torchaudio are resolved from the CUDA-specific PyTorch index.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-editable --index https://download.pytorch.org/whl/cu121

# PyG extension wheels are installed separately from the wheel page to avoid build isolation issues.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python --no-deps \
      --find-links https://data.pyg.org/whl/torch-2.4.0+cu121.html \
      torch-scatter torch-sparse torch-cluster torch-spline-conv

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers \
    XDG_CACHE_HOME=/workspace/.cache \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/CurBench

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-distutils \
    python3-pip \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY pyproject.toml README.md ./
COPY curbench ./curbench
COPY examples ./examples
COPY run.py ./
COPY docs ./docs

RUN mkdir -p /workspace/data /workspace/runs /workspace/.cache/huggingface

CMD ["bash"]
