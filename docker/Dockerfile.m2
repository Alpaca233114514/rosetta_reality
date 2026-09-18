FROM python:3.13.5-slim-bookworm

ARG PYTORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/rosetta
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install \
        "torch==${PYTORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install ".[dev,data,qwen]" \
    && python -m pip check

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

CMD ["bash"]
