# syntax=docker/dockerfile:1

# Base image pinned by digest for reproducible builds. The digest is a multi-arch
# index, so the same line works for linux/amd64 and linux/arm64. To update it:
#   docker buildx imagetools inspect python:3.11-slim
ARG PYTHON_IMAGE=python:3.11-slim@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534

# --- Build stage: install the locked dependencies into /app/.venv ----------------
FROM ${PYTHON_IMAGE} AS builder

# Same uv version as the CI workflow.
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # pyproject.toml prefers uv-managed Pythons for local development; in the image,
    # use the base image's Python and never download another one.
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1. Dependencies only: this layer is rebuilt only when the lockfile changes.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2. The project itself (README.md is declared in pyproject.toml, so it is required).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- Runtime stage: only what the app needs, no uv, non-root user -------------------
FROM ${PYTHON_IMAGE} AS runtime

LABEL org.opencontainers.image.title="yield-curve-recession" \
      org.opencontainers.image.description="Can the bond market predict recessions? A Streamlit app exploring the US yield curve." \
      org.opencontainers.image.source="https://github.com/Alexandre-Le-Bacon/yield-curve-recession" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --create-home app

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY src ./src
COPY app ./app
COPY data/raw ./data/raw

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    YIELD_CURVE_DATA_DIR=/app/data/raw \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

USER app

EXPOSE 8501

# python:3.11-slim has no curl: use Python's standard library for the check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --start-interval=2s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "app/Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
