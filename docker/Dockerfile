# syntax=docker/dockerfile:1.7
# Imagen productiva para la app Streamlit Datathon SUNASS 2026.
# Base slim + uv para que el build sea reproducible contra pyproject.toml + uv.lock.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencias de sistema: libgomp para xgboost/lightgbm, build-essential para wheels fallback.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv oficial (binario estatico).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Capa 1: deps (cacheable). Solo copiamos lo que define el resolver.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Capa 2: codigo del proyecto.
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit

# Reinstala el paquete con el codigo copiado.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8501

# Healthcheck pegado al endpoint interno de Streamlit.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "app/Home.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
