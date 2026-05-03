# layer-rag-query: FastMCP HTTP (default port 8000). In k3s, map Service NodePort (e.g. 30183) → targetPort 8000.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HTTP_PORT=8000

WORKDIR /app

# Install deps before copying main.py so image layers cache when only main.py changes.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install --no-cache-dir ".[mcp]" \
    && useradd --create-home --shell /usr/sbin/nologin --uid 1000 appuser \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser main.py ./main.py

USER appuser

EXPOSE 8000

# Override listen port without rebuilding (e.g. rare cases where containerPort must match process port).
CMD ["/bin/sh", "-c", "exec fastmcp run main.py:mcp --transport http --host 0.0.0.0 --port ${HTTP_PORT}"]
