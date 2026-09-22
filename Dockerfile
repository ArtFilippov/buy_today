# syntax=docker/dockerfile:1

FROM ghcr.io/astral-sh/uv:0.12.6 AS uv

FROM python:3.13-slim AS base
ENV VIRTUAL_ENV=/opt/prak/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/cache \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    IPYTHONDIR=/tmp/ipython \
    JUPYTER_CONFIG_DIR=/tmp/jupyter/config \
    JUPYTER_DATA_DIR=/tmp/jupyter/data \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter/runtime
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

FROM base AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/prak/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/tmp/uv-cache
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev --no-install-project --no-editable

FROM dependencies AS package
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev --no-editable

# This target has no dependency on the Olist download stage.
FROM package AS tests
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --group dev --no-editable
COPY tests ./tests
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]

# Only the stdlib downloader affects this layer's cache, not the application.
FROM base AS olist-data
COPY src/prak/bundled_data.py /tmp/bundled_data.py
RUN python /tmp/bundled_data.py /opt/prak/olist

# Keep runtime last so an ordinary build selects the application image.
FROM base AS runtime
COPY --from=package /opt/prak/venv /opt/prak/venv
COPY --from=olist-data /opt/prak/olist /opt/prak/olist
WORKDIR /workspace
ENTRYPOINT ["python", "-m", "prak.container_cli"]
CMD ["--help"]
