# syntax=docker/dockerfile:1.7
#
# Slim CPU image for jax-spring-sim.
#   * builder  -- resolves the locked environment (CPU jaxlib) with uv
#   * runtime  -- carries only the venv, the package, and the benchmarks
#
# Build:  docker build -t jax-spring-sim .
# Run:    docker run --rm jax-spring-sim
#         (runs the quick benchmark sweep and prints the results)

# --------------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first, so source edits do not re-download jaxlib.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# --------------------------------------------------------------------- runtime
FROM python:3.14-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home /app app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app benchmarks ./benchmarks

USER app

CMD ["python", "benchmarks/profile_engine.py", "--quick", "--out", "/tmp/BENCHMARKS.md"]
