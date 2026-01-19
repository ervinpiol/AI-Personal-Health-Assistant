# stage 1: builder
FROM python:3.13.11-slim-bookworm as builder


COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# stage 2: runner
FROM python:3.13.11-slim-bookworm AS runner

COPY --from=builder /app/.venv /app/.venv

# Set the virtual environment path explicitly
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY app ./app

# create a non-root user (named "appuser") and set access to the /app directory
RUN useradd -m appuser && chown -R appuser /app

# set created user as the current user
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]