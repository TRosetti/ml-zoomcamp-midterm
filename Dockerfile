# Base image: small Linux with Python 3.12
FROM python:3.12-slim

# Copy the uv binary from the official image (faster than installing it with pip)
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/

WORKDIR /app

# UV_PYTHON_DOWNLOADS=never: use the image's Python instead of downloading another one
# PATH: makes "uvicorn" point to the one inside .venv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Dependency files first, so Docker caches this (slow) layer when only the code changes
COPY pyproject.toml uv.lock .python-version ./

# --locked: use exactly the versions in uv.lock
# --no-dev: skip jupyter and requests, which the service does not need
RUN uv sync --locked --no-dev

# Then the service code and the trained model
COPY predict.py model.bin ./

EXPOSE 9696

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
