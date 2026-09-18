FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv

RUN uv sync --frozen --no-install-project

COPY backend ./backend
COPY data ./data

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]