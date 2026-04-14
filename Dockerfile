FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY auto_insurance ./auto_insurance

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/uvicorn auto_insurance.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
