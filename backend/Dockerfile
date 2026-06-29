FROM python:3.12-slim

RUN pip install uv --no-cache-dir

WORKDIR /workspace

COPY backend/pyproject.toml backend/

WORKDIR /workspace/backend
RUN uv sync --no-dev

COPY backend/app /workspace/backend/app
COPY backend/scripts /workspace/backend/scripts

ENV PATH="/workspace/backend/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
