FROM python:3.11-slim

WORKDIR /app

# System deps for faiss-cpu + pdfplumber + uvicorn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev,serve]"

# Source
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]