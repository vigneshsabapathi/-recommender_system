FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps (install before copying code for better layer caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy project source
COPY src/ /app/src/
COPY backend/ /app/backend/
COPY data/processed/ /app/data/processed/
COPY models/ /app/models/

# Set env defaults (can be overridden at runtime)
ENV MODEL_DIR=/app/models
ENV DATA_DIR=/app/data
ENV DB_PATH=/app/backend/movies.db
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
