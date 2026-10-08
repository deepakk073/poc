# Production Dockerfile for Vertex AI Gemini PDF Classification & Extraction Pipeline
# Base image: Official lightweight Python 3.12 slim
FROM python:3.12-slim AS builder

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build essentials if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies into temporary wheel directory
COPY requirements.txt .
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


# Final runtime stage
FROM python:3.12-slim AS runner

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Create non-root system user for security compliance
RUN groupadd -r pipeline && useradd -r -g pipeline -u 1001 pipeline

# Install runtime dependencies from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy application source code and SQL definitions
COPY pdf_pipeline /app/pdf_pipeline
COPY sql /app/sql
COPY README.md /app/

# Ensure non-root permissions
RUN chown -R pipeline:pipeline /app
USER pipeline

# Default execution entrypoint
ENTRYPOINT ["python3", "-m", "pdf_pipeline.services.orchestrator"]
CMD ["--help"]
