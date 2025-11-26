# Python 3.11 slim image (no GPU required)
FROM python:3.11-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies BEFORE copying code (better caching)
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    # Verify installation
    python -c "import fastapi; import uvicorn; print('✓ All dependencies installed')"

# Copy application files
COPY app.py .
COPY templates/ ./templates/
COPY services/ ./services/

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application with extended timeout for AI processing
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "3600"]
