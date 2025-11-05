# NVIDIA CUDA 12.2 with Python 3.11 for GPU acceleration
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install Python 3.11 and system dependencies in single RUN to reduce layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ca-certificates \
    git \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python -m pip install --upgrade pip setuptools wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies BEFORE copying code (better caching)
COPY requirements.txt .

# Install all Python dependencies in one go
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    # Verify installation
    python -c "import torch; import fastapi; import uvicorn; print('✓ All dependencies installed')"

# Copy application files
COPY app.py .
COPY image_resizer.py .
COPY transformer_flux.py .
COPY controlnet_flux.py .
COPY pipeline_flux_controlnet_inpaint.py .
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
