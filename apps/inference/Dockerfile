# CrescendAI Inference Handler
# RunPod serverless container for piano performance analysis

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies with uv (use requirements.txt for Docker layer caching)
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download HuggingFace models (cached in image)
# This prevents cold start from downloading ~1.3GB each time
RUN python3 -c "\
from transformers import AutoModel, AutoProcessor; \
print('Downloading MERT-v1-330M...'); \
AutoProcessor.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); \
AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); \
print('Done!'); \
"

# Copy application code
COPY constants.py .
COPY handler.py .
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/

# Create checkpoints directory (will be mounted or copied)
RUN mkdir -p /app/checkpoints/mert /app/checkpoints/percepiano /app/checkpoints/fusion

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# Entry point
CMD ["python3", "handler.py"]
