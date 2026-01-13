# D9c AsymmetricGatedFusion Inference Handler
# HuggingFace Inference Endpoints container for piano performance analysis

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

# Install Python dependencies with uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Pre-download HuggingFace models (cached in image)
# This prevents cold start from downloading ~2.5GB each time (MERT + MuQ)
RUN python3 -c "\
from transformers import AutoModel, AutoProcessor; \
print('Downloading MERT-v1-330M...'); \
AutoProcessor.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); \
AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True); \
print('Downloading MuQ-large-msd-iter...'); \
from muq import MuQ; \
MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter'); \
print('Done!'); \
"

# Copy application code
COPY constants.py .
COPY handler_hf.py .
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/

# Create checkpoints directory structure for D9c
RUN mkdir -p /app/checkpoints/fold0 /app/checkpoints/fold1 /app/checkpoints/fold2 /app/checkpoints/fold3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# HuggingFace Inference Endpoints expects handler.py or handler_hf.py
# The EndpointHandler class will be automatically detected
