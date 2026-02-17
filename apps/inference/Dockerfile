# M1c MuQ L9-12 Inference Handler
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
# MuQ only
RUN python3 -c "\
print('Downloading MuQ-large-msd-iter...'); \
from muq import MuQ; \
MuQ.from_pretrained('OpenMuQ/MuQ-large-msd-iter'); \
print('Done!'); \
"

# Copy application code
COPY constants.py .
COPY handler.py .
COPY models/ ./models/
COPY preprocessing/ ./preprocessing/

# Create checkpoints directory structure
RUN mkdir -p /app/checkpoints/fold0 /app/checkpoints/fold1 /app/checkpoints/fold2 /app/checkpoints/fold3

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# HuggingFace Inference Endpoints expects handler.py
# The EndpointHandler class will be automatically detected
