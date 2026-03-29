# syntax=docker/dockerfile:1
# Target platform: linux/amd64 (CF Containers requirement)
#
# Hybrid approach: ONNX encoder (graph-optimized) + PyTorch decoder (KV cache).
# The ONNX export runs at build time; the runtime image includes both
# ONNX Runtime and PyTorch CPU for the decoder.

# Stage 1: builder -- exports ONNX encoder from safetensors checkpoint
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Install torch CPU from the official wheel index (avoids dependency confusion)
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN uv pip install --system --no-cache \
    aria-amt \
    safetensors \
    onnx \
    onnxruntime \
    numpy

WORKDIR /build

ARG CHECKPOINT_PATH
COPY ${CHECKPOINT_PATH} /build/checkpoint.safetensors

COPY scripts/export_onnx.py scripts/export_onnx.py
COPY transcription.py transcription.py

# Export ONNX encoder (decoder stays in PyTorch)
RUN python scripts/export_onnx.py \
    --checkpoint /build/checkpoint.safetensors \
    --output-dir /build/onnx_models/ \
    --config medium-double


# Stage 2: runtime -- PyTorch CPU (decoder) + ONNX RT (encoder) + ffmpeg
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install PyTorch CPU (for decoder) from official wheel index
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install remaining runtime dependencies from PyPI
COPY requirements-container.txt requirements-container.txt
RUN uv pip install --system --no-cache -r requirements-container.txt

# Copy ONNX encoder from builder
COPY --from=builder /build/onnx_models/ /app/models/

# Copy checkpoint for PyTorch decoder
COPY --from=builder /build/checkpoint.safetensors /app/checkpoint.safetensors

# Copy server code + transcription module (needed for _load_weight)
COPY server.py server.py
COPY transcription.py transcription.py

RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV CHECKPOINT_PATH=/app/checkpoint.safetensors
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
