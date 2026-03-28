# syntax=docker/dockerfile:1
# Target platform: linux/amd64 (CF Containers requirement)

# Stage 1: builder -- exports ONNX models from safetensors checkpoint
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Install torch from the official CPU wheel index only (avoids dependency confusion
# from mixing --extra-index-url with PyPI for the same package name).
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

RUN python scripts/export_onnx.py \
    --checkpoint /build/checkpoint.safetensors \
    --output-dir /build/onnx_models/ \
    --config medium-double


# Stage 2: runtime -- slim image, no PyTorch
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY requirements-container.txt requirements-container.txt
RUN uv pip install --system --no-cache -r requirements-container.txt

COPY --from=builder /build/onnx_models/ /app/models/

COPY server.py server.py

RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV MODEL_DIR=/app/models
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
