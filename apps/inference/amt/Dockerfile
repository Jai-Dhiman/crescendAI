# syntax=docker/dockerfile:1
# Target platform: linux/amd64 (CF Containers requirement)
#
# Transkun transcription server. Single runtime stage: Transkun (which bundles its
# own pretrained weights) + torch CPU + ffmpeg. The former ONNX-export builder stage
# is gone -- its only rationale was aria-amt CPU throughput, which Transkun replaces.

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 && rm -rf /var/lib/apt/lists/*

# uv is used to install deps AND stays on PATH so transcription.resolve_transcriber()'s
# CLI fallback (`uv run --with transkun ...`) is available if the warm in-process import
# ever fails. The primary path is the warm in-process transkun import (installed below).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Install PyTorch CPU from the official wheel index first so transkun's torch dep
# resolves to the CPU build (not the large CUDA one) and to avoid dependency confusion.
RUN uv pip install --system --no-cache \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install remaining runtime dependencies (transkun + fastapi/uvicorn/audio libs).
COPY requirements-container.txt requirements-container.txt
RUN uv pip install --system --no-cache -r requirements-container.txt

# Copy server code + shared transcription helpers.
COPY server.py server.py
COPY transcription.py transcription.py
COPY transkun_cli.py transkun_cli.py

RUN useradd --system --no-create-home --shell /usr/sbin/nologin appuser && \
    chown -R appuser /app

USER appuser

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

EXPOSE 8080

CMD ["python", "server.py"]
