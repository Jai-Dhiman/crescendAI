#!/bin/bash
# Run MIDI alignment using Docker to execute Linux alignment tool on macOS
#
# Usage:
#   cd model
#   ./scripts/data_prep/run_docker_alignment.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PERCEPIANO_ROOT="$MODEL_ROOT/data/raw/PercePiano"
DOCKER_DIR="$SCRIPT_DIR/docker_alignment"

echo "============================================================"
echo "PercePiano MIDI Alignment via Docker"
echo "============================================================"
echo "Model root: $MODEL_ROOT"
echo "PercePiano root: $PERCEPIANO_ROOT"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if data exists
if [ ! -d "$PERCEPIANO_ROOT/virtuoso/data" ]; then
    echo "Error: PercePiano data not found at $PERCEPIANO_ROOT/virtuoso/data"
    exit 1
fi

# Create a temporary build context with AlignmentTool
BUILD_CONTEXT=$(mktemp -d)
trap "rm -rf $BUILD_CONTEXT" EXIT

echo "Copying alignment tool to build context..."
cp "$DOCKER_DIR/Dockerfile" "$BUILD_CONTEXT/"
cp -r "$PERCEPIANO_ROOT/virtuoso/data/AlignmentTool" "$BUILD_CONTEXT/"

# Build Docker image
echo ""
echo "Building Docker image..."
docker build --platform linux/amd64 -t percepiano-alignment "$BUILD_CONTEXT"

# Run alignment
echo ""
echo "Running alignment in Docker container..."
docker run --rm --platform linux/amd64 \
    -v "$PERCEPIANO_ROOT/virtuoso/data:/data:rw" \
    -v "$PERCEPIANO_ROOT:/labels:ro" \
    -v "$DOCKER_DIR/align_all.py:/alignment/align_all.py:ro" \
    percepiano-alignment \
    python3 /alignment/align_all.py

echo ""
echo "Alignment complete!"
echo "You can now run the preprocessing script:"
echo "  uv run python scripts/data_prep/run_percepiano_preprocessing.py"
