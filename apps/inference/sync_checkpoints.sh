#!/bin/bash
# Sync MERT model checkpoints from Google Drive to local directory
# Run this before building the Docker image or uploading to HuggingFace

set -e

CHECKPOINT_DIR="./checkpoints"

echo "Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/mert"

echo "Syncing MERT checkpoints (4-fold ensemble)..."
rclone sync gdrive:crescendai_data/checkpoints/audio_baseline/ "$CHECKPOINT_DIR/mert/" \
    --include "fold*_best.ckpt" \
    --progress

echo "Done! Checkpoints synced to $CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR/mert"

echo ""
echo "Expected files for HuggingFace upload:"
echo "  checkpoints/mert/fold0_best.ckpt"
echo "  checkpoints/mert/fold1_best.ckpt"
echo "  checkpoints/mert/fold2_best.ckpt"
echo "  checkpoints/mert/fold3_best.ckpt"
