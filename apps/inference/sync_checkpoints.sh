#!/bin/bash
# Sync model checkpoints from Google Drive to local directory
# Run this before building the Docker image

set -e

CHECKPOINT_DIR="./checkpoints"

echo "Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/mert"
mkdir -p "$CHECKPOINT_DIR/percepiano"
mkdir -p "$CHECKPOINT_DIR/fusion"

echo "Syncing MERT checkpoints..."
rclone sync gdrive:crescendai_data/checkpoints/audio_baseline/ "$CHECKPOINT_DIR/mert/" \
    --include "fold*_best.ckpt" \
    --progress

echo "Syncing PercePiano checkpoints..."
rclone sync gdrive:crescendai_data/checkpoints/percepiano_original/ "$CHECKPOINT_DIR/percepiano/" \
    --include "fold*_best.pt" \
    --progress

echo "Syncing symbolic predictions..."
rclone copy gdrive:crescendai_data/predictions/symbolic_predictions.json "$CHECKPOINT_DIR/" \
    --progress

echo "Done! Checkpoints synced to $CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR"
ls -la "$CHECKPOINT_DIR/mert"
ls -la "$CHECKPOINT_DIR/percepiano"
