#!/bin/bash
# Sync D9c AsymmetricGatedFusion checkpoints from Google Drive
# Run this before building the Docker image or uploading to HuggingFace

set -e

CHECKPOINT_DIR="./checkpoints"
GDRIVE_PATH="gdrive:crescendai_data/checkpoints/audio_phase2/checkpoints/D9c_asymmetric_gated_fusion"

echo "D9c AsymmetricGatedFusion Checkpoint Sync"
echo "=========================================="
echo ""

echo "Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/fold0"
mkdir -p "$CHECKPOINT_DIR/fold1"
mkdir -p "$CHECKPOINT_DIR/fold2"
mkdir -p "$CHECKPOINT_DIR/fold3"

echo ""
echo "Syncing D9c checkpoints (4-fold ensemble)..."
echo "Source: $GDRIVE_PATH"
echo ""

# Sync each fold
for fold in 0 1 2 3; do
    echo "Syncing fold$fold..."
    rclone copy "$GDRIVE_PATH/fold$fold/best.ckpt" "$CHECKPOINT_DIR/fold$fold/" --progress
done

echo ""
echo "Checkpoint sync complete!"
echo ""
echo "Directory structure:"
ls -la "$CHECKPOINT_DIR"
echo ""

for fold in 0 1 2 3; do
    echo "fold$fold:"
    ls -la "$CHECKPOINT_DIR/fold$fold"
done

echo ""
echo "Expected HuggingFace repository structure:"
echo "  checkpoints/"
echo "    fold0/best.ckpt"
echo "    fold1/best.ckpt"
echo "    fold2/best.ckpt"
echo "    fold3/best.ckpt"
echo ""
echo "Model: D9c AsymmetricGatedFusion (MERT+MuQ, R2=0.531)"
