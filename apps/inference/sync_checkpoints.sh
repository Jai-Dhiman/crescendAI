#!/bin/bash
# Sync M1c MuQ L9-12 checkpoints from Google Drive
# Run this before building the Docker image or uploading to HuggingFace

set -e

CHECKPOINT_DIR="./checkpoints"
GDRIVE_PATH="gdrive:crescendai_data/checkpoints/definitive_experiments/checkpoints/M1c_muq_L9-12"

echo "M1c MuQ L9-12 Checkpoint Sync"
echo "=============================="
echo ""

echo "Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/fold0"
mkdir -p "$CHECKPOINT_DIR/fold1"
mkdir -p "$CHECKPOINT_DIR/fold2"
mkdir -p "$CHECKPOINT_DIR/fold3"

echo ""
echo "Syncing M1c checkpoints (4-fold ensemble)..."
echo "Source: $GDRIVE_PATH"
echo ""

# Sync each fold (GDrive has foldX_best.ckpt, we need foldX/best.ckpt)
for fold in 0 1 2 3; do
    echo "Syncing fold$fold..."
    rclone copyto "$GDRIVE_PATH/fold${fold}_best.ckpt" "$CHECKPOINT_DIR/fold$fold/best.ckpt" --progress
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
echo "Model: M1c MuQ L9-12 (MuQ-only, R2=0.539)"
