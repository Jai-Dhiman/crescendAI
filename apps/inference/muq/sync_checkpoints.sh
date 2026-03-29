#!/bin/bash
# Sync A1-Max MuQ LoRA checkpoints from Google Drive
# Run this before building the Docker image or uploading to HuggingFace

set -e

CHECKPOINT_DIR="./checkpoints"
GDRIVE_PATH="gdrive:crescendai_data/checkpoints/a1_max_sweep/A1max_r32_L7-12_ls0.1"

echo "A1-Max MuQ LoRA Checkpoint Sync"
echo "================================"
echo ""

echo "Creating checkpoint directories..."
mkdir -p "$CHECKPOINT_DIR/fold_0"
mkdir -p "$CHECKPOINT_DIR/fold_1"
mkdir -p "$CHECKPOINT_DIR/fold_2"
mkdir -p "$CHECKPOINT_DIR/fold_3"

echo ""
echo "Syncing A1-Max checkpoints (4-fold ensemble, 80.8% pairwise)..."
echo "Source: $GDRIVE_PATH"
echo ""

# Sync each fold's best checkpoint
for fold in 0 1 2 3; do
    echo "Syncing fold_$fold..."
    rclone copyto "$GDRIVE_PATH/fold_${fold}/best.ckpt" "$CHECKPOINT_DIR/fold_$fold/best.ckpt" --progress
done

echo ""
echo "Checkpoint sync complete!"
echo ""
echo "Directory structure:"
ls -la "$CHECKPOINT_DIR"
echo ""

for fold in 0 1 2 3; do
    echo "fold_$fold:"
    ls -la "$CHECKPOINT_DIR/fold_$fold"
done

echo ""
echo "Expected HuggingFace repository structure:"
echo "  checkpoints/"
echo "    fold_0/best.ckpt"
echo "    fold_1/best.ckpt"
echo "    fold_2/best.ckpt"
echo "    fold_3/best.ckpt"
echo ""
echo "Model: A1-Max MuQ LoRA r32 L7-12 (6-dim, 80.8% pairwise, R2=0.50)"
