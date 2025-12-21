#!/usr/bin/env python3
"""
Train MIDI-only model on PercePiano dataset.

Usage:
    python scripts/train_midi_only.py --config configs/midi_only_percepiano.yaml

For Thunder Compute:
    python scripts/train_midi_only.py --config configs/midi_only_percepiano.yaml --gpus 1
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import OmegaConf

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.crescendai.models.midi_only_module import MIDIOnlyModule
from src.percepiano.data.percepiano_dataset import create_dataloaders


def train(config_path: str, gpus: int = 1, wandb: bool = False):
    """Main training function."""
    # Load config
    config = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    print(OmegaConf.to_yaml(config))

    # Set seed for reproducibility
    pl.seed_everything(42, workers=True)

    # Create dataloaders
    data_dir = Path(config.data.data_dir)
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config.training.batch_size,
        max_seq_length=config.data.max_seq_length,
        num_workers=config.data.num_workers,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = MIDIOnlyModule(
        # Encoder params
        midi_hidden_dim=config.model.midi_encoder.hidden_dim,
        midi_num_layers=config.model.midi_encoder.num_layers,
        midi_num_heads=config.model.midi_encoder.num_heads,
        max_seq_length=config.model.midi_encoder.max_seq_length,
        # Aggregation params
        lstm_hidden=config.model.aggregation.lstm_hidden,
        lstm_layers=config.model.aggregation.lstm_layers,
        attention_heads=config.model.aggregation.attention_heads,
        # MTL head params
        shared_hidden=config.model.mtl_head.shared_hidden,
        task_hidden=config.model.mtl_head.task_hidden,
        # Training params
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        dropout=config.model.midi_encoder.dropout,
        # Task params
        dimensions=list(config.model.dimensions),
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.checkpoints.dirpath,
            filename=config.checkpoints.filename,
            monitor=config.logging.monitor,
            mode=config.logging.mode,
            save_top_k=config.logging.save_top_k,
            save_last=config.checkpoints.save_last,
        ),
        EarlyStopping(
            monitor=config.training.early_stopping_metric,
            patience=config.training.early_stopping_patience,
            mode=config.training.early_stopping_mode,
        ),
        LearningRateMonitor(logging_interval="step"),
        RichProgressBar(),
    ]

    # Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if wandb:
        logger = WandbLogger(
            project=config.logging.project,
            name=f"{config.logging.name}_{timestamp}",
            save_dir="logs",
        )
    else:
        logger = TensorBoardLogger(
            save_dir="logs",
            name=config.logging.name,
            version=timestamp,
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.hardware.accelerator,
        devices=gpus if gpus > 0 else config.hardware.devices,
        strategy=config.hardware.strategy,
        precision=config.training.precision,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.gradient_accumulation_steps,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps,
        deterministic=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test with best checkpoint
    print("\nRunning test with best checkpoint...")
    best_path = callbacks[0].best_model_path
    print(f"Best checkpoint: {best_path}")

    if best_path:
        trainer.test(model, test_loader, ckpt_path=best_path)

    # Save final model for inference
    final_path = Path(config.checkpoints.dirpath) / "midi_scorer_final.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "hparams": dict(model.hparams),
        "dimensions": model.dimensions,
    }, final_path)
    print(f"Saved final model to {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MIDI-only model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/midi_only_percepiano.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (0 for CPU)",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    train(args.config, args.gpus, args.wandb)


if __name__ == "__main__":
    main()
