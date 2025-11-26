#!/usr/bin/env python3
"""
Training script for piano performance evaluation model.

Supports 3 fusion modes for comparison experiments:
- crossattn: Cross-attention fusion (baseline)
- gated: Gated Multimodal Unit fusion (recommended)
- concat: Simple concatenation fusion

Usage:
    # Train with default (gated) fusion
    python train.py --config configs/experiment.yaml

    # Train specific fusion type
    python train.py --config configs/experiment.yaml --fusion_type crossattn
    python train.py --config configs/experiment.yaml --fusion_type gated
    python train.py --config configs/experiment.yaml --fusion_type concat

    # Resume from checkpoint
    python train.py --config configs/experiment.yaml --checkpoint checkpoints/gated/best.ckpt
"""

import argparse
from pathlib import Path
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.data.dataset import create_dataloaders
from src.models.lightning_module import PerformanceEvaluationModel

# Suppress known warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics')


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_fusion_type_to_paths(config: dict, fusion_type: str) -> dict:
    """Replace {fusion_type} placeholders in config paths."""
    # Checkpoint path
    if "callbacks" in config and "checkpoint" in config["callbacks"]:
        ckpt = config["callbacks"]["checkpoint"]
        if "dirpath" in ckpt and "{fusion_type}" in ckpt["dirpath"]:
            ckpt["dirpath"] = ckpt["dirpath"].replace("{fusion_type}", fusion_type)
        if "filename" in ckpt and "{fusion_type}" in ckpt["filename"]:
            ckpt["filename"] = ckpt["filename"].replace("{fusion_type}", fusion_type)

    # Logging path
    if "logging" in config and "tensorboard_logdir" in config["logging"]:
        if "{fusion_type}" in config["logging"]["tensorboard_logdir"]:
            config["logging"]["tensorboard_logdir"] = config["logging"]["tensorboard_logdir"].replace(
                "{fusion_type}", fusion_type
            )

    return config


def setup_callbacks(config: dict) -> list:
    """Setup training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_config = config["callbacks"]["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        monitor=checkpoint_config["monitor"],
        mode=checkpoint_config["mode"],
        save_top_k=checkpoint_config["save_top_k"],
        save_last=checkpoint_config["save_last"],
        dirpath=checkpoint_config["dirpath"],
        filename=checkpoint_config["filename"],
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    early_stop_config = config["callbacks"]["early_stopping"]
    early_stop_callback = EarlyStopping(
        monitor=early_stop_config["monitor"],
        mode=early_stop_config["mode"],
        patience=early_stop_config["patience"],
        min_delta=early_stop_config["min_delta"],
        verbose=True,
    )
    callbacks.append(early_stop_callback)

    # Learning rate monitor
    if "lr_monitor" in config["callbacks"]:
        lr_monitor = LearningRateMonitor(
            logging_interval=config["callbacks"]["lr_monitor"]["logging_interval"]
        )
        callbacks.append(lr_monitor)
    else:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    return callbacks


def setup_loggers(config: dict) -> list:
    """Setup training loggers."""
    loggers = []

    # TensorBoard
    if config["logging"].get("use_tensorboard", True):
        tb_logger = TensorBoardLogger(
            save_dir=config["logging"]["tensorboard_logdir"],
            name="",
            version=None,
        )
        loggers.append(tb_logger)

    # WandB (optional)
    if config["logging"].get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity", None),
            name=config["logging"].get("wandb_run_name", None),
        )
        loggers.append(wandb_logger)

    return loggers


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train piano performance evaluation model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run a single batch for debugging")
    parser.add_argument(
        "--fusion_type",
        type=str,
        choices=["crossattn", "gated", "concat"],
        default=None,
        help="Fusion type: crossattn, gated, or concat (overrides config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["audio", "midi", "fusion"],
        default=None,
        help="Legacy mode: audio-only, midi-only, or fusion (for backward compatibility)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Determine fusion type
    fusion_type = args.fusion_type or config.get("model", {}).get("fusion_type", "gated")
    print(f"Fusion type: {fusion_type}")

    # Apply fusion type presets if available
    if "fusion_presets" in config and fusion_type in config["fusion_presets"]:
        preset = config["fusion_presets"][fusion_type]
        print(f"Applying preset: {preset.get('description', fusion_type)}")
        for key, value in preset.items():
            if key != "description":
                config["model"][key] = value

    # Apply fusion type to paths
    config = apply_fusion_type_to_paths(config, fusion_type)

    # Handle legacy mode argument
    if args.mode:
        print(f"Warning: --mode is deprecated. Use --fusion_type instead.")
        if "modes" in config and args.mode in config["modes"]:
            config["model"].update(config["modes"][args.mode])

    # Set seed for reproducibility
    pl.seed_everything(config.get("seed", 42))

    # Enable Tensor Cores for better performance
    torch.set_float32_matmul_precision("high")

    # Create checkpoint directory
    checkpoint_dir = Path(config["callbacks"]["checkpoint"]["dirpath"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_annotation_path=config["data"]["train_path"],
        val_annotation_path=config["data"]["val_path"],
        test_annotation_path=config["data"].get("test_path", None),
        dimension_names=config["data"]["dimensions"],
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        augmentation_config=config["data"].get("augmentation", None),
        audio_sample_rate=config["data"]["audio_sample_rate"],
        max_audio_length=config["data"]["max_audio_length"],
        max_midi_events=config["data"]["max_midi_events"],
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"Test samples: {len(test_loader.dataset)}")

    # Get model config
    model_config = config["model"]
    training_config = config["training"]
    loss_config = config.get("loss", {})

    # Initialize model
    print("Initializing model...")
    checkpoint_path = args.checkpoint or config.get("resume_from_checkpoint", None)

    model_kwargs = {
        "audio_dim": model_config.get("audio_dim", 768),
        "midi_dim": model_config.get("midi_dim", 256),
        "shared_dim": model_config.get("shared_dim", 512),
        "aggregator_dim": model_config.get("aggregator_dim", 512),
        "num_dimensions": model_config.get("num_dimensions", 8),
        "dimension_names": config["data"]["dimensions"],
        "fusion_type": fusion_type,
        "use_projection": model_config.get("use_projection", True),
        "mert_model_name": model_config.get("mert_model_name", "m-a-p/MERT-v1-95M"),
        "freeze_audio_encoder": model_config.get("freeze_audio_encoder", False),
        "gradient_checkpointing": model_config.get("gradient_checkpointing", True),
        # Loss weights
        "mse_weight": loss_config.get("mse_weight", 1.0),
        "ranking_weight": loss_config.get("ranking_weight", 0.2),
        "contrastive_weight": loss_config.get("contrastive_weight", 0.1),
        "ranking_margin": loss_config.get("ranking_margin", 5.0),
        "contrastive_temperature": loss_config.get("contrastive_temperature", 0.07),
        # Training params
        "learning_rate": training_config.get("learning_rate", 1e-5),
        "backbone_lr": training_config.get("backbone_lr", 5e-6),
        "heads_lr": training_config.get("heads_lr", 1e-4),
        "weight_decay": training_config.get("weight_decay", 0.01),
        "warmup_steps": training_config.get("warmup_steps", 500),
        "max_epochs": training_config.get("max_epochs", 50),
    }

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PerformanceEvaluationModel.load_from_checkpoint(
            checkpoint_path,
            **model_kwargs,
        )
    else:
        print("Creating new model from scratch")
        model = PerformanceEvaluationModel(**model_kwargs)

    # Print model summary
    print("\nModel Architecture:")
    print(f"- Audio encoder: {model_config.get('mert_model_name', 'm-a-p/MERT-v1-95M')}")
    print(f"- MIDI encoder: {model_config.get('midi_dim', 256)}d")
    print(f"- Shared dimension: {model_config.get('shared_dim', 512)}")
    print(f"- Fusion type: {fusion_type}")
    print(f"- Use projection: {model_config.get('use_projection', True)}")
    print(f"- Number of dimensions: {model_config.get('num_dimensions', 8)}")
    print(f"- Dimensions: {config['data']['dimensions']}")
    print(f"\nLoss weights:")
    print(f"- MSE: {loss_config.get('mse_weight', 1.0)}")
    print(f"- Ranking: {loss_config.get('ranking_weight', 0.2)}")
    print(f"- Contrastive: {loss_config.get('contrastive_weight', 0.1)}")

    # Setup callbacks and loggers
    callbacks = setup_callbacks(config)
    loggers = setup_loggers(config)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        max_epochs=training_config["max_epochs"],
        precision=training_config.get("precision", 16),
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config["logging"].get("log_every_n_steps", 50),
        gradient_clip_val=training_config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_config.get("accumulate_grad_batches", 1),
        val_check_interval=training_config.get("val_check_interval", 1.0),
        limit_val_batches=training_config.get("limit_val_batches", 1.0),
        fast_dev_run=args.fast_dev_run,
    )

    # Print training info
    print("\nTraining Configuration:")
    print(f"- Max epochs: {training_config['max_epochs']}")
    print(f"- Batch size: {config['data']['batch_size']}")
    print(f"- Gradient accumulation: {training_config.get('accumulate_grad_batches', 1)}")
    effective_batch = config['data']['batch_size'] * training_config.get('accumulate_grad_batches', 1)
    print(f"- Effective batch size: {effective_batch}")
    print(f"- Backbone LR: {training_config.get('backbone_lr', 5e-6)}")
    print(f"- Heads LR: {training_config.get('heads_lr', 1e-4)}")
    print(f"- Warmup steps: {training_config.get('warmup_steps', 500)}")
    print(f"- Precision: {training_config.get('precision', 16)}")

    # Train
    print("\nStarting training...\n")
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test (if test set available)
    if test_loader is not None:
        print("\nRunning test evaluation...")
        trainer.test(model, dataloaders=test_loader)

    # Print best model path
    best_model_path = callbacks[0].best_model_path
    print(f"\nTraining complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"Best validation loss: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
