#!/usr/bin/env python3
"""
Training script for piano performance evaluation model.

Usage:
    # Pseudo-label pre-training
    python train.py --config configs/pseudo_pretrain.yaml

    # Expert label fine-tuning
    python train.py --config configs/expert_finetune.yaml

    # Resume from checkpoint
    python train.py --config configs/expert_finetune.yaml --checkpoint checkpoints/pseudo_pretrain/best.ckpt
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
    lr_monitor = LearningRateMonitor(
        logging_interval=config["callbacks"]["lr_monitor"]["logging_interval"]
    )
    callbacks.append(lr_monitor)

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
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--fast-dev-run", action="store_true", help="Run a single batch for debugging"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["audio", "midi", "fusion"],
        default=None,
        help="Model mode: audio-only, midi-only, or fusion (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Apply mode overrides if specified
    if args.mode:
        print(f"Applying mode: {args.mode}")
        mode_overrides = config.get("modes", {}).get(args.mode, {})
        config["model"].update(mode_overrides)
        # Update checkpoint path to include mode
        if "{mode}" in config["callbacks"]["checkpoint"]["dirpath"]:
            config["callbacks"]["checkpoint"]["dirpath"] = config["callbacks"]["checkpoint"][
                "dirpath"
            ].replace("{mode}", args.mode)
        if "{mode}" in config["callbacks"]["checkpoint"]["filename"]:
            config["callbacks"]["checkpoint"]["filename"] = config["callbacks"]["checkpoint"][
                "filename"
            ].replace("{mode}", args.mode)
        if "{mode}" in config["logging"]["tensorboard_logdir"]:
            config["logging"]["tensorboard_logdir"] = config["logging"][
                "tensorboard_logdir"
            ].replace("{mode}", args.mode)

    # Set seed for reproducibility
    pl.seed_everything(config.get("seed", 42))

    # Enable Tensor Cores for better performance on modern GPUs
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

    # Initialize model
    print("Initializing model...")

    # Check for checkpoint to resume from
    checkpoint_path = args.checkpoint or config.get("resume_from_checkpoint", None)

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model from checkpoint: {checkpoint_path}")
        model = PerformanceEvaluationModel.load_from_checkpoint(
            checkpoint_path,
            # Override config params
            audio_dim=config["model"]["audio_dim"],
            midi_dim=config["model"]["midi_dim"],
            fusion_dim=config["model"]["fusion_dim"],
            aggregator_dim=config["model"]["aggregator_dim"],
            num_dimensions=config["model"]["num_dimensions"],
            dimension_names=config["data"]["dimensions"],
            mert_model_name=config["model"]["mert_model_name"],
            freeze_audio_encoder=config["model"]["freeze_audio_encoder"],
            gradient_checkpointing=config["model"]["gradient_checkpointing"],
            learning_rate=config["training"]["learning_rate"],
            backbone_lr=config["training"]["backbone_lr"],
            heads_lr=config["training"]["heads_lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_epochs=config["training"]["max_epochs"],
        )
    else:
        print("Creating new model from scratch")
        model = PerformanceEvaluationModel(
            audio_dim=config["model"]["audio_dim"],
            midi_dim=config["model"]["midi_dim"],
            fusion_dim=config["model"]["fusion_dim"],
            aggregator_dim=config["model"]["aggregator_dim"],
            num_dimensions=config["model"]["num_dimensions"],
            dimension_names=config["data"]["dimensions"],
            mert_model_name=config["model"]["mert_model_name"],
            freeze_audio_encoder=config["model"]["freeze_audio_encoder"],
            gradient_checkpointing=config["model"]["gradient_checkpointing"],
            learning_rate=config["training"]["learning_rate"],
            backbone_lr=config["training"]["backbone_lr"],
            heads_lr=config["training"]["heads_lr"],
            weight_decay=config["training"]["weight_decay"],
            warmup_steps=config["training"]["warmup_steps"],
            max_epochs=config["training"]["max_epochs"],
        )

    # Print model summary
    print("\nModel Architecture:")
    print(f"- Audio encoder: {config['model']['mert_model_name']}")
    print(f"- MIDI encoder: {config['model']['midi_dim']}d")
    print(f"- Fusion dimension: {config['model']['fusion_dim']}")
    print(f"- Aggregator dimension: {config['model']['aggregator_dim']}")
    print(f"- Number of dimensions: {config['model']['num_dimensions']}")
    print(f"- Dimensions: {config['data']['dimensions']}")

    # Setup callbacks
    callbacks = setup_callbacks(config)

    # Setup loggers
    loggers = setup_loggers(config)

    # Create trainer
    print("\nInitializing trainer...")
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"]["precision"],
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        val_check_interval=config["training"]["val_check_interval"],
        limit_val_batches=config["training"]["limit_val_batches"],
        fast_dev_run=args.fast_dev_run,
    )

    # Print training info
    print("\nTraining Configuration:")
    print(f"- Max epochs: {config['training']['max_epochs']}")
    print(f"- Batch size: {config['data']['batch_size']}")
    print(f"- Gradient accumulation: {config['training']['accumulate_grad_batches']}")
    print(
        f"- Effective batch size: {config['data']['batch_size'] * config['training']['accumulate_grad_batches']}"
    )
    print(f"- Backbone LR: {config['training']['backbone_lr']}")
    print(f"- Heads LR: {config['training']['heads_lr']}")
    print(f"- Warmup steps: {config['training']['warmup_steps']}")
    print(f"- Precision: {config['training']['precision']}")

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
