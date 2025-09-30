"""Training utilities and Lightning trainer setup."""

import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Optional, Dict, Any
import os


def create_trainer(
    max_epochs: int = 100,
    accelerator: str = "auto",
    devices: str = "auto",
    precision: str = "16-mixed",
    log_every_n_steps: int = 10,
    val_check_interval: float = 1.0,
    experiment_name: str = "crescendai",
    project_name: str = "crescendai-model",
    use_wandb: bool = False,
    checkpoint_dir: str = "./checkpoints",
    **kwargs
) -> L.Trainer:
    """Create a configured PyTorch Lightning trainer.
    
    Args:
        max_epochs: Maximum number of training epochs
        accelerator: Hardware accelerator ("auto", "gpu", "cpu")
        devices: Number/list of devices to use
        precision: Training precision ("32", "16-mixed", "bf16-mixed")
        log_every_n_steps: How often to log metrics
        val_check_interval: How often to run validation
        experiment_name: Name for this experiment
        project_name: Project name for logging
        use_wandb: Whether to use Weights & Biases logging
        checkpoint_dir: Directory to save checkpoints
        **kwargs: Additional trainer arguments
    
    Returns:
        Configured Lightning trainer
    """
    # Callbacks
    callbacks = []
    
    # Model checkpointing
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{experiment_name}-{{epoch:02d}}-{{val_loss:.3f}}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=10,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Logger setup
    loggers = []
    
    if use_wandb:
        wandb_logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            save_dir="./logs"
        )
        loggers.append(wandb_logger)
    
    # Always include TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir="./logs",
        name=experiment_name
    )
    loggers.append(tb_logger)
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        callbacks=callbacks,
        logger=loggers,
        enable_progress_bar=True,
        enable_model_summary=True,
        **kwargs
    )
    
    return trainer


def train_model(
    model: L.LightningModule,
    train_dataloader,
    val_dataloader,
    trainer_config: Optional[Dict[str, Any]] = None,
    resume_from_checkpoint: Optional[str] = None
) -> L.Trainer:
    """Train a model with the given data loaders.
    
    Args:
        model: Lightning module to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        trainer_config: Configuration for trainer
        resume_from_checkpoint: Path to checkpoint to resume from
    
    Returns:
        Trained Lightning trainer
    """
    if trainer_config is None:
        trainer_config = {}
    
    trainer = create_trainer(**trainer_config)
    
    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=resume_from_checkpoint
    )
    
    return trainer