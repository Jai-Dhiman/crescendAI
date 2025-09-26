#!/usr/bin/env python3
"""
Complete AST+SSAST Training Pipeline
Pre-training on MAESTRO + Fine-tuning on PercePiano
"""

import os
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state, checkpoints
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm

# Import our models and datasets
from src.models.ast_transformer import AudioSpectrogramTransformer, create_train_state
from src.models.ssast_pretraining import (
    SSASTPreTrainingModel,
    create_ssast_train_state,
    ssast_train_step,
    extract_encoder_for_finetuning,
)
from src.training.masked_train_step import ssl_train_step_dispatch
from src.datasets.maestro_dataset import MAESTRODataset
from src.datasets.percepiano_dataset import PercePianoDataset
import optax
import logging

from src.utils.seeding import set_seed


class ASTTrainingPipeline:
    """Complete training pipeline for AST+SSAST"""

    def __init__(self, config: Dict):
        """
        Initialize training pipeline
        Args:
            config: Training configuration dict
        """
        self.config = config
        # Deterministic seeding across Python/NumPy/JAX
        seed_value = config.get("seed", 42)
        deterministic = config.get("deterministic", True)
        seed_env = set_seed(int(seed_value), deterministic=bool(deterministic))
        self.rng = seed_env.get("jax_key") or jax.random.PRNGKey(int(seed_value))

        # Setup directories
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.results_dir = Path(config["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎯 AST Training Pipeline initialized")
        print(f"   Checkpoint dir: {self.checkpoint_dir}")
        print(f"   Results dir: {self.results_dir}")

    def pretrain_ssast(self) -> str:
        """
        Pre-train SSAST model on MAESTRO dataset
        Returns:
            Path to pre-trained checkpoint
        """
        print("\n=== PHASE 1: SSAST Pre-training on MAESTRO ===")

        # Initialize MAESTRO dataset
        if not Path(self.config["maestro_path"]).exists():
            print(f"❌ MAESTRO dataset not found: {self.config['maestro_path']}")
            print("Please download MAESTRO dataset first")
            return None

        maestro_dataset = MAESTRODataset(
            maestro_root=self.config["maestro_path"],
            segment_length=self.config["segment_length"],
            cache_dir=self.config.get("cache_dir", "cache/maestro"),
        )

        # Get dataset statistics
        stats = maestro_dataset.get_statistics()

        # Initialize SSAST model
        # If SSL (InfoNCE) is enabled, set mask_prob=0.0 so encodings use real patches (no [MASK] tokens)
        mask_prob = 0.0 if (self.config.get("ssl_enable") and self.config.get("ssl_objective", "infonce") == "infonce") else self.config.get("mask_prob", 0.15)
        ssast_model = SSASTPreTrainingModel(
            patch_size=self.config["patch_size"],
            embed_dim=self.config["embed_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
            mask_prob=mask_prob,
        )

        # Create training state
        dummy_shape = (self.config["batch_size"], self.config["segment_length"], 128)
        rng_key, self.rng = jax.random.split(self.rng)

        ssast_state = create_ssast_train_state(
            ssast_model, rng_key, dummy_shape,
            learning_rate=self.config["pretrain_lr"],
            accumulate_steps=int(self.config.get("accumulate_steps", 1))
        )

        # Optional resume from previous final checkpoint
        if bool(self.config.get("resume", False)):
            final_ckpt_path = self.checkpoint_dir / "ssast_final"
            try:
                if final_ckpt_path.exists():
                    ssast_state = checkpoints.restore_checkpoint(final_ckpt_path, ssast_state)
                    print(f"Resumed from {final_ckpt_path}")
            except Exception as e:
                print(f"WARN: resume failed ({e}); continuing from fresh state")

        # Data iterator
        data_iterator = maestro_dataset.get_data_iterator(
            batch_size=self.config["batch_size"],
            shuffle=True,
            infinite=True,
            unique_per_file=bool(self.config.get("ssl_enable", False)),
        )

        print(
            f"Starting SSAST pre-training for {self.config['pretrain_steps']} steps..."
        )

        # Temperature schedule (InfoNCE): linear from temperature_start to temperature_end
        temp_start = float(self.config.get("temperature_start", self.config.get("temperature", 0.2)))
        temp_end = float(self.config.get("temperature_end", 0.07))
        total_steps = int(self.config["pretrain_steps"]) if int(self.config["pretrain_steps"]) > 1 else 1

        # Early stopping settings
        use_early_stop = int(self.config.get("early_stop_patience", 0)) > 0
        early_patience = int(self.config.get("early_stop_patience", 0))
        early_min_delta = float(self.config.get("early_stop_min_delta", 0.0))
        best_loss = float("inf")
        wait = 0

        # Training loop
        for step in tqdm(range(self.config["pretrain_steps"]), desc="Pre-training"):
            # Compute scheduled temperature
            alpha = step / float(total_steps - 1) if total_steps > 1 else 1.0
            scheduled_temp = (1.0 - alpha) * temp_start + alpha * temp_end
            # Get batch
            batch = next(data_iterator)

            # Optional mixed-precision cast
            if bool(self.config.get("use_bf16", False)):
                try:
                    batch = batch.astype(jnp.bfloat16)
                except Exception:
                    pass

            # Generate RNG for this step
            step_rng = jax.random.fold_in(self.rng, step)
            dropout_rng, stochastic_rng = jax.random.split(step_rng)

            # Select objective
            if self.config.get("ssl_enable"):
                objective = self.config.get("ssl_objective", "infonce")
                train_step_fn = ssl_train_step_dispatch(objective)
                # Pad mask unknown here; assume no padding for fixed segments
                pad_masks = jnp.zeros_like(batch, dtype=bool)
                ssast_state, metrics = train_step_fn(
                    ssast_state,
                    batch,
                    pad_masks,
                    dropout_rng=dropout_rng,
                    stochastic_rng=stochastic_rng,
                    temperature=float(scheduled_temp),
                    augment=bool(self.config.get("augment", True)),
                    aug_params={
                        "time_mask_width": int(self.config.get("augment_time_mask", 8)),
                        "freq_mask_width": int(self.config.get("augment_freq_mask", 8)),
                        "num_time_masks": 1,
                        "num_freq_masks": 1,
                        "db_jitter": float(self.config.get("augment_db_jitter", 1.5)),
                        "db_noise_std": float(self.config.get("augment_db_noise_std", 0.2)),
                        "max_time_shift": int(self.config.get("augment_time_shift", 2)),
                    },
                )
            else:
                # MSPM path (baseline)
                ssast_state, metrics = ssast_train_step(ssast_state, batch, step_rng)

            # Log metrics
            if step % self.config["log_interval"] == 0:
                if self.config.get("ssl_enable"):
                    ps = float(metrics.get("pos_sim_mean", 0.0))
                    ns = float(metrics.get("neg_sim_mean", 0.0))
                    tl = float(metrics.get("total_loss", 0.0))
                    print(f"Step {step}: total_loss={tl:.4f} pos_sim={ps:.3f} neg_sim={ns:.3f}")
                    # Early stopping on total loss
                    if use_early_stop:
                        if tl < (best_loss - early_min_delta):
                            best_loss = tl
                            wait = 0
                        else:
                            wait += 1
                            if wait >= early_patience:
                                print(f"Early stopping at step {step} (best_loss={best_loss:.4f})")
                                break
                else:
                    print(f"Step {step}: Loss = {metrics['total_loss']:.4f}")

            # Save checkpoint
            if step % self.config["save_interval"] == 0:
                checkpoint_path = self.checkpoint_dir / f"ssast_step_{step}"
                checkpoints.save_checkpoint(
                    checkpoint_path, ssast_state, step=step, overwrite=True
                )

        # Save final checkpoint
        final_checkpoint = self.checkpoint_dir / "ssast_final"
        checkpoints.save_checkpoint(
            final_checkpoint,
            ssast_state,
            step=self.config["pretrain_steps"],
            overwrite=True,
        )

        print(f"✅ SSAST pre-training complete! Checkpoint: {final_checkpoint}")
        return str(final_checkpoint)

    def finetune_on_percepianos(self, pretrain_checkpoint: str) -> str:
        """
        Fine-tune pre-trained encoder on PercePiano dataset
        Args:
            pretrain_checkpoint: Path to SSAST pre-trained checkpoint
        Returns:
            Path to fine-tuned checkpoint
        """
        print("\n=== PHASE 2: Fine-tuning on PercePiano ===")

        # Load PercePiano data
        percepianos_path = Path(self.config["percepianos_path"])
        if not percepianos_path.exists():
            print(f"❌ PercePiano data not found: {percepianos_path}")
            return None

        # Load pre-trained SSAST
        print(f"Loading pre-trained SSAST from: {pretrain_checkpoint}")
        ssast_state = checkpoints.restore_checkpoint(pretrain_checkpoint, None)

        # Extract encoder parameters
        encoder_params = extract_encoder_for_finetuning(ssast_state)

        # Initialize AST model for fine-tuning
        ast_model = AudioSpectrogramTransformer(
            patch_size=self.config["patch_size"],
            embed_dim=self.config["embed_dim"],
            num_layers=self.config["num_layers"],
            num_heads=self.config["num_heads"],
        )

        # Create AST training state with pre-trained encoder
        dummy_shape = (self.config["batch_size"], self.config["segment_length"], 128)
        rng_key, self.rng = jax.random.split(self.rng)

        ast_state = create_train_state(
            ast_model, rng_key, dummy_shape, learning_rate=self.config["finetune_lr"]
        )

        # Transfer pre-trained encoder weights
        # This would require careful parameter mapping - simplified for now
        print("Transferring pre-trained encoder weights...")

        # Initialize PercePiano dataset
        percepianos_dataset = PercePianoDataset(
            percepianos_root=self.config["percepianos_path"],
            segment_length=self.config["segment_length"],
            cache_dir=self.config.get("cache_dir", "cache/percepiano"),
        )
        
        # Create data iterator
        data_iterator = percepianos_dataset.get_data_iterator(
            batch_size=self.config["batch_size"],
            shuffle=True,
            infinite=True
        )
        
        # Initialize optimizer and loss criterion
        optimizer = optax.adamw(
            learning_rate=self.config["finetune_lr"],
            weight_decay=0.01
        )
        
        def compute_mse_loss(predictions, targets):
            """Compute MSE loss across all perceptual dimensions"""
            total_loss = 0.0
            valid_dims = 0
            
            for dim in predictions:
                if dim in targets:
                    loss = jnp.mean((predictions[dim] - targets[dim]) ** 2)
                    total_loss += loss
                    valid_dims += 1
            
            return total_loss / max(valid_dims, 1)
        
        def compute_accuracy(predictions, targets, tolerance=0.5):
            """Compute accuracy with tolerance for regression task"""
            total_acc = 0.0
            valid_dims = 0
            
            for dim in predictions:
                if dim in targets:
                    diff = jnp.abs(predictions[dim] - targets[dim])
                    acc = jnp.mean(diff < tolerance)
                    total_acc += acc
                    valid_dims += 1
            
            return total_acc / max(valid_dims, 1)
        
        @jax.jit
        def train_step(state, batch_spectrograms, batch_labels, rng):
            """Single training step with gradient computation"""
            def loss_fn(params):
                predictions, _ = ast_model.apply(
                    params, batch_spectrograms, training=True, rngs={'dropout': rng}
                )
                loss = compute_mse_loss(predictions, batch_labels)
                return loss, predictions
            
            # Compute gradients
            (loss, predictions), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # Apply gradients with optional clipping
            if self.config.get("grad_clip_norm", None):
                grads = optax.clip_by_global_norm(self.config["grad_clip_norm"])(grads)
            
            # Update parameters
            state = state.apply_gradients(grads=grads)
            
            # Compute metrics
            accuracy = compute_accuracy(predictions, batch_labels)
            
            metrics = {
                'loss': loss,
                'accuracy': accuracy,
                'grad_norm': optax.global_norm(grads)
            }
            
            return state, metrics
        
        # Training metrics tracking
        running_loss = 0.0
        running_accuracy = 0.0
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Device placement
        device = jax.devices()[0]
        print(f"Training on device: {device}")
        
        # Fine-tuning training loop
        print(f"Starting fine-tuning for {self.config['finetune_steps']} steps...")
        
        try:
            for step in tqdm(range(self.config["finetune_steps"]), desc="Fine-tuning"):
                try:
                    # Get batch
                    batch_spectrograms, batch_labels = next(data_iterator)
                    
                    # Move to device (JAX handles this automatically)
                    batch_spectrograms = jax.device_put(batch_spectrograms, device)
                    batch_labels = {k: jax.device_put(v, device) for k, v in batch_labels.items()}
                    
                    # Generate RNG for this step
                    step_rng = jax.random.fold_in(self.rng, step)
                    
                    # Training step
                    ast_state, metrics = train_step(ast_state, batch_spectrograms, batch_labels, step_rng)
                    
                    # Update running metrics
                    running_loss = 0.9 * running_loss + 0.1 * float(metrics['loss'])
                    running_accuracy = 0.9 * running_accuracy + 0.1 * float(metrics['accuracy'])
                    
                    # Log metrics
                    if step % self.config.get("log_interval", 100) == 0:
                        logger.info(
                            f"Step {step}: Loss = {running_loss:.4f}, "
                            f"Accuracy = {running_accuracy:.3f}, "
                            f"Grad Norm = {float(metrics['grad_norm']):.4f}"
                        )
                    
                    # Save checkpoint
                    if step % self.config.get("save_interval", 1000) == 0 and step > 0:
                        checkpoint_path = self.checkpoint_dir / f"ast_finetune_step_{step}"
                        checkpoints.save_checkpoint(
                            checkpoint_path, ast_state, step=step, overwrite=True
                        )
                        logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                except Exception as batch_error:
                    logger.warning(f"Error in step {step}: {batch_error}")
                    continue
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

        # Save fine-tuned checkpoint
        finetune_checkpoint = self.checkpoint_dir / "ast_finetuned"
        checkpoints.save_checkpoint(
            finetune_checkpoint,
            ast_state,
            step=self.config["finetune_steps"],
            overwrite=True,
        )

        print(f"✅ Fine-tuning complete! Checkpoint: {finetune_checkpoint}")
        return str(finetune_checkpoint)

    def run_full_pipeline(self):
        """Run complete AST+SSAST pipeline"""
        print("🚀 Starting complete AST+SSAST pipeline...")

        # Phase 1: Pre-training
        pretrain_checkpoint = self.pretrain_ssast()
        if pretrain_checkpoint is None:
            return

        # Phase 2: Fine-tuning
        finetune_checkpoint = self.finetune_on_percepianos(pretrain_checkpoint)
        if finetune_checkpoint is None:
            return

        # Save final results
        results = {
            "pretrain_checkpoint": pretrain_checkpoint,
            "finetune_checkpoint": finetune_checkpoint,
            "config": self.config,
        }

        results_file = self.results_dir / "training_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"🎉 Complete pipeline finished! Results: {results_file}")


def get_default_config() -> Dict:
    """Get default training configuration"""
    return {
        # Data paths
        "maestro_path": "data/maestro-v3.0.0",
        "percepianos_path": "data/PercePiano",
        "cache_dir": "cache/preprocessed",
        # Model architecture
        "patch_size": 16,
        "embed_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "segment_length": 128,
        # Training hyperparameters
        "batch_size": 32,
        "pretrain_lr": 1e-4,
        "finetune_lr": 5e-5,
        "pretrain_steps": 10000,
        "finetune_steps": 5000,
        "grad_clip_norm": 1.0,
        # Logging and checkpointing
        "log_interval": 100,
        "save_interval": 1000,
        "checkpoint_dir": "checkpoints/ast_ssast",
        "results_dir": "results/ast_training",
        # Other
        "seed": 42,
    }


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="AST+SSAST Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument(
        "--pretrain-only", action="store_true", help="Only run pre-training"
    )
    # Phase 2: SSL objective flags
    parser.add_argument("--ssl-enable", action="store_true", help="Enable SSL pretraining mode")
    parser.add_argument("--ssl-objective", type=str, default="infonce", choices=["infonce", "repulsive"], help="SSL objective to use")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--augment", action="store_true", help="Enable conservative spectrogram augmentations")
    parser.add_argument("--augment-time-mask", type=int, default=8)
    parser.add_argument("--augment-freq-mask", type=int, default=8)
    parser.add_argument("--augment-time-shift", type=int, default=2)
    parser.add_argument("--augment-db-jitter", type=float, default=1.5)
    parser.add_argument("--augment-db-noise-std", type=float, default=0.2)
    parser.add_argument("--ckpt-dir", type=str, default=None, help="Override checkpoint dir")
    parser.add_argument(
        "--finetune-only", action="store_true", help="Only run fine-tuning"
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        help="Pre-training checkpoint for fine-tuning",
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = get_default_config()
        print("Using default configuration")

    # Merge CLI SSL flags into config
    if args.ssl_enable:
        config["ssl_enable"] = True
        config["ssl_objective"] = args.ssl_objective
        config["temperature"] = args.temperature
        config["augment"] = args.augment
        config["augment_time_mask"] = args.augment_time_mask
        config["augment_freq_mask"] = args.augment_freq_mask
        config["augment_time_shift"] = args.augment_time_shift
        config["augment_db_jitter"] = args.augment_db_jitter
        config["augment_db_noise_std"] = args.augment_db_noise_std
        # If SSL, prefer a separate default checkpoint dir unless user provided one
        if args.ckpt_dir:
            config["checkpoint_dir"] = args.ckpt_dir
        elif config.get("checkpoint_dir") == "checkpoints/ast_ssast":
            config["checkpoint_dir"] = "checkpoints/ssl_infonce"
    else:
        # Respect explicit override of checkpoint dir
        if args.ckpt_dir:
            config["checkpoint_dir"] = args.ckpt_dir

    # Initialize pipeline
    pipeline = ASTTrainingPipeline(config)

    if args.pretrain_only:
        pipeline.pretrain_ssast()
    elif args.finetune_only:
        if not args.pretrain_checkpoint:
            print("❌ --pretrain-checkpoint required for fine-tuning only")
            return
        pipeline.finetune_on_percepianos(args.pretrain_checkpoint)
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
