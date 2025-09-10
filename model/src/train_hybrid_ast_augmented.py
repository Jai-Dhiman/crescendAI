#!/usr/bin/env python3
"""
Hybrid AST Training with Smart Data Augmentation
Enhanced training script with piano-specific augmentation pipeline
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
from tqdm import tqdm
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd

from models.hybrid_ast import HybridAudioSpectrogramTransformer, create_hybrid_train_state
from enhanced_feature_extraction import EnhancedFeatureExtractor
from piano_data_augmentation import PianoDataAugmentationPipeline


def load_percepton_dataset(data_path: Path = Path("../data")) -> Tuple[Dict, pd.DataFrame]:
    """Load PercePiano dataset with audio and perceptual ratings"""
    print("🎹 Loading PercePiano dataset...")
    
    # Load perceptual ratings
    ratings_file = data_path / "PercePiano_ratings.csv"
    if not ratings_file.exists():
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
    
    ratings_df = pd.read_csv(ratings_file)
    print(f"   Loaded {len(ratings_df)} performance ratings")
    
    # Load audio metadata
    metadata_file = data_path / "PercePiano_metadata.json" 
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"   Found metadata for {len(metadata)} performances")
    return metadata, ratings_df


def prepare_augmented_hybrid_dataset(
    data_path: Path,
    ratings_df: pd.DataFrame,
    max_samples: Optional[int] = None,
    n_augmentations: int = 2,
    use_augmentation: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[Dict], StandardScaler]:
    """
    Prepare augmented dataset for hybrid model training
    Returns:
        spectrograms: AST input features (with augmentations)
        traditional_features: Traditional audio features 
        targets: Perceptual dimension targets
        performance_names: List of performance identifiers
        augmentation_metadata: List of augmentation parameters
        scaler: Fitted scaler for traditional features
    """
    print("\n🎵 Preparing augmented hybrid dataset...")
    
    # Initialize processors
    feature_extractor = EnhancedFeatureExtractor(sr=22050)
    augmentation_pipeline = PianoDataAugmentationPipeline(sr=22050) if use_augmentation else None
    
    # Get perceptual dimension columns
    metadata_cols = ['Performance', 'Performer', 'Composer', 'Piece']
    perceptual_dims = [col for col in ratings_df.columns if col not in metadata_cols]
    print(f"   Target perceptual dimensions: {len(perceptual_dims)}")
    
    # Initialize lists for data
    all_spectrograms = []
    all_traditional_features = []
    all_targets = []
    all_performance_names = []
    all_augmentation_metadata = []
    
    # Get audio files directory
    audio_dir = data_path / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Process samples
    n_samples = min(len(ratings_df), max_samples) if max_samples else len(ratings_df)
    print(f"   Processing {n_samples} performances...")
    
    if use_augmentation:
        print(f"   Creating {n_augmentations + 1} samples per performance (1 original + {n_augmentations} augmented)")
    
    successful_extractions = 0
    feature_names = None
    
    for idx in tqdm(range(n_samples), desc="Extracting augmented features"):
        row = ratings_df.iloc[idx]
        performance_name = row['Performance']
        
        # Find audio file
        audio_file = None
        for ext in ['.wav', '.mp3', '.flac']:
            candidate = audio_dir / f"{performance_name}{ext}"
            if candidate.exists():
                audio_file = candidate
                break
        
        if audio_file is None:
            continue
        
        try:
            # Load audio once
            y, sr = librosa.load(audio_file, sr=22050, mono=True)
            
            # Get target values for perceptual dimensions
            target_values = []
            for dim in perceptual_dims:
                value = row[dim] if pd.notna(row[dim]) else 0.5
                target_values.append(float(value))
            
            if use_augmentation:
                # Create augmented samples using pipeline
                augmented_samples = augmentation_pipeline.create_augmented_samples(y, n_augmentations)
                
                for spec, aug_params in augmented_samples:
                    # Extract traditional features from the original audio for each sample
                    # (we keep traditional features consistent since they're less sensitive to subtle augmentations)
                    trad_features_dict = feature_extractor.extract_all_features(y)
                    
                    # Get consistent feature ordering
                    if feature_names is None:
                        feature_names = sorted(trad_features_dict.keys())
                    
                    trad_features_array = feature_extractor.features_to_array(
                        trad_features_dict, feature_names
                    )
                    
                    # Store data
                    all_spectrograms.append(spec)
                    all_traditional_features.append(trad_features_array)
                    all_targets.append(target_values)
                    all_performance_names.append(f"{performance_name}_{aug_params['augmentation_type']}")
                    all_augmentation_metadata.append(aug_params)
            
            else:
                # No augmentation - just original sample
                spec = augmentation_pipeline.extract_spectrogram(y) if augmentation_pipeline else None
                if spec is None:
                    # Fallback spectrogram extraction
                    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    spec = mel_spec_db.T[:128]  # Truncate/pad to 128 time frames
                
                trad_features_dict = feature_extractor.extract_all_features(y)
                
                if feature_names is None:
                    feature_names = sorted(trad_features_dict.keys())
                
                trad_features_array = feature_extractor.features_to_array(
                    trad_features_dict, feature_names
                )
                
                all_spectrograms.append(spec)
                all_traditional_features.append(trad_features_array)
                all_targets.append(target_values)
                all_performance_names.append(performance_name)
                all_augmentation_metadata.append({'augmentation_type': 'original'})
            
            successful_extractions += 1
            
        except Exception as e:
            print(f"Warning: Failed to process {performance_name}: {e}")
            continue
    
    total_samples = len(all_spectrograms)
    print(f"   Successfully processed {successful_extractions} performances")
    print(f"   Generated {total_samples} total samples")
    
    if total_samples == 0:
        raise ValueError("No successful feature extractions!")
    
    # Convert to numpy arrays
    spectrograms = np.array(all_spectrograms)
    traditional_features = np.array(all_traditional_features)
    targets = np.array(all_targets)
    
    print(f"   Final dataset shapes:")
    print(f"     Spectrograms: {spectrograms.shape}")
    print(f"     Traditional features: {traditional_features.shape}")
    print(f"     Targets: {targets.shape}")
    
    # Normalize traditional features
    scaler = StandardScaler()
    traditional_features_scaled = scaler.fit_transform(traditional_features)
    
    effective_multiplier = total_samples / successful_extractions
    print(f"   🎯 Dataset multiplier: {effective_multiplier:.1f}x")
    
    return (spectrograms, traditional_features_scaled, targets, 
            all_performance_names, all_augmentation_metadata, scaler)


def compute_correlation_loss(predictions: Dict[str, jnp.ndarray], targets: jnp.ndarray, 
                           dimension_names: List[str]) -> jnp.ndarray:
    """Compute correlation-based loss function optimized for small datasets"""
    total_loss = 0.0
    valid_dims = 0
    
    for i, dim_name in enumerate(dimension_names):
        if dim_name in predictions:
            pred = predictions[dim_name]
            target = targets[:, i]
            
            # Pearson correlation loss with numerical stability
            pred_centered = pred - jnp.mean(pred)
            target_centered = target - jnp.mean(target)
            
            numerator = jnp.sum(pred_centered * target_centered)
            pred_std = jnp.sqrt(jnp.sum(pred_centered**2) + 1e-8)
            target_std = jnp.sqrt(jnp.sum(target_centered**2) + 1e-8)
            
            correlation = numerator / (pred_std * target_std + 1e-8)
            corr_loss = 1.0 - correlation
            
            # MSE loss for stability
            mse_loss = jnp.mean((pred - target)**2)
            
            # Weighted combination: prioritize correlation for final metric
            combined_loss = 0.8 * corr_loss + 0.2 * mse_loss
            total_loss += combined_loss
            valid_dims += 1
    
    return total_loss / valid_dims if valid_dims > 0 else 0.0


def train_step(state, spectrograms_batch, traditional_batch, targets_batch, dimension_names):
    """Single training step with gradient clipping for stability"""
    
    def loss_fn(params):
        predictions, _, _ = state.apply_fn(params, spectrograms_batch, traditional_batch, training=True)
        return compute_correlation_loss(predictions, targets_batch, dimension_names)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    
    # Gradient clipping for stability with small datasets
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    state = state.apply_gradients(grads=grads)
    
    return state, loss


def evaluate_model(state, spectrograms, traditional_features, targets, dimension_names):
    """Evaluate model performance using correlation metrics"""
    # Get predictions
    predictions, _, _ = state.apply_fn(state.params, spectrograms, traditional_features, training=False)
    
    # Compute correlations for each dimension
    correlations = {}
    mse_scores = {}
    
    for i, dim_name in enumerate(dimension_names):
        if dim_name in predictions:
            pred = np.array(predictions[dim_name])
            target = targets[:, i]
            
            # Pearson correlation
            if np.std(pred) > 1e-8 and np.std(target) > 1e-8:
                correlation = np.corrcoef(pred, target)[0, 1]
                correlations[dim_name] = correlation if not np.isnan(correlation) else 0.0
            else:
                correlations[dim_name] = 0.0
            
            # MSE
            mse = mean_squared_error(target, pred)
            mse_scores[dim_name] = mse
    
    # Overall metrics
    avg_correlation = np.mean(list(correlations.values()))
    avg_mse = np.mean(list(mse_scores.values()))
    
    return correlations, mse_scores, avg_correlation, avg_mse


def train_augmented_hybrid_model(
    data_path: Path = Path("../data"),
    output_dir: Path = Path("../results/hybrid_ast_augmented"),
    max_samples: Optional[int] = None,
    n_augmentations: int = 2,
    use_augmentation: bool = True,
    embed_dim: int = 384,
    num_layers: int = 4,
    num_heads: int = 6,
    fusion_strategy: str = 'attention',
    learning_rate: float = 3e-5,
    num_epochs: int = 200,
    batch_size: int = 8
):
    """Train hybrid AST model with smart data augmentation"""
    
    print("=== Hybrid AST Training with Smart Augmentation ===\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    metadata, ratings_df = load_percepton_dataset(data_path)
    
    # Prepare augmented dataset
    (spectrograms, traditional_features, targets, 
     performance_names, aug_metadata, scaler) = prepare_augmented_hybrid_dataset(
        data_path, ratings_df, max_samples, n_augmentations, use_augmentation
    )
    
    # Get dimension names
    metadata_cols = ['Performance', 'Performer', 'Composer', 'Piece']
    dimension_names = [col for col in ratings_df.columns if col not in metadata_cols]
    
    # Split dataset (80/20 train/val)
    n_samples = len(spectrograms)
    split_idx = int(0.8 * n_samples)
    
    train_spectrograms = spectrograms[:split_idx]
    train_traditional = traditional_features[:split_idx] 
    train_targets = targets[:split_idx]
    
    val_spectrograms = spectrograms[split_idx:]
    val_traditional = traditional_features[split_idx:]
    val_targets = targets[split_idx:]
    
    print(f"\nAugmented dataset split:")
    print(f"   Training samples: {len(train_spectrograms)}")
    print(f"   Validation samples: {len(val_spectrograms)}")
    
    # Create hybrid model (smaller architecture for augmented data)
    print(f"\nCreating optimized hybrid model:")
    print(f"   Architecture: {embed_dim}D, {num_layers}L, {num_heads}H")
    print(f"   Fusion strategy: {fusion_strategy}")
    print(f"   Augmentation: {'ON' if use_augmentation else 'OFF'}")
    
    model = HybridAudioSpectrogramTransformer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        fusion_strategy=fusion_strategy,
        traditional_feature_dim=traditional_features.shape[1],
        dropout_rate=0.3  # Higher dropout for augmented data
    )
    
    # Initialize training state with lower learning rate for augmented data
    rng = jax.random.PRNGKey(42)
    state = create_hybrid_train_state(
        model, rng, 
        train_spectrograms[:1].shape,
        train_traditional[:1].shape,
        learning_rate=learning_rate
    )
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"   Total parameters: {param_count:,}")
    
    # Training loop with augmented data
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_correlation = -1.0
    best_epoch = 0
    patience = 30
    patience_counter = 0
    train_history = []
    
    for epoch in range(num_epochs):
        # Training epoch
        epoch_losses = []
        n_batches = (len(train_spectrograms) + batch_size - 1) // batch_size
        
        # Shuffle training data
        indices = np.random.permutation(len(train_spectrograms))
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_spectrograms))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch
            spec_batch = train_spectrograms[batch_indices]
            trad_batch = train_traditional[batch_indices]
            targets_batch = train_targets[batch_indices]
            
            # Training step
            state, loss = train_step(state, spec_batch, trad_batch, targets_batch, dimension_names)
            epoch_losses.append(float(loss))
        
        avg_train_loss = np.mean(epoch_losses)
        
        # Validation evaluation (every 10 epochs)
        if epoch % 10 == 0:
            val_correlations, val_mse, avg_val_corr, avg_val_mse = evaluate_model(
                state, val_spectrograms, val_traditional, val_targets, dimension_names
            )
            
            print(f"Epoch {epoch:3d}: Loss={avg_train_loss:.4f}, Val_Corr={avg_val_corr:.4f}, Val_MSE={avg_val_mse:.4f}")
            
            # Early stopping and model saving
            if avg_val_corr > best_val_correlation:
                best_val_correlation = avg_val_corr
                best_epoch = epoch
                patience_counter = 0
                
                # Save model checkpoint
                with open(output_dir / "best_model_params.pkl", "wb") as f:
                    pickle.dump(state.params, f)
                
                print(f"   🎯 New best model! Correlation: {best_val_correlation:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch} (patience: {patience})")
                break
        else:
            print(f"Epoch {epoch:3d}: Loss={avg_train_loss:.4f}")
        
        # Save training history
        train_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
        })
    
    # Final evaluation
    print(f"\n=== Final Evaluation ===")
    
    # Load best model
    with open(output_dir / "best_model_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    
    best_state = state.replace(params=best_params)
    
    # Evaluate on validation set
    val_correlations, val_mse, avg_val_corr, avg_val_mse = evaluate_model(
        best_state, val_spectrograms, val_traditional, val_targets, dimension_names
    )
    
    print(f"Best model (epoch {best_epoch}):")
    print(f"   Average correlation: {avg_val_corr:.4f}")
    print(f"   Average MSE: {avg_val_mse:.4f}")
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_correlation': float(best_val_correlation),
        'final_correlations': {k: float(v) for k, v in val_correlations.items()},
        'final_mse': {k: float(v) for k, v in val_mse.items()},
        'model_config': {
            'embed_dim': embed_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'fusion_strategy': fusion_strategy,
            'parameters': param_count,
            'augmentation_enabled': use_augmentation,
            'n_augmentations': n_augmentations if use_augmentation else 0
        },
        'training_history': train_history,
        'dataset_info': {
            'original_samples': successful_extractions if 'successful_extractions' in locals() else 'unknown',
            'total_samples': n_samples,
            'augmentation_multiplier': n_samples / successful_extractions if 'successful_extractions' in locals() else 1.0
        }
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save scaler and metadata
    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    with open(output_dir / "augmentation_metadata.json", "w") as f:
        json.dump(aug_metadata[:100], f, indent=2)  # Save first 100 for inspection
    
    print(f"\n✅ Training complete! Results saved to {output_dir}")
    
    # Compare with Random Forest baseline
    rf_baseline = 0.5869
    improvement = avg_val_corr - rf_baseline
    
    print(f"\n🎯 Comparison with Random Forest baseline:")
    print(f"   Random Forest: {rf_baseline:.4f}")
    print(f"   Hybrid AST (augmented): {avg_val_corr:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/rf_baseline*100:+.1f}%)")
    
    if avg_val_corr > rf_baseline:
        print("   ✅ BEAT RANDOM FOREST BASELINE!")
        print("   🏆 Hybrid architecture + augmentation successful!")
    else:
        print("   ⚠️  Still below Random Forest baseline")
        gap_remaining = rf_baseline - avg_val_corr
        print(f"   💡 Gap remaining: {gap_remaining:.4f} correlation points")
    
    return best_state, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Augmented Hybrid AST model")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to PercePiano dataset")
    parser.add_argument("--output-dir", type=str, default="../results/hybrid_ast_augmented", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to use")
    parser.add_argument("--augmentations", type=int, default=2, help="Number of augmentations per sample")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable augmentation")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension (small for augmented data)")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--fusion", type=str, default="attention", choices=['concat', 'attention', 'gated'])
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    
    args = parser.parse_args()
    
    train_augmented_hybrid_model(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        n_augmentations=args.augmentations,
        use_augmentation=not args.no_augmentation,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        fusion_strategy=args.fusion,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )