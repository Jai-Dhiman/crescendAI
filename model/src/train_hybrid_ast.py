#!/usr/bin/env python3
"""
Hybrid AST Training Script
Train hybrid model combining AST with traditional features to beat Random Forest baseline
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


def extract_spectrogram_features(audio_path: Path, target_length: int = 128) -> np.ndarray:
    """Extract mel-spectrogram features for AST input"""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target length (truncate or pad)
        if mel_spec_db.shape[1] > target_length:
            # Truncate from center
            start = (mel_spec_db.shape[1] - target_length) // 2
            mel_spec_db = mel_spec_db[:, start:start + target_length]
        elif mel_spec_db.shape[1] < target_length:
            # Pad
            pad_width = target_length - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        
        # Transpose to (time, freq) for transformer
        return mel_spec_db.T  # Shape: (128, 128)
        
    except Exception as e:
        print(f"Warning: Failed to process {audio_path}: {e}")
        # Return zero spectrogram as fallback
        return np.zeros((target_length, 128))


def prepare_hybrid_dataset(
    data_path: Path,
    ratings_df: pd.DataFrame,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Prepare dataset for hybrid model training
    Returns:
        spectrograms: AST input features
        traditional_features: Traditional audio features 
        targets: Perceptual dimension targets
        performance_names: List of performance identifiers
        scaler: Fitted scaler for traditional features
    """
    print("\n🎵 Preparing hybrid dataset...")
    
    # Initialize feature extractor
    feature_extractor = EnhancedFeatureExtractor(sr=22050)
    
    # Get perceptual dimension columns (exclude metadata columns)
    metadata_cols = ['Performance', 'Performer', 'Composer', 'Piece']
    perceptual_dims = [col for col in ratings_df.columns if col not in metadata_cols]
    print(f"   Target perceptual dimensions: {len(perceptual_dims)}")
    
    # Initialize lists for data
    spectrograms = []
    traditional_features = []
    targets = []
    performance_names = []
    
    # Get audio files directory
    audio_dir = data_path / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    
    # Process samples
    n_samples = min(len(ratings_df), max_samples) if max_samples else len(ratings_df)
    print(f"   Processing {n_samples} performances...")
    
    successful_extractions = 0
    
    for idx in tqdm(range(n_samples), desc="Extracting features"):
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
            print(f"Warning: Audio file not found for {performance_name}")
            continue
        
        try:
            # Extract spectrogram features for AST
            spectrogram = extract_spectrogram_features(audio_file)
            
            # Extract traditional features  
            y, sr = librosa.load(audio_file, sr=22050, mono=True)
            trad_features_dict = feature_extractor.extract_all_features(y)
            
            # Get consistent feature ordering
            if not traditional_features:  # First sample
                feature_names = sorted(trad_features_dict.keys())
            
            trad_features_array = feature_extractor.features_to_array(
                trad_features_dict, feature_names
            )
            
            # Get target values for perceptual dimensions
            target_values = []
            for dim in perceptual_dims:
                value = row[dim] if pd.notna(row[dim]) else 0.5  # Default to middle value
                target_values.append(float(value))
            
            # Store data
            spectrograms.append(spectrogram)
            traditional_features.append(trad_features_array)
            targets.append(target_values)
            performance_names.append(performance_name)
            
            successful_extractions += 1
            
        except Exception as e:
            print(f"Warning: Failed to process {performance_name}: {e}")
            continue
    
    print(f"   Successfully processed {successful_extractions}/{n_samples} performances")
    
    if successful_extractions == 0:
        raise ValueError("No successful feature extractions!")
    
    # Convert to numpy arrays
    spectrograms = np.array(spectrograms)  # Shape: (N, 128, 128)
    traditional_features = np.array(traditional_features)  # Shape: (N, 145)
    targets = np.array(targets)  # Shape: (N, 19)
    
    print(f"   Final dataset shapes:")
    print(f"     Spectrograms: {spectrograms.shape}")
    print(f"     Traditional features: {traditional_features.shape}")
    print(f"     Targets: {targets.shape}")
    
    # Normalize traditional features
    scaler = StandardScaler()
    traditional_features_scaled = scaler.fit_transform(traditional_features)
    
    return spectrograms, traditional_features_scaled, targets, performance_names, scaler


def compute_correlation_loss(predictions: Dict[str, jnp.ndarray], targets: jnp.ndarray, 
                           dimension_names: List[str]) -> jnp.ndarray:
    """Compute correlation-based loss function"""
    total_loss = 0.0
    valid_dims = 0
    
    for i, dim_name in enumerate(dimension_names):
        if dim_name in predictions:
            pred = predictions[dim_name]
            target = targets[:, i]
            
            # Pearson correlation loss (maximize correlation = minimize negative correlation)
            pred_centered = pred - jnp.mean(pred)
            target_centered = target - jnp.mean(target)
            
            numerator = jnp.sum(pred_centered * target_centered)
            denominator = jnp.sqrt(jnp.sum(pred_centered**2) * jnp.sum(target_centered**2))
            
            correlation = numerator / (denominator + 1e-8)
            corr_loss = 1.0 - correlation  # Convert to loss (minimize)
            
            # Also include MSE for stability
            mse_loss = jnp.mean((pred - target)**2)
            
            # Combined loss
            combined_loss = 0.7 * corr_loss + 0.3 * mse_loss
            total_loss += combined_loss
            valid_dims += 1
    
    return total_loss / valid_dims if valid_dims > 0 else 0.0


def train_step(state, spectrograms_batch, traditional_batch, targets_batch, dimension_names):
    """Single training step for hybrid model"""
    
    def loss_fn(params):
        predictions, _, _ = state.apply_fn(params, spectrograms_batch, traditional_batch, training=True)
        return compute_correlation_loss(predictions, targets_batch, dimension_names)
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
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
            correlation = np.corrcoef(pred, target)[0, 1]
            correlations[dim_name] = correlation if not np.isnan(correlation) else 0.0
            
            # MSE
            mse = mean_squared_error(target, pred)
            mse_scores[dim_name] = mse
    
    # Overall metrics
    avg_correlation = np.mean(list(correlations.values()))
    avg_mse = np.mean(list(mse_scores.values()))
    
    return correlations, mse_scores, avg_correlation, avg_mse


def train_hybrid_model(
    data_path: Path = Path("../data"),
    output_dir: Path = Path("../results/hybrid_ast"),
    max_samples: Optional[int] = None,
    embed_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    fusion_strategy: str = 'attention',
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    batch_size: int = 16
):
    """Train hybrid AST model"""
    
    print("=== Hybrid Audio Spectrogram Transformer Training ===\n")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    metadata, ratings_df = load_percepton_dataset(data_path)
    
    # Prepare hybrid dataset
    spectrograms, traditional_features, targets, performance_names, scaler = prepare_hybrid_dataset(
        data_path, ratings_df, max_samples
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
    
    print(f"\nDataset split:")
    print(f"   Training samples: {len(train_spectrograms)}")
    print(f"   Validation samples: {len(val_spectrograms)}")
    
    # Create hybrid model
    print(f"\nCreating hybrid model:")
    print(f"   Architecture: {embed_dim}D, {num_layers}L, {num_heads}H")
    print(f"   Fusion strategy: {fusion_strategy}")
    
    model = HybridAudioSpectrogramTransformer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        fusion_strategy=fusion_strategy,
        traditional_feature_dim=traditional_features.shape[1]
    )
    
    # Initialize training state
    rng = jax.random.PRNGKey(42)
    state = create_hybrid_train_state(
        model, rng, 
        train_spectrograms[:1].shape,  # (1, 128, 128)
        train_traditional[:1].shape,   # (1, 145)
        learning_rate=learning_rate
    )
    
    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"   Total parameters: {param_count:,}")
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    
    best_val_correlation = -1.0
    best_epoch = 0
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
            
            # Save best model
            if avg_val_corr > best_val_correlation:
                best_val_correlation = avg_val_corr
                best_epoch = epoch
                
                # Save model checkpoint
                with open(output_dir / "best_model_params.pkl", "wb") as f:
                    pickle.dump(state.params, f)
                
                print(f"   New best model! Correlation: {best_val_correlation:.4f}")
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
    
    print(f"\nPer-dimension correlations:")
    for dim, corr in val_correlations.items():
        print(f"   {dim}: {corr:.4f}")
    
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
            'parameters': param_count
        },
        'training_history': train_history
    }
    
    with open(output_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save scaler
    with open(output_dir / "feature_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"\n✅ Training complete! Results saved to {output_dir}")
    
    # Compare with Random Forest baseline (0.5869)
    rf_baseline = 0.5869
    improvement = avg_val_corr - rf_baseline
    
    print(f"\n🎯 Comparison with Random Forest baseline:")
    print(f"   Random Forest: {rf_baseline:.4f}")
    print(f"   Hybrid AST: {avg_val_corr:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/rf_baseline*100:+.1f}%)")
    
    if avg_val_corr > rf_baseline:
        print("   ✅ BEAT RANDOM FOREST BASELINE!")
    else:
        print("   ❌ Did not beat Random Forest baseline")
        print("   💡 Consider: smaller architecture, more features, better fusion")
    
    return best_state, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid AST model")
    parser.add_argument("--data-path", type=str, default="../data", help="Path to PercePiano dataset")
    parser.add_argument("--output-dir", type=str, default="../results/hybrid_ast", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to use")
    parser.add_argument("--embed-dim", type=int, default=384, help="Embedding dimension (smaller for less overfitting)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--fusion", type=str, default="attention", choices=['concat', 'attention', 'gated'])
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    train_hybrid_model(
        data_path=Path(args.data_path),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        fusion_strategy=args.fusion,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )