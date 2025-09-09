#!/usr/bin/env python3
"""
Hybrid Piano Perception Model
Combines Audio Spectrogram Transformer with traditional audio features
to address the limitation of pure end-to-end learning on small datasets
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Callable
from functools import partial


class AudioFeatureExtractor:
    """Extract traditional audio features that work well with Random Forest"""
    
    def __init__(self, sr: int = 22050, n_mels: int = 128):
        self.sr = sr
        self.n_mels = n_mels
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract spectral features"""
        # Compute spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        features = {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_rolloff_std': np.std(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_bandwidth_std': np.std(spectral_bandwidth),
            'zcr_mean': np.mean(zero_crossing_rate),
            'zcr_std': np.std(zero_crossing_rate),
        }
        
        return features
    
    def extract_mfcc_features(self, audio: np.ndarray, n_mfcc: int = 13) -> Dict[str, float]:
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        
        features = {}
        for i in range(n_mfcc):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        return features
    
    def extract_harmonic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract harmonic and percussive components"""
        y_harmonic, y_percussive = librosa.effects.hpss(audio)
        
        features = {
            'harmonic_energy': np.sum(y_harmonic ** 2),
            'percussive_energy': np.sum(y_percussive ** 2),
            'harmonic_ratio': np.sum(y_harmonic ** 2) / (np.sum(audio ** 2) + 1e-8),
        }
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract temporal features"""
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        # Tempo estimation
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        features = {
            'onset_rate': len(onset_times) / (len(audio) / self.sr),  # onsets per second
            'tempo': tempo,
            'beat_consistency': np.std(np.diff(beats)) if len(beats) > 1 else 0.0,
        }
        
        return features
    
    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract all traditional audio features as a single vector"""
        all_features = {}
        
        # Combine all feature types
        all_features.update(self.extract_spectral_features(audio))
        all_features.update(self.extract_mfcc_features(audio))
        all_features.update(self.extract_harmonic_features(audio))
        all_features.update(self.extract_temporal_features(audio))
        
        # Convert to ordered array
        feature_vector = np.array(list(all_features.values()), dtype=np.float32)
        
        # Handle NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_vector


class MusicalStructureExtractor:
    """Extract high-level musical structure features"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
    
    def extract_chroma_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract chroma features (key/harmony information)"""
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        
        features = []
        # Mean and std of each chroma bin
        for i in range(12):
            features.extend([np.mean(chroma[i]), np.std(chroma[i])])
        
        # Key strength (how concentrated is the chroma)
        chroma_var = np.var(np.mean(chroma, axis=1))
        features.append(chroma_var)
        
        return np.array(features, dtype=np.float32)
    
    def extract_rhythm_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract rhythm and meter features"""
        # Tempogram for rhythm analysis
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        features = []
        features.append(tempo)
        
        # Beat regularity
        if len(beats) > 2:
            beat_intervals = np.diff(librosa.frames_to_time(beats, sr=self.sr))
            features.append(np.mean(beat_intervals))
            features.append(np.std(beat_intervals))
        else:
            features.extend([0.0, 0.0])
        
        # Pulse clarity (strength of periodic patterns)
        tempogram = librosa.feature.tempogram(y=audio, sr=self.sr)
        pulse_clarity = np.max(np.mean(tempogram, axis=1))
        features.append(pulse_clarity)
        
        return np.array(features, dtype=np.float32)
    
    def extract_dynamics_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract dynamic range and envelope features"""
        # RMS energy for dynamics
        rms = librosa.feature.rms(y=audio)[0]
        
        features = []
        features.append(np.mean(rms))  # Average loudness
        features.append(np.std(rms))   # Dynamic range
        features.append(np.max(rms))   # Peak loudness
        features.append(np.min(rms))   # Minimum loudness
        
        # Dynamic contrast
        dynamic_range = np.max(rms) - np.min(rms)
        features.append(dynamic_range)
        
        return np.array(features, dtype=np.float32)
    
    def extract_all_structure_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract all musical structure features"""
        chroma = self.extract_chroma_features(audio)
        rhythm = self.extract_rhythm_features(audio)
        dynamics = self.extract_dynamics_features(audio)
        
        return np.concatenate([chroma, rhythm, dynamics])


class CompactAST(nn.Module):
    """Smaller, more efficient AST for hybrid model"""
    
    patch_size: int = 16
    embed_dim: int = 512  # Reduced from 768
    num_layers: int = 8   # Reduced from 12
    num_heads: int = 8    # Reduced from 12
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        """Compact AST forward pass"""
        batch_size, time_frames, freq_bins = x.shape
        
        # Patch embedding
        patch_size = self.patch_size
        time_pad = (patch_size - time_frames % patch_size) % patch_size
        freq_pad = (patch_size - freq_bins % patch_size) % patch_size
        
        if time_pad > 0 or freq_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, time_pad), (0, freq_pad)), 
                        mode='constant', constant_values=-80.0)
        
        time_patches = x.shape[1] // patch_size
        freq_patches = x.shape[2] // patch_size
        num_patches = time_patches * freq_patches
        
        # Reshape to patches
        x = x.reshape(batch_size, time_patches, patch_size, freq_patches, patch_size)
        x = x.transpose(0, 1, 3, 2, 4)
        x = x.reshape(batch_size, num_patches, patch_size * patch_size)
        
        # Linear projection
        x = nn.Dense(self.embed_dim, name='patch_projection')(x)
        
        # Positional encoding
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.truncated_normal(stddev=0.02),
            (1, num_patches, self.embed_dim)
        )
        x = x + pos_embedding
        
        # Transformer layers (fewer layers for efficiency)
        for layer_idx in range(self.num_layers):
            # Self-attention block
            residual = x
            x = nn.LayerNorm(name=f'norm1_layer{layer_idx}')(x)
            
            attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f'attention_layer{layer_idx}'
            )(x, x, deterministic=not training)
            
            x = residual + nn.Dropout(self.dropout_rate)(attention, deterministic=not training)
            
            # MLP block
            residual = x
            x = nn.LayerNorm(name=f'norm2_layer{layer_idx}')(x)
            
            mlp_hidden = int(self.embed_dim * self.mlp_ratio)
            mlp = nn.Dense(mlp_hidden, name=f'mlp1_layer{layer_idx}')(x)
            mlp = nn.gelu(mlp)
            mlp = nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
            mlp = nn.Dense(self.embed_dim, name=f'mlp2_layer{layer_idx}')(mlp)
            
            x = residual + nn.Dropout(self.dropout_rate)(mlp, deterministic=not training)
        
        # Final layer norm
        x = nn.LayerNorm(name='final_norm')(x)
        
        # Global average pooling
        x = jnp.mean(x, axis=1)  # [batch, embed_dim]
        
        return x


class HybridPerceptionModel(nn.Module):
    """
    Hybrid model combining AST with traditional audio features
    
    This addresses the Random Forest advantage by incorporating
    domain knowledge while maintaining transformer capabilities
    """
    
    # AST configuration (smaller for efficiency)
    ast_embed_dim: int = 512
    ast_num_layers: int = 8
    ast_num_heads: int = 8
    
    # Feature processing dimensions
    audio_feature_dim: int = 256
    structure_feature_dim: int = 128
    
    # Fusion and output
    fusion_dim: int = 512
    num_outputs: int = 19
    dropout_rate: float = 0.15
    
    def setup(self):
        # AST encoder (compact version)
        self.ast_encoder = CompactAST(
            embed_dim=self.ast_embed_dim,
            num_layers=self.ast_num_layers,
            num_heads=self.ast_num_heads,
            dropout_rate=self.dropout_rate
        )
        
        # Traditional audio feature encoder
        self.audio_feature_encoder = nn.Sequential([
            nn.Dense(self.audio_feature_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.audio_feature_dim // 2),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
        ])
        
        # Musical structure feature encoder  
        self.structure_encoder = nn.Sequential([
            nn.Dense(self.structure_feature_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.structure_feature_dim // 2),
            nn.gelu,
        ])
        
        # Fusion network
        total_features = (self.ast_embed_dim + 
                         self.audio_feature_dim // 2 + 
                         self.structure_feature_dim // 2)
        
        self.fusion_network = nn.Sequential([
            nn.Dense(self.fusion_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.fusion_dim // 2),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
        ])
        
        # Dimension-specific heads (grouped by musical similarity)
        self.timing_head = self._create_dimension_head(4)      # Rubato, tempo-related
        self.dynamics_head = self._create_dimension_head(6)    # Softness, attack, strength
        self.expression_head = self._create_dimension_head(5)  # Tension, fluidity, emotion
        self.technical_head = self._create_dimension_head(4)   # Precision, articulation
        
    def _create_dimension_head(self, num_dims: int):
        """Create a specialized head for a group of related dimensions"""
        return nn.Sequential([
            nn.Dense(64),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(32),
            nn.gelu,
            nn.Dense(num_dims)
        ])
    
    @nn.compact
    def __call__(self, 
                 spectrogram: jnp.ndarray,
                 audio_features: jnp.ndarray, 
                 structure_features: jnp.ndarray,
                 training: bool = True) -> jnp.ndarray:
        """
        Forward pass combining all information sources
        
        Args:
            spectrogram: [batch, time, freq] mel-spectrogram
            audio_features: [batch, n_audio_features] traditional features  
            structure_features: [batch, n_structure_features] musical structure
            training: training mode flag
            
        Returns:
            predictions: [batch, 19] perceptual dimension predictions
        """
        
        # Process each input stream
        ast_features = self.ast_encoder(spectrogram, training=training)
        audio_features = self.audio_feature_encoder(audio_features)
        structure_features = self.structure_encoder(structure_features)
        
        # Concatenate all features
        combined_features = jnp.concatenate([
            ast_features, 
            audio_features, 
            structure_features
        ], axis=-1)
        
        # Fusion network
        fused_features = self.fusion_network(combined_features)
        
        # Dimension-specific predictions
        timing_pred = self.timing_head(fused_features)      # 4 dimensions
        dynamics_pred = self.dynamics_head(fused_features)  # 6 dimensions  
        expression_pred = self.expression_head(fused_features) # 5 dimensions
        technical_pred = self.technical_head(fused_features)   # 4 dimensions
        
        # Combine all predictions (ensure we get exactly 19 dimensions)
        all_predictions = jnp.concatenate([
            timing_pred, dynamics_pred, expression_pred, technical_pred
        ], axis=-1)
        
        return all_predictions


class HybridDataProcessor:
    """Process data for hybrid model training"""
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.audio_extractor = AudioFeatureExtractor(sr=sr)
        self.structure_extractor = MusicalStructureExtractor(sr=sr)
    
    def process_audio_file(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process audio into all required feature types
        
        Returns:
            spectrogram: mel-spectrogram for AST
            audio_features: traditional audio features
            structure_features: musical structure features
        """
        # Generate mel-spectrogram (for AST)
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128, hop_length=512, n_fft=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).T
        
        # Ensure fixed size
        if mel_spec_db.shape[0] >= 128:
            spectrogram = mel_spec_db[:128, :]
        else:
            pad_width = 128 - mel_spec_db.shape[0]
            spectrogram = np.pad(mel_spec_db, ((0, pad_width), (0, 0)), 
                               mode='constant', constant_values=-80.0)
        
        # Extract traditional features (for Random Forest-style processing)
        audio_features = self.audio_extractor.extract_all_features(audio)
        
        # Extract musical structure features
        structure_features = self.structure_extractor.extract_all_structure_features(audio)
        
        return spectrogram, audio_features, structure_features


# Expected performance improvement: +0.05 to +0.10 correlation
# This hybrid approach should beat Random Forest by combining:
# 1. Transformer's ability to learn complex patterns
# 2. Traditional features that Random Forest excels at
# 3. Domain-specific musical knowledge

if __name__ == "__main__":
    print("🎹 Hybrid Piano Perception Model")
    print("   Combines AST + Traditional Features + Musical Structure")
    print("   Expected improvement: +0.05-0.10 correlation")
    print("   Target: Beat Random Forest (0.5869) and reach 0.65+")
