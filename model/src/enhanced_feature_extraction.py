#!/usr/bin/env python3
"""
Enhanced Traditional Feature Extraction for Hybrid AST
Comprehensive feature extraction to match Random Forest's domain knowledge advantage
"""

import numpy as np
import librosa
import librosa.display
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import warnings

# Suppress librosa warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


class EnhancedFeatureExtractor:
    """
    Comprehensive traditional audio feature extractor
    Designed to match/exceed Random Forest's feature engineering advantage
    """
    
    def __init__(self, 
                 sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 13,
                 n_chroma: int = 12):
        
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        
        print(f"🎵 Enhanced Feature Extractor initialized")
        print(f"   SR: {sr}Hz, FFT: {n_fft}, Hop: {hop_length}")
        print(f"   MFCCs: {n_mfcc}, Chroma: {n_chroma}")

    def extract_mfcc_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive MFCC-based features"""
        features = {}
        
        # Standard MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, 
                                    n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Statistical measures for each MFCC coefficient
        for i in range(self.n_mfcc):
            mfcc_coeff = mfccs[i]
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc_coeff))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc_coeff))
            features[f'mfcc_{i}_skew'] = float(self._safe_skew(mfcc_coeff))
            features[f'mfcc_{i}_kurtosis'] = float(self._safe_kurtosis(mfcc_coeff))
        
        # Delta and delta-delta MFCCs for temporal dynamics
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Statistical measures for deltas
        for i in range(min(3, self.n_mfcc)):  # Focus on first 3 coefficients for efficiency
            features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
            features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
            features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
            features[f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))
        
        return features

    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive spectral features for timbre analysis"""
        features = {}
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, 
                                                    n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features['spectral_centroid_mean'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))
        features['spectral_centroid_skew'] = float(self._safe_skew(centroid))
        features['spectral_centroid_range'] = float(np.max(centroid) - np.min(centroid))
        
        # Spectral rolloff (energy distribution)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr, 
                                                  n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        features['spectral_rolloff_std'] = float(np.std(rolloff))
        
        # Spectral bandwidth (spectral shape)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr,
                                                      n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features['spectral_bandwidth_mean'] = float(np.mean(bandwidth))
        features['spectral_bandwidth_std'] = float(np.std(bandwidth))
        
        # Spectral contrast (peak-to-valley ratio in different frequency bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr,
                                                    n_fft=self.n_fft, hop_length=self.hop_length)
        for i in range(contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = float(np.mean(contrast[i]))
            features[f'spectral_contrast_{i}_std'] = float(np.std(contrast[i]))
        
        # Spectral flatness (noisiness measure)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
        features['spectral_flatness_mean'] = float(np.mean(flatness))
        features['spectral_flatness_std'] = float(np.std(flatness))
        
        return features

    def extract_harmonic_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract harmonic and tonal features"""
        features = {}
        
        # Separate harmonic and percussive components
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic-to-percussive ratio
        harmonic_energy = np.sum(y_harmonic ** 2)
        percussive_energy = np.sum(y_percussive ** 2)
        total_energy = harmonic_energy + percussive_energy
        
        if total_energy > 0:
            features['harmonic_ratio'] = float(harmonic_energy / total_energy)
            features['percussive_ratio'] = float(percussive_energy / total_energy)
        else:
            features['harmonic_ratio'] = 0.0
            features['percussive_ratio'] = 0.0
        
        # Chromagram features (harmonic content)
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Chromagram statistics
        for i in range(self.n_chroma):
            pitch_class = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][i]
            features[f'chroma_{pitch_class}_mean'] = float(np.mean(chroma[i]))
            features[f'chroma_{pitch_class}_std'] = float(np.std(chroma[i]))
        
        # Tonal features
        features['chroma_energy'] = float(np.sum(chroma))
        features['chroma_centroid'] = float(np.sum(np.arange(12) * np.mean(chroma, axis=1)) / np.sum(np.mean(chroma, axis=1)))
        
        # Tonnetz (tonal network) features - harmonic relationships
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=self.sr)
        for i in range(tonnetz.shape[0]):
            features[f'tonnetz_{i}_mean'] = float(np.mean(tonnetz[i]))
            features[f'tonnetz_{i}_std'] = float(np.std(tonnetz[i]))
        
        return features

    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract temporal and rhythmic features"""
        features = {}
        
        # Zero-crossing rate (articulation proxy)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        features['zcr_range'] = float(np.max(zcr) - np.min(zcr))
        
        # Tempo and beat features
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=self.sr, hop_length=self.hop_length)
            tempo_val = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            features['tempo'] = tempo_val
            
            if len(beats) > 2:
                beat_intervals = np.diff(beats) * (self.hop_length / self.sr)
                features['beat_consistency'] = float(1.0 / (1.0 + np.std(beat_intervals)))
                features['beat_strength'] = float(np.mean(librosa.util.normalize(
                    librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
                )))
            else:
                features['beat_consistency'] = 0.0
                features['beat_strength'] = 0.0
        except:
            features['tempo'] = 0.0
            features['beat_consistency'] = 0.0 
            features['beat_strength'] = 0.0
        
        # Onset detection features (articulation)
        onset_strength = librosa.onset.onset_strength(y=y, sr=self.sr, hop_length=self.hop_length)
        features['onset_strength_mean'] = float(np.mean(onset_strength))
        features['onset_strength_std'] = float(np.std(onset_strength))
        
        # Estimate number of onsets
        onsets = librosa.onset.onset_detect(onset_envelope=onset_strength, sr=self.sr, 
                                           hop_length=self.hop_length, units='time')
        features['onset_rate'] = float(len(onsets) / (len(y) / self.sr)) if len(y) > 0 else 0.0
        
        return features

    def extract_dynamic_features(self, y: np.ndarray) -> Dict[str, float]:
        """Extract dynamic and energy-related features"""
        features = {}
        
        # RMS energy (loudness proxy)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['rms_range'] = float(np.max(rms) - np.min(rms))
        features['rms_skew'] = float(self._safe_skew(rms))
        
        # Dynamic range and contrast
        features['dynamic_range'] = float(np.max(rms) - np.min(rms)) if len(rms) > 0 else 0.0
        
        # Peak-to-average ratio
        peak_energy = np.max(np.abs(y)) ** 2
        avg_energy = np.mean(y ** 2)
        features['peak_to_avg_ratio'] = float(peak_energy / avg_energy) if avg_energy > 0 else 0.0
        
        # Energy distribution across time
        frame_energy = rms ** 2
        if len(frame_energy) > 0:
            features['energy_entropy'] = float(self._entropy(frame_energy))
            features['energy_centroid'] = float(np.sum(np.arange(len(frame_energy)) * frame_energy) / np.sum(frame_energy))
        else:
            features['energy_entropy'] = 0.0
            features['energy_centroid'] = 0.0
        
        return features

    def extract_all_features(self, y: np.ndarray, normalize: bool = True) -> Dict[str, float]:
        """
        Extract all traditional features for hybrid model
        Args:
            y: Audio signal
            normalize: Whether to normalize audio first
        Returns:
            Dictionary with all extracted features
        """
        # Normalize audio if requested
        if normalize and np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        # Extract all feature groups
        all_features = {}
        
        # MFCC features (26 features: 13 coefficients * 2 stats each)
        mfcc_features = self.extract_mfcc_features(y)
        all_features.update(mfcc_features)
        
        # Spectral features (~15 features)
        spectral_features = self.extract_spectral_features(y)
        all_features.update(spectral_features)
        
        # Harmonic features (~30 features: 12 chroma + 6 tonnetz + others)
        harmonic_features = self.extract_harmonic_features(y)
        all_features.update(harmonic_features)
        
        # Temporal features (~8 features)
        temporal_features = self.extract_temporal_features(y)
        all_features.update(temporal_features)
        
        # Dynamic features (~8 features)
        dynamic_features = self.extract_dynamic_features(y)
        all_features.update(dynamic_features)
        
        return all_features

    def features_to_array(self, features: Dict[str, float], feature_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert feature dictionary to numpy array for neural network input
        Args:
            features: Feature dictionary
            feature_order: Fixed order of features (for consistency across samples)
        Returns:
            Feature array
        """
        if feature_order is None:
            feature_order = sorted(features.keys())
        
        feature_array = np.array([features.get(key, 0.0) for key in feature_order])
        
        # Handle any NaN or infinite values
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return feature_array

    def get_feature_names(self, sample_audio: np.ndarray) -> List[str]:
        """Get list of all feature names for consistent ordering"""
        sample_features = self.extract_all_features(sample_audio)
        return sorted(sample_features.keys())

    # Helper methods
    def _safe_skew(self, x: np.ndarray) -> float:
        """Calculate skewness, handling edge cases"""
        if len(x) < 3 or np.std(x) == 0:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / np.std(x)) ** 3))

    def _safe_kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis, handling edge cases"""
        if len(x) < 4 or np.std(x) == 0:
            return 0.0
        return float(np.mean(((x - np.mean(x)) / np.std(x)) ** 4) - 3)

    def _entropy(self, x: np.ndarray) -> float:
        """Calculate entropy of signal"""
        # Normalize to probability distribution
        x = x / np.sum(x) if np.sum(x) > 0 else x
        x = x[x > 0]  # Remove zeros for log calculation
        return float(-np.sum(x * np.log2(x))) if len(x) > 0 else 0.0


def test_enhanced_features():
    """Test enhanced feature extraction"""
    print("=== Enhanced Traditional Feature Extraction Test ===\n")
    
    # Create test audio (synthetic piano-like signal)
    sr = 22050
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Synthetic piano-like signal with harmonics
    fundamental = 440  # A4
    y = (0.7 * np.sin(2 * np.pi * fundamental * t) +
         0.3 * np.sin(2 * np.pi * fundamental * 2 * t) +
         0.1 * np.sin(2 * np.pi * fundamental * 3 * t))
    
    # Add some envelope and noise
    envelope = np.exp(-t * 2)  # Decay
    noise = 0.02 * np.random.randn(len(t))
    y = y * envelope + noise
    
    # Initialize feature extractor
    extractor = EnhancedFeatureExtractor(sr=sr)
    
    # Extract features
    print("Extracting enhanced traditional features...")
    features = extractor.extract_all_features(y)
    
    print(f"\n📊 Extracted {len(features)} traditional features:")
    
    # Group features by type
    feature_groups = {
        'MFCC': [k for k in features.keys() if k.startswith('mfcc')],
        'Spectral': [k for k in features.keys() if k.startswith('spectral')],
        'Harmonic': [k for k in features.keys() if k.startswith(('chroma', 'tonnetz', 'harmonic'))],
        'Temporal': [k for k in features.keys() if k.startswith(('zcr', 'tempo', 'beat', 'onset'))],
        'Dynamic': [k for k in features.keys() if k.startswith(('rms', 'dynamic', 'peak', 'energy'))]
    }
    
    for group, feature_list in feature_groups.items():
        print(f"\n{group} features ({len(feature_list)}):")
        for feat in feature_list[:5]:  # Show first 5 features
            print(f"  {feat}: {features[feat]:.4f}")
        if len(feature_list) > 5:
            print(f"  ... and {len(feature_list) - 5} more")
    
    # Convert to array format
    feature_names = extractor.get_feature_names(y)
    feature_array = extractor.features_to_array(features, feature_names)
    
    print(f"\nFeature array shape: {feature_array.shape}")
    print(f"Feature range: [{np.min(feature_array):.4f}, {np.max(feature_array):.4f}]")
    
    print("\n✅ Enhanced feature extraction test complete!")
    return features, feature_array, feature_names


if __name__ == "__main__":
    test_enhanced_features()