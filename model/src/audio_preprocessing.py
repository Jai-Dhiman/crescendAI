#!/usr/bin/env python3
"""
My Piano Performance Analysis - Audio Preprocessing
Clean implementation built from scratch for Phase 1 recreation
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    import librosa
    import librosa.display
except ImportError:
    print("❌ librosa required: pip install librosa")
    raise

class PianoAudioPreprocessor:
    """Clean audio preprocessing for piano performance analysis"""
    
    def __init__(self, 
                 target_sr: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 n_mfcc: int = 13):
        
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        print(f"🎹 Piano Audio Preprocessor initialized:")
        print(f"   Target SR: {target_sr}Hz, FFT: {n_fft}, Hop: {hop_length}")
        print(f"   Mel bands: {n_mels}, MFCCs: {n_mfcc}")
    
    def load_and_normalize_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio with consistent preprocessing"""
        y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        return y, sr
    
    def extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract spectral features optimized for piano analysis"""
        features = {}
        
        # Mel-spectrogram (primary feature for neural networks)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCCs (compact representation)
        features['mfccs'] = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Chromagram (harmonic content)
        features['chromagram'] = librosa.feature.chroma_stft(
            y=y, sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return features
    
    def extract_scalar_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract scalar features for correlation analysis"""
        features = {}
        
        # Tempo and rhythm
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
            features['beat_consistency'] = float(np.std(np.diff(beats))) if len(beats) > 1 else 0.0
        except:
            features['tempo'] = 0.0
            features['beat_consistency'] = 0.0
        
        # Spectral features (timbre)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(centroid))
        features['spectral_centroid_std'] = float(np.std(centroid))
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = float(np.mean(rolloff))
        
        # Dynamic features
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['dynamic_range'] = float(np.max(rms) - np.min(rms))
        
        # Articulation features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        return features
    
    def process_audio_file(self, audio_path: Path, performance_name: str) -> Dict:
        """Complete preprocessing pipeline for one audio file"""
        print(f"\n🎵 Processing: {performance_name}")
        
        # Load audio
        y, sr = self.load_and_normalize_audio(audio_path)
        duration = len(y) / sr
        print(f"   Loaded: {duration:.2f}s at {sr}Hz")
        
        # Extract features
        spectral_features = self.extract_spectral_features(y, sr)
        scalar_features = self.extract_scalar_features(y, sr)
        
        # Package results
        return {
            'performance_name': performance_name,
            'duration': duration,
            'sample_rate': sr,
            'spectral_features': {
                'mel_spectrogram': spectral_features['mel_spectrogram'].tolist(),
                'mfccs': spectral_features['mfccs'].tolist(),
                'chromagram': spectral_features['chromagram'].tolist()
            },
            'scalar_features': scalar_features
        }

def test_preprocessing():
    """Test preprocessing on sample data"""
    print("=== My Piano Performance Analysis - Preprocessing Test ===\n")
    
    # Initialize preprocessor
    preprocessor = PianoAudioPreprocessor()
    
    # Test on sample audio
    audio_path = Path('../data/Beethoven_WoO80_var27_8bars_3_15.wav')
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("Please copy the audio file to my_implementation/data/")
        return
    
    # Process
    result = preprocessor.process_audio_file(audio_path, audio_path.stem)
    
    print(f"\n📊 Results:")
    print(f"   Duration: {result['duration']:.2f}s")
    print(f"   Scalar features: {len(result['scalar_features'])}")
    
    # Show spectral feature shapes
    for name, data in result['spectral_features'].items():
        shape = np.array(data).shape
        print(f"   {name}: {shape}")
    
    # Save results
    output_path = Path('../results/preprocessing_test.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Test complete! Results saved to {output_path}")
    return result

if __name__ == "__main__":
    test_preprocessing()