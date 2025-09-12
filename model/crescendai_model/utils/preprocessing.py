#!/usr/bin/env python3
"""
Audio Preprocessing Helpers for Cloudflare Workers Integration
Lightweight preprocessing functions that can be called from Rust backend
"""

import numpy as np
import librosa
import json
from typing import Dict, List, Tuple, Optional, Union
import base64
import io

class CloudflareAudioPreprocessor:
    """
    Lightweight audio preprocessing optimized for Cloudflare Workers integration
    Handles basic audio processing before sending to Modal for inference
    """
    
    def __init__(self, 
                 target_sr: int = 22050,
                 n_mels: int = 128,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None):
        """
        Initialize preprocessing with AST-compatible parameters
        
        Args:
            target_sr: Target sample rate (22050 for efficiency)
            n_mels: Number of mel-frequency bins (128 for AST)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            fmin: Minimum frequency
            fmax: Maximum frequency (None = sr/2)
        """
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax or target_sr // 2
        
    def process_audio_bytes(self, audio_bytes: bytes, original_sr: Optional[int] = None) -> Dict[str, any]:
        """
        Process raw audio bytes into mel-spectrogram
        Main function for Cloudflare Workers integration
        
        Args:
            audio_bytes: Raw audio file bytes
            original_sr: Original sample rate (if known)
            
        Returns:
            Dict containing mel-spectrogram and metadata
        """
        try:
            # Load audio from bytes
            audio_array, sr = librosa.load(
                io.BytesIO(audio_bytes), 
                sr=self.target_sr,
                mono=True
            )
            
            # Validate audio
            if len(audio_array) == 0:
                raise ValueError("Empty audio file")
            
            # Generate mel-spectrogram
            mel_spec = self._generate_mel_spectrogram(audio_array, sr)
            
            # Prepare response
            return {
                "status": "success",
                "mel_spectrogram": mel_spec.tolist(),  # Convert to list for JSON serialization
                "metadata": {
                    "duration_seconds": len(audio_array) / sr,
                    "sample_rate": sr,
                    "shape": mel_spec.shape,
                    "n_frames": mel_spec.shape[0],
                    "n_mels": mel_spec.shape[1]
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def process_audio_file(self, file_path: str) -> Dict[str, any]:
        """
        Process audio file by path
        Useful for testing and direct file processing
        """
        try:
            with open(file_path, 'rb') as f:
                audio_bytes = f.read()
            return self.process_audio_bytes(audio_bytes)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def _generate_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Generate mel-spectrogram compatible with AST model
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Mel-spectrogram array [time, frequency]
        """
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0  # Power spectrum
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to [time, frequency] format expected by AST
        mel_spec_db = mel_spec_db.T
        
        # Normalize to [0, 1] range
        mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        return mel_spec_normalized


class ModalAPIClient:
    """
    Client for calling Modal API from Cloudflare Workers
    Handles HTTP communication with deployed Modal service
    """
    
    def __init__(self, modal_endpoint: str, api_key: Optional[str] = None):
        """
        Initialize Modal API client
        
        Args:
            modal_endpoint: Modal deployment endpoint URL
            api_key: Optional API key for authentication
        """
        self.modal_endpoint = modal_endpoint.rstrip('/')
        self.api_key = api_key
        
    def analyze_performance(self, 
                          mel_spectrogram: Union[np.ndarray, List[List[float]]], 
                          metadata: Optional[Dict] = None) -> Dict[str, any]:
        """
        Send mel-spectrogram to Modal for analysis
        
        Args:
            mel_spectrogram: Preprocessed mel-spectrogram
            metadata: Optional metadata
            
        Returns:
            Analysis results from Modal service
        """
        import requests
        
        try:
            # Prepare payload
            if isinstance(mel_spectrogram, np.ndarray):
                mel_data = mel_spectrogram.tolist()
            else:
                mel_data = mel_spectrogram
                
            payload = {
                "mel_spectrogram": mel_data,
                "metadata": metadata or {}
            }
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make request to Modal
            response = requests.post(
                f"{self.modal_endpoint}/analyze_piano_performance",
                json=payload,
                headers=headers,
                timeout=30  # 30s timeout for inference
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            return {
                "status": "error",
                "error": "Request timeout - Modal service took too long",
                "error_type": "TimeoutError"
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error", 
                "error": f"HTTP request failed: {str(e)}",
                "error_type": "RequestError"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }


def create_preprocessing_pipeline(target_sr: int = 22050) -> CloudflareAudioPreprocessor:
    """
    Factory function to create preprocessor with optimal settings
    
    Args:
        target_sr: Target sample rate
        
    Returns:
        Configured preprocessor instance
    """
    return CloudflareAudioPreprocessor(
        target_sr=target_sr,
        n_mels=128,  # AST standard
        n_fft=2048,  # Good balance of frequency resolution
        hop_length=512,  # 25% overlap
        fmin=0.0,
        fmax=target_sr // 2
    )


# Utility functions for Rust/WASM integration
def encode_mel_spectrogram(mel_spec: np.ndarray) -> str:
    """Encode mel-spectrogram as base64 for transport"""
    return base64.b64encode(mel_spec.tobytes()).decode('utf-8')


def decode_mel_spectrogram(encoded_data: str, shape: Tuple[int, int]) -> np.ndarray:
    """Decode base64 mel-spectrogram"""
    data = base64.b64decode(encoded_data.encode('utf-8'))
    return np.frombuffer(data, dtype=np.float32).reshape(shape)


def validate_audio_input(audio_bytes: bytes, max_duration: float = 180.0) -> Dict[str, any]:
    """
    Validate audio input before processing
    
    Args:
        audio_bytes: Raw audio bytes
        max_duration: Maximum allowed duration in seconds
        
    Returns:
        Validation result
    """
    try:
        # Quick audio load to check validity
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, duration=1.0)
        
        # Get full duration without loading entire file
        duration = librosa.get_duration(y=audio, sr=sr)
        
        if duration > max_duration:
            return {
                "valid": False,
                "error": f"Audio too long: {duration:.1f}s (max: {max_duration}s)"
            }
        
        if duration < 5.0:  # Minimum 5 seconds for meaningful analysis
            return {
                "valid": False,
                "error": f"Audio too short: {duration:.1f}s (min: 5.0s)"
            }
            
        return {
            "valid": True,
            "duration": duration,
            "sample_rate": sr
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": f"Invalid audio file: {str(e)}"
        }


if __name__ == "__main__":
    # Test preprocessing pipeline
    processor = create_preprocessing_pipeline()
    print("🎹 Audio preprocessing helpers ready for Cloudflare Workers integration")
    print(f"   Target SR: {processor.target_sr}Hz")
    print(f"   Mel bands: {processor.n_mels}")
    print(f"   FFT size: {processor.n_fft}")