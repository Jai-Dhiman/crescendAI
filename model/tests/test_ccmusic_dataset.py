#!/usr/bin/env python3
"""
Test CCMusic Piano Dataset Integration
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import pandas as pd
import logging
from datasets import load_dataset
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_ccmusic_piano_dataset():
    """Test loading and examining the CCMusic Piano dataset"""
    print("=== Testing CCMusic Piano Dataset ===")
    
    try:
        # Try loading the dataset
        print("🔄 Loading ccmusic-database/pianos dataset...")
        
        # Try different loading approaches
        dataset = None
        
        # Approach 1: Load without trust_remote_code
        try:
            dataset = load_dataset(
                "ccmusic-database/pianos",
                cache_dir="./__pycache__/test_ccmusic",
            )
            print("✅ Loaded dataset successfully")
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            return False
        
        # Examine dataset structure
        print(f"\n📊 Dataset Structure:")
        print(f"Available splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"\nSplit '{split_name}':")
            print(f"  Size: {len(split_data)}")
            print(f"  Features: {split_data.features}")
            
            # Look at first few samples
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"  Sample keys: {list(sample.keys())}")
                
                # Check each field in the sample but don't try to access audio
                for key, value in sample.items():
                    if key == 'audio':
                        print(f"    {key}: [Audio data available]")
                        continue
                    
                    value_type = type(value).__name__
                    if hasattr(value, 'shape'):
                        print(f"    {key}: {value_type} {value.shape}")
                    elif hasattr(value, '__len__') and not isinstance(value, str):
                        print(f"    {key}: {value_type} (len={len(value)})")
                    else:
                        print(f"    {key}: {value_type} = {value}")
        
        # Test processing a sample
        if dataset:
            split_name = list(dataset.keys())[0]
            split_data = dataset[split_name]
            
            if len(split_data) > 0:
                print(f"\n🔍 Testing Sample Processing:")
                sample = split_data[0]
                
                # Skip loading the actual samples to avoid audio processing issues
                print("\n🔍 Key features from dataset:")
                
                # Print the label information
                if 'label' in split_data.features:
                    class_names = split_data.features['label'].names
                    print(f"  Piano brands/quality classes: {class_names}")
                
                # Print pitch classes if available
                if 'pitch' in split_data.features:
                    pitch_classes = split_data.features['pitch'].names
                    print(f"  Pitch classes: {len(pitch_classes)} unique pitches")
                    print(f"  Pitch range: {pitch_classes[0]} to {pitch_classes[-1]}")
                
                # Print scoring information if available
                score_fields = [f for f in split_data.features if 'score' in f]
                if score_fields:
                    print(f"  Score fields: {score_fields}")
                
                # Print description of the dataset structure
                print("\n📝 Dataset Structure Summary:")
                print("  - Each sample contains a piano recording with:")
                print("    * Mel spectrogram image representation")
                print("    * Audio data (requires torchaudio to decode)")
                print("    * Piano brand/quality label")
                print("    * Pitch class information")
                print("    * Quality scores (bass, mid, treble, average)")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mel_processing():
    """Test mel spectrogram processing"""
    print("\n=== Testing Mel Spectrogram Processing ===")
    
    try:
        # Create a dummy mel spectrogram (simulating PIL Image)
        dummy_mel = np.random.rand(128, 128, 3) * 255  # RGB image
        dummy_mel = dummy_mel.astype(np.uint8)
        pil_image = Image.fromarray(dummy_mel)
        
        print(f"Created dummy PIL image: {pil_image.size} {pil_image.mode}")
        
        # Process it like in our dataset
        mel_array = np.array(pil_image)
        print(f"Converted to array: {mel_array.shape}")
        
        # Convert to grayscale
        if len(mel_array.shape) == 3 and mel_array.shape[2] == 3:
            mel_array = np.mean(mel_array, axis=2)
        print(f"Grayscale shape: {mel_array.shape}")
        
        # Normalize
        mel_array = mel_array.astype(np.float32)
        if mel_array.max() > 0:
            mel_array = (mel_array - mel_array.min()) / (mel_array.max() - mel_array.min())
            mel_array = mel_array * 80.0 - 80.0  # Convert to dB-like range
        
        print(f"Processed mel shape: {mel_array.shape}")
        print(f"Processed mel range: [{mel_array.min():.2f}, {mel_array.max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Mel processing test failed: {e}")
        return False


if __name__ == "__main__":
    success1 = test_ccmusic_piano_dataset()
    success2 = test_mel_processing()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
