#!/usr/bin/env python3
import numpy as np
from pathlib import Path

from src.data.audio_io import mel_db_128x128


def test_mel_db_shape_and_range():
    # Generate ~3 seconds of 440 Hz sine at 22050 Hz
    sr = 22050
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False, dtype=np.float32)
    y = 0.5 * np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    S = mel_db_128x128(y, sr=sr)

    assert S.shape == (128, 128), f"Expected (128,128), got {S.shape}"
    assert S.dtype == np.float32, f"Expected float32, got {S.dtype}"
    # Values should be within [-80, 0]
    assert S.min() >= -80.0001, f"Min out of range: {S.min()}"
    assert S.max() <= 0.0001, f"Max out of range: {S.max()}"
    # Should not be all padding
    assert np.any(S > -80.0), "All values at padding level (-80 dB)"
