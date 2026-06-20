"""Tests for the public practice-augmentation API (issue #76).

Room-IR convolution and practice-noise mixing already exist as private helpers
inside AudioAugmentor; #76 exposes them as deterministic public methods so the
practice-synthesis pipeline can compose MIDI corruption + acoustic degradation
end to end.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from model_improvement.augmentation import AudioAugmentor


SR = 16000


def _tone(seconds: float = 0.5, freq: float = 220.0) -> torch.Tensor:
    t = torch.arange(int(seconds * SR)) / SR
    return torch.sin(2 * np.pi * freq * t).unsqueeze(0).float()  # [1, T]


def test_room_ir_convolve_preserves_shape_synthetic():
    random.seed(0)
    np.random.seed(0)
    aug = AudioAugmentor(augment_prob=1.0)  # no ir_dir -> synthetic IR
    wav = _tone()
    out = aug.room_ir_convolve(wav, SR, p=1.0)
    assert out.shape == wav.shape
    assert torch.is_tensor(out)
    # Convolution changed the signal.
    assert not torch.allclose(out, wav)


def test_room_ir_convolve_p_zero_is_identity():
    random.seed(0)
    aug = AudioAugmentor(augment_prob=1.0)
    wav = _tone()
    out = aug.room_ir_convolve(wav, SR, p=0.0)
    assert torch.allclose(out, wav)


def test_add_practice_noise_lowers_snr():
    random.seed(1)
    np.random.seed(1)
    aug = AudioAugmentor(augment_prob=1.0)
    wav = _tone()
    out = aug.add_practice_noise(wav, SR, snr_range=(10.0, 10.0), p=1.0)
    assert out.shape == wav.shape
    # Noise added => residual is non-trivial.
    residual = (out - wav).abs().mean().item()
    assert residual > 0.0


def test_add_practice_noise_p_zero_is_identity():
    random.seed(1)
    aug = AudioAugmentor(augment_prob=1.0)
    wav = _tone()
    out = aug.add_practice_noise(wav, SR, p=0.0)
    assert torch.allclose(out, wav)
