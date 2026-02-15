import random

import torch
import pytest
from model_improvement.augmentation import AudioAugmentor


def test_augmentor_returns_same_length():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None)
    waveform = torch.randn(1, 24000)  # 1 second at 24kHz
    result = aug(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_augmentor_no_augmentation_when_prob_zero():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=0.0)
    waveform = torch.randn(1, 24000)
    result = aug(waveform, sample_rate=24000)
    assert torch.allclose(result, waveform)


def test_augmentor_always_augments_when_prob_one():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 48000)
    # Seed both Python random and torch to get deterministic augmentation.
    # With room_irs_dir=None and noise_dir=None, only phone_sim (p=0.2),
    # pitch_shift (p=0.1), and EQ (p=0.2) can fire. Run multiple attempts
    # to confirm that the pipeline can modify the waveform when active.
    modified = False
    for seed in range(20):
        random.seed(seed)
        torch.manual_seed(seed)
        result = aug(waveform, sample_rate=24000)
        if not torch.allclose(result, waveform):
            modified = True
            break
    assert modified, "Augmentor with augment_prob=1.0 should modify waveform for at least one seed"


def test_phone_simulation():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 24000)
    result = aug._apply_phone_simulation(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_eq_variation():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 24000)
    result = aug._apply_eq_variation(waveform, sample_rate=24000)
    assert result.shape == waveform.shape
