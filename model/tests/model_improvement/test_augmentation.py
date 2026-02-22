import random

import torch
import pytest
from model_improvement.augmentation import AudioAugmentor


def test_augmentor_returns_same_shape():
    aug = AudioAugmentor(augment_prob=1.0)
    waveform = torch.randn(1, 24000)  # 1 second at 24kHz
    result = aug(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_augmentor_no_augmentation_when_prob_zero():
    aug = AudioAugmentor(augment_prob=0.0)
    waveform = torch.randn(1, 24000)
    result = aug(waveform, sample_rate=24000)
    assert torch.allclose(result, waveform)


def test_augmentor_always_augments_when_prob_one():
    aug = AudioAugmentor(augment_prob=1.0)
    waveform = torch.randn(1, 48000)
    # With augment_prob=1.0, at least one of the augmentations
    # (reverb, phone_sim, pitch_shift, eq, noise) should fire
    # across multiple random seeds.
    modified = False
    for seed in range(20):
        random.seed(seed)
        result = aug(waveform, sample_rate=24000)
        if not torch.allclose(result, waveform):
            modified = True
            break
    assert modified, "Augmentor with augment_prob=1.0 should modify waveform for at least one seed"


def test_augmentor_stereo():
    aug = AudioAugmentor(augment_prob=1.0)
    waveform = torch.randn(2, 24000)  # stereo
    random.seed(0)
    result = aug(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_augmentor_invalid_prob():
    with pytest.raises(ValueError, match="augment_prob must be in"):
        AudioAugmentor(augment_prob=1.5)
    with pytest.raises(ValueError, match="augment_prob must be in"):
        AudioAugmentor(augment_prob=-0.1)


def test_augmentor_rejects_1d_input():
    aug = AudioAugmentor(augment_prob=1.0)
    waveform = torch.randn(24000)  # 1D -- invalid
    with pytest.raises(ValueError, match="Expected waveform of shape"):
        aug(waveform, sample_rate=24000)
