import torch
import pytest
from model_improvement.lora import apply_lora_to_muq, count_trainable_params, create_mock_encoder


def test_apply_lora_reduces_trainable_params():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    trainable_before = count_trainable_params(model)
    # Freeze all, then apply LoRA
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_muq(model, rank=16, target_layers=(2, 3))
    trainable_after = count_trainable_params(model)
    assert trainable_after < trainable_before
    assert trainable_after > 0


def test_lora_forward_preserves_output_shape():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    x = torch.randn(2, 50, 256)
    out_before = model(x)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_muq(model, rank=16, target_layers=(2, 3))
    out_after = model(x)
    assert out_before.shape == out_after.shape


def test_count_trainable_params():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    count = count_trainable_params(model)
    assert isinstance(count, int)
    assert count > 0
