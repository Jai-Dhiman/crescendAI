"""LoRA adapter integration for MuQ fine-tuning."""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def apply_lora_to_muq(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 32,
    target_layers: tuple[int, ...] = (9, 10, 11, 12),
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Apply LoRA adapters to self-attention layers of a transformer encoder in-place.

    Identifies q_proj and v_proj linear layers in the specified transformer
    layers and wraps them with low-rank adapters. Unfreezes only the LoRA
    parameters while keeping the base model frozen.

    Args:
        model: Transformer encoder model with named layers accessible
            via ``model.layers`` (e.g., MuQ or MockEncoder).
        rank: Rank of LoRA decomposition.
        alpha: LoRA alpha scaling factor.
        target_layers: Tuple of layer indices (0-based) to apply LoRA to.
        target_modules: Explicit list of module name patterns to target.
            If None, targets ``q_proj`` and ``v_proj`` in the specified layers.

    Returns:
        The model with LoRA adapters applied (modified in-place and returned).
    """
    if target_modules is None:
        target_modules = []
        for layer_idx in target_layers:
            target_modules.append(f"layers.{layer_idx}.self_attn.q_proj")
            target_modules.append(f"layers.{layer_idx}.self_attn.v_proj")

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    peft_model = get_peft_model(model, config)

    return peft_model


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _SelfAttention(nn.Module):
    """Minimal self-attention block for testing LoRA integration."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scale = q.shape[-1] ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class _TransformerLayer(nn.Module):
    """Minimal transformer layer for testing."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = _SelfAttention(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MockEncoder(nn.Module):
    """Small transformer encoder for unit tests.

    Mimics the layer structure of MuQ so that LoRA can target
    ``layers.{i}.self_attn.q_proj`` and ``layers.{i}.self_attn.v_proj``.
    """

    def __init__(self, hidden_size: int = 256, num_layers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_TransformerLayer(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def create_mock_encoder(
    hidden_size: int = 256, num_layers: int = 4
) -> MockEncoder:
    """Create a small transformer encoder for unit tests.

    Args:
        hidden_size: Hidden dimension of each layer.
        num_layers: Number of transformer layers.

    Returns:
        A MockEncoder instance with the same layer naming convention as MuQ.
    """
    return MockEncoder(hidden_size=hidden_size, num_layers=num_layers)
