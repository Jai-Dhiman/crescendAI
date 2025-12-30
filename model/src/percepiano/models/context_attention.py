"""
Context Attention modules for hierarchical aggregation.

Ported from PercePiano's virtuoso/module.py (line 348-383).

Two versions are provided:
1. ContextAttention - Multi-head attention for HIERARCHY processing (beat/measure aggregation)
   Used by make_higher_node() in hierarchy_utils.py. Matches original exactly.

2. FinalContextAttention - Simplified attention for PIECE-LEVEL aggregation
   Used for final aggregation before prediction head. Fixed architecture.
"""

from typing import Optional

import torch
import torch.nn as nn


class ContextAttention(nn.Module):
    """
    Multi-head context attention for HIERARCHY processing.

    Used by make_higher_node() to aggregate notes->beats and beats->measures.
    This matches the original PercePiano implementation exactly.

    Key design:
    - Projects input to same dimension with tanh: Linear(size, size) -> tanh
    - Splits into num_head heads, each of size head_size
    - Uses learnable context vectors for attention scoring
    - get_attention() returns [B, T, num_head] for use in make_higher_node
    """

    def __init__(self, size: int, num_head: int, temperature: float = 1.0):
        """
        Args:
            size: Input/output feature dimension
            num_head: Number of attention heads (must divide size evenly)
            temperature: Softmax temperature (< 1.0 sharpens attention, > 1.0 softens)
        """
        super().__init__()

        if size % num_head != 0:
            raise ValueError(
                f"size ({size}) must be divisible by num_head ({num_head})"
            )

        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head
        self.head_size = size // num_head
        self.temperature = temperature

        # Learnable context vector for each head
        self.context_vector = nn.Parameter(torch.Tensor(num_head, self.head_size, 1))

        # Initialize attention_net with Xavier for better gradient flow
        # Default kaiming_uniform produces too small weights (std ~0.04 for size=512)
        # causing near-uniform attention at initialization
        nn.init.xavier_uniform_(self.attention_net.weight)
        nn.init.zeros_(self.attention_net.bias)

        # Wider context vector initialization for sharper initial attention
        # Original uses uniform(-1, 1), we use (-2, 2) for stronger initial signal
        nn.init.uniform_(self.context_vector, a=-2, b=2)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores for use in make_higher_node.

        Args:
            x: Input tensor of shape (B, T, C)

        Returns:
            Attention scores of shape (B, T, num_head)
        """
        # Project to attention space and apply tanh (original PercePiano)
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)

        # Split into heads and compute similarity with context vectors
        attention_split = torch.stack(
            attention_tanh.split(split_size=self.head_size, dim=2), dim=0
        )  # (num_head, B, T, head_size)

        # Compute similarity with context vector
        similarity = torch.bmm(
            attention_split.view(self.num_head, -1, self.head_size),
            self.context_vector,
        )  # (num_head, B*T, 1)

        # Reshape to (B, T, num_head)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)

        return similarity

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, diagnose: bool = False
    ) -> torch.Tensor:
        """
        Apply context attention and return weighted sum.

        NOTE: This forward() is used for piece-level aggregation in the original,
        but has shape issues. Use FinalContextAttention for that instead.

        Args:
            x: Input tensor of shape (B, T, C)
            mask: Optional mask of shape (B, T) where True indicates valid positions
            diagnose: If True, print attention statistics

        Returns:
            Weighted sum of shape (B, C)
        """
        # Project and apply tanh
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)

        if diagnose:
            print(f"\n    [ContextAttention DIAGNOSE] (temperature={self.temperature})")
            print(f"      input x:        mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"      attention_tanh: mean={attention_tanh.mean():.4f}, std={attention_tanh.std():.4f}")
            print(f"      context_vector: mean={self.context_vector.mean():.4f}, std={self.context_vector.std():.4f}")

        # Split into heads
        attention_split = torch.stack(
            attention_tanh.split(split_size=self.head_size, dim=2), dim=0
        )  # (num_head, B, T, head_size)

        # Compute similarity with context vectors
        similarity = torch.bmm(
            attention_split.view(self.num_head, -1, self.head_size),
            self.context_vector,
        )  # (num_head, B*T, 1)

        # Reshape to (B, T, num_head)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)

        # Mask out zero-padded positions
        is_zero_padded = x.sum(-1) == 0
        similarity[is_zero_padded] = -1e10

        # Apply optional explicit mask
        if mask is not None:
            similarity[~mask] = -1e10

        # Softmax over time dimension (with optional temperature scaling)
        # Temperature < 1.0 sharpens attention, > 1.0 softens
        softmax_weight = torch.softmax(similarity / self.temperature, dim=1)

        if diagnose:
            # Check for uniform attention (indicates gradient vanishing)
            seq_len = x.shape[1]
            uniform_weight = 1.0 / seq_len
            max_weight = softmax_weight.max().item()
            min_weight = softmax_weight[softmax_weight > 0].min().item() if (softmax_weight > 0).any() else 0
            print(f"      similarity:     mean={similarity.mean():.4f}, std={similarity.std():.4f}")
            print(f"      softmax_weight: max={max_weight:.4f}, min={min_weight:.4f}, uniform={uniform_weight:.4f}")
            if max_weight < 2 * uniform_weight:
                print(f"      [WARN] Attention is near-uniform! max/uniform={max_weight/uniform_weight:.2f}x")
                print(f"      [WARN] This causes vanishing gradients through softmax!")

        # Split x into heads and weight
        x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(
            1, 1, 1, x_split.shape[-1]
        )

        # Merge heads back
        attention_out = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])

        # Sum over time
        sum_attention = torch.sum(attention_out, dim=1)

        return sum_attention


class FinalContextAttention(nn.Module):
    """
    Simplified context attention for PIECE-LEVEL aggregation.

    Used as the final attention layer before prediction head.
    Simpler architecture that avoids the shape issues in original ContextAttention.forward().

    Key design:
    - Projects input to num_head scores per position
    - Softmax over sequence dimension
    - Averages attention across heads for weighted sum
    """

    def __init__(self, size: int, num_head: int):
        """
        Args:
            size: Input feature dimension
            num_head: Number of attention heads
        """
        super().__init__()

        self.size = size
        self.num_head = num_head

        # Project to num_head attention scores per position
        self.attention_net = nn.Linear(size, num_head)

        # Softmax over sequence dimension
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply attention and return weighted sum.

        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional mask of shape (B, T) where True = valid position

        Returns:
            Weighted sum of shape (B, D)
        """
        # Compute attention scores: [B, T, num_head]
        attention = self.attention_net(x)

        # Mask out invalid positions before softmax
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(attention)
            attention = attention.masked_fill(~mask_expanded, -1e9)

        # Also mask zero-padded positions
        is_zero_padded = (x.abs().sum(dim=-1) < 1e-8)
        if is_zero_padded.any():
            zero_mask = is_zero_padded.unsqueeze(-1).expand_as(attention)
            attention = attention.masked_fill(zero_mask, -1e9)

        # Softmax over sequence dimension
        attention_weights = self.softmax(attention)  # [B, T, num_head]

        # Average attention weights across heads
        avg_attention = attention_weights.mean(dim=-1, keepdim=True)  # [B, T, 1]

        # Weighted sum over sequence
        weighted_x = x * avg_attention  # [B, T, D]
        output = weighted_x.sum(dim=1)  # [B, D]

        return output


class SimpleAttention(nn.Module):
    """
    Simple attention without multiple heads or context vectors.

    Projects input to attention scores and computes weighted sum.
    """

    def __init__(self, size: int):
        """
        Args:
            size: Input/output feature dimension
        """
        super().__init__()
        self.attention_net = nn.Linear(size, size)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw attention scores."""
        return self.attention_net(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention and return weighted sum.

        Args:
            x: Input tensor of shape (N, T, C)

        Returns:
            Weighted sum of shape (N, C)
        """
        attention = self.attention_net(x)
        softmax_weight = torch.softmax(attention, dim=1)
        attention = softmax_weight * x
        sum_attention = torch.sum(attention, dim=1)
        return sum_attention
