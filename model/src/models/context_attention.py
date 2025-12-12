"""
Context Attention module for hierarchical aggregation.

Ported from PercePiano's virtuoso/module.py.
Implements multi-head attention with learned context vectors.
"""

import torch
import torch.nn as nn
from typing import Optional


class ContextAttention(nn.Module):
    """
    Multi-head context attention with learned context vectors.

    Used in hierarchical networks to aggregate lower-level representations
    (e.g., notes -> beats, beats -> measures) using attention.

    Each head has a learnable context vector that determines what to attend to.
    """

    def __init__(self, size: int, num_head: int):
        """
        Args:
            size: Input/output feature dimension
            num_head: Number of attention heads (must divide size evenly)
        """
        super().__init__()

        if size % num_head != 0:
            raise ValueError(f"size ({size}) must be divisible by num_head ({num_head})")

        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head
        self.head_size = size // num_head

        # Learnable context vector for each head
        self.context_vector = nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores without applying softmax.

        Args:
            x: Input tensor of shape (N, T, C)

        Returns:
            Attention scores of shape (N, T, num_head)
        """
        # Project to attention space
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)

        # Split into heads and compute similarity with context vectors
        attention_split = torch.stack(
            attention_tanh.split(split_size=self.head_size, dim=2), dim=0
        )  # (num_head, N, T, head_size)

        # Compute similarity with context vector
        similarity = torch.bmm(
            attention_split.view(self.num_head, -1, self.head_size),
            self.context_vector,
        )  # (num_head, N*T, 1)

        # Reshape to (N, T, num_head)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)

        return similarity

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply context attention and return weighted sum.

        Args:
            x: Input tensor of shape (N, T, C)
            mask: Optional mask of shape (N, T) where True indicates valid positions

        Returns:
            Weighted sum of shape (N, C)
        """
        # Project and apply tanh
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)

        if self.head_size != 1:
            # Split into heads
            attention_split = torch.stack(
                attention_tanh.split(split_size=self.head_size, dim=2), dim=0
            )  # (num_head, N, T, head_size)

            # Compute similarity with context vectors
            similarity = torch.bmm(
                attention_split.view(self.num_head, -1, self.head_size),
                self.context_vector,
            )  # (num_head, N*T, 1)

            # Reshape to (N, T, num_head)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1, 2, 0)

            # Mask out zero-padded positions
            is_zero_padded = x.sum(-1) == 0
            similarity[is_zero_padded] = -1e10

            # Apply optional explicit mask
            if mask is not None:
                similarity[~mask] = -1e10

            # Softmax over time dimension
            softmax_weight = torch.softmax(similarity, dim=1)

            # Split x into heads and weight
            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])

            # Merge heads back
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
        else:
            # Single head case
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        # Sum over time
        sum_attention = torch.sum(attention, dim=1)

        return sum_attention


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
