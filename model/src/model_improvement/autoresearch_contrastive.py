"""Contrastive pretraining pipeline for MuQ and Aria encoders.

Symmetric quality-aware contrastive training using piece-based InfoNCE
and ordinal margin losses. Trains encoder + projection head pairs that
can be transferred to the downstream multi-task model (Phase C).

Architecture mirrors MuQLoRAModel exactly (attn, encoder, projection)
so that pretrained weights can be loaded directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MuQContrastiveEncoder(nn.Module):
    """MuQ contrastive encoder with attention pooling.

    Architecture matches MuQLoRAModel (attn, encoder, projection) exactly
    so Phase C can initialize from these pretrained weights.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Attention pooling over temporal frames
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Shared encoder (2-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-pool variable-length frames, then encode.

        Args:
            x: Frame embeddings [B, T, D].
            mask: Valid frame mask [B, T], True for real frames.

        Returns:
            Encoded representation [B, hidden_dim].
        """
        # Attention scores [B, T, 1]
        scores = self.attn(x)

        # Mask padded frames with -inf before softmax
        scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        pooled = (weights * x).sum(dim=1)  # [B, D]

        return self.encoder(pooled)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoded representation to contrastive space.

        Args:
            z: Encoded representation [B, hidden_dim].

        Returns:
            L2-normalized projection [B, projection_dim].
        """
        return F.normalize(self.projection(z), dim=-1)


class AriaContrastiveEncoder(nn.Module):
    """Aria contrastive encoder for fixed-dim symbolic embeddings.

    No attention pooling needed -- Aria produces a single vector per segment.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        projection_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Encoder (2-layer MLP)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Encode fixed-dim symbolic embedding.

        Args:
            x: Symbolic embeddings [B, D].
            mask: Ignored (accepted for API compatibility).

        Returns:
            Encoded representation [B, hidden_dim].
        """
        return self.encoder(x)

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project encoded representation to contrastive space.

        Args:
            z: Encoded representation [B, hidden_dim].

        Returns:
            L2-normalized projection [B, projection_dim].
        """
        return F.normalize(self.projection(z), dim=-1)
