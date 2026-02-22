"""Loss functions for model improvement experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def piece_based_infonce_loss(
    embeddings: torch.Tensor,
    piece_ids: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """InfoNCE contrastive loss using piece membership as positive signal.

    Positive pairs: same piece, different sample.
    Negative pairs: different piece.

    Reference:
        https://lilianweng.github.io/posts/2021-05-31-contrastive/
    """
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1, eps=eps)

    # Compute pairwise similarities [B, B]
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Create positive mask: same piece, different sample
    piece_mask = piece_ids.unsqueeze(0) == piece_ids.unsqueeze(1)  # [B, B]
    eye = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    positive_mask = piece_mask & ~eye  # Same piece, not self

    # Check if we have any positive pairs
    has_positives = positive_mask.any(dim=1)  # [B]

    if not has_positives.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # For numerical stability
    sim_max = sim.max(dim=-1, keepdim=True)[0].detach()
    sim = sim - sim_max

    # Compute InfoNCE loss for samples with positives
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(batch_size):
        if not has_positives[i]:
            continue

        pos_mask = positive_mask[i]  # Positives for anchor i
        neg_mask = ~piece_mask[i] & ~eye[i]  # Different piece, not self

        if not pos_mask.any() or not neg_mask.any():
            continue

        pos_sims = sim[i][pos_mask]
        neg_sims = sim[i][neg_mask]

        for pos_sim in pos_sims:
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            log_sum_exp = torch.logsumexp(all_sims, dim=0)
            loss = loss - (pos_sim - log_sum_exp)
            count += 1

    if count > 0:
        loss = loss / count

    return loss


class DimensionWiseRankingLoss(nn.Module):
    """Per-dimension binary cross-entropy ranking loss with ambiguity filtering.

    For each quality dimension, predicts which of two performances is better.
    Pairs where the label difference is below ambiguous_threshold are excluded.
    """

    def __init__(
        self,
        margin: float = 0.3,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.margin = margin
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
    ) -> torch.Tensor:
        # Compute true ranking direction
        label_diff = labels_a - labels_b  # [B, D]

        # Create targets: 1 if A > B, 0 if B > A
        targets = (label_diff > 0).float()  # [B, D]

        # Mask for non-ambiguous pairs
        non_ambiguous = label_diff.abs() >= self.ambiguous_threshold

        if not non_ambiguous.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss with logits, only on non-ambiguous pairs
        loss = F.binary_cross_entropy_with_logits(
            logits[non_ambiguous],
            targets[non_ambiguous],
            reduction="mean",
        )

        return loss
