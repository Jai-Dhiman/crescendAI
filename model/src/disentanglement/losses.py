"""Loss functions for disentanglement experiments.

Implements:
- InfoNCE with piece-based positives (same piece = positive)
- Margin-based pairwise ranking loss
- Triplet loss for performer discrimination
- Combined losses for multi-objective training
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def piece_based_infonce_loss(
    embeddings: torch.Tensor,
    piece_ids: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """InfoNCE loss where same-piece samples are positives.

    For each anchor, samples of the same piece are positives and
    samples of different pieces are negatives. This encourages the
    model to learn piece-invariant representations.

    Args:
        embeddings: Normalized embeddings [B, D].
        piece_ids: Integer piece IDs [B].
        temperature: Temperature for softmax scaling.
        eps: Small constant for numerical stability.

    Returns:
        Scalar InfoNCE loss.

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
        # No positive pairs - return zero loss (cannot compute contrastive)
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

        # Log-sum-exp over all negatives and the positive
        pos_sims = sim[i][pos_mask]
        neg_sims = sim[i][neg_mask]

        # For each positive, compute -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        for pos_sim in pos_sims:
            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            log_sum_exp = torch.logsumexp(all_sims, dim=0)
            loss = loss - (pos_sim - log_sum_exp)
            count += 1

    if count > 0:
        loss = loss / count

    return loss


def pairwise_margin_ranking_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.2,
    reduction: str = "mean",
) -> torch.Tensor:
    """Margin-based pairwise ranking loss.

    For each pair (A, B), predicts which performance is better and
    applies margin loss to enforce separation.

    Args:
        z_a: Embeddings for sample A [B, D].
        z_b: Embeddings for sample B [B, D].
        targets: Ranking targets {-1, 0, 1} [B, num_dims].
            1 = A > B, -1 = B > A, 0 = ambiguous (ignored).
        margin: Margin for ranking separation.
        reduction: "mean", "sum", or "none".

    Returns:
        Margin ranking loss.

    Reference:
        DirectRanker: https://arxiv.org/abs/1909.02768
    """
    # Compute difference in embeddings
    diff = z_a - z_b  # [B, D]

    # Score difference (using L2 norm as proxy for quality difference)
    score_diff = diff.norm(dim=-1)  # [B]

    # We want: if target=1, score_A > score_B + margin
    # Loss: max(0, margin - target * score_diff)
    # But we need per-dimension targets...

    # For simplicity, use mean target across dimensions
    mean_target = targets.float().mean(dim=-1)  # [B]

    # Ignore ambiguous pairs (target close to 0)
    valid_mask = mean_target.abs() > 0.3
    if not valid_mask.any():
        return torch.tensor(0.0, device=z_a.device, requires_grad=True)

    # Margin ranking loss
    loss = F.relu(margin - mean_target[valid_mask] * score_diff[valid_mask])

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DimensionWiseRankingLoss(nn.Module):
    """Per-dimension pairwise ranking loss.

    For each dimension, applies margin loss based on whether A or B
    is rated higher for that dimension.

    Handles ambiguous pairs (score difference < threshold) by ignoring them.
    """

    def __init__(
        self,
        margin: float = 0.3,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        """Initialize dimension-wise ranking loss.

        Args:
            margin: Margin for ranking separation.
            ambiguous_threshold: Score difference below which pairs are ambiguous.
            label_smoothing: Smooth targets toward 0.5 (reduces overconfidence).
        """
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
        """Compute dimension-wise ranking loss.

        Args:
            logits: Predicted ranking logits [B, num_dims] where >0 means A>B.
            labels_a: Ground truth labels for A [B, num_dims].
            labels_b: Ground truth labels for B [B, num_dims].

        Returns:
            Scalar loss.
        """
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


class ContrastiveRankingLoss(nn.Module):
    """Combined contrastive + ranking loss.

    L_total = L_ranking + lambda_contrastive * L_infonce

    The contrastive loss encourages piece-invariant representations,
    while the ranking loss enforces correct pairwise orderings.
    """

    def __init__(
        self,
        lambda_contrastive: float = 0.3,
        margin: float = 0.2,
        temperature: float = 0.07,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.temperature = temperature
        self.ranking_loss = DimensionWiseRankingLoss(
            margin=margin,
            ambiguous_threshold=ambiguous_threshold,
            label_smoothing=label_smoothing,
        )

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        ranking_logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        piece_ids_a: torch.Tensor,
        piece_ids_b: torch.Tensor,
    ) -> dict:
        """Compute combined loss.

        Args:
            z_a: Projected embeddings for sample A [B, D].
            z_b: Projected embeddings for sample B [B, D].
            ranking_logits: Predicted A>B logits [B, num_dims].
            labels_a: Ground truth for A [B, num_dims].
            labels_b: Ground truth for B [B, num_dims].
            piece_ids_a: Piece IDs for A [B].
            piece_ids_b: Piece IDs for B [B].

        Returns:
            Dict with total_loss, ranking_loss, contrastive_loss.
        """
        # Ranking loss
        l_rank = self.ranking_loss(ranking_logits, labels_a, labels_b)

        # Contrastive loss on combined embeddings
        # Stack A and B embeddings to form a larger batch
        all_z = torch.cat([z_a, z_b], dim=0)  # [2B, D]
        all_pieces = torch.cat([piece_ids_a, piece_ids_b], dim=0)  # [2B]

        l_contrast = piece_based_infonce_loss(
            all_z, all_pieces, temperature=self.temperature
        )

        total = l_rank + self.lambda_contrastive * l_contrast

        return {
            "total_loss": total,
            "ranking_loss": l_rank,
            "contrastive_loss": l_contrast,
        }


class DisentanglementLoss(nn.Module):
    """Loss for disentangled dual-encoder model.

    L_total = L_regression + lambda_adv * L_adversarial + lambda_style * L_contrastive

    - L_regression: MSE on dimension predictions
    - L_adversarial: Cross-entropy for piece classification (on style encoder)
    - L_contrastive: InfoNCE to cluster same-performer samples
    """

    def __init__(
        self,
        lambda_adversarial: float = 0.5,
        lambda_style_contrastive: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_adversarial = lambda_adversarial
        self.lambda_style_contrastive = lambda_style_contrastive
        self.temperature = temperature
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        piece_logits: torch.Tensor,
        piece_ids: torch.Tensor,
        style_embeddings: torch.Tensor,
    ) -> dict:
        """Compute disentanglement loss.

        Args:
            predictions: Predicted dimension scores [B, 19].
            labels: Ground truth labels [B, 19].
            piece_logits: Piece classification logits [B, num_pieces].
            piece_ids: Ground truth piece IDs [B].
            style_embeddings: Style encoder output [B, D].

        Returns:
            Dict with total_loss and component losses.
        """
        # Regression loss
        l_reg = self.mse(predictions, labels)

        # Adversarial piece classification loss
        # The GRL is applied before this, so minimizing this loss
        # actually maximizes piece classification error on style encoder
        l_adv = self.ce(piece_logits, piece_ids)

        # Style contrastive loss (optional)
        # Encourages same-performer representations to cluster
        l_style = torch.tensor(0.0, device=predictions.device)
        # Note: Would need performer IDs to implement this properly

        total = l_reg + self.lambda_adversarial * l_adv

        return {
            "total_loss": total,
            "regression_loss": l_reg,
            "adversarial_loss": l_adv,
            "style_contrastive_loss": l_style,
        }


class TripletPerformerLoss(nn.Module):
    """Triplet loss for same-piece performer discrimination.

    This loss encourages the model to learn quality differences within
    the same piece by using triplet sampling:

    - Anchor: A performance of piece P
    - Positive: A higher-quality performance of piece P
    - Negative: A lower-quality performance of piece P

    The loss encourages the anchor to be closer to the positive (better
    performance) than to the negative (worse performance).

    This differs from standard triplet loss in that:
    1. All three samples are from the SAME piece
    2. Positive/negative are determined by quality scores
    3. Forces model to learn performer variance, not piece similarity

    Reference:
        V7Labs triplet loss literature for fine-grained discrimination.
    """

    def __init__(
        self,
        margin: float = 0.5,
        distance_fn: str = "euclidean",
        swap: bool = False,
    ):
        """Initialize triplet loss.

        Args:
            margin: Minimum distance margin between positive and negative.
            distance_fn: Distance function ("euclidean" or "cosine").
            swap: Use the semi-hard negative mining swap trick.
        """
        super().__init__()
        self.margin = margin
        self.distance_fn = distance_fn
        self.swap = swap

    def _distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance."""
        if self.distance_fn == "cosine":
            # Cosine distance = 1 - cosine_similarity
            x_norm = F.normalize(x, dim=-1)
            y_norm = F.normalize(y, dim=-1)
            return 1 - (x_norm * y_norm).sum(dim=-1)
        else:
            # Euclidean distance
            return F.pairwise_distance(x, y)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """Compute triplet loss.

        Args:
            anchor: Anchor embeddings [B, D].
            positive: Positive (higher quality) embeddings [B, D].
            negative: Negative (lower quality) embeddings [B, D].

        Returns:
            Scalar triplet loss.
        """
        pos_dist = self._distance(anchor, positive)
        neg_dist = self._distance(anchor, negative)

        if self.swap:
            # Semi-hard mining: also consider dist(positive, negative)
            neg_dist_swap = self._distance(positive, negative)
            neg_dist = torch.min(neg_dist, neg_dist_swap)

        # Triplet margin loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()


class TripletRankingLoss(nn.Module):
    """Combined triplet loss with dimension-wise ranking.

    Combines TripletPerformerLoss with per-dimension ranking constraints.
    The triplet component ensures global quality ordering, while the
    ranking component ensures correct per-dimension comparisons.
    """

    def __init__(
        self,
        margin: float = 0.5,
        lambda_ranking: float = 0.5,
        ambiguous_threshold: float = 0.05,
    ):
        """Initialize combined triplet ranking loss.

        Args:
            margin: Margin for triplet loss.
            lambda_ranking: Weight for ranking loss component.
            ambiguous_threshold: Threshold for ambiguous pair filtering.
        """
        super().__init__()
        self.triplet_loss = TripletPerformerLoss(margin=margin)
        self.ranking_loss = DimensionWiseRankingLoss(
            ambiguous_threshold=ambiguous_threshold
        )
        self.lambda_ranking = lambda_ranking

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        ranking_logits: torch.Tensor,
        labels_anchor: torch.Tensor,
        labels_pos: torch.Tensor,
        labels_neg: torch.Tensor,
    ) -> dict:
        """Compute combined triplet and ranking loss.

        Args:
            anchor: Anchor embeddings [B, D].
            positive: Positive embeddings [B, D].
            negative: Negative embeddings [B, D].
            ranking_logits: Ranking logits for anchor vs negative [B, 19].
            labels_anchor: Labels for anchor [B, 19].
            labels_pos: Labels for positive [B, 19].
            labels_neg: Labels for negative [B, 19].

        Returns:
            Dict with total_loss, triplet_loss, ranking_loss.
        """
        l_triplet = self.triplet_loss(anchor, positive, negative)

        # Ranking loss: anchor vs negative (anchor should be closer to positive)
        l_ranking = self.ranking_loss(ranking_logits, labels_anchor, labels_neg)

        total = l_triplet + self.lambda_ranking * l_ranking

        return {
            "total_loss": total,
            "triplet_loss": l_triplet,
            "ranking_loss": l_ranking,
        }
