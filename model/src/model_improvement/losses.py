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


class ListMLELoss(nn.Module):
    """ListMLE ranking loss (Xia et al. 2008).

    Computes the negative log-likelihood of the ground-truth permutation
    under the Plackett-Luce model. Applied independently per dimension.

    For lists of length 1, returns 0. For length 2, equivalent to pairwise
    logistic loss.
    """

    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            predictions: Model scores [n_items, n_dims].
            labels: Ground-truth scores [n_items, n_dims].

        Returns:
            Scalar loss averaged across dimensions.
        """
        n_items, n_dims = predictions.shape

        if n_items <= 1:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=predictions.device)

        for d in range(n_dims):
            # Sort by ground-truth label descending
            sorted_indices = torch.argsort(labels[:, d], descending=True)
            sorted_preds = predictions[sorted_indices, d]

            # ListMLE: sum of (pred_i - log(sum(exp(pred_j) for j >= i)))
            # Use reverse cumulative logsumexp for numerical stability
            max_pred = sorted_preds.max().detach()
            shifted = sorted_preds - max_pred
            rev_cumsumexp = torch.logcumsumexp(shifted.flip(0), dim=0).flip(0)
            loss_d = -(shifted - rev_cumsumexp).sum()

            total_loss = total_loss + loss_d

        return total_loss / n_dims


def ccc_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Concordance Correlation Coefficient loss (1 - CCC).

    CCC measures agreement between predictions and targets, penalizing
    both correlation and mean/variance shift. CCC = 1 is perfect agreement.

    Args:
        predictions: Predicted scores, any shape (flattened internally).
        targets: Ground-truth scores, same shape as predictions.
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss in [0, 2]. 0 = perfect concordance, 2 = anti-concordance.
    """
    predictions = predictions.flatten()
    targets = targets.flatten()

    mean_pred = predictions.mean()
    mean_targ = targets.mean()
    var_pred = predictions.var(correction=0)
    var_targ = targets.var(correction=0)
    covar = ((predictions - mean_pred) * (targets - mean_targ)).mean()

    ccc = (2 * covar) / (var_pred + var_targ + (mean_pred - mean_targ) ** 2 + eps)

    return 1.0 - ccc


def ordinal_margin_loss(
    embeddings: torch.Tensor,
    piece_ids: torch.Tensor,
    quality_scores: torch.Tensor,
    margin_scale: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Ordinal margin loss using cross-piece anchors.

    For each ordered pair (better, worse) within the same piece,
    samples a random anchor from a different piece and enforces:
        sim(anchor, better) > sim(anchor, worse) + margin

    Args:
        embeddings: L2-normalized projected embeddings [B, D].
        piece_ids: Piece membership per segment [B].
        quality_scores: Normalized quality in [0,1], higher=better [B].
        margin_scale: Scales the quality gap into a similarity margin.
        eps: Small constant for numerical stability.

    Returns:
        Scalar loss (0 if no valid pairs exist).
    """
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Pairwise cosine similarity [B, B]
    sim = torch.matmul(embeddings, embeddings.T)

    # Masks
    same_piece = piece_ids.unsqueeze(0) == piece_ids.unsqueeze(1)
    diff_piece = ~same_piece

    if not diff_piece.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Ordered pairs: (i, j) where quality_i > quality_j, same piece
    quality_diff = quality_scores.unsqueeze(1) - quality_scores.unsqueeze(0)
    ordered_mask = same_piece & (quality_diff > eps)

    if not ordered_mask.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    pair_indices = ordered_mask.nonzero(as_tuple=False)  # [N, 2]

    # NOTE: Python for-loop over pairs is O(n^2) in same-piece segments.
    # Acceptable for batch_size=16. Vectorize if batch sizes increase.
    loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    count = 0

    for idx in range(pair_indices.size(0)):
        i, j = pair_indices[idx]
        anchor_mask = diff_piece[i]
        if not anchor_mask.any():
            continue

        anchor_indices = anchor_mask.nonzero(as_tuple=True)[0]
        k = anchor_indices[torch.randint(len(anchor_indices), (1,))]

        margin = margin_scale * (quality_scores[i] - quality_scores[j])
        sim_better = sim[k, i]
        sim_worse = sim[k, j]

        pair_loss = torch.clamp(margin - (sim_better - sim_worse), min=0.0)
        loss = loss + pair_loss.squeeze()
        count += 1

    if count > 0:
        loss = loss / count

    return loss


def semi_sup_con_loss(
    embeddings: torch.Tensor,
    piece_ids: torch.Tensor,
    ordinal_group_ids: torch.Tensor | None = None,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Semi-supervised InfoNCE with labeled positive groups.

    Extends piece_based_infonce_loss with additional labeled positives from
    T5 skill-level annotations. Positive pairs are the union of:
      - Same piece_id (existing MAESTRO / T2 round signal)
      - Same ordinal_group_id where ordinal_group_id >= 0 (T5 skill bucket)

    T1/T2 items should pass ordinal_group_ids=-1 (or omit the tensor).
    T5 items set ordinal_group_id to their skill bucket (0–4 after zero-indexing).

    Args:
        embeddings: L2-normalized projected embeddings [B, D].
        piece_ids: Piece membership per segment [B].
        ordinal_group_ids: T5 skill ordinal bucket per segment [B], or None.
            Use -1 for items without an ordinal label.
        temperature: InfoNCE softmax temperature.
        eps: Small constant for numerical stability.

    Returns:
        Scalar InfoNCE loss (0 if no anchor has any positive in its row).
    """
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    embeddings = F.normalize(embeddings, dim=-1, eps=eps)
    sim = torch.matmul(embeddings, embeddings.T) / temperature

    eye = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)

    # Piece-based positives (T1 / T2 / MAESTRO)
    piece_mask = piece_ids.unsqueeze(0) == piece_ids.unsqueeze(1)
    positive_mask = piece_mask & ~eye

    # Labeled positives from T5 ordinal buckets
    if ordinal_group_ids is not None:
        valid = ordinal_group_ids >= 0  # [B]
        # Same ordinal bucket, both items have a valid label, not self
        ordinal_same = ordinal_group_ids.unsqueeze(0) == ordinal_group_ids.unsqueeze(1)
        both_valid = valid.unsqueeze(0) & valid.unsqueeze(1)
        ordinal_positive = ordinal_same & both_valid & ~eye
        positive_mask = positive_mask | ordinal_positive

    has_positives = positive_mask.any(dim=1)

    if not has_positives.any():
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    sim_max = sim.max(dim=-1, keepdim=True)[0].detach()
    sim = sim - sim_max

    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(batch_size):
        if not has_positives[i]:
            continue

        pos_mask = positive_mask[i]
        # Negatives: anything that is not a positive and not self
        neg_mask = ~positive_mask[i] & ~eye[i]

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


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.

    Args:
        mu: Predicted means, any shape.
        sigma: Predicted stds (positive), same shape as mu.
        target: Ground-truth values, same shape as mu.

    Returns:
        Scalar mean NLL.
    """
    return F.gaussian_nll_loss(mu, target, sigma ** 2, eps=1e-6)


class DimensionWiseRankingLoss(nn.Module):
    """Per-dimension binary cross-entropy ranking loss with ambiguity filtering.

    For each quality dimension, predicts which of two performances is better.
    Pairs where the label difference is below ambiguous_threshold are excluded.

    When apply_to_tiers is set, only samples whose tier appears in that list
    contribute to the loss.  Pass tiers=batch["tiers"] in forward() to activate
    the filter.  Tier filtering happens before the ambiguity mask so T5 samples
    are fully excluded from the per-dim regression head while still contributing
    to ListMLE / SemiSupCon ranking losses.
    """

    def __init__(
        self,
        margin: float = 0.3,
        ambiguous_threshold: float = 0.05,
        label_smoothing: float = 0.0,
        apply_to_tiers: list[str] | None = None,
    ):
        super().__init__()
        self.margin = margin
        self.ambiguous_threshold = ambiguous_threshold
        self.label_smoothing = label_smoothing
        self.apply_to_tiers = apply_to_tiers

    def forward(
        self,
        logits: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor,
        tiers: list[str] | None = None,
    ) -> torch.Tensor:
        # Filter to allowed tiers before any other computation
        if self.apply_to_tiers is not None and tiers is not None:
            keep = torch.tensor(
                [t in self.apply_to_tiers for t in tiers],
                dtype=torch.bool,
                device=logits.device,
            )
            if not keep.any():
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            logits = logits[keep]
            labels_a = labels_a[keep]
            labels_b = labels_b[keep]

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
