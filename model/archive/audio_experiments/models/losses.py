"""Loss functions for audio experiments.

Custom loss functions for multi-task learning and auxiliary objectives.
"""

import torch
import torch.nn.functional as F


def contrastive_auxiliary_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    similarity_threshold: float = 0.8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Contrastive loss that pulls together samples with similar labels.

    This auxiliary loss encourages the model to learn representations where
    performances with similar ratings are closer together in embedding space.

    For each sample pair (i, j):
    - Compute label similarity: sim_labels = 1 - mean(|labels_i - labels_j|)
    - If sim_labels > threshold: treat as positive pair
    - Use InfoNCE-style loss

    Args:
        embeddings: Normalized pooled representations [B, D].
        labels: Ground truth labels [B, 19] in range [0, 1].
        temperature: Temperature for softmax scaling.
        similarity_threshold: Threshold for considering pairs as positive.
        eps: Small constant for numerical stability.

    Returns:
        Scalar contrastive loss.
    """
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Compute pairwise label similarity [B, B]
    # L1 distance normalized to [0, 1], then convert to similarity
    label_diff = torch.cdist(labels, labels, p=1) / labels.size(1)  # [B, B]
    label_sim = 1.0 - label_diff  # Higher = more similar

    # Create positive mask (pairs with high label similarity)
    positive_mask = label_sim > similarity_threshold  # [B, B]
    # Remove diagonal (self-pairs)
    eye = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    positive_mask = positive_mask & ~eye

    # Compute embedding similarity [B, B]
    # embeddings should already be normalized, but ensure it
    embeddings = F.normalize(embeddings, dim=-1, eps=eps)
    emb_sim = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]

    # For numerical stability, subtract max
    emb_sim_stable = emb_sim - emb_sim.max(dim=-1, keepdim=True)[0].detach()

    # Compute InfoNCE-style loss for each sample
    # For each anchor, we want to maximize similarity with positives
    # and minimize similarity with negatives

    # Check if we have any positive pairs
    has_positives = positive_mask.any(dim=1)  # [B]

    if not has_positives.any():
        # No positive pairs in batch - return soft contrastive loss
        # Encourage similar samples to be closer using label similarity as soft target
        soft_target = label_sim.detach() / temperature
        soft_target = F.softmax(soft_target, dim=-1)
        log_pred = F.log_softmax(emb_sim, dim=-1)
        # KL divergence
        loss = F.kl_div(log_pred, soft_target, reduction="batchmean")
        return loss

    # Compute loss for samples with positive pairs
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(batch_size):
        if not has_positives[i]:
            continue

        pos_mask_i = positive_mask[i]  # [B]
        neg_mask_i = ~pos_mask_i & ~eye[i]  # [B]

        if not pos_mask_i.any() or not neg_mask_i.any():
            continue

        # Get positive and negative similarities
        pos_sims = emb_sim_stable[i][pos_mask_i]  # [num_pos]
        neg_sims = emb_sim_stable[i][neg_mask_i]  # [num_neg]

        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # For multiple positives, average over positive samples
        for pos_sim in pos_sims:
            # Denominator: exp(pos) + sum(exp(neg))
            all_neg = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            log_sum_exp = torch.logsumexp(all_neg, dim=0)
            loss = loss - (pos_sim - log_sum_exp)
            count += 1

    if count > 0:
        loss = loss / count

    return loss


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervised contrastive loss using continuous label similarity.

    Instead of binary positive/negative, uses label similarity as a
    soft weighting for the contrastive objective.

    Args:
        embeddings: Normalized pooled representations [B, D].
        labels: Ground truth labels [B, 19] in range [0, 1].
        temperature: Temperature for softmax scaling.
        eps: Small constant for numerical stability.

    Returns:
        Scalar contrastive loss.
    """
    batch_size = embeddings.size(0)

    if batch_size < 2:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1, eps=eps)

    # Compute pairwise embedding similarity [B, B]
    emb_sim = torch.matmul(embeddings, embeddings.T) / temperature

    # Compute pairwise label similarity [B, B]
    # Use cosine similarity on labels for consistency
    labels_norm = F.normalize(labels, dim=-1, eps=eps)
    label_sim = torch.matmul(labels_norm, labels_norm.T)  # [B, B]

    # Create mask to exclude diagonal
    mask = ~torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)

    # Apply mask
    emb_sim_masked = emb_sim.masked_fill(~mask, float("-inf"))

    # Soft targets: use label similarity as target distribution
    # Mask out diagonal from label similarity
    label_sim_masked = label_sim.masked_fill(~mask, 0)
    # Normalize to create probability distribution
    target_dist = F.softmax(label_sim_masked / temperature, dim=-1)

    # Log softmax of embedding similarities
    log_pred = F.log_softmax(emb_sim_masked, dim=-1)

    # KL divergence loss
    loss = F.kl_div(log_pred, target_dist, reduction="batchmean")

    return loss


def ranking_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Margin-based ranking loss for contrastive learning.

    Enforces that if label_sim(i,j) > label_sim(i,k), then
    emb_sim(i,j) > emb_sim(i,k) + margin.

    This is a triplet-style loss that preserves label similarity rankings
    in the embedding space.

    Args:
        embeddings: Normalized pooled representations [B, D].
        labels: Ground truth labels [B, 19] in range [0, 1].
        margin: Margin for ranking constraint.
        eps: Small constant for numerical stability.

    Returns:
        Scalar ranking loss.
    """
    batch_size = embeddings.size(0)

    if batch_size < 3:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=-1, eps=eps)

    # Compute pairwise similarities
    emb_sim = torch.matmul(embeddings, embeddings.T)  # [B, B]

    # Label similarity
    label_diff = torch.cdist(labels, labels, p=1) / labels.size(1)
    label_sim = 1.0 - label_diff

    # For each anchor i, compare pairs (i,j) and (i,k) where label_sim(i,j) > label_sim(i,k)
    # Loss: max(0, margin + emb_sim(i,k) - emb_sim(i,j))

    # Sample triplets efficiently
    loss = torch.tensor(0.0, device=embeddings.device)
    count = 0

    for i in range(batch_size):
        for j in range(batch_size):
            if i == j:
                continue
            for k in range(batch_size):
                if i == k or j == k:
                    continue
                # Check if j is more similar to i than k (based on labels)
                if label_sim[i, j] > label_sim[i, k] + 0.05:  # Small buffer
                    # Enforce embedding similarity ranking
                    triplet_loss = F.relu(margin + emb_sim[i, k] - emb_sim[i, j])
                    loss = loss + triplet_loss
                    count += 1

    if count > 0:
        loss = loss / count

    return loss
