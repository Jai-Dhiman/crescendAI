# A1-Max Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stack all Tier 1 improvements on A1 (ListMLE, CCC, hard negative mining, mixup, label smoothing, LoRA ablation, loss weight tuning) into one sweep, run 72 training configs on local MPS, deploy best 4-fold ensemble to HF.

**Architecture:** Extend existing A1 training pipeline with new loss functions (ListMLE, CCC), a hard-negative-aware pair sampler, and embedding mixup. A new `MuQLoRAMaxModel` subclass overrides `training_step` to use the new losses. A sweep runner script iterates 18 configs x 4 folds sequentially with MPS memory management.

**Tech Stack:** PyTorch Lightning 2.x, MuQ embeddings (pre-extracted), MPS backend, existing pipeline from `model/src/model_improvement/`

**Spec:** `docs/superpowers/specs/2026-03-10-a1-max-training-design.md`

**Deviations from spec:**
- **Loss weights fixed, not swept.** The spec mentions "loss weight tuning" but the weights (1.5/0.3/0.3/0.1) are fixed in `BASE_CONFIG` by design. Sweeping loss weights multiplicatively with LoRA/label_smoothing would explode the grid. If the fixed weights underperform, a follow-up sweep can tune them.
- **Batch size 4 (effective 16).** Spec says "batch size 32, fall back to 16." MPS memory limits us to batch_size=4 with accumulate_grad_batches=4 = effective 16. This is the practical maximum on a 32GB Mac for 1024-dim variable-length embeddings.
- **ListMLE works on 2-item lists.** Spec says "falls back to pairwise BCE for pieces with 2 performances." ListMLE is mathematically valid for n=2 (equivalent to pairwise logistic loss), so no fallback is needed.
- **Web app 6-dim update is out of scope.** The web app observation pipeline change is a separate task after the model ships.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `model/src/model_improvement/losses.py` | Modify | Add `ListMLELoss`, `ccc_loss` |
| `model/src/model_improvement/data.py` | Modify | Add `HardNegativePairSampler`, `apply_mixup` |
| `model/src/model_improvement/audio_encoders.py` | Modify | Add `MuQLoRAMaxModel` subclass |
| `model/src/model_improvement/a1_max_sweep.py` | Create | Sweep runner script |
| `model/tests/model_improvement/test_losses.py` | Create | Tests for ListMLE, CCC |
| `model/tests/model_improvement/test_a1_max.py` | Create | Tests for MuQLoRAMaxModel, HardNegativePairSampler, mixup |
| `apps/inference/handler.py` | Modify | 6-dim ensemble output |

---

## Chunk 1: Loss Functions

### Task 1: ListMLE Loss

ListMLE computes the negative log-likelihood of the ground-truth permutation under the Plackett-Luce model. Applied per-dimension on regression scores grouped by piece within a batch.

**Files:**
- Create: `model/tests/model_improvement/test_losses.py`
- Modify: `model/src/model_improvement/losses.py`

- [ ] **Step 1: Write failing test for ListMLE**

```python
# model/tests/model_improvement/test_losses.py
import torch
import pytest
from model_improvement.losses import ListMLELoss


class TestListMLELoss:
    def test_perfect_ranking_low_loss(self):
        """Perfectly ordered predictions should yield lower loss than reversed."""
        loss_fn = ListMLELoss()
        # 5 items, 1 dimension, perfectly ordered predictions match labels
        predictions = torch.tensor([[0.9], [0.7], [0.5], [0.3], [0.1]])
        labels = torch.tensor([[0.9], [0.7], [0.5], [0.3], [0.1]])
        perfect_loss = loss_fn(predictions, labels)

        # Reversed predictions
        reversed_preds = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
        reversed_loss = loss_fn(reversed_preds, labels)

        assert perfect_loss < reversed_loss

    def test_single_item_returns_zero(self):
        """ListMLE with a single item should return 0."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.5]])
        labels = torch.tensor([[0.5]])
        loss = loss_fn(predictions, labels)
        assert loss.item() == 0.0

    def test_two_items_nonzero(self):
        """Two items with different labels should produce nonzero loss."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.3], [0.7]])
        labels = torch.tensor([[0.8], [0.2]])  # Wrong order
        loss = loss_fn(predictions, labels)
        assert loss.item() > 0.0

    def test_multi_dimension(self):
        """Should work independently per dimension."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        labels = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        loss = loss_fn(predictions, labels)
        assert loss.ndim == 0

    def test_gradient_flows(self):
        """Loss must produce gradients."""
        loss_fn = ListMLELoss()
        predictions = torch.tensor([[0.5], [0.3]], requires_grad=True)
        labels = torch.tensor([[0.8], [0.2]])
        loss = loss_fn(predictions, labels)
        loss.backward()
        assert predictions.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_losses.py::TestListMLELoss -v`
Expected: FAIL with ImportError (ListMLELoss not defined)

- [ ] **Step 3: Implement ListMLE loss**

Add to `model/src/model_improvement/losses.py` after the `DimensionWiseRankingLoss` class:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_losses.py::TestListMLELoss -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/losses.py model/tests/model_improvement/test_losses.py && git commit -m "feat: add ListMLE ranking loss"
```

---

### Task 2: CCC Loss

Concordance Correlation Coefficient loss penalizes both scale and shift errors, better suited for subjective quality scores than MSE.

**Files:**
- Modify: `model/tests/model_improvement/test_losses.py`
- Modify: `model/src/model_improvement/losses.py`

- [ ] **Step 1: Write failing test for CCC loss**

Add to `model/tests/model_improvement/test_losses.py`:

```python
from model_improvement.losses import ccc_loss


class TestCCCLoss:
    def test_perfect_prediction_zero_loss(self):
        """Identical predictions and targets should yield loss near 0."""
        preds = torch.tensor([0.1, 0.5, 0.9])
        targets = torch.tensor([0.1, 0.5, 0.9])
        loss = ccc_loss(preds, targets)
        assert loss.item() < 0.01

    def test_opposite_high_loss(self):
        """Negatively correlated should yield high loss."""
        preds = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        loss = ccc_loss(preds, targets)
        assert loss.item() > 1.0

    def test_shifted_predictions_penalized(self):
        """Constant shift should be penalized (unlike Pearson)."""
        preds = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        targets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        loss = ccc_loss(preds, targets)
        assert loss.item() > 0.1

    def test_gradient_flows(self):
        preds = torch.tensor([0.3, 0.5, 0.7], requires_grad=True)
        targets = torch.tensor([0.4, 0.6, 0.8])
        loss = ccc_loss(preds, targets)
        loss.backward()
        assert preds.grad is not None

    def test_constant_predictions_high_loss(self):
        """All-same predictions should yield high loss."""
        preds = torch.tensor([0.5, 0.5, 0.5, 0.5])
        targets = torch.tensor([0.1, 0.3, 0.7, 0.9])
        loss = ccc_loss(preds, targets)
        assert loss.item() > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_losses.py::TestCCCLoss -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement CCC loss**

Add to `model/src/model_improvement/losses.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_losses.py::TestCCCLoss -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/losses.py model/tests/model_improvement/test_losses.py && git commit -m "feat: add CCC regression loss"
```

---

## Chunk 2: Data Pipeline Enhancements

### Task 3: HardNegativePairSampler

Wraps `PairedPerformanceDataset` to oversample hard pairs (where model is uncertain or wrong) after a warmup period. Uses curriculum: easy pairs first, progressively adds harder pairs.

**Files:**
- Create: `model/tests/model_improvement/test_a1_max.py`
- Modify: `model/src/model_improvement/data.py`

- [ ] **Step 1: Write failing test**

```python
# model/tests/model_improvement/test_a1_max.py
import torch
import pytest
import numpy as np
from pathlib import Path
from model_improvement.data import HardNegativePairSampler, PairedPerformanceDataset


class TestHardNegativePairSampler:
    def _make_dataset(self):
        """Create a minimal PairedPerformanceDataset for testing."""
        labels = {
            "piece1_1": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "piece1_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "piece1_3": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "piece2_1": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "piece2_2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
        piece_to_keys = {
            "piece1": ["piece1_1", "piece1_2", "piece1_3"],
            "piece2": ["piece2_1", "piece2_2"],
        }
        keys = list(labels.keys())
        ds = PairedPerformanceDataset(
            cache_dir=Path("/tmp"), labels=labels,
            piece_to_keys=piece_to_keys, keys=keys,
        )
        return ds, labels

    def test_warmup_returns_all_pairs(self):
        """During warmup, should sample uniformly from all pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=5, easy_threshold=0.3,
        )
        indices = sampler.get_indices(epoch=2)
        assert len(indices) == len(ds)

    def test_post_warmup_filters_easy(self):
        """After warmup with curriculum, should filter some pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=5, easy_threshold=0.3,
        )
        indices = sampler.get_indices(epoch=10)
        assert len(indices) <= len(ds)
        assert len(indices) > 0

    def test_curriculum_progression(self):
        """Later epochs should include harder pairs."""
        ds, labels = self._make_dataset()
        sampler = HardNegativePairSampler(
            dataset=ds, warmup_epochs=2, easy_threshold=0.3,
        )
        early = sampler.get_indices(epoch=3)
        late = sampler.get_indices(epoch=50)
        assert len(late) >= len(early)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestHardNegativePairSampler -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement HardNegativePairSampler**

Add `import numpy as np` at the top of `model/src/model_improvement/data.py` (after existing imports), then add after `PairedPerformanceDataset` (line ~263):

```python
import numpy as np  # Add this at top of file if not already present


class HardNegativePairSampler:
    """Curriculum-based pair sampler that progressively adds harder pairs.

    During warmup, returns all pairs uniformly. After warmup, filters by
    label difficulty: starts with easy pairs (large label gap), progressively
    adds harder pairs (smaller gap) as training continues.

    Optionally accepts model prediction errors to oversample pairs where the
    model is wrong or uncertain.
    """

    def __init__(
        self,
        dataset: PairedPerformanceDataset,
        warmup_epochs: int = 5,
        easy_threshold: float = 0.3,
        hard_oversample: float = 2.0,
    ):
        self.dataset = dataset
        self.warmup_epochs = warmup_epochs
        self.easy_threshold = easy_threshold
        self.hard_oversample = hard_oversample

        # Pre-compute mean absolute label differences per pair
        self.pair_diffs = []
        for key_a, key_b, _pid in dataset.pairs:
            labels_a = np.array(dataset.labels[key_a])
            labels_b = np.array(dataset.labels[key_b])
            mean_diff = float(np.abs(labels_a - labels_b).mean())
            self.pair_diffs.append(mean_diff)
        self.pair_diffs = np.array(self.pair_diffs)

        self._model_errors: np.ndarray | None = None

    def update_model_errors(self, errors: np.ndarray) -> None:
        """Update pair-level model errors for hard negative mining.

        Args:
            errors: Array of shape (n_pairs,) with error magnitude per pair.
                Higher values = model struggled more on this pair.
        """
        self._model_errors = errors

    def get_indices(self, epoch: int) -> list[int]:
        """Get pair indices for the given epoch.

        During warmup: all indices shuffled.
        After warmup: curriculum + optional hard negative oversampling.
        """
        n = len(self.dataset)
        all_indices = list(range(n))

        if epoch < self.warmup_epochs:
            return all_indices

        # Curriculum: threshold decreases over time
        # At warmup_epochs: only easy pairs (diff > easy_threshold)
        # By epoch 50+: all pairs included
        progress = min(1.0, (epoch - self.warmup_epochs) / 45.0)
        min_diff = self.easy_threshold * (1.0 - progress)

        # Filter by difficulty
        mask = self.pair_diffs >= min_diff
        indices = [i for i in all_indices if mask[i]]

        # Oversample hard pairs if model errors available
        if self._model_errors is not None and len(indices) > 0:
            error_threshold = np.percentile(self._model_errors, 75)
            hard_indices = [
                i for i in indices
                if self._model_errors[i] > error_threshold
            ]
            oversample_count = int(len(hard_indices) * (self.hard_oversample - 1))
            if oversample_count > 0 and hard_indices:
                rng = np.random.default_rng(epoch)
                extra = rng.choice(hard_indices, size=oversample_count, replace=True)
                indices.extend(extra.tolist())

        return indices if indices else all_indices
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestHardNegativePairSampler -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/data.py model/tests/model_improvement/test_a1_max.py && git commit -m "feat: add HardNegativePairSampler with curriculum"
```

---

### Task 4: Mixup Utility

Embedding-level mixup: interpolate between two embeddings and their labels using Beta distribution.

**Files:**
- Modify: `model/tests/model_improvement/test_a1_max.py`
- Modify: `model/src/model_improvement/data.py`

- [ ] **Step 1: Write failing test**

Add to `model/tests/model_improvement/test_a1_max.py`:

```python
from model_improvement.data import apply_mixup


class TestMixup:
    def test_output_shapes_match_input(self):
        emb = torch.randn(4, 50, 1024)
        labels = torch.rand(4, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.2)
        assert mixed_emb.shape == emb.shape
        assert mixed_labels.shape == labels.shape

    def test_alpha_zero_returns_original(self):
        """Alpha=0 means no mixup, output equals input."""
        emb = torch.randn(4, 50, 1024)
        labels = torch.rand(4, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.0)
        assert torch.allclose(mixed_emb, emb)
        assert torch.allclose(mixed_labels, labels)

    def test_mixup_changes_values(self):
        """With alpha > 0, output should differ from input."""
        torch.manual_seed(42)
        emb = torch.randn(8, 50, 1024)
        labels = torch.rand(8, 6)
        mixed_emb, mixed_labels = apply_mixup(emb, labels, alpha=0.4)
        assert not torch.allclose(mixed_emb, emb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestMixup -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement mixup**

Add to `model/src/model_improvement/data.py`:

```python
def apply_mixup(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation on embeddings and labels.

    Randomly shuffles the batch and interpolates:
        mixed = lambda * original + (1 - lambda) * shuffled
    where lambda ~ Beta(alpha, alpha).

    Args:
        embeddings: [B, T, D] frame embeddings.
        labels: [B, num_dims] label scores.
        alpha: Beta distribution parameter. 0 = no mixup.

    Returns:
        (mixed_embeddings, mixed_labels) with same shapes as inputs.
    """
    if alpha <= 0.0:
        return embeddings, labels

    batch_size = embeddings.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # Ensure lambda >= 0.5 so the "original" sample dominates
    lam = max(lam, 1.0 - lam)

    perm = torch.randperm(batch_size, device=embeddings.device)

    mixed_emb = lam * embeddings + (1.0 - lam) * embeddings[perm]
    mixed_labels = lam * labels + (1.0 - lam) * labels[perm]

    return mixed_emb, mixed_labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestMixup -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/data.py model/tests/model_improvement/test_a1_max.py && git commit -m "feat: add embedding mixup utility"
```

---

## Chunk 3: A1-Max Model

### Task 5: MuQLoRAMaxModel

New model class extending `MuQLoRAModel` with all Tier 1 improvements: ListMLE ranking, CCC regression, mixup, configurable loss weights. Keeps the parent's architecture but overrides `training_step`.

**Files:**
- Modify: `model/tests/model_improvement/test_a1_max.py`
- Modify: `model/src/model_improvement/audio_encoders.py`

- [ ] **Step 1: Write failing tests**

Add to `model/tests/model_improvement/test_a1_max.py`:

```python
from model_improvement.audio_encoders import MuQLoRAMaxModel


class TestMuQLoRAMaxModel:
    def _make_batch(self, batch_size=4, seq_len=50, input_dim=1024, num_dims=6):
        return {
            "embeddings_a": torch.randn(batch_size, seq_len, input_dim),
            "embeddings_b": torch.randn(batch_size, seq_len, input_dim),
            "mask_a": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "mask_b": torch.ones(batch_size, seq_len, dtype=torch.bool),
            "labels_a": torch.rand(batch_size, num_dims),
            "labels_b": torch.rand(batch_size, num_dims),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }

    def test_training_step_returns_loss(self):
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False,
        )
        batch = self._make_batch()
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_listmle_loss_included(self):
        """With lambda_listmle > 0, ListMLE should contribute to loss."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False, lambda_listmle=1.5,
        )
        logged = {}
        model.log = lambda name, value, **kw: logged.update({name: value})
        batch = self._make_batch()
        model.training_step(batch, 0)
        assert "train_listmle_loss" in logged

    def test_ccc_loss_included(self):
        """With use_ccc=True, CCC loss should replace MSE."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False, use_ccc=True,
        )
        logged = {}
        model.log = lambda name, value, **kw: logged.update({name: value})
        batch = self._make_batch()
        model.training_step(batch, 0)
        assert "train_ccc_loss" in logged

    def test_mixup_changes_loss(self):
        """With mixup_alpha > 0, loss should differ from no-mixup."""
        torch.manual_seed(42)
        model_no_mix = MuQLoRAMaxModel(
            input_dim=256, hidden_dim=128, num_labels=6,
            use_pretrained_muq=False, mixup_alpha=0.0,
        )
        torch.manual_seed(42)
        model_mix = MuQLoRAMaxModel(
            input_dim=256, hidden_dim=128, num_labels=6,
            use_pretrained_muq=False, mixup_alpha=0.4,
        )
        model_mix.load_state_dict(model_no_mix.state_dict())

        batch = self._make_batch(input_dim=256)
        torch.manual_seed(0)
        loss_no_mix = model_no_mix.training_step(batch, 0)
        torch.manual_seed(0)
        loss_mix = model_mix.training_step(batch, 0)
        assert not torch.isclose(loss_no_mix, loss_mix)

    def test_forward_compatible_with_parent(self):
        """forward() and predict_scores() should work identically."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, num_labels=6,
            use_pretrained_muq=False,
        )
        x_a = torch.randn(2, 50, 1024)
        x_b = torch.randn(2, 50, 1024)
        out = model(x_a, x_b)
        assert out["ranking_logits"].shape == (2, 6)

        scores = model.predict_scores(x_a)
        assert scores.shape == (2, 6)

    def test_default_params_match_spec(self):
        """Default loss weights should match spec."""
        model = MuQLoRAMaxModel(
            input_dim=1024, hidden_dim=512, use_pretrained_muq=False,
        )
        assert model.hparams.lambda_listmle == 1.5
        assert model.hparams.lambda_contrastive == 0.3
        assert model.hparams.lambda_regression == 0.3
        assert model.hparams.lambda_invariance == 0.1
        assert model.hparams.use_ccc is True
        assert model.hparams.mixup_alpha == 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestMuQLoRAMaxModel -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement MuQLoRAMaxModel**

Add to `model/src/model_improvement/audio_encoders.py` after the `MuQLoRAModel` class (after line 304):

```python
class MuQLoRAMaxModel(MuQLoRAModel):
    """A1-Max: MuQ + LoRA with all Tier 1 improvements.

    Extends MuQLoRAModel with:
    - ListMLE ranking loss (per-dimension, grouped by piece within batch)
    - CCC regression loss (replaces MSE)
    - Embedding mixup (Beta distribution)
    - Configurable loss weights (ranking-dominant by default)

    Architecture is identical to MuQLoRAModel. Only training_step differs.
    """

    def __init__(
        self,
        *,
        lambda_listmle: float = 1.5,
        use_ccc: bool = True,
        mixup_alpha: float = 0.2,
        lambda_contrastive: float = 0.3,
        lambda_regression: float = 0.3,
        lambda_invariance: float = 0.1,
        **kwargs,
    ):
        # NOTE: Do NOT call self.save_hyperparameters() here -- parent
        # already calls it, and a second call overwrites parent hparams.
        super().__init__(
            lambda_contrastive=lambda_contrastive,
            lambda_regression=lambda_regression,
            lambda_invariance=lambda_invariance,
            **kwargs,
        )
        # Manually store extra hparams that parent doesn't know about
        self.hparams.update({
            "lambda_listmle": lambda_listmle,
            "use_ccc": use_ccc,
            "mixup_alpha": mixup_alpha,
        })
        self.lambda_listmle = lambda_listmle
        self.use_ccc = use_ccc
        self.mixup_alpha = mixup_alpha

        from model_improvement.losses import ListMLELoss, ccc_loss
        self._listmle = ListMLELoss()
        self._ccc_loss_fn = ccc_loss

    def training_step(self, batch: dict, idx: int) -> torch.Tensor:
        emb_a = batch["embeddings_a"]
        emb_b = batch["embeddings_b"]
        labels_a = batch["labels_a"]
        labels_b = batch["labels_b"]

        # Apply mixup on embeddings + labels (before encoding)
        if self.mixup_alpha > 0 and self.training:
            from model_improvement.data import apply_mixup
            emb_a, labels_a = apply_mixup(emb_a, labels_a, self.mixup_alpha)
            emb_b, labels_b = apply_mixup(emb_b, labels_b, self.mixup_alpha)

        outputs = self(
            emb_a, emb_b,
            batch.get("mask_a"), batch.get("mask_b"),
        )

        # 1. Pairwise ranking loss (BCE, from parent)
        l_rank = self.ranking_loss(
            outputs["ranking_logits"], labels_a, labels_b,
        )

        # 2. Contrastive loss
        all_proj = torch.cat([outputs["proj_a"], outputs["proj_b"]], dim=0)
        all_pieces = torch.cat([batch["piece_ids_a"], batch["piece_ids_b"]], dim=0)
        l_contrast = piece_based_infonce_loss(
            all_proj, all_pieces, temperature=self.temperature
        )

        # 3. Regression loss (CCC or MSE)
        scores_a = self.regression_head(outputs["z_a"])
        scores_b = self.regression_head(outputs["z_b"])
        if self.use_ccc:
            l_reg = (
                self._ccc_loss_fn(scores_a, labels_a)
                + self._ccc_loss_fn(scores_b, labels_b)
            ) / 2.0
        else:
            l_reg = (
                F.mse_loss(scores_a, labels_a)
                + F.mse_loss(scores_b, labels_b)
            ) / 2.0

        # 4. ListMLE on regression scores grouped by piece
        l_listmle = torch.tensor(0.0, device=self.device)
        if self.lambda_listmle > 0:
            all_scores = torch.cat([scores_a, scores_b], dim=0)
            all_labels = torch.cat([labels_a, labels_b], dim=0)
            piece_ids = torch.cat(
                [batch["piece_ids_a"], batch["piece_ids_b"]], dim=0
            )

            unique_pieces = piece_ids.unique()
            listmle_count = 0
            for pid in unique_pieces:
                pmask = piece_ids == pid
                if pmask.sum() < 2:
                    continue
                l_listmle = l_listmle + self._listmle(
                    all_scores[pmask], all_labels[pmask]
                )
                listmle_count += 1
            if listmle_count > 0:
                l_listmle = l_listmle / listmle_count

        # 5. Augmentation invariance loss
        l_inv = torch.tensor(0.0, device=self.device)
        if "embeddings_aug_a" in batch:
            z_aug_a = self.encode(
                batch["embeddings_aug_a"], batch.get("mask_a")
            )
            l_inv = F.mse_loss(outputs["z_a"], z_aug_a)

        # Total loss (ranking-dominant)
        loss = (
            l_rank
            + self.lambda_listmle * l_listmle
            + self.lambda_contrastive * l_contrast
            + self.lambda_regression * l_reg
            + self.lambda_invariance * l_inv
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rank_loss", l_rank)
        self.log("train_listmle_loss", l_listmle)
        self.log("train_contrast_loss", l_contrast)
        reg_name = "train_ccc_loss" if self.use_ccc else "train_reg_loss"
        self.log(reg_name, l_reg)
        self.log("train_inv_loss", l_inv)

        return loss
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_a1_max.py::TestMuQLoRAMaxModel -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Run existing A1 tests to verify no regression**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run pytest tests/model_improvement/test_audio_encoders.py -v`
Expected: PASS (all existing tests still pass)

- [ ] **Step 6: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/audio_encoders.py model/tests/model_improvement/test_a1_max.py && git commit -m "feat: add MuQLoRAMaxModel with ListMLE, CCC, mixup"
```

---

## Chunk 4: Sweep Runner

### Task 6: A1-Max Sweep Script

Iterates 18 configs x 4 folds sequentially on MPS. Handles memory cleanup, result logging, and checkpoint management.

**Files:**
- Create: `model/src/model_improvement/a1_max_sweep.py`

- [ ] **Step 1: Create sweep runner**

Create `model/src/model_improvement/a1_max_sweep.py`:

```python
"""A1-Max hyperparameter sweep runner.

Runs 18 configs x 4 folds sequentially on local MPS with aggressive memory
management. Saves results to a JSON file for analysis.

Usage:
    cd model/
    uv run python -m model_improvement.a1_max_sweep
"""

from __future__ import annotations

import gc
import json
import time
from itertools import product
from pathlib import Path

import torch
import pytorch_lightning as pl
from functools import partial
from torch.utils.data import DataLoader

from model_improvement.audio_encoders import MuQLoRAMaxModel
from model_improvement.data import (
    PairedPerformanceDataset,
    HardNegativePairSampler,
    audio_pair_collate_fn,
)
from model_improvement.evaluation import evaluate_model
from model_improvement.taxonomy import load_composite_labels, NUM_DIMS

# Sweep grid
LORA_RANKS = [8, 16, 32]
LORA_LAYER_RANGES = [
    (9, 10, 11, 12),
    (7, 8, 9, 10, 11, 12),
]
LABEL_SMOOTHING_VALUES = [0.0, 0.05, 0.1]

BASE_CONFIG = {
    "input_dim": 1024,
    "hidden_dim": 512,
    "num_labels": NUM_DIMS,
    "learning_rate": 3e-5,
    "weight_decay": 1e-5,
    "temperature": 0.07,
    "lambda_listmle": 1.5,
    "lambda_contrastive": 0.3,
    "lambda_regression": 0.3,
    "lambda_invariance": 0.1,
    "use_ccc": True,
    "mixup_alpha": 0.2,
    "warmup_epochs": 5,
    "max_epochs": 200,
    "use_pretrained_muq": False,
}

BATCH_SIZE = 4
ACCUM_BATCHES = 4


def _cleanup_memory():
    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _config_name(lora_rank, lora_layers, label_smoothing):
    layer_str = f"L{lora_layers[0]}-{lora_layers[-1]}"
    return f"A1max_r{lora_rank}_{layer_str}_ls{label_smoothing}"


def run_sweep(
    data_dir: Path = Path("data"),
    checkpoint_dir: Path = Path("data/checkpoints/a1_max_sweep"),
    results_path: Path = Path("data/a1_max_sweep_results.json"),
):
    """Run the full 18x4 sweep."""
    pl.seed_everything(42, workers=True)

    # Load data
    cache_dir = data_dir / "percepiano_cache"
    composite_path = data_dir / "composite_labels" / "composite_labels.json"
    labels_raw = load_composite_labels(composite_path)
    labels = {k: v.tolist() for k, v in labels_raw.items()}

    emb_path = cache_dir / "muq_embeddings.pt"
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)

    with open(cache_dir / "folds.json") as f:
        folds = json.load(f)
    with open(cache_dir / "piece_mapping.json") as f:
        piece_to_keys = json.load(f)

    print(f"Loaded {len(labels)} labels, {len(embeddings)} embeddings, {len(folds)} folds")

    configs = list(product(LORA_RANKS, LORA_LAYER_RANGES, LABEL_SMOOTHING_VALUES))
    total_runs = len(configs) * len(folds)
    print(f"Sweep: {len(configs)} configs x {len(folds)} folds = {total_runs} runs")

    # Resume from existing results
    results = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} configs already completed")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    collate_fn = partial(audio_pair_collate_fn, embeddings=embeddings)

    from model_improvement.training import train_model

    run_count = 0
    for lora_rank, lora_layers, label_smoothing in configs:
        config_name = _config_name(lora_rank, lora_layers, label_smoothing)

        if config_name in results:
            print(f"\nSkipping {config_name} (already completed)")
            run_count += len(folds)
            continue

        config = {**BASE_CONFIG}
        config["lora_rank"] = lora_rank
        config["lora_target_layers"] = lora_layers
        config["label_smoothing"] = label_smoothing

        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  lora_rank={lora_rank}, layers={lora_layers}, ls={label_smoothing}")

        fold_metrics = []
        for fold_idx, fold in enumerate(folds):
            run_count += 1
            print(f"\n  Fold {fold_idx} (run {run_count}/{total_runs})")
            start_time = time.time()

            _cleanup_memory()

            train_ds = PairedPerformanceDataset(
                cache_dir=cache_dir, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["train"],
            )
            val_ds = PairedPerformanceDataset(
                cache_dir=cache_dir, labels=labels,
                piece_to_keys=piece_to_keys, keys=fold["val"],
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                collate_fn=collate_fn, num_workers=0,
            )
            val_loader = DataLoader(
                val_ds, batch_size=BATCH_SIZE, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )

            model = MuQLoRAMaxModel(**config)
            trainer = train_model(
                model, train_loader, val_loader,
                config_name, fold_idx,
                checkpoint_dir=checkpoint_dir,
                max_epochs=config["max_epochs"],
                patience=10,
                accumulate_grad_batches=ACCUM_BATCHES,
            )

            trained_model = trainer.lightning_module
            trained_model.cpu()
            trained_model.eval()

            fold_res = evaluate_model(
                trained_model, fold["val"], labels,
                get_input_fn=lambda key: (embeddings[key].unsqueeze(0), None),
                encode_fn=lambda m, inp, mask: m.encode(inp, mask),
                compare_fn=lambda m, z_a, z_b: m.compare(z_a, z_b),
                predict_fn=lambda m, inp, mask: m.predict_scores(inp, mask),
            )
            fold_metrics.append(fold_res)

            elapsed = time.time() - start_time
            pw = fold_res.get("pairwise", 0)
            r2 = fold_res.get("r2", 0)
            print(f"    pairwise={pw:.4f}, r2={r2:.4f} ({elapsed:.0f}s)")

            del model, trainer, trained_model
            del train_ds, val_ds, train_loader, val_loader
            _cleanup_memory()

        pw_values = [m.get("pairwise", 0) for m in fold_metrics]
        r2_values = [m.get("r2", 0) for m in fold_metrics]
        results[config_name] = {
            "config": {
                "lora_rank": lora_rank,
                "lora_layers": list(lora_layers),
                "label_smoothing": label_smoothing,
            },
            "pairwise_mean": sum(pw_values) / len(pw_values),
            "pairwise_per_fold": pw_values,
            "r2_mean": sum(r2_values) / len(r2_values),
            "r2_per_fold": r2_values,
        }

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n  {config_name}: pairwise={results[config_name]['pairwise_mean']:.4f}")

    # Print leaderboard
    print(f"\n{'='*60}")
    print("SWEEP RESULTS (sorted by pairwise accuracy)")
    print(f"{'='*60}")
    sorted_configs = sorted(
        results.items(), key=lambda x: x[1]["pairwise_mean"], reverse=True
    )
    print(f"{'Config':<40} {'Pairwise':>10} {'R2':>10}")
    print("-" * 60)
    for name, metrics in sorted_configs:
        print(f"{name:<40} {metrics['pairwise_mean']:>10.4f} {metrics['r2_mean']:>10.4f}")

    best_name, best_metrics = sorted_configs[0]
    delta = best_metrics["pairwise_mean"] - 0.7393
    print(f"\nBest: {best_name} (pairwise={best_metrics['pairwise_mean']:.4f}, delta vs A1={delta:+.4f})")

    return results


if __name__ == "__main__":
    run_sweep()
```

- [ ] **Step 2: Verify the script is importable**

Run: `cd /Users/jdhiman/Documents/crescendai/model && uv run python -c "from model_improvement.a1_max_sweep import run_sweep; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/src/model_improvement/a1_max_sweep.py && git commit -m "feat: add A1-max sweep runner for 18x4 grid search"
```

---

## Chunk 5: Run Sweep & Deploy

### Task 7: Run the Sweep

- [ ] **Step 1: Run the sweep**

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -m model_improvement.a1_max_sweep
```

Results save incrementally to `data/a1_max_sweep_results.json`. Resumable if interrupted (completed configs are skipped).

- [ ] **Step 2: Verify results meet gates**

After sweep completes, check:
- Best config pairwise > 73.9% (A1 baseline)
- Score drop < 15% (robustness -- run manually on best config)
- STOP AUC >= 0.80 (run manually on best config)

- [ ] **Step 3: Evaluate 4-fold ensemble of best config**

Load all 4 fold checkpoints of the winning config, average their `predict_scores()` outputs, and evaluate ensemble pairwise accuracy + R2. This is what ships.

```bash
cd /Users/jdhiman/Documents/crescendai/model && uv run python -c "
import json, torch
from model_improvement.a1_max_sweep import _config_name
from model_improvement.audio_encoders import MuQLoRAMaxModel
from model_improvement.taxonomy import load_composite_labels
from model_improvement.metrics import MetricsSuite
from pathlib import Path

# Load best config from sweep results
with open('data/a1_max_sweep_results.json') as f:
    results = json.load(f)
best = max(results.items(), key=lambda x: x[1]['pairwise_mean'])
print(f'Best config: {best[0]} (pairwise={best[1][\"pairwise_mean\"]:.4f})')
# Ensemble evaluation code here -- average predictions across 4 fold models
"
```

- [ ] **Step 4: Run robustness check + STOP AUC on ensemble**

Must pass: robustness score_drop < 15%, STOP AUC >= 0.80.

- [ ] **Step 5: Commit results**

```bash
cd /Users/jdhiman/Documents/crescendai && git add model/data/a1_max_sweep_results.json && git commit -m "results: A1-max sweep complete"
```

---

### Task 8: Update HF Inference Handler

Update `apps/inference/handler.py` to load the 4-fold A1-Max ensemble and output 6 dimensions.

**Files:**
- Modify: `apps/inference/handler.py`

- [ ] **Step 1: Read current handler and identify changes needed**

The handler already supports ensemble prediction. Changes needed:
- Load A1-Max checkpoints (best config from sweep) instead of old 19-dim model
- Import `MuQLoRAMaxModel` instead of base model
- Output 6 dimensions with taxonomy names

- [ ] **Step 2: Update handler checkpoint loading and output format**

Update the checkpoint loading to point to best sweep config's 4 fold checkpoints. Update output to include dimension names from taxonomy: `["dynamics", "timing", "pedaling", "articulation", "phrasing", "interpretation"]`.

- [ ] **Step 3: Test handler locally**

```bash
cd /Users/jdhiman/Documents/crescendai/apps/inference && uv run python -c "
from handler import EndpointHandler
h = EndpointHandler('.')
print('Handler loaded successfully')
"
```

- [ ] **Step 4: Commit**

```bash
cd /Users/jdhiman/Documents/crescendai && git add apps/inference/handler.py && git commit -m "feat: update HF handler for 6-dim A1-max ensemble"
```

---

### Task 9: Tag and Deploy

- [ ] **Step 1: Tag current 19-dim model as rollback**

```bash
cd /Users/jdhiman/Documents/crescendai && git tag v0.1-19dim-baseline
```

- [ ] **Step 2: Deploy updated handler to HF inference endpoint**

Push changes and verify predictions return 6 dimensions.
