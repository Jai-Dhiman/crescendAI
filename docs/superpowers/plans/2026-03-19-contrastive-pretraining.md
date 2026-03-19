# Phase B: Contrastive Pretraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autoresearch-style contrastive pretraining script that trains quality-aware embeddings for MuQ and Aria encoders using a hybrid loss (piece-clustering InfoNCE + cross-piece-anchor ordinal margin).

**Architecture:** Single new script (`autoresearch_contrastive.py`) with pluggable encoder classes, a unified multi-tier dataset, weighted sampling, and structured output. One new loss function (`ordinal_margin_loss`) added to `losses.py`. All training uses pre-extracted embeddings.

**Tech Stack:** PyTorch, PyTorch Lightning, existing `piece_based_infonce_loss`, `train_linear_probe` from `aria_linear_probe.py`

**Spec:** `docs/superpowers/specs/2026-03-19-contrastive-pretraining-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `model/src/model_improvement/losses.py` | Add `ordinal_margin_loss` function |
| `model/src/model_improvement/autoresearch_contrastive.py` | **New.** Encoder classes, dataset, sampler, collate, Lightning module, linear probe diagnostic, CLI |
| `model/tests/model_improvement/test_ordinal_margin_loss.py` | **New.** Tests for the new loss function |
| `model/tests/model_improvement/test_contrastive_pretrain.py` | **New.** Tests for encoders, dataset, sampler, Lightning module, integration |

---

### Task 1: `ordinal_margin_loss` in `losses.py`

**Files:**
- Modify: `model/src/model_improvement/losses.py` (append after `ccc_loss`)
- Create: `model/tests/model_improvement/test_ordinal_margin_loss.py`

- [ ] **Step 1: Write failing tests for `ordinal_margin_loss`**

Create `model/tests/model_improvement/test_ordinal_margin_loss.py`:

```python
import torch
import pytest
from model_improvement.losses import ordinal_margin_loss


class TestOrdinalMarginLoss:
    def test_no_same_piece_pairs_returns_zero(self):
        """All segments from unique pieces -> no ordinal pairs -> loss 0."""
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 1, 2, 3])
        quality = torch.tensor([0.8, 0.6, 0.4, 0.2])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_same_quality_returns_zero(self):
        """Same piece, same quality -> no ordered pairs -> loss 0."""
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.5, 0.5, 0.5, 0.5])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_violated_margin_produces_nonzero_loss(self):
        """Better and worse embeddings equidistant from anchor -> margin violated."""
        torch.manual_seed(42)
        emb = torch.randn(3, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1])
        quality = torch.tensor([0.9, 0.1, 0.5])
        loss = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=1.0)
        assert loss.item() > 0.0

    def test_gradient_flows(self):
        """Loss must produce gradients."""
        emb = torch.randn(4, 8, requires_grad=True)
        emb_norm = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.9, 0.1, 0.8, 0.2])
        loss = ordinal_margin_loss(emb_norm, piece_ids, quality, margin_scale=1.0)
        loss.backward()
        assert emb.grad is not None

    def test_no_cross_piece_anchors_returns_zero(self):
        """All segments same piece -> no cross-piece anchor -> loss 0."""
        emb = torch.randn(3, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 0])
        quality = torch.tensor([0.9, 0.5, 0.1])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0

    def test_larger_margin_scale_increases_loss(self):
        """Larger margin_scale should produce >= loss."""
        torch.manual_seed(42)
        emb = torch.randn(4, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0, 0, 1, 1])
        quality = torch.tensor([0.9, 0.1, 0.8, 0.2])
        loss_small = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=0.01)
        loss_large = ordinal_margin_loss(emb, piece_ids, quality, margin_scale=1.0)
        assert loss_large >= loss_small

    def test_batch_size_one_returns_zero(self):
        """Single segment -> no pairs -> loss 0."""
        emb = torch.randn(1, 8)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        piece_ids = torch.tensor([0])
        quality = torch.tensor([0.5])
        loss = ordinal_margin_loss(emb, piece_ids, quality)
        assert loss.item() == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_ordinal_margin_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'ordinal_margin_loss'`

- [ ] **Step 3: Implement `ordinal_margin_loss`**

Add to `model/src/model_improvement/losses.py` after the `ccc_loss` function (before `DimensionWiseRankingLoss`):

```python
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
    quality_diff = quality_scores.unsqueeze(0) - quality_scores.unsqueeze(1)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_ordinal_margin_loss.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/losses.py model/tests/model_improvement/test_ordinal_margin_loss.py
git commit -m "feat: add ordinal_margin_loss with cross-piece anchors"
```

---

### Task 2: Encoder Classes (MuQ + Aria)

**Files:**
- Create: `model/src/model_improvement/autoresearch_contrastive.py` (start with encoder classes only)
- Create: `model/tests/model_improvement/test_contrastive_pretrain.py`

**Reference:** MuQ attention pooling + encoder MLP architecture from `model/src/model_improvement/audio_encoders.py:72-94`.

- [ ] **Step 1: Write failing tests for encoder classes**

Create `model/tests/model_improvement/test_contrastive_pretrain.py`:

```python
import torch
import pytest


class TestMuQContrastiveEncoder:
    def test_encode_shape(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 50, 1024)
        mask = torch.ones(2, 50, dtype=torch.bool)
        z = enc.encode(x, mask)
        assert z.shape == (2, 512)

    def test_project_shape(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        assert proj.shape == (2, 256)

    def test_project_is_normalized(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_mask_affects_output(self):
        from model_improvement.autoresearch_contrastive import MuQContrastiveEncoder
        enc = MuQContrastiveEncoder(input_dim=1024, hidden_dim=512, projection_dim=256)
        enc.eval()
        x = torch.randn(1, 50, 1024)
        mask_full = torch.ones(1, 50, dtype=torch.bool)
        mask_half = torch.ones(1, 50, dtype=torch.bool)
        mask_half[0, 25:] = False
        with torch.no_grad():
            z_full = enc.encode(x, mask_full)
            z_half = enc.encode(x, mask_half)
        assert not torch.allclose(z_full, z_half, atol=1e-3)


class TestAriaContrastiveEncoder:
    def test_encode_shape(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 512)
        z = enc.encode(x)
        assert z.shape == (2, 512)

    def test_project_shape(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        assert proj.shape == (2, 256)

    def test_project_is_normalized(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        z = torch.randn(2, 512)
        proj = enc.project(z)
        norms = proj.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_no_mask_required(self):
        from model_improvement.autoresearch_contrastive import AriaContrastiveEncoder
        enc = AriaContrastiveEncoder(input_dim=512, hidden_dim=512, projection_dim=256)
        x = torch.randn(2, 512)
        z = enc.encode(x)  # no mask arg
        assert z.shape == (2, 512)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement encoder classes**

Create `model/src/model_improvement/autoresearch_contrastive.py` with `MuQContrastiveEncoder` and `AriaContrastiveEncoder`. Each has `encode(embeddings, mask=None) -> [B, hidden_dim]` and `project(z) -> [B, proj_dim]` (L2-normalized via `F.normalize`).

**MuQ architecture MUST match `MuQLoRAModel` exactly** (Phase C will initialize from these weights):
- `attn`: `Sequential(Linear(1024, 256), Tanh(), Linear(256, 1))` -- attention pooling
- `encoder`: `Sequential(Linear(1024, 512), LayerNorm(512), GELU(), Dropout(0.2), Linear(512, 512), LayerNorm(512), GELU(), Dropout(0.2))`
- `projection`: `Sequential(Linear(512, 512), GELU(), Linear(512, 256))`

**Aria architecture:**
- `encoder`: `Sequential(Linear(512, 512), LayerNorm(512), GELU(), Dropout(0.2), Linear(512, 512), LayerNorm(512), GELU(), Dropout(0.2))`
- `projection`: `Sequential(Linear(512, 512), GELU(), Linear(512, 256))`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/autoresearch_contrastive.py model/tests/model_improvement/test_contrastive_pretrain.py
git commit -m "feat: add MuQ and Aria contrastive encoder classes"
```

---

### Task 3: `ContrastiveSegmentDataset` and Collation

**Files:**
- Modify: `model/src/model_improvement/autoresearch_contrastive.py` (add dataset + collate)
- Modify: `model/tests/model_improvement/test_contrastive_pretrain.py` (add dataset tests)

**Reference:** Competition manifest format: `model/data/manifests/competition/recordings.jsonl` has one JSON per line with keys `recording_id`, `competition`, `edition`, `round`, `placement`, `performer`, `piece`. **IMPORTANT:** The `piece` field is free-text (performer-specific, e.g. "BRUCE (XIAOYU) LIU -- second round"), NOT a canonical piece name. The correct grouping key for ordinal ranking is `(competition, edition, round)` -- all performers in the same round are ranked by placement.

MuQ segment files: `model/data/embeddings/competition/muq_embeddings/{recording_id}_seg{NNN}.pt` -- each is a `[T, 1024]` tensor.

T1 embedding files: `Embeddings.percepiano / "muq_embeddings.pt"` is a **dict** mapping segment keys to `[T, 1024]` tensors (for MuQ) or `Embeddings.percepiano / "aria_embedding.pt"` is a **dict** mapping segment keys to `[512]` tensors (for Aria). These are loaded as a single `.pt` file via `torch.load`, not individual per-segment files.

T1 labels: via `load_composite_labels(Labels.composite / "composite_labels.json")` which returns `{segment_id: np.ndarray[6]}` with values already in [0,1].

**Aria competition embeddings do not exist yet.** When `--encoder aria` is used, `from_t2()` should raise `FileNotFoundError` with a clear message: "Aria competition embeddings not extracted yet. Run aria_embeddings.py on competition MIDIs first." The script still works for `--encoder aria` with T1-only (set `--t2-weight 0`).

- [ ] **Step 1: Write failing tests for dataset and collation**

Add `TestContrastiveSegmentDataset` and `TestContrastiveCollate` to `test_contrastive_pretrain.py`:

```python
class TestContrastiveSegmentDataset:
    def _make_t1_data(self):
        """Synthetic T1 data mimicking the real format."""
        # muq_embeddings.pt is a dict: segment_key -> [T, 1024] tensor
        embeddings = {
            "seg_a1": torch.randn(10, 1024),
            "seg_a2": torch.randn(12, 1024),
            "seg_b1": torch.randn(8, 1024),
        }
        # composite_labels.json returns {key: [6-dim array in 0..1]}
        labels = {
            "seg_a1": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            "seg_a2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "seg_b1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
        # piece_mapping.json: piece_name -> [segment_keys]
        piece_to_keys = {"piece_X": ["seg_a1", "seg_a2"], "piece_Y": ["seg_b1"]}
        keys = ["seg_a1", "seg_a2", "seg_b1"]
        return embeddings, labels, piece_to_keys, keys

    def test_t1_items_have_correct_schema(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        item = ds[0]
        assert "embedding" in item
        assert "piece_id" in item
        assert "quality_score" in item
        assert isinstance(item["piece_id"], int)
        assert 0.0 <= item["quality_score"] <= 1.0

    def test_t1_piece_ids_are_consistent(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        # seg_a1 and seg_a2 are same piece -> same piece_id
        items = [ds[i] for i in range(len(ds))]
        ids = {item["piece_id"] for item in items}
        assert len(ids) == 2  # two pieces

    def test_piece_id_offset(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=100)
        item = ds[0]
        assert item["piece_id"] >= 100

    def test_length_matches_keys(self):
        from model_improvement.autoresearch_contrastive import ContrastiveSegmentDataset
        emb, labels, p2k, keys = self._make_t1_data()
        ds = ContrastiveSegmentDataset.from_t1(emb, labels, p2k, keys, piece_id_offset=0)
        assert len(ds) == 3


class TestContrastiveCollate:
    def test_muq_collate_pads_and_masks(self):
        from model_improvement.autoresearch_contrastive import contrastive_collate_muq
        batch = [
            {"embedding": torch.randn(10, 1024), "piece_id": 0, "quality_score": 0.8},
            {"embedding": torch.randn(20, 1024), "piece_id": 0, "quality_score": 0.3},
        ]
        out = contrastive_collate_muq(batch)
        assert out["embeddings"].shape == (2, 20, 1024)
        assert out["mask"].shape == (2, 20)
        assert out["mask"][0, 9].item() is True
        assert out["mask"][0, 10].item() is False
        assert out["mask"][1].all()
        assert out["piece_ids"].shape == (2,)
        assert out["quality_scores"].shape == (2,)

    def test_aria_collate_stacks(self):
        from model_improvement.autoresearch_contrastive import contrastive_collate_aria
        batch = [
            {"embedding": torch.randn(512), "piece_id": 0, "quality_score": 0.8},
            {"embedding": torch.randn(512), "piece_id": 1, "quality_score": 0.3},
        ]
        out = contrastive_collate_aria(batch)
        assert out["embeddings"].shape == (2, 512)
        assert "mask" not in out
        assert out["piece_ids"].shape == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestContrastiveSegmentDataset tests/model_improvement/test_contrastive_pretrain.py::TestContrastiveCollate -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `ContrastiveSegmentDataset` with `from_t1()` and `from_t2()` classmethods, plus collate functions**

`ContrastiveSegmentDataset`: stores list of `{embedding, piece_id, quality_score}` dicts.

`from_t1(embeddings_dict, labels_dict, piece_to_keys, keys, piece_id_offset)`: maps segment keys to piece_ids via `piece_to_keys` (sorted for determinism), quality = mean of 6-dim labels (already in [0,1]).

`from_t2(embeddings_dir, records, piece_id_offset)`: groups records by `(competition, edition, round)` (NOT `piece` -- `piece` is free-text per-performer). Within each group, normalizes placement: `1.0 - (placement - min_p) / (max_p - min_p)`. Single-competitor groups get quality 1.0. Loads `.pt` segment files from `embeddings_dir / f"{recording_id}_seg*.pt"`. If `embeddings_dir` does not exist, raises `FileNotFoundError` with clear message.

`contrastive_collate_muq`: pads variable-length `[T, D]` to batch max, creates boolean masks, stacks piece_ids and quality_scores.

`contrastive_collate_aria`: stacks fixed-dim `[D]` embeddings, no mask.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestContrastiveSegmentDataset tests/model_improvement/test_contrastive_pretrain.py::TestContrastiveCollate -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/autoresearch_contrastive.py model/tests/model_improvement/test_contrastive_pretrain.py
git commit -m "feat: add ContrastiveSegmentDataset and collate functions"
```

---

### Task 4: `WeightedTierSampler`

**Files:**
- Modify: `model/src/model_improvement/autoresearch_contrastive.py` (add sampler)
- Modify: `model/tests/model_improvement/test_contrastive_pretrain.py`

- [ ] **Step 1: Write failing tests for sampler**

Add `TestWeightedTierSampler` to test file. Tests: samples roughly follow tier weights (50/50 split yields ~balanced counts with 20% tolerance), guarantees multiple pieces in output indices, length matches `total_samples`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestWeightedTierSampler -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `WeightedTierSampler(Sampler)`**

Draws `total_samples` indices by: (1) pick tier by weight, (2) pick random piece from that tier, (3) pick random segment within that piece. Groups indices by piece during `__init__` for efficient sampling. Uses `random.Random(seed)` for reproducibility.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestWeightedTierSampler -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/autoresearch_contrastive.py model/tests/model_improvement/test_contrastive_pretrain.py
git commit -m "feat: add WeightedTierSampler for multi-tier contrastive training"
```

---

### Task 5: `ContrastivePretrainModel` (Lightning Module)

**Files:**
- Modify: `model/src/model_improvement/autoresearch_contrastive.py`
- Modify: `model/tests/model_improvement/test_contrastive_pretrain.py`

**Reference:** `MuQLoRAModel.training_step` at `model/src/model_improvement/audio_encoders.py:196-246`, `configure_optimizers` at line 292-303.

- [ ] **Step 1: Write failing tests for Lightning module**

Add `TestContrastivePretrainModel`. Tests: `training_step` returns scalar loss with gradients (MuQ and Aria), `validation_step` doesn't raise, `configure_optimizers` returns dict with optimizer and lr_scheduler.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestContrastivePretrainModel -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `ContrastivePretrainModel`**

`_shared_step`: encode -> project -> `piece_based_infonce_loss` + `ordinal_margin_loss` -> weighted sum. `training_step`: logs train_loss, train_infonce, train_ordinal. `validation_step`: logs val_loss, val_infonce, val_ordinal. `configure_optimizers`: AdamW with warmup (5 epochs, start_factor=0.01) + cosine decay to 1e-6, using `SequentialLR`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestContrastivePretrainModel -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/autoresearch_contrastive.py model/tests/model_improvement/test_contrastive_pretrain.py
git commit -m "feat: add ContrastivePretrainModel Lightning module"
```

---

### Task 6: CLI, Data Loading, Linear Probe Diagnostic, and Structured Output

**Files:**
- Modify: `model/src/model_improvement/autoresearch_contrastive.py` (add `run_single_fold`, `main`)

**Reference:** `autoresearch_loss_weights.py` for full pattern: argparse, data loading, `train_model` call, best checkpoint loading, structured output. `aria_linear_probe.py:117-178` for `train_linear_probe` and `compute_pairwise_from_regression`.

- [ ] **Step 1: Write tests for `_load_t2_records`**

Add to `test_contrastive_pretrain.py`:

```python
class TestLoadT2Records:
    def test_train_excludes_cliburn_2022(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        train = _load_t2_records("train")
        for r in train:
            assert not (r["competition"] == "cliburn" and r["edition"] == 2022)

    def test_val_is_cliburn_2022_only(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        val = _load_t2_records("val")
        for r in val:
            assert r["competition"] == "cliburn" and r["edition"] == 2022

    def test_train_and_val_are_disjoint(self):
        from model_improvement.autoresearch_contrastive import _load_t2_records
        train_ids = {r["recording_id"] for r in _load_t2_records("train")}
        val_ids = {r["recording_id"] for r in _load_t2_records("val")}
        assert train_ids.isdisjoint(val_ids)
```

- [ ] **Step 2: Implement `_load_t2_records`, `run_single_fold`, and `main`**

`_load_t2_records(split)`: loads `Manifests.competition / "recordings.jsonl"`, splits by holding out Cliburn 2022 for validation.

`run_single_fold(...)`: seeds, loads T1+T2 data per encoder type, builds datasets with disjoint piece_id offsets, creates samplers + dataloaders, builds encoder + model, calls `train_model` with patience=7, loads best checkpoint, extracts val metrics, runs linear probe diagnostic (freeze encoder, extract `encode()` embeddings for T1, train `Linear(hidden_dim, 6)`, compute pairwise accuracy + R2), returns structured dict. Include `_cleanup_memory()` (gc.collect + MPS cache clear) before training and after cleanup, matching `autoresearch_loss_weights.py` pattern.

`main()`: argparse with all CLI flags, calls `run_single_fold`, prints `AUTORESEARCH_RESULT` block + `AUTORESEARCH_JSON=` line.

- [ ] **Step 3: Run `_load_t2_records` tests**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestLoadT2Records -v`
Expected: All 3 tests PASS

- [ ] **Step 4: Verify the file runs with `--help`**

Run: `cd model && uv run python -m model_improvement.autoresearch_contrastive --help`
Expected: Shows argparse help with `--encoder`, `--lambda-infonce`, `--lambda-ordinal`, `--temperature`, `--ordinal-margin`, `--t1-weight`, `--t2-weight`, `--learning-rate`, `--weight-decay`, `--max-epochs`

- [ ] **Step 5: Commit**

```bash
git add model/src/model_improvement/autoresearch_contrastive.py
git commit -m "feat: add CLI, data loading, linear probe eval, and structured output"
```

---

### Task 7: Integration Tests

**Files:**
- Modify: `model/tests/model_improvement/test_contrastive_pretrain.py`

- [ ] **Step 1: Write integration tests with synthetic data**

Add `TestIntegration` class. Two tests: `test_full_training_loop_muq` and `test_full_training_loop_aria`. Each builds synthetic `ContrastiveSegmentDataset` (2 pieces, 4 segments each), creates sampler + dataloader, instantiates encoder + `ContrastivePretrainModel`, runs `pl.Trainer` for 2 epochs on CPU, asserts `train_loss` is finite.

- [ ] **Step 2: Run integration tests**

Run: `cd model && uv run pytest tests/model_improvement/test_contrastive_pretrain.py::TestIntegration -v`
Expected: Both tests PASS

- [ ] **Step 3: Run full test suite to check regressions**

Run: `cd model && uv run pytest tests/model_improvement/test_ordinal_margin_loss.py tests/model_improvement/test_contrastive_pretrain.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add model/tests/model_improvement/test_contrastive_pretrain.py
git commit -m "test: add integration tests for contrastive pretraining"
```

---

### Task 8: Smoke Test with Real Data

**Files:** None modified -- verification only.

- [ ] **Step 1: Run MuQ contrastive pretraining for 2 epochs**

```bash
cd model
uv run python -m model_improvement.autoresearch_contrastive \
  --encoder muq \
  --max-epochs 2 \
  --lambda-infonce 1.0 \
  --lambda-ordinal 0.5
```

Expected: Training completes, outputs `AUTORESEARCH_RESULT` with finite values.

- [ ] **Step 2: Verify checkpoint was saved**

```bash
ls model/data/checkpoints/contrastive_pretrain/muq/fold_0/
```

Expected: At least one `.ckpt` file.

- [ ] **Step 3: Run existing test suite for regressions**

```bash
cd model && uv run pytest tests/model_improvement/test_losses.py tests/model_improvement/test_data.py tests/model_improvement/test_audio_encoders.py -v
```

Expected: All existing tests PASS.

- [ ] **Step 4: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: address smoke test issues in contrastive pretraining"
```
