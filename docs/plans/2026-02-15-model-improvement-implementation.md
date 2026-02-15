# Model Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a dual-encoder (audio + symbolic) piano performance evaluation model, tested across 6 independent architectures, fused for score-conditioned quality assessment.

**Architecture:** Adapt MuQ via LoRA/unfreezing for audio; build a piano-specific symbolic encoder (Transformer/GNN/continuous); fuse the best of each track via cross-attention. Train on four data tiers (labeled, competition, paired, unlabeled).

**Tech Stack:** PyTorch Lightning, transformers (HuggingFace), torch-geometric (GNN), peft (LoRA), miditok (REMI tokenizer), torchaudio-augmentations

**Design doc:** `docs/plans/2026-02-15-model-improvement-design.md`

---

## Dependency Graph

```
Task 1 (scaffold) --> all tasks
Task 2 (augmentation) --> Tasks 7, 8, 9
Task 3 (tokenizer) --> Tasks 11, 12, 13
Task 4 (metrics suite) --> Tasks 10, 14
Task 5 (data pipeline) --> Tasks 7-9, 11-13

Tasks 7-9 (audio models) --> Task 10 (audio comparison)
Tasks 11-13 (symbolic models) --> Task 14 (symbolic comparison)

Tasks 10, 14 --> Task 15 (fusion)
Task 15 --> Task 16 (robustness validation)
```

**Parallelizable:** Tasks 2+3+4+5 (after Task 1). Tasks 7-9 (audio) || Tasks 11-13 (symbolic).

---

## Phase 0: Shared Infrastructure

### Task 1: Module Scaffold

**Files:**

- Create: `model/src/model_improvement/__init__.py`
- Create: `model/src/model_improvement/audio_encoders.py`
- Create: `model/src/model_improvement/symbolic_encoders.py`
- Create: `model/src/model_improvement/fusion.py`
- Create: `model/src/model_improvement/losses.py`
- Create: `model/src/model_improvement/data.py`
- Create: `model/src/model_improvement/tokenizer.py`
- Create: `model/src/model_improvement/metrics.py`
- Create: `model/src/model_improvement/augmentation.py`
- Create: `model/tests/model_improvement/__init__.py`
- Modify: `model/pyproject.toml` (add `src/model_improvement` to build targets, add new deps)

**Step 1:** Create directory structure

```bash
mkdir -p model/src/model_improvement
mkdir -p model/tests/model_improvement
```

**Step 2:** Create `__init__.py` with module docstring

```python
"""
Model improvement experiments: dual-encoder architecture with domain adaptation.

Audio track: A1 (LoRA), A2 (staged adaptation), A3 (full unfreeze)
Symbolic track: S1 (Transformer), S2 (GNN), S3 (continuous)
Fusion: F1 (cross-attention), F2 (concat), F3 (gated)
"""
```

**Step 3:** Create empty module files with docstrings only (one per file listed above)

**Step 4:** Update `pyproject.toml`:

- Add `"src/model_improvement"` to `[tool.hatch.build.targets.wheel] packages`
- Add dependencies:
  - `peft>=0.7.0` (LoRA adapters)
  - `miditok>=3.0.0` (REMI tokenization)
  - `torch-geometric>=2.4.0` (GNN, optional)
  - `audiomentations>=0.34.0` (audio augmentation)

**Step 5:** Verify installation

```bash
cd model && uv sync
```

**Step 6:** Commit

```bash
git add model/src/model_improvement/ model/tests/model_improvement/ model/pyproject.toml
git commit -m "scaffold model_improvement module with dependency updates"
```

---

### Task 2: Audio Augmentation Pipeline

**Files:**

- Create: `model/src/model_improvement/augmentation.py`
- Create: `model/tests/model_improvement/test_augmentation.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_augmentation.py
import torch
import pytest
from model_improvement.augmentation import AudioAugmentor


def test_augmentor_returns_same_length():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None)
    waveform = torch.randn(1, 24000)  # 1 second at 24kHz
    result = aug(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_augmentor_no_augmentation_when_prob_zero():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=0.0)
    waveform = torch.randn(1, 24000)
    result = aug(waveform, sample_rate=24000)
    assert torch.allclose(result, waveform)


def test_augmentor_always_augments_when_prob_one():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 48000)
    torch.manual_seed(42)
    result = aug(waveform, sample_rate=24000)
    assert not torch.allclose(result, waveform)


def test_phone_simulation():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 24000)
    result = aug._apply_phone_simulation(waveform, sample_rate=24000)
    assert result.shape == waveform.shape


def test_eq_variation():
    aug = AudioAugmentor(room_irs_dir=None, noise_dir=None, augment_prob=1.0)
    waveform = torch.randn(1, 24000)
    result = aug._apply_eq_variation(waveform, sample_rate=24000)
    assert result.shape == waveform.shape
```

**Step 2:** Run tests, verify they fail

```bash
cd model && python -m pytest tests/model_improvement/test_augmentation.py -v
```

**Step 3: Implement AudioAugmentor**

Key interface:

```python
class AudioAugmentor:
    def __init__(
        self,
        room_irs_dir: Optional[Path],
        noise_dir: Optional[Path],
        augment_prob: float = 0.5,
    ): ...

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Apply random augmentations to waveform [C, T]."""
        ...

    def _apply_room_ir(self, waveform, sample_rate) -> torch.Tensor: ...
    def _apply_additive_noise(self, waveform, sample_rate, snr_range=(10, 30)) -> torch.Tensor: ...
    def _apply_phone_simulation(self, waveform, sample_rate) -> torch.Tensor: ...
    def _apply_pitch_shift(self, waveform, sample_rate, cents_range=(-50, 50)) -> torch.Tensor: ...
    def _apply_eq_variation(self, waveform, sample_rate) -> torch.Tensor: ...
```

Each augmentation applied independently with its own probability:

- Room IR: p=0.3 (convolve with random IR from directory, skip if dir is None)
- Additive noise: p=0.3 (mix with random noise clip at random SNR)
- Phone sim: p=0.2 (low-pass at 8kHz + dynamic range compression)
- Pitch shift: p=0.1 (torchaudio functional pitch_shift)
- EQ: p=0.2 (random 3-band parametric EQ via biquad filters)

**Step 4:** Run tests, verify they pass

**Step 5:** Commit

```bash
git commit -m "add AudioAugmentor with room IR, noise, phone sim, pitch shift, EQ"
```

---

### Task 3: REMI MIDI Tokenizer

**Files:**

- Create: `model/src/model_improvement/tokenizer.py`
- Create: `model/tests/model_improvement/test_tokenizer.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_tokenizer.py
import pytest
import numpy as np
from pathlib import Path
from model_improvement.tokenizer import PianoTokenizer, extract_continuous_features


@pytest.fixture
def sample_midi(tmp_path):
    """Create a minimal MIDI file for testing."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    piano.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
    piano.notes.append(pretty_midi.Note(velocity=90, pitch=67, start=1.0, end=1.5))
    pm.instruments.append(piano)
    path = tmp_path / "test.mid"
    pm.write(str(path))
    return path


def test_tokenizer_encodes_midi(sample_midi):
    tok = PianoTokenizer()
    tokens = tok.encode(sample_midi)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(t, int) for t in tokens)


def test_tokenizer_vocab_size():
    tok = PianoTokenizer()
    assert tok.vocab_size > 0
    assert tok.vocab_size < 1000


def test_tokenizer_roundtrip(sample_midi, tmp_path):
    tok = PianoTokenizer()
    tokens = tok.encode(sample_midi)
    output_path = tmp_path / "roundtrip.mid"
    tok.decode(tokens, output_path)
    assert output_path.exists()


def test_tokenizer_special_tokens():
    tok = PianoTokenizer()
    assert hasattr(tok, 'pad_token_id')
    assert hasattr(tok, 'mask_token_id')


def test_continuous_features(sample_midi):
    features = extract_continuous_features(sample_midi, frame_rate=50)
    assert features.ndim == 2  # [T, D]
    assert features.shape[1] >= 3  # pitch, velocity, timing at minimum
```

**Step 2:** Run tests, verify failure

**Step 3: Implement PianoTokenizer**

Wrapper around miditok REMI with piano-specific config:

```python
class PianoTokenizer:
    """REMI tokenizer for piano MIDI, wrapping miditok."""
    def __init__(self, max_seq_len: int = 2048): ...
    def encode(self, midi_path: Path) -> list[int]: ...
    def decode(self, tokens: list[int], output_path: Path) -> None: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def pad_token_id(self) -> int: ...
    @property
    def mask_token_id(self) -> int: ...


def extract_continuous_features(midi_path: Path, frame_rate: int = 50) -> np.ndarray:
    """Extract continuous feature curves from MIDI for S3 experiment.
    Returns [T, D] array: pitch, velocity, density, pedal, IOI."""
    ...
```

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add PianoTokenizer (REMI) and continuous MIDI feature extraction"
```

---

### Task 4: Shared Metrics Suite

**Files:**

- Create: `model/src/model_improvement/metrics.py`
- Create: `model/tests/model_improvement/test_metrics.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_metrics.py
import torch
import numpy as np
import pytest
from model_improvement.metrics import (
    MetricsSuite,
    compute_robustness_metrics,
    format_comparison_table,
)


def test_metrics_pairwise_accuracy():
    suite = MetricsSuite(ambiguous_threshold=0.05)
    logits = torch.tensor([[1.0, -1.0], [0.5, 0.8]])
    labels_a = torch.tensor([[0.8, 0.3], [0.6, 0.9]])
    labels_b = torch.tensor([[0.3, 0.8], [0.4, 0.5]])
    result = suite.pairwise_accuracy(logits, labels_a, labels_b)
    assert "overall" in result
    assert "per_dimension" in result
    assert 0.0 <= result["overall"] <= 1.0


def test_metrics_regression_r2():
    suite = MetricsSuite()
    predictions = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    targets = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    r2 = suite.regression_r2(predictions, targets)
    assert r2 > 0.99


def test_metrics_difficulty_correlation():
    suite = MetricsSuite()
    predictions = torch.randn(100, 19)
    difficulties = torch.randn(100)
    result = suite.difficulty_correlation(predictions, difficulties)
    assert "overall_rho" in result
    assert "per_dimension" in result


def test_robustness_metrics():
    clean_scores = torch.randn(50, 19)
    augmented_scores = clean_scores + torch.randn(50, 19) * 0.1
    result = compute_robustness_metrics(clean_scores, augmented_scores)
    assert "pearson_r" in result
    assert "score_drop_pct" in result


def test_format_comparison_table():
    results = {
        "A1": {"r2": 0.55, "pairwise": 0.85, "difficulty_rho": 0.63, "robustness": 0.92, "gpu_hours": 2.5},
        "A2": {"r2": 0.58, "pairwise": 0.87, "difficulty_rho": 0.65, "robustness": 0.94, "gpu_hours": 8.0},
    }
    table = format_comparison_table(results)
    assert isinstance(table, str)
    assert "A1" in table
    assert "A2" in table
```

**Step 2:** Run tests, verify failure

**Step 3: Implement MetricsSuite**

Reuse patterns from `src/disentanglement/training/metrics.py` but generalized:

```python
class MetricsSuite:
    """Shared metrics suite for all model improvement experiments."""
    def __init__(self, ambiguous_threshold: float = 0.05): ...
    def pairwise_accuracy(self, logits, labels_a, labels_b) -> dict: ...
    def regression_r2(self, predictions, targets) -> float: ...
    def difficulty_correlation(self, predictions, difficulties) -> dict: ...
    def full_report(self, model, test_data: dict) -> dict:
        """Run all applicable metrics and return unified results dict."""
        ...

def compute_robustness_metrics(clean_scores, augmented_scores) -> dict: ...
def format_comparison_table(results: dict[str, dict]) -> str: ...
```

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add shared MetricsSuite for cross-experiment comparison"
```

---

### Task 5: Data Pipeline for T2-T4

**Files:**

- Create: `model/src/model_improvement/data.py`
- Create: `model/tests/model_improvement/test_data.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_data.py
import torch
import pytest
from pathlib import Path
from model_improvement.data import (
    CompetitionDataset,
    PairedPerformanceDataset,
    AugmentedEmbeddingDataset,
    MIDIPretrainingDataset,
    multi_task_collate_fn,
)


def test_competition_dataset_returns_ordinal_labels():
    mock_data = [
        {"recording_id": "r1", "placement": 1, "embeddings": torch.randn(100, 1024)},
        {"recording_id": "r2", "placement": 2, "embeddings": torch.randn(80, 1024)},
        {"recording_id": "r3", "placement": 3, "embeddings": torch.randn(120, 1024)},
    ]
    ds = CompetitionDataset(mock_data, max_frames=1000)
    item = ds[0]
    assert "embeddings" in item
    assert "placement" in item


def test_paired_performance_dataset():
    mock_labels = {"k1": [0.5] * 19, "k2": [0.7] * 19}
    mock_pieces = {"piece1": ["k1", "k2"]}
    ds = PairedPerformanceDataset(
        cache_dir=Path("/tmp/test"),
        labels=mock_labels,
        piece_to_keys=mock_pieces,
        keys=["k1", "k2"],
    )
    assert len(ds) > 0


def test_midi_pretraining_dataset():
    tokens = list(range(100))
    ds = MIDIPretrainingDataset(
        token_sequences=[tokens],
        max_seq_len=512,
        mask_prob=0.15,
        vocab_size=500,
    )
    item = ds[0]
    assert "input_ids" in item
    assert "labels" in item
    assert "attention_mask" in item


def test_multi_task_collate():
    batch = [
        {"embeddings_a": torch.randn(50, 1024), "labels_a": torch.rand(19)},
        {"embeddings_a": torch.randn(30, 1024), "labels_a": torch.rand(19)},
    ]
    collated = multi_task_collate_fn(batch)
    assert collated["embeddings_a"].shape[0] == 2
    assert collated["embeddings_a"].shape[1] == 50  # padded to max
```

**Step 2:** Run tests, verify failure

**Step 3: Implement dataset classes**

Extend existing patterns from `src/disentanglement/data/`:

```python
class CompetitionDataset(Dataset):
    """T2: Competition recordings with ordinal placement labels."""
    ...

class PairedPerformanceDataset(Dataset):
    """T3: Extended PairwiseRankingDataset supporting ATEPP + competition data."""
    ...

class AugmentedEmbeddingDataset(Dataset):
    """T4: Returns clean + augmented embedding pairs for invariance training."""
    ...

class MIDIPretrainingDataset(Dataset):
    """T4: Masked token prediction for symbolic encoder pretraining."""
    ...

def multi_task_collate_fn(batch: list[dict]) -> dict:
    """Collate for multi-task training (handles variable fields per sample)."""
    ...
```

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add T2-T4 dataset classes for competition, paired, and pretraining data"
```

---

## Phase 1: Audio Encoder Track

### Task 6: LoRA Adapter Module for MuQ

**Files:**

- Create: `model/src/model_improvement/lora.py`
- Create: `model/tests/model_improvement/test_lora.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_lora.py
import torch
import pytest
from model_improvement.lora import apply_lora_to_muq, count_trainable_params, create_mock_encoder


def test_apply_lora_reduces_trainable_params():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    trainable_before = count_trainable_params(model)
    # Freeze all, then apply LoRA
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_muq(model, rank=16, target_layers=(2, 3))
    trainable_after = count_trainable_params(model)
    assert trainable_after < trainable_before
    assert trainable_after > 0


def test_lora_forward_preserves_output_shape():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    x = torch.randn(2, 50, 256)
    out_before = model(x)
    for p in model.parameters():
        p.requires_grad = False
    apply_lora_to_muq(model, rank=16, target_layers=(2, 3))
    out_after = model(x)
    assert out_before.shape == out_after.shape


def test_count_trainable_params():
    model = create_mock_encoder(hidden_size=256, num_layers=4)
    count = count_trainable_params(model)
    assert isinstance(count, int)
    assert count > 0
```

**Step 2:** Run tests, verify failure

**Step 3: Implement LoRA integration**

Use HuggingFace `peft` library:

```python
from peft import LoraConfig, get_peft_model

def apply_lora_to_muq(model, rank=16, alpha=32, target_layers=(9,10,11,12), target_modules=None):
    """Apply LoRA adapters to MuQ self-attention layers in-place."""
    ...

def count_trainable_params(model) -> int: ...

def create_mock_encoder(hidden_size=256, num_layers=4):
    """Small transformer for unit tests."""
    ...
```

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add LoRA adapter integration for MuQ fine-tuning"
```

---

### Task 7: A1 - MuQ + LoRA Multi-Task Model

**Files:**

- Create: `model/src/model_improvement/audio_encoders.py`
- Create: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_audio_encoders.py
import torch
import pytest
from model_improvement.audio_encoders import MuQLoRAModel


class TestMuQLoRAModel:
    def test_forward_shape(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        x_a = torch.randn(4, 100, 1024)
        x_b = torch.randn(4, 80, 1024)
        mask_a = torch.ones(4, 100, dtype=torch.bool)
        mask_b = torch.ones(4, 80, dtype=torch.bool)
        out = model(x_a, x_b, mask_a, mask_b)
        assert out["ranking_logits"].shape == (4, 19)
        assert out["z_a"].shape == (4, 512)
        assert out["z_b"].shape == (4, 512)

    def test_regression_forward(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        x = torch.randn(4, 100, 1024)
        mask = torch.ones(4, 100, dtype=torch.bool)
        scores = model.predict_scores(x, mask)
        assert scores.shape == (4, 19)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_training_step_returns_loss(self):
        model = MuQLoRAModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
        )
        batch = {
            "embeddings_a": torch.randn(4, 50, 1024),
            "embeddings_b": torch.randn(4, 50, 1024),
            "mask_a": torch.ones(4, 50, dtype=torch.bool),
            "mask_b": torch.ones(4, 50, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
        assert loss.item() > 0
```

**Step 2:** Run tests, verify failure

**Step 3: Implement MuQLoRAModel**

Extends ContrastivePairwiseRankingModel pattern with:

- Optional LoRA-adapted MuQ backbone (`use_pretrained_muq=True` loads real MuQ + LoRA)
- Multi-task loss: ranking + contrastive + regression + invariance
- `predict_scores()` for absolute quality via regression head with sigmoid
- Augmentation invariance loss when augmented pairs in batch

Follow existing Lightning patterns from `disentanglement/models/contrastive_ranking.py`.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add MuQLoRAModel (A1) with multi-task ranking + regression"
```

---

### Task 8: A2 - Staged Domain Adaptation Model

**Files:**

- Modify: `model/src/model_improvement/audio_encoders.py` (add MuQStagedModel)
- Modify: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write failing tests**

```python
from model_improvement.audio_encoders import MuQStagedModel


class TestMuQStagedModel:
    def test_stage1_self_supervised_step(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="self_supervised",
        )
        batch = {
            "embeddings_clean": torch.randn(4, 50, 1024),
            "embeddings_augmented": torch.randn(4, 50, 1024),
            "mask": torch.ones(4, 50, dtype=torch.bool),
            "piece_ids": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_stage2_supervised_step(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="supervised",
        )
        batch = {
            "embeddings_a": torch.randn(4, 50, 1024),
            "embeddings_b": torch.randn(4, 50, 1024),
            "mask_a": torch.ones(4, 50, dtype=torch.bool),
            "mask_b": torch.ones(4, 50, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_switch_stage(self):
        model = MuQStagedModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False, stage="self_supervised",
        )
        model.switch_to_supervised()
        assert model.stage == "supervised"
```

**Step 2:** Run tests, verify failure

**Step 3: Implement MuQStagedModel**

Two-stage model:

- Stage 1 (`self_supervised`): contrastive_cross_performer + augmentation_invariance. T3+T4 data.
- Stage 2 (`supervised`): same multi-task loss as A1. T1+T2+T3 data.
- `switch_to_supervised()` transitions stages (adjusts optimizer, loss weighting).
- Shares encoder architecture with MuQLoRAModel.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add MuQStagedModel (A2) with self-supervised + supervised stages"
```

---

### Task 9: A3 - Full Unfreeze with Gradual Layer Unfreezing

**Files:**

- Modify: `model/src/model_improvement/audio_encoders.py` (add MuQFullUnfreezeModel)
- Modify: `model/tests/model_improvement/test_audio_encoders.py`

**Step 1: Write failing tests**

```python
from model_improvement.audio_encoders import MuQFullUnfreezeModel


class TestMuQFullUnfreezeModel:
    def test_gradual_unfreezing(self):
        model = MuQFullUnfreezeModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
            unfreeze_schedule={0: [12], 10: [11], 20: [10], 30: [9]},
        )
        model.on_train_epoch_start()
        unfrozen = model.get_unfrozen_layers()
        assert 12 in unfrozen

    def test_discriminative_lr(self):
        model = MuQFullUnfreezeModel(
            input_dim=1024, hidden_dim=512, num_labels=19,
            use_pretrained_muq=False,
            unfreeze_schedule={0: [12], 10: [11]},
            lr_decay_factor=0.5,
        )
        optim_config = model.configure_optimizers()
        param_groups = optim_config["optimizer"].param_groups
        assert len(param_groups) > 1
```

**Step 2:** Run tests, verify failure

**Step 3: Implement MuQFullUnfreezeModel**

- `unfreeze_schedule`: dict mapping epoch -> list of layer indices to unfreeze
- `lr_decay_factor`: multiplier for deeper layers (layer 9 = factor^3 * base_lr)
- `on_train_epoch_start()`: check schedule, unfreeze layers
- `configure_optimizers()`: discriminative LRs per layer group
- Same multi-task loss as A1.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add MuQFullUnfreezeModel (A3) with gradual unfreezing and discriminative LR"
```

---

### Task 10: Audio Comparison Notebook

**Files:**

- Create: `model/notebooks/model_improvement/07_audio_comparison.ipynb`

**Step 1:** Create notebook with cells:

1. Setup -- imports, paths, load checkpoints for A1/A2/A3
2. Load shared validation data (PercePiano 4-fold, PSyllabus)
3. Run MetricsSuite on each model checkpoint
4. Compute robustness metrics (augmented test set)
5. Format comparison table
6. Per-dimension breakdown visualization
7. Winner selection (primary: pairwise accuracy, tiebreak: R2, veto: robustness >15% drop)

**Step 2:** Verify notebook structure

**Step 3:** Commit

```bash
git commit -m "add audio comparison notebook (07)"
```

---

## Phase 2: Symbolic Encoder Track

### Task 11: S1 - Transformer on MIDI Tokens

**Files:**

- Create: `model/src/model_improvement/symbolic_encoders.py`
- Create: `model/tests/model_improvement/test_symbolic_encoders.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_symbolic_encoders.py
import torch
import pytest
from model_improvement.symbolic_encoders import TransformerSymbolicEncoder


class TestTransformerSymbolicEncoder:
    def test_forward_shape(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500, d_model=512, nhead=8, num_layers=6,
            hidden_dim=512, num_labels=19,
        )
        input_ids = torch.randint(0, 500, (4, 128))
        attention_mask = torch.ones(4, 128, dtype=torch.bool)
        out = model(input_ids, attention_mask)
        assert out["z_symbolic"].shape == (4, 512)
        assert out["scores"].shape == (4, 19)

    def test_masked_lm_step(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500, d_model=512, nhead=8, num_layers=6,
            hidden_dim=512, num_labels=19, stage="pretrain",
        )
        batch = {
            "input_ids": torch.randint(0, 500, (4, 128)),
            "labels": torch.randint(0, 500, (4, 128)),
            "attention_mask": torch.ones(4, 128, dtype=torch.bool),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_pairwise_ranking_step(self):
        model = TransformerSymbolicEncoder(
            vocab_size=500, d_model=512, nhead=8, num_layers=6,
            hidden_dim=512, num_labels=19, stage="finetune",
        )
        batch = {
            "input_ids_a": torch.randint(0, 500, (4, 128)),
            "input_ids_b": torch.randint(0, 500, (4, 128)),
            "mask_a": torch.ones(4, 128, dtype=torch.bool),
            "mask_b": torch.ones(4, 128, dtype=torch.bool),
            "labels_a": torch.rand(4, 19),
            "labels_b": torch.rand(4, 19),
            "piece_ids_a": torch.tensor([0, 0, 1, 1]),
            "piece_ids_b": torch.tensor([0, 0, 1, 1]),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
```

**Step 2:** Run tests, verify failure

**Step 3: Implement TransformerSymbolicEncoder**

```python
class TransformerSymbolicEncoder(pl.LightningModule):
    """S1: BERT-style Transformer on REMI-tokenized MIDI."""
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 hidden_dim, num_labels, dropout=0.1, max_seq_len=2048,
                 stage="pretrain", learning_rate=1e-4, weight_decay=1e-5,
                 max_epochs=200): ...

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        """Pooled embedding z_symbolic [B, hidden_dim].""" ...

    def forward(self, input_ids, attention_mask) -> dict: ...
```

Architecture: token embed + positional + N TransformerEncoder layers + attention pooling + projection + ranking/regression heads. Pretrain: masked token prediction. Finetune: ranking + regression.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add TransformerSymbolicEncoder (S1) with pretrain/finetune stages"
```

---

### Task 12: S2 - GNN on Score Graph

**Files:**

- Modify: `model/src/model_improvement/symbolic_encoders.py` (add GNNSymbolicEncoder)
- Modify: `model/tests/model_improvement/test_symbolic_encoders.py`

**Step 1: Write failing tests**

```python
from model_improvement.symbolic_encoders import GNNSymbolicEncoder


class TestGNNSymbolicEncoder:
    def test_forward_shape(self):
        model = GNNSymbolicEncoder(
            node_features=6, hidden_dim=512, num_layers=4, num_labels=19,
        )
        x = torch.randn(20, 6)  # pitch, velocity, onset, duration, pedal, voice
        edge_index = torch.randint(0, 20, (2, 40))
        batch_vec = torch.zeros(20, dtype=torch.long)
        out = model(x, edge_index, batch_vec)
        assert out["z_symbolic"].shape == (1, 512)
        assert out["scores"].shape == (1, 19)

    def test_link_prediction_step(self):
        model = GNNSymbolicEncoder(
            node_features=6, hidden_dim=512, num_layers=4, num_labels=19,
            stage="pretrain",
        )
        batch = {
            "x": torch.randn(20, 6),
            "edge_index": torch.randint(0, 20, (2, 40)),
            "batch": torch.zeros(20, dtype=torch.long),
            "pos_edges": torch.randint(0, 20, (2, 10)),
            "neg_edges": torch.randint(0, 20, (2, 10)),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
```

**Step 2:** Run tests, verify failure

**Step 3: Implement GNNSymbolicEncoder**

Uses torch-geometric GATConv layers. Node features: [pitch, velocity, onset, duration, pedal, voice]. Edge types: temporal adjacency, harmonic interval, voice. Pretrain: link prediction. Finetune: ranking + regression.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add GNNSymbolicEncoder (S2) with graph attention and link prediction"
```

---

### Task 13: S3 - Continuous MIDI Encoder

**Files:**

- Modify: `model/src/model_improvement/symbolic_encoders.py` (add ContinuousSymbolicEncoder)
- Modify: `model/tests/model_improvement/test_symbolic_encoders.py`

**Step 1: Write failing tests**

```python
from model_improvement.symbolic_encoders import ContinuousSymbolicEncoder


class TestContinuousSymbolicEncoder:
    def test_forward_shape(self):
        model = ContinuousSymbolicEncoder(
            input_channels=5, hidden_dim=512, num_labels=19,
        )
        x = torch.randn(4, 200, 5)  # [B, T, C]
        mask = torch.ones(4, 200, dtype=torch.bool)
        out = model(x, mask)
        assert out["z_symbolic"].shape == (4, 512)
        assert out["scores"].shape == (4, 19)

    def test_contrastive_pretrain_step(self):
        model = ContinuousSymbolicEncoder(
            input_channels=5, hidden_dim=512, num_labels=19,
            stage="pretrain",
        )
        batch = {
            "features": torch.randn(4, 200, 5),
            "mask": torch.ones(4, 200, dtype=torch.bool),
            "masked_features": torch.randn(4, 200, 5),
            "masked_positions": torch.zeros(4, 200, dtype=torch.bool),
        }
        loss = model.training_step(batch, 0)
        assert loss.ndim == 0
```

**Step 2:** Run tests, verify failure

**Step 3: Implement ContinuousSymbolicEncoder**

1D CNN (multi-scale kernels: 3, 7, 15) + Transformer encoder. wav2vec-style contrastive pretraining: quantize via Gumbel-softmax codebook, predict masked frames. Attention pooling + ranking/regression heads.

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add ContinuousSymbolicEncoder (S3) with 1D-CNN + Transformer"
```

---

### Task 14: Symbolic Comparison Notebook

**Files:**

- Create: `model/notebooks/model_improvement/08_symbolic_comparison.ipynb`

**Step 1:** Create notebook:

1. Setup -- imports, paths, load checkpoints for S1/S2/S3
2. Load validation data (PercePiano MIDI path, ASAP for alignment)
3. Run MetricsSuite on each model
4. Score alignment assessment (ASAP -- measure-level accuracy, onset error)
5. Comparison table + per-dimension breakdown
6. Winner selection

**Step 2:** Commit

```bash
git commit -m "add symbolic comparison notebook (08)"
```

---

## Phase 3: Fusion

### Task 15: Fusion Modules

**Files:**

- Create: `model/src/model_improvement/fusion.py`
- Create: `model/tests/model_improvement/test_fusion.py`

**Step 1: Write failing tests**

```python
# tests/model_improvement/test_fusion.py
import torch
import pytest
from model_improvement.fusion import (
    CrossAttentionFusion,
    ConcatFusion,
    GatedFusion,
    FusedPerformanceModel,
)


class TestCrossAttentionFusion:
    def test_output_shape(self):
        fusion = CrossAttentionFusion(audio_dim=512, symbolic_dim=512, fused_dim=512)
        z_audio = torch.randn(4, 512)
        z_symbolic = torch.randn(4, 512)
        z_fused = fusion(z_audio, z_symbolic)
        assert z_fused.shape == (4, 512)


class TestGatedFusion:
    def test_output_shape(self):
        fusion = GatedFusion(audio_dim=512, symbolic_dim=512, fused_dim=512, num_dims=19)
        z_audio = torch.randn(4, 512)
        z_symbolic = torch.randn(4, 512)
        z_fused = fusion(z_audio, z_symbolic)
        assert z_fused.shape == (4, 512)

    def test_gate_values_bounded(self):
        fusion = GatedFusion(audio_dim=512, symbolic_dim=512, fused_dim=512, num_dims=19)
        z_audio = torch.randn(4, 512)
        z_symbolic = torch.randn(4, 512)
        gates = fusion.get_gates(z_audio, z_symbolic)
        assert gates.shape == (4, 19)
        assert (gates >= 0).all() and (gates <= 1).all()


class TestFusedPerformanceModel:
    def test_forward_from_embeddings(self):
        model = FusedPerformanceModel(
            audio_encoder=None, symbolic_encoder=None,
            fusion_type="cross_attention", num_labels=19,
            audio_dim=512, symbolic_dim=512,
        )
        batch = {"z_audio": torch.randn(4, 512), "z_symbolic": torch.randn(4, 512)}
        out = model(batch)
        assert out["scores"].shape == (4, 19)

    def test_score_conditioned_quality(self):
        model = FusedPerformanceModel(
            audio_encoder=None, symbolic_encoder=None,
            fusion_type="cross_attention", num_labels=19,
            audio_dim=512, symbolic_dim=512,
            use_score_conditioning=True,
        )
        batch = {
            "z_audio": torch.randn(4, 512),
            "z_performance_midi": torch.randn(4, 512),
            "z_score_midi": torch.randn(4, 512),
        }
        out = model(batch)
        assert out["scores"].shape == (4, 19)
```

**Step 2:** Run tests, verify failure

**Step 3: Implement fusion modules**

```python
class CrossAttentionFusion(nn.Module):
    """F1: Bidirectional cross-attention.""" ...

class ConcatFusion(nn.Module):
    """F2: Concatenation baseline.""" ...

class GatedFusion(nn.Module):
    """F3: Per-dimension learned gating.""" ...
    def get_gates(self, z_audio, z_symbolic) -> torch.Tensor: ...

class FusedPerformanceModel(pl.LightningModule):
    """Frozen encoders + fusion + heads. Supports score conditioning."""
    def __init__(self, audio_encoder, symbolic_encoder, fusion_type,
                 num_labels, audio_dim, symbolic_dim,
                 use_score_conditioning=False, freeze_encoders=True): ...
```

**Step 4:** Run tests, verify pass

**Step 5:** Commit

```bash
git commit -m "add fusion modules (cross-attention, concat, gated) and FusedPerformanceModel"
```

---

### Task 16: Fusion + Robustness Notebooks

**Files:**

- Create: `model/notebooks/model_improvement/09_fusion_experiments.ipynb`
- Create: `model/notebooks/model_improvement/10_robustness_validation.ipynb`

**Step 1:** Create fusion notebook (09):

1. Load best audio + symbolic encoder checkpoints
2. Extract embeddings on PercePiano
3. Train F1/F2/F3, compare
4. Score-conditioned quality experiment
5. Winner selection

**Step 2:** Create robustness notebook (10):

1. Load final model
2. Augmented test set metrics
3. Per-augmentation breakdown
4. Real phone recording pilot (qualitative)
5. Score alignment on ASAP (symbolic)
6. Final pass/fail against robustness veto threshold

**Step 3:** Commit

```bash
git commit -m "add fusion (09) and robustness validation (10) notebooks"
```

---

## Remote Training Environment (Thunder Compute)

All training notebooks run on Thunder Compute GPU instances. Follow the existing pattern from `notebooks/disentanglement/disentanglement_experiments.ipynb`.

### Setup Pattern for Each Notebook

Every training notebook should include a setup cell at the top:

```python
# -- Thunder Compute Setup --
# 1. Clone repo
# !git clone <repo-url> /workspace/crescendai
# %cd /workspace/crescendai/model

# 2. Install dependencies
# !uv sync

# 3. Pull cached data + embeddings from Google Drive via rclone
# !rclone sync gdrive:crescendai/model/data ./data --progress

# 4. Configure paths
import os
IS_REMOTE = os.environ.get("THUNDER_COMPUTE", False)
if IS_REMOTE:
    DATA_DIR = Path("/workspace/crescendai/model/data")
    CHECKPOINT_DIR = Path("/workspace/crescendai/model/checkpoints/model_improvement")
else:
    DATA_DIR = Path("../data")
    CHECKPOINT_DIR = Path("../checkpoints/model_improvement")
```

### Persistence via rclone

Checkpoints and results must be synced back to Google Drive after each fold/experiment completes, since Thunder Compute instances are ephemeral:

```python
def upload_checkpoint(local_path: Path, remote_subdir: str):
    """Sync checkpoint to Google Drive after each fold completes."""
    import subprocess
    remote = f"gdrive:crescendai/model/checkpoints/model_improvement/{remote_subdir}"
    subprocess.run(["rclone", "copy", str(local_path), remote, "--progress"], check=True)
```

Use as the `on_fold_complete` callback in the training runner (same pattern as existing disentanglement experiments).

### Thunder Compute Best Practices

Before implementing the notebooks, research and document:

- Optimal instance type for these experiments (A100 40GB vs 80GB vs H100)
- Maximum session duration and auto-save strategy
- Whether rclone is pre-installed or needs setup per instance
- GPU memory requirements for MuQ fine-tuning with LoRA (estimate: ~20GB for batch_size=32)
- GPU memory requirements for full MuQ unfreeze (estimate: ~40GB+, may need A100 80GB)
- Cost estimates per experiment

Reference the existing Thunder Compute setup in `notebooks/disentanglement/disentanglement_experiments.ipynb` cells 2-7 for the proven rclone + Google Drive pattern.

### Data Sync Strategy

```
Google Drive (persistent):
  crescendai/model/data/
    percepiano_cache/        -- MuQ embeddings + labels
    maestro_cache/           -- MIDI + audio pairs
    atepp_cache/             -- new: download and cache
    giant_midi_cache/        -- new: tokenized MIDI
    competition_cache/       -- new: audio + metadata
    augmentation_assets/     -- room IRs, noise clips
  crescendai/model/checkpoints/
    model_improvement/
      A1/, A2/, A3/          -- audio encoder checkpoints per fold
      S1/, S2/, S3/          -- symbolic encoder checkpoints per fold
      F1/, F2/, F3/          -- fusion checkpoints

Thunder Compute (ephemeral):
  /workspace/crescendai/     -- cloned repo
  Sync down at start: rclone sync gdrive:crescendai/model/data ./data
  Sync up after each fold: rclone copy ./checkpoints gdrive:crescendai/model/checkpoints
```

---

## Execution Notes

- **Phase 0 (Tasks 1-5):** Sequential start with Task 1, then 2-5 in parallel. Run locally.
- **Phase 1 (Tasks 6-10)** and **Phase 2 (Tasks 11-14):** Fully parallel. Code locally, train on Thunder Compute.
- **Phase 3 (Tasks 15-16):** After Phase 1 and 2 winners selected. Train on Thunder Compute.
- **Total: 16 tasks across 4 phases.**
- All models follow existing PyTorch Lightning patterns from `src/disentanglement/`.
- All tests run locally with: `cd model && python -m pytest tests/model_improvement/ -v`
- All training runs on Thunder Compute with rclone persistence to Google Drive.
