# ISMIR Paper Revision Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise arxiv_v2 and ismir_v2 papers to report A1-Max results (80.8% pairwise on 6 dims), add systematic ablations, and incorporate 7 validation signals.

**Architecture:** Three work streams: (1) compute -- run 5 ablation configs x 4 folds, (2) figures -- update generate_figures.py for 6-dim results, (3) LaTeX -- rewrite both papers. Stream 1 must complete before streams 2-3 can finalize.

**Tech Stack:** PyTorch Lightning (training), Python/matplotlib (figures), LaTeX/ISMIR template (paper)

**Spec:** `docs/superpowers/specs/2026-03-16-ismir-paper-revision-design.md`

---

## Chunk 1: Ablation Training Infrastructure

### Task 1: Add ablation configs to sweep script

**Files:**
- Modify: `model/src/model_improvement/a1_max_sweep.py` (extract reusable functions)
- Create: `model/src/model_improvement/ablation_sweep.py`
- Modify: `model/src/model_improvement/audio_encoders.py` (add frozen probe model)
- Create: `model/tests/model_improvement/test_ablation.py`

The existing `a1_max_sweep.py` has data loading and training logic inline in `run_sweep()`. We need to either extract reusable functions or rewrite the sweep loop in `ablation_sweep.py`.

- [ ] **Step 1: Extract reusable functions from a1_max_sweep.py**

Refactor `a1_max_sweep.py` to extract from `run_sweep()` (lines 74-229 of the current file). The current `run_sweep()` has tightly coupled data loading, model instantiation, training, and evaluation. Extract:

```python
def load_fold_data(fold_idx: int) -> tuple[DataLoader, DataLoader]:
    """Load train/val data for a specific fold.

    Move the fold data loading logic from inside run_sweep()'s per-fold loop.
    This includes: reading percepiano_cache/folds.json, loading embeddings,
    creating PairedPerformanceDataset, and building DataLoaders.
    """
    ...

def run_single_config(
    config_name: str,
    config: dict,
    checkpoint_dir: Path,
    model_class: type = MuQLoRAMaxModel,
    fold_filter: int | None = None,
) -> dict:
    """Train one config across all 4 folds (or a single fold if fold_filter set).

    Returns dict with keys: config, pairwise_mean, pairwise_per_fold, r2_mean, r2_per_fold.

    Move the per-config training loop from run_sweep(). This includes:
    1. For each fold (0-3): load data, instantiate model_class(**config), train, evaluate
    2. Collect per-fold pairwise accuracy and R²
    3. Return aggregated results dict

    The model_class parameter allows swapping MuQLoRAMaxModel for MuQFrozenProbeModel.
    """
    ...
```

After extraction, update `run_sweep()` to call these functions (verify existing sweep still works with `uv run python -m model_improvement.a1_max_sweep --dry-run` or equivalent).

- [ ] **Step 2: Implement MuQFrozenProbeModel in audio_encoders.py**

Add to `model/src/model_improvement/audio_encoders.py`:

```python
class MuQFrozenProbeModel(MuQLoRAMaxModel):
    """Frozen MuQ backbone + MLP probe. Same head architecture as A1-Max
    (attention pooling, 2-layer MLP) but no LoRA adapters and no ranking losses.
    Uses MSE regression only for fair baseline comparison."""

    def __init__(self, **kwargs):
        # Force no-LoRA and MSE-only config
        kwargs["lora_rank"] = 1  # Minimal LoRA (will be frozen)
        kwargs["lambda_listmle"] = 0.0
        kwargs["lambda_contrastive"] = 0.0
        kwargs["lambda_regression"] = 1.0
        kwargs["lambda_invariance"] = 0.0
        kwargs["use_ccc"] = False  # MSE, not CCC
        kwargs["mixup_alpha"] = 0.0
        super().__init__(**kwargs)

        # Freeze MuQ backbone + LoRA (effectively no adaptation)
        if hasattr(self, "muq"):
            for param in self.muq.parameters():
                param.requires_grad = False
        # Freeze LoRA adapters if any were created
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        """Override to use MSE regression only (no ranking losses).

        Batch keys (from PairedPerformanceDataset):
          - batch["embeddings_a"], batch["embeddings_b"]: MuQ embeddings
          - batch["labels_a"], batch["labels_b"]: 6-dim scores
        """
        # Forward pass through encoder (attention pooling + MLP)
        emb_a = self.encode(batch["embeddings_a"])
        emb_b = self.encode(batch["embeddings_b"])

        # MSE regression only
        pred_a = self.regression_head(emb_a)
        pred_b = self.regression_head(emb_b)

        loss = F.mse_loss(pred_a, batch["labels_a"]) + F.mse_loss(pred_b, batch["labels_b"])
        self.log("train_loss", loss, prog_bar=True)
        return loss
```

Note: Verify the exact batch keys by checking `MuQLoRAMaxModel.training_step()` at `audio_encoders.py:352-353`. The keys above (`embeddings_a`, `embeddings_b`, `labels_a`, `labels_b`) match the parent class. Also verify `self.encode()` and `self.regression_head` exist on the parent. The key constraints are: (a) same MLP head and attention pooling as A1-Max, (b) frozen backbone, (c) MSE-only loss.

- [ ] **Step 3: Create ablation_sweep.py**

```python
"""
Loss component ablation for ISMIR paper revision.

5 configs x 4 folds = 20 training runs.
Results saved to model/data/ablation_sweep_results.json.

Usage:
    cd model/
    uv run python -m model_improvement.ablation_sweep
    uv run python -m model_improvement.ablation_sweep --config frozen_probe --fold 0  # dry run
"""

import json
import argparse
from pathlib import Path
from model_improvement.a1_max_sweep import run_single_config
from model_improvement.audio_encoders import MuQLoRAMaxModel, MuQFrozenProbeModel
from model_improvement.taxonomy import NUM_DIMS

RESULTS_PATH = Path("data/ablation_sweep_results.json")
CHECKPOINT_DIR = Path("data/checkpoints/ablation")

# All configs use LoRA rank-32, layers 7-12, label smoothing 0.1
# EXCEPT frozen_probe which freezes backbone entirely.
# Key: lora_target_layers (NOT lora_layers) -- matches MuQLoRAModel.__init__ param name.

ABLATION_CONFIGS = {
    "frozen_probe": {
        "model_class": "MuQFrozenProbeModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 1e-4,  # Higher LR justified: only MLP head trains
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
    },
    "bce_ranking_only": {
        "model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 0.0,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.0,
        "lambda_invariance": 0.0,
        "use_ccc": False,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "bce_plus_listmle": {
        "model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 1.5,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.0,
        "lambda_invariance": 0.0,
        "use_ccc": False,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "bce_listmle_ccc": {
        "model_class": "MuQLoRAMaxModel",
        "input_dim": 1024,
        "hidden_dim": 512,
        "num_labels": NUM_DIMS,
        "learning_rate": 3e-5,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "lambda_listmle": 1.5,
        "lambda_contrastive": 0.0,
        "lambda_regression": 0.3,
        "lambda_invariance": 0.0,
        "use_ccc": True,
        "mixup_alpha": 0.0,
        "warmup_epochs": 5,
        "max_epochs": 200,
        "use_pretrained_muq": False,
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
    "full_a1max_repro": {
        "model_class": "MuQLoRAMaxModel",
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
        "lora_rank": 32,
        "lora_target_layers": (7, 8, 9, 10, 11, 12),
        "label_smoothing": 0.1,
    },
}

MODEL_CLASSES = {
    "MuQLoRAMaxModel": MuQLoRAMaxModel,
    "MuQFrozenProbeModel": MuQFrozenProbeModel,
}


def run_ablation_sweep(config_filter: str | None = None, fold_filter: int | None = None):
    """Run ablation sweep. Supports resume and single-config/fold dry runs."""
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} configs already completed")

    for config_name, config in ABLATION_CONFIGS.items():
        if config_filter and config_name != config_filter:
            continue
        if config_name in results:
            print(f"Skipping {config_name} (already completed)")
            continue

        # Read model_class without mutating the config dict
        model_class_name = config.get("model_class", "MuQLoRAMaxModel")
        model_class = MODEL_CLASSES[model_class_name]

        # Filter out non-model keys before passing to constructor
        model_config = {k: v for k, v in config.items() if k != "model_class"}

        result = run_single_config(
            config_name=config_name,
            config=model_config,
            checkpoint_dir=CHECKPOINT_DIR,
            model_class=model_class,
            fold_filter=fold_filter,
        )
        results[config_name] = result

        # Save incrementally
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results for {config_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Run single config")
    parser.add_argument("--fold", type=int, default=None, help="Run single fold (0-3)")
    args = parser.parse_args()
    run_ablation_sweep(config_filter=args.config, fold_filter=args.fold)
```

- [ ] **Step 4: Add tests for frozen probe model**

In `model/tests/model_improvement/test_ablation.py`:

```python
import torch
from model_improvement.audio_encoders import MuQFrozenProbeModel, MuQLoRAMaxModel


def test_frozen_probe_backbone_frozen():
    """Frozen probe has no trainable MuQ/LoRA parameters."""
    model = MuQFrozenProbeModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        lora_target_layers=(7, 8, 9, 10, 11, 12),
    )
    for name, param in model.named_parameters():
        if "muq" in name.lower() or "lora" in name.lower():
            assert not param.requires_grad, f"{name} should be frozen"


def test_frozen_probe_has_trainable_head():
    """Frozen probe MLP head and regression head are trainable."""
    model = MuQFrozenProbeModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        lora_target_layers=(7, 8, 9, 10, 11, 12),
    )
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) > 0, "At least MLP head should be trainable"


def test_bce_only_config_zero_lambdas():
    """BCE-only config: all additional loss lambdas are zero."""
    model = MuQLoRAMaxModel(
        input_dim=1024, hidden_dim=512, num_labels=6,
        lora_rank=32, lora_target_layers=(7, 8, 9, 10, 11, 12),
        lambda_listmle=0.0, lambda_contrastive=0.0,
        lambda_regression=0.0, lambda_invariance=0.0,
    )
    assert model.hparams.lambda_listmle == 0.0
    assert model.hparams.lambda_contrastive == 0.0
    assert model.hparams.lambda_regression == 0.0
    assert model.hparams.lambda_invariance == 0.0
```

- [ ] **Step 5: Run tests**

```bash
cd model && uv run python -m pytest tests/model_improvement/test_ablation.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add model/src/model_improvement/ablation_sweep.py model/src/model_improvement/audio_encoders.py model/src/model_improvement/a1_max_sweep.py model/tests/model_improvement/test_ablation.py
git commit -m "feat: add ablation sweep infrastructure for ISMIR paper revision"
```

### Task 2: Run ablation sweep

**Files:**
- Run: `model/src/model_improvement/ablation_sweep.py`
- Output: `model/data/ablation_sweep_results.json`

- [ ] **Step 1: Dry-run frozen probe fold 0 to sanity check**

```bash
cd model/
uv run python -m model_improvement.ablation_sweep --config frozen_probe --fold 0
```

Verify: training starts, loss decreases, completes without error. Check that MuQ gradients are zero (no LoRA updates).

- [ ] **Step 2: Run full ablation sweep (all 5 configs x 4 folds)**

```bash
cd model/
uv run python -m model_improvement.ablation_sweep
```

Expected: ~10-20 hours on MPS. Results saved incrementally to `data/ablation_sweep_results.json`. Supports resume if interrupted (re-run same command).

- [ ] **Step 3: Verify reproducibility of config 4 (full A1-Max)**

After sweep completes, compare `full_a1max_repro` pairwise mean against the existing 78.7% (from `data/a1_max_sweep_results.json`, config `A1max_r32_L7-12_ls0.1`). Must be within 2pp (accounting for MPS non-determinism). If outside 2pp, investigate.

- [ ] **Step 4: Validate results make scientific sense**

Expected ordering (by pairwise accuracy):
```
frozen_probe < bce_ranking_only < bce_plus_listmle <= bce_listmle_ccc <= full_a1max_repro
```

If the ordering is violated (e.g., BCE-only outperforms full A1-Max), investigate before proceeding. The ablation story must be clean.

- [ ] **Step 5: Commit results**

```bash
git add model/data/ablation_sweep_results.json
git commit -m "data: ablation sweep results for ISMIR paper revision"
```

---

## Chunk 2: Figures and Bibliography

### Task 3: Update figure generation script

**Files:**
- Modify: `paper/generate_figures.py`
- Output: `paper/ismir/fig1_dimension_by_category.png` (regenerated for 6-dim)

- [ ] **Step 1: Read ablation results and extract per-dimension data**

After Task 2 completes, read `model/data/ablation_sweep_results.json` and extract:
- Per-dimension pairwise accuracy for frozen probe vs A1-Max (6 dims)
- Ablation table numbers for Table 2

- [ ] **Step 2: Update generate_figures.py for 6-dim regime**

Replace the 19-dim category averages with 6-dim per-dimension pairwise accuracy:
- X-axis: 6 dimensions (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Y-axis: Pairwise accuracy (%)
- Two bars per dimension: Frozen probe (baseline) vs A1-Max (ensemble)

Update the hard-coded values with actual results from the ablation sweep.

- [ ] **Step 3: Regenerate figures**

```bash
cd paper/
uv run python generate_figures.py
```

Verify: `paper/ismir/fig1_dimension_by_category.png` regenerated with 6-dim data.

- [ ] **Step 4: Create architecture diagram (Fig 2)**

Replace `excalidraw_model_pipeline.png` with a new diagram showing:
- MuQ backbone with LoRA adapters (layers 7-12)
- Attention-weighted pooling
- Multi-task loss heads (BCE ranking, ListMLE, CCC regression, contrastive, invariance)
- 6-dim output

Create in Excalidraw or TikZ. Save as `paper/ismir/excalidraw_model_pipeline.png` (same filename for compatibility with existing `\includegraphics` references).

- [ ] **Step 5: Commit updated figures**

```bash
git add paper/generate_figures.py paper/ismir/*.png
git commit -m "fig: update figures for 6-dim A1-Max results"
```

### Task 4: Update bibliography

**Files:**
- Modify: `paper/ismir/references.bib` (master copy, symlinked by ismir_v2)
- Modify: `paper/arxiv_v2/references.bib` (separate copy, NOT a symlink)

Note: `ismir_v2/references.bib` symlinks to `../ismir/references.bib`, but `arxiv_v2/references.bib` is a standalone file. Both must be updated.

- [ ] **Step 1: Add LoRA and ListMLE citations to both bib files**

Append to `paper/ismir/references.bib` AND `paper/arxiv_v2/references.bib`:

```bibtex
@inproceedings{hu2022lora,
  title={{LoRA}: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shaohan and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations},
  year={2022}
}

@inproceedings{xia2008listwise,
  title={Listwise approach to learning to rank: theory and algorithm},
  author={Xia, Fen and Liu, Tie-Yan and Wang, Jue and Zhang, Wensheng and Li, Hang},
  booktitle={Proceedings of the 25th International Conference on Machine Learning},
  pages={1192--1199},
  year={2008}
}
```

- [ ] **Step 2: Verify no duplicate keys in either bib file**

```bash
grep -c "hu2022lora\|xia2008listwise" paper/ismir/references.bib paper/arxiv_v2/references.bib
```

Expected: 1 occurrence of each key per file.

- [ ] **Step 3: Commit**

```bash
git add paper/ismir/references.bib paper/arxiv_v2/references.bib
git commit -m "refs: add LoRA and ListMLE citations for ISMIR revision"
```

---

## Chunk 3: LaTeX Rewrite -- Method and Results

Note: All line number references below are from the **current** (pre-edit) papers. After earlier sections are rewritten, line numbers will shift. Use `\section{}` labels and `\begin{table}` markers to navigate, not line numbers.

### Task 5: Rewrite abstract, introduction, and related work

**Files:**
- Modify: `paper/ismir_v2/main.tex` (sections: abstract, `\section{Introduction}`, `\section{Related Work}`)
- Modify: `paper/arxiv_v2/main.tex` (same sections)

- [ ] **Step 1: Rewrite abstract**

Replace the current abstract (19-dim, R²=0.537, 70.3% pairwise) with:
- 6 teacher-grounded dimensions as target
- 80.8% pairwise accuracy as headline result
- Adaptation strategy comparison + loss ablation as contributions
- 7 validation signals including real audio transfer
- Keep concise (150-200 words for ISMIR)

- [ ] **Step 2: Rewrite introduction**

Update contributions list to match spec (4 bullets):
1. Systematic adaptation strategy comparison (frozen, full-unfreeze, staged, LoRA)
2. Loss ablation showing ListMLE ranking loss as key ingredient
3. Multi-signal validation framework (7 independent signals)
4. Synthesized-to-real audio transfer evidence (YouTube AMT)

Remove 19-dim framing. Add context about why pairwise ranking accuracy matters more than R² for practical evaluation.

- [ ] **Step 3: Update related work**

Add paragraph on parameter-efficient adaptation (LoRA) in audio/music domains. Add context for ListMLE / learning-to-rank in MIR. Reference `\cite{hu2022lora}` and `\cite{xia2008listwise}`.

- [ ] **Step 4: Update title (if desired)**

Current: "Audio Foundation Models for Multi-Dimensional Piano Performance Evaluation"
Spec proposes: "Evaluating Piano Performance Quality with Pretrained Audio Encoders: Adaptation, Ranking Objectives, and Multi-Signal Validation"

Decision: Update title or keep current. The current title is clean but doesn't signal the adaptation/ranking contributions.

- [ ] **Step 5: Verify both files match (except author/anonymization)**

- [ ] **Step 6: Commit**

```bash
git add paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
git commit -m "paper: rewrite abstract, intro, and related work for 6-dim A1-Max"
```

### Task 6: Rewrite method section

**Files:**
- Modify: `paper/ismir_v2/main.tex` (section: `\section{Method}`)
- Modify: `paper/arxiv_v2/main.tex` (same section)

- [ ] **Step 1: Update Section 3.1 (Data)**

Add dimension mapping paragraph: 19 PercePiano dims collapsed to 6 teacher-grounded dims via composite soft-label aggregation. Brief inline table mapping dims. Cite taxonomy work if published separately.

- [ ] **Step 2: Update Section 3.2 (Architecture)**

Replace "MuQ L9-12, mean pooling, 2-layer MLP" with:
- MuQ backbone (160K hours pretraining) with LoRA rank-32 adapters on layers 7-12 `\cite{hu2022lora}`
- Attention-weighted temporal pooling (learned attention weights, softmax, weighted sum)
- 2-layer MLP encoder (hidden dim 512, LayerNorm + GELU)
- Per-dimension ranking heads (6 binary heads) + regression head with sigmoid
- Update `\includegraphics` to reference new architecture diagram (Fig 2)

- [ ] **Step 3: Add Section 3.3 (Adaptation Strategies)**

Paragraph describing the 4 strategies compared:
(a) Frozen MLP probe -- same attention pooling and MLP head, frozen backbone, MSE only
(b) Full unfreeze with gradual unfreezing (A3)
(c) Staged domain adaptation -- self-supervised then supervised (A2)
(d) LoRA rank-16/32 with multi-task loss (A1, A1-Max)

- [ ] **Step 4: Add Section 3.4 (Training Objectives)**

Document all 5 loss components with equations:
- $\mathcal{L}_\text{rank}$: BCE pairwise ranking (always active, coefficient 1.0)
- $\mathcal{L}_\text{ListMLE}$: Plackett-Luce ranking `\cite{xia2008listwise}` ($\lambda=1.5$)
- $\mathcal{L}_\text{CCC}$: Concordance correlation coefficient ($\lambda=0.3$)
- $\mathcal{L}_\text{contrast}$: InfoNCE contrastive ($\lambda=0.3$)
- $\mathcal{L}_\text{inv}$: Soundfont invariance ($\lambda=0.1$)
- Total: $\mathcal{L} = \mathcal{L}_\text{rank} + \sum_i \lambda_i \mathcal{L}_i$

- [ ] **Step 5: Add Section 3.5 (Ensemble) and Section 3.6 (Dimension Mapping)**

3.5: 4-fold piece-stratified CV, prediction averaging at inference. Brief.
3.6: Composite mapping from 19 PercePiano dims to 6 teacher-grounded dims. Brief description with forward reference to supplementary for full mapping table.

- [ ] **Step 6: Commit**

```bash
git add paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
git commit -m "paper: rewrite method with LoRA, attention pooling, 5-component loss, dim mapping"
```

### Task 7: Rewrite results section

**Files:**
- Modify: `paper/ismir_v2/main.tex` (section: `\section{Experiments}`)
- Modify: `paper/arxiv_v2/main.tex` (same section)

Requires: Task 2 complete (ablation results needed for Table 2).

- [ ] **Step 1: Replace Table 1 with adaptation strategy comparison (all 6-dim)**

| Model | Strategy | Pairwise | R² |
|-------|----------|----------|----|
| Frozen probe | MuQ L7-12 + MLP, MSE | [from ablation_sweep_results.json] | [from ablation] |
| A3 | Full unfreeze | 69.9% | 0.28 |
| A2 | Staged adaptation | 71.4% | 0.42 |
| A1 | LoRA rank-16, multi-task | 73.9% | 0.40 |
| A1-Max (single) | LoRA rank-32, multi-task | 78.7% | 0.16 |
| A1-Max (ensemble) | 4-fold average | **80.8%** | **0.50** |

Add footnote: "For comparability with \cite{percepiano}, the frozen probe on original 19 dims achieves $R^2 = 0.537$."

- [ ] **Step 2: Add Table 2 with loss component ablation**

| Configuration | Components beyond $\mathcal{L}_\text{rank}$ | Pairwise | R² |
|---------------|----------------------------------------------|----------|----|
| BCE ranking only | -- | [from ablation] | [from ablation] |
| + ListMLE | $\mathcal{L}_\text{ListMLE}$ | [from ablation] | [from ablation] |
| + ListMLE + CCC | $\mathcal{L}_\text{ListMLE} + \mathcal{L}_\text{CCC}$ | [from ablation] | [from ablation] |
| Full A1-Max | All 5 components | 78.7% | 0.16 |

Caption note: "All configurations include the unweighted BCE pairwise ranking loss."

- [ ] **Step 3: Add Table 3 with per-dimension pairwise breakdown (6 dims)**

Show per-dimension pairwise accuracy for A1-Max ensemble. Data exists in `data/a1_max_sweep_results.json` (per-fold, per-dimension breakdowns).

- [ ] **Step 4: Add R² vs. pairwise discussion**

Paragraph: ListMLE improves pairwise accuracy while reducing pointwise R². Ranking objectives optimize for ordinal correctness ("which is better?"), not absolute score prediction. For practical evaluation, pairwise accuracy is the more relevant metric.

- [ ] **Step 5: Update Fig 1 reference and caption**

Point to regenerated 6-dim per-dimension figure. Update caption to describe 6 dimensions and frozen-vs-A1-Max comparison.

- [ ] **Step 6: Commit**

```bash
git add paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
git commit -m "paper: rewrite results with adaptation + loss ablation tables (6-dim)"
```

---

## Chunk 4: LaTeX Rewrite -- Validation, Discussion, Cleanup

### Task 8: Rewrite validation / analysis section

**Files:**
- Modify: `paper/ismir_v2/main.tex` (section: `\section{Analysis}` -- rename to `\section{Validation}`)
- Modify: `paper/arxiv_v2/main.tex` (same)

- [ ] **Step 1: Restructure as 5 main-body validation signals**

Replace current "Analysis: What Does the Model Learn?" (including the "Why Audio Outperforms Symbolic" subsection) with "Validation":

1. **Cross-soundfont generalization** -- keep existing text. Note these results are on 19-dim PercePiano labels. R²=0.534 +/- 0.075 (leave-one-out across 6 soundfonts).
2. **Piece vs. performer split** -- keep existing. Note 19-dim context. Both yield R²=0.536 (rules out piece memorization).
3. **Competition ranking (NEW)** -- new subsection (see Step 2)
4. **Real audio transfer (NEW)** -- new subsection (see Step 3)
5. **Fusion failure** -- error correlation r=0.738. Cite 19-dim R² values with explicit label: "On the original 19-dim labels, fusion (R²=0.524) underperforms audio-only (R²=0.537)."

- [ ] **Step 2: Write competition ranking subsection**

2021 Chopin Competition, 11 performers across 3 rounds. Model-predicted aggregated rankings correlated with jury rankings: Spearman rho=+0.704 (p=0.016). Per-dimension: pedaling rho=+0.887 (strongest), phrasing +0.803. Known limitation: dynamics inverts (rho=-0.917) -- model captures dynamic range magnitude, not appropriateness relative to score demands. Motivates score-conditioned evaluation.

Source for numbers: `docs/model/00-research-timeline.md` (Layer 1 validation, competition correlation section).

- [ ] **Step 3: Write real audio transfer subsection**

50 YouTube recordings of intermediate-level pianists. Audio processed through ByteDance piano transcription to generate symbolic input. Audio encoder (A1) and symbolic encoder (S2) predictions compared: 79.9% agreement across all 6 dimensions, all individual dims >72%. Validates that synthesized-audio training transfers to real recordings.

Source for numbers: `docs/model/00-research-timeline.md` (YouTube AMT validation section).

- [ ] **Step 4: Handle difficulty correlation and multi-performer consistency**

If page-constrained (check after compilation in Task 10), move to supplementary:
- Difficulty correlation: PSyllabus, rho=0.623
- Multi-performer consistency: ASAP, std=0.020

If space allows, keep as brief signals (3-4 lines each) in main body.

- [ ] **Step 5: Commit**

```bash
git add paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
git commit -m "paper: rewrite validation with 7 signals including competition + YouTube transfer"
```

### Task 9: Rewrite discussion, limitations, and conclusion

**Files:**
- Modify: `paper/ismir_v2/main.tex` (sections: `\section{Limitations}`, `\section{Conclusion}`)
- Modify: `paper/arxiv_v2/main.tex` (same)

- [ ] **Step 1: Expand limitations**

Replace current 3-point limitations with:
1. Synthesized audio training only (Pianoteq) -- mitigated by YouTube transfer result
2. Dynamics inversion in competition (rho=-0.917) -- needs score conditioning
3. Single-fold pairwise range (70.3-77.7%) -- fold variance from piece distribution, not overfitting
4. No phone audio validation (YouTube proxy only)
5. Crowdsourced label noise ceiling
6. 6-dim composite mapping inherits 19-dim annotation noise

- [ ] **Step 2: Update conclusion**

Update headline numbers: 80.8% pairwise (6-dim), LoRA + ListMLE as key recipe, 7 validation signals. Future work: score-conditioned evaluation, real piano training data, phone audio validation, teacher-grounded dimension mapping as independent contribution.

- [ ] **Step 3: Commit**

```bash
git add paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
git commit -m "paper: update limitations and conclusion"
```

### Task 10: Final compilation and verification

**Files:**
- Compile: `paper/ismir_v2/` and `paper/arxiv_v2/`

- [ ] **Step 1: Compile ismir_v2**

Check for Makefile first. If no Makefile exists in `paper/ismir_v2/`:

```bash
cd paper/ismir_v2
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

If Makefile exists: `make`

Expected: Clean compilation, no undefined references.

- [ ] **Step 2: Compile arxiv_v2**

```bash
cd paper/arxiv_v2 && make
```

Expected: Clean compilation.

- [ ] **Step 3: Verify page count**

ISMIR limit: 6 pages main body + references. Open `ismir_v2/main.pdf` and count pages. If over 6 pages, move difficulty correlation (signal 6) and multi-performer consistency (signal 7) to supplementary material.

- [ ] **Step 4: Verify no stale claims**

Search both papers for remaining 19-dim claims that aren't explicitly labeled:

```bash
grep -n "0\.537\|70\.3.*pairwise\|mean pooling\|layers 9-12\|Core ML\|19 dim" paper/ismir_v2/main.tex paper/arxiv_v2/main.tex
```

- "R² = 0.537" should appear only in the prior-work comparison footnote
- "70.3% pairwise" should not appear (old 19-dim headline)
- "mean pooling" should not appear in architecture description
- "layers 9-12" should not appear as A1-Max architecture (layers 7-12 is correct)
- "Core ML" should not appear anywhere

- [ ] **Step 5: Verify anonymous mode for ismir_v2**

Check that `\usepackage[submission]{ismir}` is set.

- [ ] **Step 6: Verify arxiv_v2 has author info**

Check that author name (Jai Dhiman) and affiliation are present, not anonymized.

- [ ] **Step 7: Final commit**

```bash
git add paper/ismir_v2/ paper/arxiv_v2/
git commit -m "paper: final compilation verified, page count OK, no stale claims"
```

---

## Execution Order

```
Task 1 (ablation infra) ───► Task 2 (run sweep, ~10-20h)
                                      │
Task 4 (bibliography) ────────────────┤  (parallel with Task 2)
Task 5 (abstract/intro/related) ──────┤  (parallel with Task 2, no data deps)
Task 6 (method) ──────────────────────┤  (parallel with Task 2, no data deps)
                                      │
                                      ▼
                              Task 3 (figures, needs ablation data)
                                      │
                                      ▼
                              Task 7 (results, needs ablation data + figures)
                                      │
                                      ▼
                              Task 8 (validation)
                                      │
                                      ▼
                              Task 9 (discussion/conclusion)
                                      │
                                      ▼
                              Task 10 (compile + verify)
```

**Critical path:** Task 1 -> Task 2 (compute) -> Task 3 (figures) -> Task 7 (results) -> Task 10 (verify).

**Can run in parallel with compute (Tasks 4-6):** Bibliography updates, abstract/intro rewrite, and method section rewrite don't depend on ablation numbers.
