# #138 Phase 1: LoRA Fine-Tune of MoonBeam-839M for Difficulty Ranking

**Goal:** Produce a LoRA fine-tune of MoonBeam-839M whose mean-pooled embeddings,
scored through `bakeoff_cv.py`'s own composer-disjoint folds, beat the 37 hand
features (tau-c **0.8048**) by a paired-bootstrap-significant margin at n=900,
with the same delta holding on the 709-piece real-audio subset.

**Not in scope:**
- The R3 cross-rendering check (synthesized vs human renderings). Separate follow-up issue.
- Recovering the 191 bot-blocked WAVs. Optional; would tighten the real-audio CI from ±0.017 to ±0.0155.
- Any change to the 37 hand features, to `tk_ablation.py`, or to the #137 feature frontier.
- Deploying the fine-tuned encoder anywhere. This phase produces a measurement, not a release.
- Launching HF Jobs. Every paid run is human-lit; this phase produces the scripts and the runbook.

## Problem

`docs/mirex/track-a-difficulty-prediction.md` records seven converging nulls: the
hand-crafted symbolic feature frontier is closed. #137's new Transkun-unlocked
offset/pedal family bought only +0.0064 tau-c. Phase 0 of #138 established that a
frozen MoonBeam-839M encoder with mean-over-tokens pooling reaches tau-c 0.8257,
against the 37 hand features' 0.8048 measured on the same folds — an honest lead of
only **+0.0244**. Frozen Aria loses outright to the hand features.

An encoder that is merely *read out* is therefore worth ~0.02. The unrefuted lever
is moving the encoder weights themselves. Nothing in the repo can do that today:
`moonbeam_extract_script.py` runs a forward pass under `torch.no_grad()`, there is
no training objective, no fold-aware training-set construction, and no way to score
embeddings that differ per fold. Every one of those is missing code.

The failure mode this design most guards against is not "the fine-tune doesn't
help". It is "the fine-tune appears to help because of leakage". Two specific
leaks are live here:

1. The encoder training set overlapping the 900 pieces the ridge head is fit and
   scored on.
2. The encoder training set sharing composers with the fold it is later tested on.

Both are prevented in exactly one module (`fold_plan.py`), which is why that module
carries the heaviest tests in the plan.

## Solution (from the operator's perspective)

The user runs, in order:

1. `push_train_dataset.py` once — uploads a hermetic private HF dataset (5,798
   Transkun MIDIs, labels, the five fold plans, and a pinned snapshot of the
   MoonBeam fork). Prints a bundle report with per-file counts and a checksum.
2. `hf jobs uv run --flavor a100-large ... train_fold.py --fold 0` — the pilot.
   Returns a LoRA adapter plus `emb_fold0.npz` (900 × 1920 f32, 6.9 MB), and a
   measured wall-clock that either confirms or refutes the ~1 GPU-hr estimate.
3. If the pilot moves: four more jobs, folds 1–4, same seed. ~$13 total.
4. `ft_eval.py` locally on CPU — prints the gate:
   `moonbeam_ft_mean|ridge - features37|ridge: +0.0XXX CI95[+a,+b] SIG|noise`.
5. If gate (i) passes: `realaudio_check.py`, which transcribes the 709 WAVs
   (resumable), scores each through its own fold's adapter, and prints the
   real-audio delta, the matched symbolic delta on the same 709, and MIDI drift.

At every stage the operator reads `docs/mirex/phase1-lora-runbook.md`, which holds
the exact `hf jobs uv run | ps | logs | inspect | stats | cancel` sequence and the
abort criteria.

## Design

### Training set: option D, per fold

For fold `f`, the training pool excludes **all 900 eval pieces** and **every piece
whose composer appears in fold `f`'s test set**. Re-derived against live data this
session, the pools are **3815 / 4082 / 4283 / 4028 / 4149** pieces (510/511/510/508/515
composers) — matching the issue exactly.

Composer-disjointness is a **per-fold** constraint. The "765 fully-disjoint pieces"
figure applies only if a single fine-tune must serve every fold; it does not apply
here. The consequence is that a set of per-fold adapters is **welded to one seed**:
seed 2027's test fold contains composers that seed 2026's adapters trained on, so
seed-2026 embeddings must never be re-scored under another seed.

A ~12% validation slice is carved from each pool, itself composer-disjoint from the
remaining train pieces, for early stopping.

### Objective

Pairwise logistic ranking loss over all strictly-grade-ordered pairs within a
forward batch (pairs are formed from embeddings already computed, so they are free),
plus a small-weight cumulative-link ordinal auxiliary: 10 binary "grade > k" logits
for the 11-level scale.

Rationale for pairwise-primary: the gate metric is Kendall tau-c, a rank
correlation. Optimizing squared error against integer grades optimizes the wrong
thing; the ordinal auxiliary is retained at low weight only to keep the score scale
from drifting freely, which pure pairwise loss does not pin down.

**Degenerate batches are a real case, not a hypothetical.** A micro-batch whose
pieces all share one grade yields zero ordered pairs. The loss must return a finite
zero that still participates in the autograd graph, never a NaN from a mean over an
empty tensor. This is tested directly.

### LoRA configuration

`r=16, alpha=32, dropout=0.05`, applied to the **top 5 of 15 layers only** (layers
10–14), on all seven projections per layer — `self_attn.{q,k,v,o}_proj` and
`mlp.{gate,up,down}_proj` — for **35 target modules**.

Verified against `moonbeam_839M.pt` (178 state-dict keys) and the fork's
`model_config.json`: 15 layers, hidden 1920, intermediate 6720, `max_len` 1024.

**Explicitly excluded:** `lm_head`, `decoder_embedding`, `fc_out`,
`summary_projection`. These are the fork's *default* `target_modules`
(`src/llama_recipes/configs/peft.py:11`), so accepting the default would adapt the
generative decoder heads we never invoke. Silence picks the wrong thing here, so the
target list is constructed explicitly and asserted in a test.

### Train-time vs extract-time windowing

`moonbeam_extract_script.py` chunks a piece into `max_len`=1024 windows, forwards
each, concatenates, and means over all tokens — full-piece coverage. Backpropagating
that for a long piece is unbounded memory, so **training samples one random 1024-token
window per piece per step** and mean-pools within it.

This is a deliberate, stated asymmetry (a random-crop augmentation), not an
oversight. Extraction — the path the gate actually measures — is byte-identical to
Phase 0's, so the comparison against frozen 0.8257 stays paired. The pilot fold
validates that a window-trained encoder still improves the full-piece readout; if it
does not, that is a genuine negative result about the objective, and it must be
reported as one rather than patched by quietly changing the extraction path.

### Gate metric

**(i) Encoder-as-feature-extractor is THE GATE.** Discard the trained head, extract
mean-pooled embeddings, score with RidgeCV through `bakeoff_cv.py`. Only the encoder
weights move, so it is exactly paired against frozen 0.8257 and features37 0.8048.

This requires one new statistical primitive, `oof_tau_per_fold`: ordinary OOF holds
one `X` and varies the fold, but here **`X` itself differs per fold** because each
fold has its own adapter. For fold `f`, both the ridge head's training rows and its
test rows are taken from `emb_fold{f}.npz` — which is why each job must emit
embeddings for **all 900** eval pieces, not just its own 180. Mixing rows across
adapters would compare a head fit on one encoder to features from another.

**(ii) End-to-end** is reported as the deployment number and is **NOT comparable to
0.8048**: features37's ridge is fit on 720 train-fold pieces while an end-to-end head
trains on ~3,800, so beating 0.8048 that way partly measures more supervision. If (i)
passes, a matched features37 arm refit on the same ~3,800 pieces makes (ii) honest.

### Real-audio second gate

709 of 900 eval pieces have local WAVs (verified by name lookup this session, not by
trusting the report), uniform 75–88% coverage across all 11 grades. At n=709 the
paired-difference CI half-width is ≈±0.017, enough to resolve the +0.024 margin. The
n≥500 "gate not floor" threshold was fixed before the yield was known.

`realaudio_check.py` reports three things, and the second two are what make the first
interpretable:

- tau-c on the audio subset, paired-bootstrapped against features37 on the same pieces;
- the **same subset's symbolic tau-c**, so any gap is attributable to audio provenance rather than to the subset being easier or harder;
- **MIDI drift** vs the stored Transkun MIDIs (note-count delta + onset F1). The re-fetch used a fresh yt-dlp path, so this is a genuine audio-provenance perturbation — but only if drift is measured rather than assumed.

Transcription is resumable: each transcribed MIDI is written atomically to a cache
directory and existing files are skipped, so the multi-hour local run survives
interruption.

### Shared `paired_boot`

`paired_boot` currently lives in `features37_compare.py`, a standalone `# /// script`
that `ft_eval.py` cannot import (lightgbm is not in `model/.venv`). It is promoted
into `bakeoff_cv.py` (numpy + scipy only) and imported by both callers, so the gate
and the baseline share one bootstrap implementation. Re-running
`features37_compare.py` afterward must still print `features37|ridge 0.8048` and
`moonbeam_mean|ridge 0.8257` — that is the regression check, and it is Task Group 0.

### Constraints carried forward (each has already bitten once)

- **`composer_index.json` is read-only on this path.** `extract.py::_composer_id`
  does an unlocked read-modify-write, and a grown index would silently change the
  folds. `train_fold.py` therefore writes `emb_fold{F}.npz` directly rather than
  going through `extract_embeddings`, and `ft_eval.py` reads grades and composer ids
  from the **existing** `emb/features37/*.npz` — the same rows `features37_compare.py`
  scores — rather than re-deriving them.
- **`bakeoff_cv.py` and `tk_ablation.py` are not interchangeable.** RidgeCV + seeded
  composer folds vs LightGBM + GroupKFold; comparing their tau-c is the #135
  cross-protocol mirage. Phase 1 is measured with `bakeoff_cv.py` only.
- **The 900 eval pieces span 900 distinct composers** (verified), so
  composer-disjointness is vacuous on that sample and the folds are effectively
  random splits. Trust paired within-protocol deltas, never absolute levels.
- **`_real_loader`'s strict checkpoint key check must be preserved.** A half-loaded
  backbone must abort, not train.
- **Never `uv run --with` from inside `model/`** — it rebuilds the shared `.venv`.
  Use `uv run --no-project --script`, as `moonbeam_extract_script.py` does.
- **Trackio is fatal at init, warn-and-continue mid-run.** A telemetry misconfig
  should stop the job before GPU spend; a telemetry hiccup at step 300 must not
  destroy a paid run.
- **`model/data/results/amt_gap_curve/wav/` (12 GB, gitignored) is kept on purpose.**
  The real-audio gate needs it and re-fetching is now bot-blocked. Do not delete.

## Modules

**`fold_plan.py`**
- Interface: `FoldPlan`, `build_fold_plans(eval_entries, pool_entries, n_folds, seed, val_frac) -> list[FoldPlan]`, `check_fold_plans(plans, eval_entries, pool_entries, n_folds, seed) -> list[str]`
- Hides: option-D exclusion, per-fold test-composer sets, the deterministic composer-disjoint val carve, and every leakage invariant.
- Tested through: `build_fold_plans` output on real manifests (pool counts must equal 3815/4082/4283/4028/4149) and `check_fold_plans` on hand-built violating plans.
- Depth: **DEEP** — one call replaces the entire leakage argument.

**`ranking_loss.py`**
- Interface: `ordered_pairs`, `pairwise_ranking_loss`, `ordinal_loss`, `combined_loss`
- Hides: pair enumeration, empty-pair degeneracy, cumulative-link target construction.
- Tested through: real `torch` on CPU, comparing losses across correctly- and reverse-ranked batches. No mocks.
- Depth: **DEEP** — factored out of the GPU script precisely so the CPU suite can reach it.

**`train_fold.py`**
- Interface: `lora_target_modules(n_layers, n_top) -> list[str]`, `main(argv, loader_factory=...)`
- Hides: checkpoint load, PEFT wrapping, window sampling, the training loop, adapter + `emb_fold{F}.npz` emission, Trackio wiring.
- Tested through: `lora_target_modules` directly, and `main` with an injected fake `loader_factory` (the pattern `moonbeam_extract_script.py` already establishes).
- Depth: **DEEP**.

**`ft_eval.py`**
- Interface: `oof_tau_per_fold(emb_by_fold, y, composers, n_folds, seed) -> np.ndarray`, `main(argv)`
- Hides: the per-fold-X OOF assembly and the paired comparison against features37.
- Tested through: synthetic per-fold embedding matrices with known rank structure.
- Depth: **DEEP** — the one new statistical primitive.
- Constraint: MUST import `composer_disjoint_folds` from `bakeoff_cv.py`. "Same folds" rests on it being literally the same function at the same seed.

**`realaudio_check.py`**
- Interface: `midi_drift(reference_notes, candidate_notes, onset_tolerance) -> dict`, `main(argv, transcriber=...)`
- Hides: resumable transcription, per-piece fold routing, the audio-vs-symbolic paired comparison.
- Tested through: `midi_drift` on constructed note lists, and `main --stage transcribe` with an injected fake transcriber to prove resume/skip behavior.
- Depth: **DEEP**.

**`push_train_dataset.py`**
- Interface: `stage_training_bundle(paths, plans, staging_dir) -> BundleReport`, `main(argv, uploader=...)`
- Hides: which files constitute a hermetic job bundle, and the staging tree layout.
- Tested through: `stage_training_bundle` against a tmp tree, and `main` with an injected fake uploader (no network).
- Depth: **DEEP** — the upload itself is three lines; all the judgment is in what gets staged.

## Verification Architecture

- **Canonical success state:** `ft_eval.py` prints
  `moonbeam_ft_mean|ridge - features37|ridge: +0.0XXX CI95[+a,+b] P(diff<=0)=p SIG`
  with `a > 0` at n=900, and `realaudio_check.py` prints a same-signed significant
  delta at n=709.
- **Automated check (offline, CPU, no GPU, no network):**
  `cd model && uv run python -m pytest src/claim_measurement/difficulty/ -q --no-cov`
  — must stay green and grow from 41 tests.
- **Harness (Task Group 0, buildable before any Phase 1 feature code):** promote
  `paired_boot` into `bakeoff_cv.py`, rewire `features37_compare.py`, and re-run it
  end to end. It must still print `features37|ridge 0.8048` and
  `moonbeam_mean|ridge 0.8257`. This locks the reference values the whole phase is
  measured against, *before* any code that could move them exists. Command:
  ```
  cd model/src/claim_measurement/difficulty && uv run --no-project --script \
      features37_compare.py --data-root /Users/jdhiman/Documents/crescendai/model/data
  ```
- **Not automatically verifiable:** whether the fine-tune actually clears 0.8048.
  That is the experiment, and it costs $13. The manual verification step is the
  staged HF Jobs sequence in the runbook, with the pilot fold as the abort gate.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/claim_measurement/difficulty/bakeoff_cv.py` | Add `paired_boot` | Modify |
| `model/src/claim_measurement/difficulty/features37_compare.py` | Import `paired_boot` instead of defining it | Modify |
| `model/src/claim_measurement/difficulty/fold_plan.py` | Option-D fold plans + leakage invariants | New |
| `model/src/claim_measurement/difficulty/test_fold_plan.py` | Tests | New |
| `model/src/claim_measurement/difficulty/ranking_loss.py` | Pairwise ranking + ordinal aux | New |
| `model/src/claim_measurement/difficulty/test_ranking_loss.py` | Tests | New |
| `model/src/claim_measurement/difficulty/train_fold.py` | HF Jobs `# /// script` entry point | New |
| `model/src/claim_measurement/difficulty/test_train_fold.py` | Tests | New |
| `model/src/claim_measurement/difficulty/ft_eval.py` | The gate | New |
| `model/src/claim_measurement/difficulty/test_ft_eval.py` | Tests | New |
| `model/src/claim_measurement/difficulty/realaudio_check.py` | Second gate | New |
| `model/src/claim_measurement/difficulty/test_realaudio_check.py` | Tests | New |
| `model/src/claim_measurement/difficulty/push_train_dataset.py` | Hermetic bundle staging + upload | New |
| `model/src/claim_measurement/difficulty/test_push_train_dataset.py` | Tests | New |
| `docs/mirex/phase1-lora-runbook.md` | Exact `hf jobs` sequence, staged gates, abort criteria | New |
| `docs/mirex/track-a-difficulty-prediction.md` | Decision-log entry once measured | Modify (at ship time) |

## Open Questions

- **Q: Learning rate.** The issue fixes `r/alpha/dropout/layers` but not LR.
  Default: `1e-4` with cosine decay and 50 warmup steps (standard LoRA starting
  point). The pilot fold reports val ranking tau per epoch; if it is flat or
  diverging, the runbook's abort criterion fires before the remaining four jobs.
- **Q: Micro-batch size.** Pairwise loss needs pairs *within* one forward pass, so
  gradient accumulation does not substitute for batch size. Default: micro-batch 8
  (up to 28 ordered pairs), grad-accum 4, bf16, gradient checkpointing on. The pilot
  reports peak memory; if 8 does not fit on `a100-large`, the runbook says to drop to
  4 rather than to switch flavor.
- **Q: Epochs.** Default 3, with early stopping on val ranking tau (patience 1
  epoch). ~4,000 pieces at micro-batch 8 is ~500 steps/epoch.
- **Q: Is the ~1 GPU-hr/fold estimate right?** It is FLOP-derived, not measured, as
  is the ~12.7M tokens/epoch figure (extrapolated from a 150-file note-count sample).
  Default: run the pilot with `--timeout 3h` and treat the measured wall-clock as the
  budget input for the remaining four.
