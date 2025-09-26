# Improvement Plan: Refactor, Model Enhancements, and Tutor Alignment

## Executive Summary

This document lays out a focused plan to:

- Refactor and split notebook logic into importable modules (keeps Colab notebooks small and robust)
- Align the model architecture and training objectives with the AI classical piano tutor goal
- Improve data hygiene, normalization parity, and self-supervised learning objectives
- Scale data and evaluation rigor for stronger generalization

Primary outcomes:

- A single ultra-small AST backbone with consistent preprocessing and multiple task heads
- A PercePiano 19D regression head (primary for tutoring), plus optional CCMusic 7-class timbre head
- Time-local outputs for actionable, bar-level feedback during lessons

---

## 1) Refactor and Splitting (Top Priority) — DONE 2025-09-20

Goals

- Move the bulk of logic from notebooks into `src/` as importable modules
- Enforce a single preprocessing contract and deterministic seeding
- Keep notebooks as orchestration only (data paths + training/eval calls)

Proposed module structure

- `src/data/audio_io.py`
  - `load_audio_mono_22050(path) -> np.ndarray`
  - `mel_db_128x128(y, sr=22050, n_fft=2048, hop=512, n_mels=128) -> np.ndarray in [-80, 0]`
  - Guarantees [time, mel] = [128,128], clip to [-80,0], identical to training

- `src/augment/piano_augment.py`
  - `conservative_spec_augment(mel_db) -> mel_db` (time/freq masks, AWGN in power domain, light gain)
  - `randomized_offset_window(y, sr, duration=…, target=…) -> y_window` for window sampling

- `src/datasets/maestro_mae.py`
  - `SmartMAESTRODataset(original_dir, augmented_dir, split, target_shape=(128,128), group_split_by_piece=True)`
  - Returns `(spec_db, pad_mask)`; performs canonicalization, cropping, padding

- `src/models/ultra_small_ast.py`
  - `UltraSmallASTForSSAST(nn.Module)` exactly matching pretraining architecture (256D, 3L, 4H)

- `src/objectives/ssl_losses.py`
  - `masked_losses(features, pad_masks)` [consistency, magnitude, diversity]
  - `info_nce(global_features, pos_index_map, temperature)` or BYOL-like teacher-student projector/predictor

- `src/training/ssast_trainer.py`
  - `train_pretrain_ultra_small(cfg, datasets) -> checkpoint`
  - Creates optimizer, handles RNG keys, checkpointing, early stopping

- `src/fine_tune/percepiano_head.py`
  - 19D regression head (primary task); MSE/MAE plus correlation-aware loss; optional ordinal/CORN head

- `src/fine_tune/ccmusic_head.py`
  - 7-class classifier head; loss, metrics, trainer

- `src/eval/eval_suite.py`
  - Performer-level CV, bootstrap CIs; guarantees preprocessing parity; saves figures + JSON

Colab notebook choreography (thin)

- Pretraining: clone repo, `uv sync`, mount Drive, run `train_pretrain_ultra_small(cfg)`
- Fine-tune (PercePiano): `train_percepiano_regression(cfg, pretrained_path)`
- Fine-tune (CCMusic): `train_ccmusic_classifier(cfg, pretrained_path)`
- Evaluate: `run_all_evals(cfg)` (uses performer-level CV, bootstrap CIs)

Determinism and config

- Centralize all seeds: Python, NumPy, JAX
- Single `cfg` object for all hyperparameters, paths, and seeds
- Explicit exceptions for data loading and preprocessing (no silent fallbacks)

---

## 2) Preprocessing & Normalization Parity (High Impact)

Problem

- Training uses mel→dB in [-80, 0]; some evaluation paths normalize to [0,1]
- CCMusic mel-images map 8‑bit intensities to dB by linear scaling (not guaranteed to match `librosa.power_to_db`)

Plan

- Enforce identical preprocessing everywhere:
  - Audio → mel → `power_to_db(..., ref=np.max)` → clip to [-80, 0] → [time, mel]=[128,128]
  - Remove (db+80)/80 scaling from evaluation; keep raw dB
  - Prefer recomputing mel from raw audio for CCMusic; if images must be used, derive an empirical mapping to match the `power_to_db` distribution (validated by Q‑Q plots)
- Document preprocessing contract at the top of each trainer and in README to avoid drift

Benefits

- Reduces domain shift across phases
- Makes the backbone features dependable and stable

---

## 3) Self-Supervised Learning Objective (High Impact)

Problem

- Current “repulsive” off-diagonal similarity pushes all pairs apart; near-duplicate positives (original ↔ augmented of same clip) become false negatives

Plan (choose one)

- InfoNCE/SimCLR: define explicit positives (orig ↔ aug) and negatives (others) with temperature scaling
- BYOL/DINO-style: a predictor network encourages invariance without negatives; works well on audio with careful augmentations
- If keeping repulsive term, enforce batch sampler constraints so orig & its aug never co-occur in the same mini-batch

Benefits

- Builds invariance to augmentations (what we want) rather than pushing positives apart
- Typically boosts downstream regression/classification by 3–10 points (task-dependent)

---

## 4) Fine-Tuning Aligned With Tutor Goals (High Impact)

Primary task: PercePiano 19D regression

- Add a dedicated 19D head trained with:
  - MAE/MSE + correlation-aware loss (maximize mean Pearson or differentiable surrogate)
  - Optional ordinal regression head (CORN) if labels are Likert/ordinal, improving stability
- Multi-task with CCMusic timbre/brand classification as auxiliary (robustness to piano timbre/recording conditions)

Time-local supervision for actionable feedback

- Add patch/window-level head to produce local versions of the 19 metrics
- Weak supervision paths:
  - Broadcast track-level labels to windows (regularized by smoothness)
  - Distill MIR-derived curves (onset density, tempo stability, loudness dynamics) into the local head via consistency loss
- Aggregation to full performance:
  - Robust pooling (median/trimmed mean), var/quantiles per dimension
  - Highlight bars/time-spans with worst predicted skill metrics

---

## 5) Data Strategy (NonCommercial posture; Medium–High Impact)

Legal posture

- This project is currently NON‑COMMERCIAL. CC BY‑NC datasets (e.g., MAESTRO, CCMusic Pianos) are allowed for training and evaluation. We will still record the license per item and keep manifests for clarity.

Portfolio v1 (what to use and why)

- Supervised (primary): PercePiano — direct 19D regression labels aligned to tutor goals; use performer‑level CV.
- SSL/pretext (backbone): MAESTRO v3 train split (~200h) — solo classical piano with aligned MIDI; ideal for representation learning and proxy targets.
- Diversity/pretext (curated): ASAP (score‑aligned classical) and/or MusicNet (piano‑only subset), MAPS (limited use due to synthetic domain).
- Auxiliary labels: CCMusic‑database/pianos (piano brand/timbre classification) to regularize timbre robustness; licensing is NC/ND, so use for research/auxiliary only.
- Robustness/OOD evaluation: CrescendAI‑PhonePilot mini‑set (3–5h) — internal, consented phone/laptop mic recordings; evaluation only, never in validation of supervised labels.

Actions

- Manifests: Create JSONL manifests for each corpus (PercePiano, MAESTRO, ASAP/MusicNet/MAPS, CCMusic, PhonePilot) with fields: path, sha1, duration_sec, performer_id (if available), split, license, source, notes. Store under data/manifests/.
- Dedup: Run Chromaprint/LSH and persist dedup pairs. Enforce in dataloaders with explicit exceptions. Never allow overlap between SSL corpora and supervised val/test. Exclude augmented items from any validation/test.
- Preprocessing parity: Recompute mel from raw audio wherever possible using the single source‑of‑truth pipeline (mono 22050 Hz; mel n_fft=2048, hop=512, n_mels=128; power_to_db ref=np.max; clip [-80,0]). For CCMusic image‑only items, apply the grayscale→[0,1]→[-80,0] mapping as a fallback and prefer audio when available.
- Time‑local proxies: From aligned MIDI/score sets (MAESTRO/ASAP) compute onset density, tempo stability (tempogram), loudness dynamics, and pedal proxies; use as consistency targets for the local head.
- Augmentations (conservative, piano‑aware): light IR‑based room reverb, mic‑distance EQ, small gain/noise. Avoid pitch/time‑stretch that harms timing/articulation semantics. Validate via AB tests on PercePiano correlation.

Acceptance (Data)

- Manifests exist for PercePiano, MAESTRO, and at least one of ASAP/MusicNet/MAPS; each entry records license and performer (if applicable).
- Dedup report saved; supervised validation/test contain no augmented or duplicate material; SSL corpora do not leak into supervised holdouts.
- Similarity probe report comparing SSL corpora vs PhonePilot (RMS, spectral centroid/rolloff, loudness range, RT60 proxy, onset density).
- Small SSL pretrain + PercePiano fine‑tune shows ≥0.05 median correlation improvement vs no‑SSL baseline (target‑dependent).

---

## 6) Evaluation Rigor (Medium Impact)

Primary metric suite

- Performer-level CV (most reflective of real-world generalization)
- Holdout evaluation with bootstrap confidence intervals for correlation
- Per-dimension analysis: best/worst dimensions; calibration plots

Parity checks

- Guarantee identical preprocessing and segment policy across train/val/test
- Ablations: with vs without InfoNCE/BYOL; with vs without CCMusic auxiliary; augmentation strength sweeps

Calibration

- Isotonic regression on validation to make predicted skill scores interpretable and stable

---

## 7) Final Production Shape (I/O Contract)

Backbone

- Ultra-small AST (3.3M params; 256D, 3L, 4H), patch size 16×16; dropout 0.3, stochastic depth 0.2

Preprocessing (strict)

- Input: mono float32 PCM at 22050 Hz
- Window: ~3.0s (128 frames × 512 hop / 22050 ≈ 2.97 s)
- Mel: n_fft=2048, hop=512, n_mels=128
- dB: `power_to_db(..., ref=np.max)` clipped to [-80, 0]
- Shape: [B, 128, 128] as [time, mel]

Heads

- 19D PercePiano regression (primary)
- Optional 7‑class CCMusic timbre/brand classification (auxiliary)
- Optional local window head for time‑localized scoring

Outputs per 3s window

- 19 continuous scores (normalized 0–1 or raw Likert mapped consistently)
- Optional 7‑class probabilities
- Confidence/uncertainty (e.g., MC dropout), and local saliency for explainability

Full‑performance aggregation

- Sliding window (e.g., hop 1s), robust pool to overall 19D scores
- Time series per dimension → actionable feedback: “bars 14–18: timing stability and articulation need work”

---

## 8) Roadmap, Priorities, and Acceptance Criteria

Phase 1 (Week 1–2): Refactor & normalization — DONE 2025-09-20

- [P0] Move preprocessing, datasets, model, training, eval into `src/`
- [P0] Enforce dB [-80, 0] parity everywhere; remove any 0–1 scaling in eval
- [P1] Deterministic seeding across Python/NumPy/JAX; explicit exception handling
- Acceptance: notebooks shrink to < 100 lines; unit test simple batch → model → loss on CPU

Phase 2 (Week 2–3): SSL objective, minimal corpus, and batching

- [P0] Replace repulsive similarity with InfoNCE or BYOL; or batch sampler guard to avoid false negatives.
- [P0] Acquire minimal MAESTRO subset (e.g., first ~50h) and build manifest with licenses; run dedup against PercePiano.
- [P1] Run a small SSL pretrain and validate on PercePiano: report median and per‑dimension correlations with bootstrap CIs.
- Acceptance: +0.05± on PercePiano median correlation vs no‑SSL baseline, no leakage per dedup checks.

Phase 3 (Week 3–4): PercePiano head & eval suite

- [P0] Implement 19D regression head + correlation-aware loss; performer‑CV + bootstrap CIs
- [P1] Optional ordinal/CORN head; calibration
- Acceptance: performer‑CV correlation meets target; detailed per‑dimension analysis saved

Phase 4 (Week 4+): Scale data & local head

- [P1] Extend SSL corpus to full MAESTRO train split; curate ASAP/MusicNet/MAPS (piano‑only) subsets with manifests; document licenses.
- [P1] Add local window head and time‑local proxies (onset/tempo/loudness/pedal) from aligned sets; enforce preprocessing parity.
- [P1] Establish CrescendAI‑PhonePilot OOD holdout (3–5h) for robustness checks; never include in supervised validation.
- [P2] Distillation from a larger teacher (PaSST/AudioMAE/BEATs) to the 3.3M AST.
- Acceptance: improved median correlation and stable calibration; robustness improved on PhonePilot; clear local diagnostics plots.

---

## 9) Risks & Mitigations

- Normalization drift across phases → Single preprocessing source-of-truth; tests
- Positive/negative mixups in SSL → InfoNCE/BYOL or batch sampler constraints
- Data scarcity for supervised tutor task → grow labeled set; active learning; calibration
- Over-augmentation harming musical semantics → conservative parameter ranges, val AB tests

---

## 10) Implementation Notes & Developer UX

- Use `uv` for Python package management in Colab and local
- Provide `scripts/` or `cli` wrappers (`uv run python -m crescendai.train.pretrain ...`)
- Keep checkpoints, manifests, and analysis outputs under a consistent tree (e.g., `/checkpoints`, `/analysis_results`)
- Prefer explicit exceptions (no silent falls backs) in data loaders and preprocessing

---

## 11) Quick Acceptance Checklist

- [x] Single preprocessing function produces [128×128] mel dB in [-80, 0]
- [ ] InfoNCE/BYOL objective implemented or batch sampler prevents false negatives
- [ ] PercePiano 19D regression head trained; performer‑CV and bootstrap CIs computed
- [ ] Evaluation pipelines use identical preprocessing and segment policies
- [x] Notebooks reduced to orchestration with repo imports
- [ ] Documentation updated with I/O contract and tutor outputs
- [ ] Dataset manifests created with license and performer metadata (PercePiano, MAESTRO, ≥1 of ASAP/MusicNet/MAPS)
- [ ] Dedup pairs persisted and enforced; no leakage from SSL corpora into supervised val/test
- [ ] Similarity probe report comparing SSL corpora vs PhonePilot OOD holdout
- [ ] SSL pretrain → fine‑tune achieves ≥0.05 median correlation gain over no‑SSL baseline
