# ISMIR Paper Revision — Design Spec

## Context

The current arxiv_v2 and ismir_v2 papers report results from a frozen MuQ probe (R²=0.537, 70.3% pairwise). The deployed model (A1-Max) uses LoRA rank-32 with ListMLE ranking loss and achieves 80.8% pairwise accuracy — a 10.5pp improvement. The papers need to reflect the current best results, add systematic ablations, and incorporate new validation signals completed since the original submission.

## Decisions

- **Primary metric:** Pairwise accuracy (R² reported as secondary for comparability)
- **Dimensions:** Keep 19 PercePiano dimensions (6-dim teacher taxonomy is future work)
- **Ablation approach:** Re-run loss component ablations (LoRA+MSE, LoRA+ListMLE, LoRA+ListMLE+CCC, full A1-Max) to show what each loss contributes
- **Validations:** Include all 7 (original 4 + competition correlation, YouTube AMT, Layer 1 gates)
- **Files to update:** `paper/arxiv_v2/main.tex` and `paper/ismir_v2/main.tex` in place
- **Timeline:** ~2 months before ISMIR deadline

## Paper Structure

### Title

"Evaluating Piano Performance Quality with Pretrained Audio Encoders: Adaptation, Ranking Objectives, and Multi-Signal Validation"

Shorter alternative: "From Frozen Features to Ranked Performances: Audio Encoders for Multi-Dimensional Piano Evaluation"

### Thesis

Pretrained audio encoders (MuQ) provide a strong foundation for multi-dimensional piano performance evaluation. Lightweight LoRA adaptation with ranking-based losses achieves 80.8% pairwise accuracy across 19 perceptual dimensions. Seven independent validation signals confirm the model captures musically meaningful quality differences, including transfer to real (non-synthesized) recordings.

### Contributions

1. Systematic comparison of adaptation strategies (frozen probe, full unfreeze, staged, LoRA) for music quality assessment on PercePiano
2. Loss ablation showing ranking objectives (ListMLE) drive most of the improvement over regression
3. Multi-signal validation framework (7 signals) going beyond held-out accuracy
4. Evidence that synthesized-audio training transfers to real recordings (YouTube AMT, 79.9% cross-encoder agreement)

### Section Outline (6 pages + references, ISMIR format)

**Section 1: Introduction (~0.75 page)**
- Problem: evaluating piano performance quality across multiple perceptual dimensions
- Gap: no systematic study of pretrained audio encoder adaptation for this task
- Contribution summary (4 bullets)

**Section 2: Related Work (~0.5 page)**
- PercePiano benchmark (Zhang et al.)
- Audio foundation models in MIR (MuQ, MERT, CLAP)
- Parameter-efficient adaptation (LoRA) in audio/music domains
- Music quality/performance assessment (distinct from note accuracy)

**Section 3: Method (~1.25 pages)**
- 3.1 Data: PercePiano (1,202 segments, 19 dims), Pianoteq rendering with 6 soundfont variants, 4-fold piece-stratified CV
- 3.2 Audio encoder: MuQ backbone (160K hours pretraining), layer 9-12 concatenation (4096-dim), mean pooling
- 3.3 Adaptation strategies: (a) frozen linear probe, (b) full unfreeze with gradual unfreezing, (c) staged domain adaptation, (d) LoRA rank-16/32
- 3.4 Training objectives: MSE regression baseline, ListMLE (Plackett-Luce ranking), CCC (concordance correlation coefficient), contrastive (InfoNCE), soundfont invariance. Multi-task weighting: lambda_ListMLE=1.5, lambda_contrastive=0.3, lambda_CCC=0.3, lambda_invariance=0.1
- 3.5 Ensemble: 4-fold piece-stratified CV, prediction averaging at inference

**Section 4: Results (~1.5 pages)**

Table 1 — Adaptation strategy comparison (existing data):

| Model | Strategy | Pairwise | R² |
|-------|----------|----------|----|
| Frozen probe | MuQ layers 9-12 + MLP, MSE | 70.3% | 0.537 |
| A3 | Full unfreeze, gradual | 69.9% | — |
| A2 | Staged domain adaptation | 71.4% | — |
| A1 | LoRA rank-16, multi-task | 73.9% | 0.40 |
| A1-Max (single best) | LoRA rank-32, multi-task | 78.7% | 0.16 |
| A1-Max (ensemble) | 4-fold average | **80.8%** | **0.50** |

Table 2 — Loss component ablation (NEW runs needed, all LoRA rank-32, L7-12):

| Loss Configuration | Pairwise | R² |
|--------------------|----------|----|
| MSE only | TBD | TBD |
| ListMLE only | TBD | TBD |
| ListMLE + CCC | TBD | TBD |
| Full multi-task (A1-Max) | 78.7% | 0.16 |

- Per-dimension breakdown showing which dimensions benefit most from ranking loss
- Discussion: R² vs. pairwise accuracy — ranking objective improves pairwise (+10.5pp) while reducing pointwise R² (0.537 to 0.50). The metric you optimize is the metric that improves.

**Section 5: Validation (~1.5 pages)**

Seven independent validation signals:

1. **Cross-soundfont generalization:** Leave-one-out CV across 6 Pianoteq soundfonts, R²=0.534 +/- 0.075. Comparable to ensemble result — model generalizes across timbres.
2. **Piece-split vs. performer-split equivalence:** Both yield R²=0.536. Rules out piece memorization — model captures performer-level variation.
3. **Competition ranking correlation (NEW):** 2021 Chopin Competition (11 performers, 3 rounds). Mean rho=+0.704 (p=0.016). Per-dimension: pedaling rho=+0.887, phrasing rho=+0.803. Known limitation: dynamics inverts (rho=-0.917) — model captures "amount" not "appropriateness."
4. **Difficulty correlation:** PSyllabus dataset (508 pieces), Spearman rho=0.623 (p<10^-50). More difficult pieces receive higher model scores, as expected.
5. **Real audio transfer (NEW):** 50 YouTube recordings of intermediate-level pianists processed through ByteDance AMT. 79.9% agreement between audio encoder (A1) and symbolic encoder (S2) predictions. All dimensions >72%. Addresses the "synthesized audio only" limitation.
6. **Fusion failure analysis:** Audio + symbolic fusion R²=0.524 < audio-only 0.537. Error correlation r=0.738 — both modalities fail on the same samples. Fusion adds no complementary signal.
7. **Multi-performer consistency:** ASAP dataset (206 pieces, 631 performances). Mean intra-piece std=0.020. Model captures ~20% of within-piece variation (ground-truth average 0.097).

**Section 6: Discussion & Limitations (~0.5 page)**
- Synthesized audio training only (Pianoteq, not real pianos) — mitigated by YouTube transfer result
- Dynamics inversion in competition (captures amount, not appropriateness relative to score) — needs score conditioning
- Single-fold R² lower than ensemble (0.15-0.39 range) — fold variance from piece distribution, not overfitting
- No formal phone audio validation (YouTube is a proxy)
- Crowdsourced labels have inherent noise ceiling
- Path forward: score-conditioned evaluation, teacher-grounded dimension mapping

### Figures and Tables

| # | Type | Content | Status |
|---|------|---------|--------|
| Fig 1 | Bar chart | Per-dimension pairwise accuracy: A1-Max vs frozen baseline (19 dims) | Update generate_figures.py with new numbers |
| Fig 2 | Diagram | Architecture: MuQ + LoRA + multi-task heads | New figure (replace old pipeline diagram) |
| Fig 3 | Scatter | Difficulty correlation (PSyllabus) | Existing fig4_difficulty_correlation.png |
| Table 1 | Results | Adaptation strategy comparison (6 rows) | Existing data |
| Table 2 | Results | Loss component ablation (4-5 rows) | NEW runs needed |
| Table 3 | Results | Per-dimension pairwise accuracy breakdown (19 dims) | Existing data (reformat) |

### Narrative Shifts

| Current v2 | Revised v2 |
|------------|------------|
| "Frozen features are sufficient" | "Frozen features are a strong baseline; LoRA + ranking loss adds 10.5pp" |
| R² = 0.537 headline | 80.8% pairwise accuracy headline |
| MSE regression | Multi-task with ListMLE as key ingredient |
| 4 external validations | 7 external validations including real audio |
| No adaptation ablation | Systematic adaptation strategy + loss component ablation |
| Pairwise analysis as secondary | Pairwise as primary metric with R²/pairwise tradeoff discussion |

## New Compute Required

4 loss ablation configs x 4 folds = 16 training runs:

1. LoRA rank-32, L7-12, ls=0.1 — **MSE only** (lambda_ListMLE=0, lambda_CCC=0, lambda_contrastive=0, lambda_invariance=0)
2. LoRA rank-32, L7-12, ls=0.1 — **ListMLE only** (lambda_ListMLE=1.5, rest=0)
3. LoRA rank-32, L7-12, ls=0.1 — **ListMLE + CCC** (lambda_ListMLE=1.5, lambda_CCC=0.3, rest=0)
4. LoRA rank-32, L7-12, ls=0.1 — **Full A1-Max** (all losses, validation run to confirm reproducibility)

Estimated time: ~2-4 hours per fold on MPS, ~8-16 hours total compute.

## Files to Modify

- `paper/arxiv_v2/main.tex` — full rewrite of sections 1, 3, 4, 5, 6; update abstract
- `paper/ismir_v2/main.tex` — same changes (these share content)
- `paper/generate_figures.py` — update with A1-Max numbers, add ablation figure
- `paper/arxiv_v2/references.bib` — add LoRA (Hu et al.), ListMLE citations if not present
- `model/src/model_improvement/a1_max_sweep.py` — add ablation configs for loss-only runs

## Verification

- [ ] All ablation runs complete with per-fold metrics saved
- [ ] Ablation config #4 reproduces existing A1-Max results (within 1pp)
- [ ] All 7 validation claims traceable to code/data in the repo
- [ ] No claims about results that don't exist yet (Table 2 TBD cells filled)
- [ ] Figures regenerated from updated generate_figures.py
- [ ] Paper compiles cleanly (make in both arxiv_v2/ and ismir_v2/)
- [ ] Page count within ISMIR 6-page limit (main body)
- [ ] Anonymous submission mode enabled for ismir_v2
