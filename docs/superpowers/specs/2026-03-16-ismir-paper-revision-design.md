# ISMIR Paper Revision -- Design Spec

## Context

The current arxiv_v2 and ismir_v2 papers report results from a frozen MuQ probe trained on 19 PercePiano dimensions (R²=0.537, 70.3% pairwise). The deployed model (A1-Max) uses LoRA rank-32 with ListMLE ranking loss, trained on 6 teacher-grounded dimensions, achieving 80.8% pairwise accuracy.

**Critical issue:** The 19-dim and 6-dim results are not directly comparable. All new results (A1, A1-Max, A2, A3) were trained on 6 dims. The revision must either operate entirely in the 6-dim regime with a re-run frozen baseline, or clearly separate the two regimes.

**Decision:** Re-run the frozen probe on 6 dims to establish a fair baseline. All comparisons in the main tables use 6-dim results. The old 19-dim results are mentioned briefly for comparability with prior work (Zhang et al.).

## Decisions

- **Primary metric:** Pairwise accuracy (R² reported as secondary for comparability)
- **Dimensions:** 6 teacher-grounded dimensions for all main results. 19-dim PercePiano results cited for prior work comparison only.
- **Ablation approach:** Re-run frozen probe on 6 dims + 4 loss component ablations to show what each loss contributes
- **Validations:** Include all 7 (original 4 + competition correlation, YouTube AMT, Layer 1 gates). Move weaker signals to supplementary if page-constrained.
- **Files to update:** `paper/arxiv_v2/main.tex` and `paper/ismir_v2/main.tex` in place
- **Timeline:** ~2 months before ISMIR deadline

## Paper Structure

### Title

"Evaluating Piano Performance Quality with Pretrained Audio Encoders: Adaptation, Ranking Objectives, and Multi-Signal Validation"

Shorter alternative: "From Frozen Features to Ranked Performances: Audio Encoders for Multi-Dimensional Piano Evaluation"

### Thesis

Pretrained audio encoders (MuQ) provide a strong foundation for multi-dimensional piano performance evaluation. Lightweight LoRA adaptation with ranking-based losses achieves 80.8% pairwise accuracy across 6 teacher-grounded dimensions. Seven independent validation signals confirm the model captures musically meaningful quality differences, including transfer to real (non-synthesized) recordings.

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
- 3.1 Data: PercePiano (1,202 segments, 19 dims collapsed to 6 teacher-grounded dims via composite mapping), Pianoteq rendering with 6 soundfont variants, 4-fold piece-stratified CV
- 3.2 Audio encoder: MuQ backbone (160K hours pretraining), LoRA adapters on layers 7-12, attention-weighted pooling across frames
- 3.3 Adaptation strategies: (a) frozen MLP probe (2-layer, hidden dim 512, frozen MuQ backbone), (b) full unfreeze with gradual unfreezing, (c) staged domain adaptation, (d) LoRA rank-16/32
- 3.4 Training objectives: 5 loss components in total:
  - L_rank: BCE pairwise ranking loss (always active, weight 1.0)
  - L_ListMLE: Plackett-Luce ranking (lambda=1.5)
  - L_CCC: Concordance correlation coefficient regression (lambda=0.3)
  - L_contrastive: InfoNCE contrastive (lambda=0.3)
  - L_invariance: Soundfont invariance (lambda=0.1)
- 3.5 Ensemble: 4-fold piece-stratified CV, prediction averaging at inference
- 3.6 Dimension mapping: Brief description of 19-to-6 dim composite mapping (dynamics, timing, pedaling, articulation, phrasing, interpretation). Full taxonomy documented in supplementary.

**Section 4: Results (~1.5 pages)**

Table 1 -- Adaptation strategy comparison (all 6-dim regime):

| Model | Strategy | Pairwise | R² |
|-------|----------|----------|----|
| Frozen probe (6-dim) | MuQ layers 7-12 + MLP, MSE | **TBD (NEW)** | **TBD (NEW)** |
| A3 | Full unfreeze, gradual | 69.9% | 0.28 |
| A2 | Staged domain adaptation | 71.4% | 0.42 |
| A1 | LoRA rank-16, multi-task | 73.9% | 0.40 |
| A1-Max (single best) | LoRA rank-32, multi-task | 78.7% | 0.16 |
| A1-Max (ensemble) | 4-fold average | **80.8%** | **0.50** |

Note: For comparability with Zhang et al., the frozen probe on original 19 PercePiano dims achieves R²=0.537 (reported in prior version of this paper).

Table 2 -- Loss component ablation (NEW runs needed, all LoRA rank-32, L7-12, ls=0.1):

Note: All configs include the unweighted BCE pairwise ranking loss (L_rank) which is always active in the architecture. The ablation varies the additional loss components.

| Loss Configuration | Components (beyond L_rank) | Pairwise | R² |
|--------------------|---------------------------|----------|----|
| BCE ranking only | None | TBD | TBD |
| + ListMLE | L_ListMLE (lambda=1.5) | TBD | TBD |
| + ListMLE + CCC | L_ListMLE + L_CCC | TBD | TBD |
| Full A1-Max | L_ListMLE + L_CCC + L_contrastive + L_invariance | 78.7% | 0.16 |

- Per-dimension breakdown showing which dimensions benefit most from ranking loss
- Discussion: R² vs. pairwise accuracy -- ranking objective improves pairwise while reducing pointwise R². The metric you optimize is the metric that improves.

**Section 5: Validation (~1.25 pages main, overflow to supplementary)**

Seven independent validation signals. If page-constrained, signals 4 and 7 move to supplementary (weakest standalone contributions).

Main body (5 signals):

1. **Cross-soundfont generalization:** Leave-one-out CV across 6 Pianoteq soundfonts. Comparable to ensemble result -- model generalizes across timbres. (Re-run on 6 dims if needed, or cite 19-dim result with clear label.)
2. **Piece-split vs. performer-split equivalence:** Both yield equivalent results. Rules out piece memorization -- model captures performer-level variation.
3. **Competition ranking correlation (NEW):** 2021 Chopin Competition (11 performers, 3 rounds). Mean rho=+0.704 (p=0.016). Per-dimension: pedaling rho=+0.887, phrasing rho=+0.803. Known limitation: dynamics inverts (rho=-0.917) -- model captures "amount" not "appropriateness."
4. **Real audio transfer (NEW):** 50 YouTube recordings of intermediate-level pianists processed through ByteDance AMT. 79.9% agreement between audio encoder (A1) and symbolic encoder (S2) predictions. All dimensions >72%. Addresses the "synthesized audio only" concern.
5. **Fusion failure analysis:** Audio + symbolic error correlation r=0.738 -- both modalities fail on the same samples. Fusion adds no complementary signal. (Cite 19-dim R² numbers with clear label for context.)

Supplementary (2 signals, if page-constrained):

6. **Difficulty correlation:** PSyllabus dataset (508 pieces), Spearman rho=0.623. More difficult pieces receive higher model scores.
7. **Multi-performer consistency:** ASAP dataset (206 pieces, 631 performances). Mean intra-piece std=0.020.

**Section 6: Discussion & Limitations (~0.5 page)**
- Synthesized audio training only (Pianoteq, not real pianos) -- mitigated by YouTube transfer result
- Dynamics inversion in competition (captures amount, not appropriateness relative to score) -- needs score conditioning
- Single-fold pairwise range (70.3-77.7%) -- fold variance from piece distribution, not overfitting
- No formal phone audio validation (YouTube is a proxy)
- Crowdsourced labels have inherent noise ceiling
- Path forward: score-conditioned evaluation, teacher-grounded dimension mapping, real piano training data

### Figures and Tables

| # | Type | Content | Status |
|---|------|---------|--------|
| Fig 1 | Bar chart | Per-dimension pairwise accuracy: A1-Max vs frozen baseline (6 dims) | Update generate_figures.py |
| Fig 2 | Diagram | Architecture: MuQ + LoRA + multi-task heads (5 loss components) | New figure |
| Fig 3 | Scatter | Difficulty correlation (PSyllabus) | Existing (move to supp if needed) |
| Table 1 | Results | Adaptation strategy comparison (6 rows, all 6-dim) | Frozen probe needs NEW run |
| Table 2 | Results | Loss component ablation (4 rows) | NEW runs needed |
| Table 3 | Results | Per-dimension pairwise accuracy breakdown (6 dims) | Existing data (reformat) |

### Narrative Shifts

| Current v2 | Revised v2 |
|------------|------------|
| 19 PercePiano dimensions | 6 teacher-grounded dimensions (19-dim cited for prior work only) |
| "Frozen features are sufficient" | "Frozen features are a strong baseline; LoRA + ranking loss improves substantially" |
| R² = 0.537 headline | 80.8% pairwise accuracy headline |
| MSE regression only | Multi-task with 5 loss components; ListMLE as key ingredient |
| 4 external validations | 7 external validations including real audio |
| No adaptation ablation | Systematic adaptation strategy + loss component ablation |
| Pairwise analysis as secondary | Pairwise as primary metric with R²/pairwise tradeoff discussion |
| Mean pooling | Attention-weighted pooling |

## New Compute Required

5 configs x 4 folds = 20 training runs:

0. **Frozen MLP probe, 6 dims, MSE** -- same MLP head architecture (2-layer, hidden 512) and attention pooling as A1-Max, but with frozen MuQ backbone and MSE-only loss (no ranking losses). Establishes fair baseline for Table 1.
1. LoRA rank-32, L7-12, ls=0.1 -- **BCE ranking only** (all lambdas=0)
2. LoRA rank-32, L7-12, ls=0.1 -- **+ ListMLE** (lambda_ListMLE=1.5, rest=0)
3. LoRA rank-32, L7-12, ls=0.1 -- **+ ListMLE + CCC** (lambda_ListMLE=1.5, lambda_regression=0.3, rest=0)
4. LoRA rank-32, L7-12, ls=0.1 -- **Full A1-Max** (all losses, reproducibility validation -- expect within 2pp of existing results given MPS non-determinism)

Estimated time: ~2-4 hours per fold on MPS, ~10-20 hours total compute.

## Files to Modify

- `paper/arxiv_v2/main.tex` -- full rewrite of sections 1, 3, 4, 5, 6; update abstract
- `paper/ismir_v2/main.tex` -- same changes (these share content)
- `paper/generate_figures.py` -- update with 6-dim A1-Max numbers, add ablation figure
- `paper/arxiv_v2/references.bib` -- add LoRA (Hu et al. 2022), ListMLE (Xia et al. 2008) citations
- `model/src/model_improvement/a1_max_sweep.py` -- add ablation configs (frozen probe + loss-only variants)

## Verification

- [ ] Frozen probe re-run on 6 dims complete (establishes fair baseline)
- [ ] All 4 loss ablation runs complete with per-fold metrics saved
- [ ] Ablation config #4 reproduces existing A1-Max results (within 2pp, accounting for MPS non-determinism)
- [ ] All 7 validation claims traceable to code/data in the repo
- [ ] No claims about results that don't exist yet (all TBD cells filled)
- [ ] No 19-dim numbers mixed into 6-dim tables without explicit labeling
- [ ] All 5 loss components documented accurately (including always-on BCE ranking)
- [ ] Architecture description matches code: layers 7-12, attention pooling (not mean pooling)
- [ ] Figures regenerated from updated generate_figures.py
- [ ] Paper compiles cleanly (make in both arxiv_v2/ and ismir_v2/)
- [ ] Page count within ISMIR 6-page limit (main body); overflow validations in supplementary
- [ ] Anonymous submission mode enabled for ismir_v2

## Open Questions

- Should we add a frozen-probe comparison with MERT alongside MuQ to preempt reviewer questions about backbone choice? (Low priority but strengthens contribution #1.)
- The cross-soundfont and piece/performer-split results were on 19 dims. Re-run on 6 dims or cite with label? (Recommend re-run if compute budget allows; otherwise cite with "[19-dim]" label.)
- Fusion failure numbers (R²=0.524 vs 0.537) are 19-dim. Re-run or cite with context? (Recommend cite with context -- fusion is a negative result, exact numbers matter less than the r=0.738 correlation.)
