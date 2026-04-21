# Label Quality Under Solo-Dev Constraints

> Last updated: 2026-04-20

This doc is the canonical treatment of label noise, rater drift, and the
explicit decisions we're making about which noise we accept vs measure vs fix.
It is not a plan — it is the stable reference the plans cite.

---

## Tier-by-tier label audit

### T1 — PercePiano (crowd-sourced, 3 pieces, 19 dims aggregated → 6)

- **Source.** PercePiano dataset: 1,202 segments across 3 pieces, 22
  performers, 19 perception dimensions, rated by multiple Amazon MTurk workers.
  Our pipeline aggregates raters to a mean per dimension (stored in
  `data/labels/percepiano/labels.json`), then composes 19 → 6 via
  `taxonomy.load_composite_labels`.
- **Noise.** Crowd-source. Per-dim rater std is not currently preserved in our
  pipeline (only the mean is stored) — see Fix #1 below.
- **Coverage.** 3 pieces limits piece-stratified CV to 3 folds. Dangerously few
  for claims about generalization, so we report per-piece pairwise alongside
  overall pairwise and flag regressions on any single piece.
- **Our status.** Sole source of per-dimension supervision. Promoted from 20%
  of training mix to 30–35% per
  `docs/plans/2026-04-20-percepiano-anchor-emphasis.md`.

### T2 — Competition placements (ordinal, scalar)

- **Source.** Public international piano competitions. Each recording has a
  placement (1st, 2nd, semifinalist, etc.).
- **Noise.** Judges are human, and placements reflect the judges' preferences
  plus program choice, not pure performance quality. Cross-competition
  comparison is unreliable (different judging panels).
- **Our status.** Used only for ranking loss (ListMLE / SemiSupCon positives
  within the same competition round). Not used for per-dim regression.

### T3 — MAESTRO (piece-paired, unlabeled)

- **Source.** 200+ hours of paired MIDI + audio from concert recordings at
  MAESTRO (International Piano-e-Competition).
- **Noise.** No perception labels at all. Used only for self-supervised
  piece-based InfoNCE: same piece → positive, different piece → negative.
- **Our status.** Stable. No changes planned.

### T5 — YouTube Skill (single-ordinal, 1–5 bucket)

- **Source.** Curated YouTube recordings labeled by the solo dev, 1–5 integer
  bucket, current cadence ~20 clips/week.
- **Noise.** Single rater. No cross-rater agreement signal available. Drift
  over time is the dominant unknown. Addressed by Fix #2 below.
- **Composite derivation.** The 6-vector in `composite_labels.json` is derived
  from the 1–5 ordinal, not independently rated. **All 6 dims carry the same
  information as the single ordinal, plus calibration offsets.** This is the
  root cause of dimension collapse (see
  `docs/model/08-uncertainty-and-diagnostics.md`).
- **Our status.** Used for ranking loss + SemiSupCon. Its per-dim contribution
  to the regression head has been zeroed out since per-dim T5 labels are
  synthetic, not real.

---

## Decisions

### Per-dim T5 relabeling is OUT OF SCOPE

A 6-dimensional rubric applied to 500+ recordings is ≈6× the labeler-hours of
the current 1–5 workflow. The solo-dev constraint makes this infeasible on the
roadmap horizon (Q2–Q4 2026). Rather than defer the whole model plan until
labelers exist, we accept that T5 contributes only skill-ranking information
and route around the lack of per-dim signal:

- PercePiano is the sole per-dim anchor.
- T5 feeds ListMLE + SemiSupCon (ranking) but not per-dim regression.
- Dimension collapse is measured (Chunk A diagnostics) but not fixed.
- Heteroscedastic σ tells the harness per-session whether the 6-vector
  actually carries 6-dim information or is effectively a scalar.

This decision is revisited if and only if external annotators become available
(partnership program in year 2, see
`docs/plans/2026-04-20-model-year-roadmap.md`).

### Expert-annotated practice set is also deferred

Getting teacher-annotated practice recordings would be the cleanest fix for
distribution shift (see `docs/model/07-distribution-shift.md`). That requires
a teacher partnership. Deferred to year 2. In the meantime,
practice-distribution augmentation covers most of the content-side gap, and a
small solo-labeled OOD set (Chunk B) covers the acoustic side.

### Confident Learning / explicit label-noise modeling is deferred

Techniques like Cleanlab's Confident Learning can down-weight noisy labels. On
our tier sizes (1,202 PercePiano segments with aggregated means; 500+ T5
single-rater labels), the expected lift is bounded: confident learning works
best when label noise is *asymmetric and informative*, not when it's
single-rater-constant. The labeler-time cost of running the CL pass, fixing
flagged labels, and validating is higher than expected gain. Reconsider once
per-dim labels exist.

---

## Fixes landing this quarter (Q2 2026)

### Fix #1 — Preserve PercePiano rater variance

PercePiano's raw per-rater annotations are available; we currently discard the
variance. Landing a patch to `scripts/extract_percepiano_muq.py` (not part of
the immediate plan, but queued) stores per-dim mean *and* std per segment. The
heteroscedastic-heads plan consumes std as a target σ floor so the model
doesn't predict narrower confidence than the raters themselves showed.

### Fix #2 — T5 single-ordinal drift instrumentation

Chunk C of the 2026-04-20 plan: `scripts/t5_label_consistency.py`. Every 50
labels, prompt for a relabel on 5 random recordings from ≥100 labels ago.
Quadratic-weighted Cohen's kappa rolling over 100 pairs. Warn if <0.6.

This doesn't *fix* drift — it *makes it visible*. If rolling κ drops below
0.6, the response is: (a) do a calibration pass on 10 anchor recordings with
known scores, (b) flag the drifted window in `calibration_log.jsonl`, (c)
optionally re-weight or drop that window from the training mix.

**T5 labeling gates the Q3 training runs.** All four Wave 1 code changes
(PercePiano mix, SemiSupCon, Practice Augmentation, Heteroscedastic heads) are
merged as infrastructure, but their experimental runs — the sweeps, the
contrastive pretraining, the calibration passes — wait until T5 labeling is
complete with rolling κ ≥ 0.6. Running those experiments on a partial or
drifted T5 pool would measure label noise, not the interventions. See the
"Execution Timing" section in each plan and the Q2/Q3 split in
`docs/plans/2026-04-20-model-year-roadmap.md`.

### Fix #3 — Dimension collapse diagnostics

Chunk A of the 2026-04-20 plan: `evaluation.py:dimension_collapse_score` and
friends. This does not reduce collapse — it reports it every sweep. The
harness (per `docs/model/08-uncertainty-and-diagnostics.md`) then decides
whether to surface per-dim feedback or collapse to overall quality.

---

## Mapping PercePiano's 19 dims onto our 6

The aggregation from 19 → 6 is lossy. It matters because PercePiano is now our
sole per-dim signal.

| Our 6-dim | PercePiano source dims (19) | Method |
|---|---|---|
| dynamics | Dynamic_Range, Dynamic_Contrast | weighted mean |
| timing | Tempo_Stability, Rhythmic_Accuracy | weighted mean |
| pedaling | Pedaling_Clarity, Pedaling_Appropriateness | weighted mean |
| articulation | Articulation_Clarity, Articulation_Variety | weighted mean |
| phrasing | Phrasing_Shape, Phrasing_Direction, Breath | weighted mean |
| interpretation | Emotion, Overall_Musicality, Expressiveness, Imagination, Style_Appropriateness, Fluency, Intentionality, Convincingness | weighted mean |

"interpretation" collects 8 of the 19 source dims — its composite score carries
disproportionate information. Treat its σ skeptically; it's a dim with a lot
of aggregated variance.

---

## Not in this doc

- The *plan* for lifting PercePiano's training weight → see
  `docs/plans/2026-04-20-percepiano-anchor-emphasis.md`.
- The *plan* for heteroscedastic σ → see
  `docs/plans/2026-04-20-heteroscedastic-heads.md`.
- The dimension-collapse measurement details → see
  `docs/model/08-uncertainty-and-diagnostics.md`.
