# Model — 1-Year Roadmap (Q2 2026 → Q2 2027)

> Last updated: 2026-04-20. Reviewed quarterly. Compact by design — details
> live in the dated plans and concept docs this file points to.

**Governing constraints.**
- Solo dev. ~40% time on model, ~60% on harness.
- Model noise is bounded (expected 82–85% pairwise, Cohen's d 0.4–0.7).
  Stacking more techniques past that point has sharply diminishing returns on
  the product goal.
- Harness noise is unbounded. It multiplies downstream. Prioritize accordingly.

---

## Q2 (now → Jul 2026)

### Ships

1. **Chunk A / B / C diagnostics** — `docs/plans/2026-04-20-*` today's chunks.
   Landed 2026-04-20. Every sweep from today forward emits collapse score,
   per-dim correlation, skill discrimination.
2. **PercePiano anchor reweight** — `2026-04-20-percepiano-anchor-emphasis.md`.
   Cheapest of the four week-scale plans. Unblocks clean per-dim measurement
   by removing T5's synthetic per-dim signal from regression.
3. **Heteroscedastic heads** — `2026-04-20-heteroscedastic-heads.md`. Highest
   harness-leverage. σ becomes the signal the confidence gate consumes.
4. **SemiSupCon loss** — `2026-04-20-semi-sup-con-loss.md`. Makes T5's tier
   labels useful without requiring per-dim relabeling.
5. **Practice augmentation** — `2026-04-20-practice-augmentation.md`. First
   solo-executable attempt at closing the distribution gap.

### Order

PercePiano reweight first (it's measurement-unblocking), then heteroscedastic
heads (highest harness-leverage), then SemiSupCon (pretraining changes),
then practice augmentation (builds on the stabilized encoder).

### Concept docs

`docs/model/06-label-quality.md`, `07-distribution-shift.md`,
`08-uncertainty-and-diagnostics.md`. Reference docs, not plans.

---

## Q3 (Aug–Oct 2026)

### Goal

Model v2 release candidate: fused MuQ+Aria with practice-distribution
training, heteroscedastic heads, SemiSupCon pretraining.

### Phases

- **Phase B** — Contrastive pretraining. Apply SemiSupCon to MuQ and Aria
  separately. Emit pretrained encoder checkpoints.
- **Phase C** — Fine-tuning on labeled data. Practice-augmented MAESTRO in
  the training mix. PercePiano at 30–35%. Heteroscedastic heads.
- **Phase D** — Fusion. Per-dim gates. Measure error correlation between
  the two encoders *after* independent contrastive pretraining (the
  prediction from `docs/model/00-research-timeline.md` is that
  error-correlation drops vs the leaked-fold baseline of r=0.738).

### Gates to move from RC → production

- Skill discrimination Cohen's d > 0.4 on OOD.
- OOD pairwise within 15pp of clean-fold pairwise.
- Per-dim ECE ≤ 0.10 on OOD.
- Dimension collapse *measured but not gated* — we accept it won't be clean.
  The harness handles it.

---

## Q4 (Nov 2026 → Jan 2027)

### Ships

- **Deploy Model v2** to HF inference endpoint. Replace production MuQ
  endpoint.
- **Beta-user data collection.** Opt-in recording uploads wired into the
  harness. Start accumulating real practice distribution.
- **Active-learning pipeline.** `ood_harness` harvests high-σ recordings from
  beta users, surfaces them for single-ordinal (1–5) relabeling. Same
  labeling capacity as current T5 — no new rubric complexity.
- **First practice corpus** starts accumulating. Target: 200 clips by end of
  Q4. Gates the year-2 partnership conversation.

---

## Year 2 candidates (not committed)

Each has a single-sentence rationale and an explicit "defer because" note.
Re-evaluated at 2027-02 roadmap review.

| Candidate | Rationale | Defer because |
|---|---|---|
| Partnership-driven per-dim relabeling | Unblocks the dimension collapse we've accepted. Fixes the root label problem. | Requires external annotators, FERPA handling, teacher-incentive design. Not solo-feasible. |
| Domain-Adversarial Neural Networks (DANN) | Domain-invariant features via adversarial training. | Practice augmentation covers most of the same ground at lower training instability. Re-enter if augmentation plateaus with gap >10pp. |
| Distillation to edge student model | On-device inference, privacy, latency. | Premature before product-market fit. Cloud inference is cheap enough at current scale. |
| Iterative Mel-RVQ refinement | Improves audio tokenization. | Compute cost vs expected gain is poor on our data sizes. The MuQ frozen representation is already strong enough. |
| Cross-session longitudinal consistency | "Same pianist, same piece, weeks apart" is a gold contrastive signal. | Beta data doesn't exist yet. Blocked on Q4 data-collection ship. |
| Teacher voice finetune (Qwen3-27B) | Part of the teaching-knowledge work, not this roadmap directly. | Gated on separate PMF + A/B + 7B probe per `project_teacher_model_finetuning.md`. |

---

## What success looks like in 12 months

Not "85% pairwise." That's bounded by label noise and will plateau regardless.
Success is:

- The harness never lies to users. High-σ dims are suppressed. Collapse-heavy
  sessions fall back to overall-quality feedback.
- OOD practice gap ≤ 10pp.
- First real beta-user cohort (~50 users) generating signal that informs
  year-2 decisions.
- The per-dim label gap is either closed (partnership lands) or formally
  accepted as a product constraint (it won't be closed solo).

---

## References

- **Plans landing this quarter:**
  `docs/plans/2026-04-20-percepiano-anchor-emphasis.md`,
  `docs/plans/2026-04-20-heteroscedastic-heads.md`,
  `docs/plans/2026-04-20-semi-sup-con-loss.md`,
  `docs/plans/2026-04-20-practice-augmentation.md`.
- **Concept docs:** `docs/model/06-label-quality.md`,
  `docs/model/07-distribution-shift.md`,
  `docs/model/08-uncertainty-and-diagnostics.md`.
- **Timeline anchor:** `docs/model/00-research-timeline.md`.
