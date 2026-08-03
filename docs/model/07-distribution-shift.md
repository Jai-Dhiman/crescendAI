# Distribution Shift — Practice vs Training

> **Status (2026-08-02):** Practice-augmentation and OOD-harness infrastructure shipped, but the planned model-v2 sweep never ran. Synthetic corruption cannot establish ecological validity. This document now owns #139's phone/room, learner, piece, restart, and longitudinal distribution-shift policy.

CrescendAI is trained on finished performances (PercePiano, MAESTRO,
competition, YouTube curated clips) but deployed on practice audio (phone mic,
untreated room, in-progress playing). This doc enumerates the three skew types,
the solo-feasible subset of mitigations, and what is deferred.

---

## Three skew types

### Skew 1 — Skill level

Training distribution is weighted toward intermediate-to-advanced playing:
PercePiano uses conservatory students and amateurs, competition data is
finalists, T5 skews to "interesting enough to curate" which means mostly not
beginner. Production distribution includes absolute beginners making first
contact with a piece.

**Failure mode.** The model is never shown "someone playing Chopin at
1/3 speed with most notes missed" — so when it sees that in production, it
extrapolates outside the training manifold. Predictions become arbitrary; σ
(if calibrated) inflates.

**Mitigation.** The repository can synthesize MIDI corruptions, but synthetic
practice remains an ablation population. Real beginner and intermediate phone
recordings are required for the headline result.

### Skew 2 — Acoustic context

Training: studio mic, treated room. Production: phone mic, untreated room,
occasional phone case rustle, HVAC hum, sibling in the next room.

**Failure mode.** The encoder learns features that correlate with "studio-
quality audio", which are spuriously associated with "good playing". A great
performance captured badly scores worse than a mediocre performance captured
cleanly.

**Mitigation.** Room-IR convolution augmentation in
`practice_synthesis.py` + a small bank of real practice-room IRs.

### Skew 3 — Event / practice-shape

Training: contiguous, uninterrupted playing. Production: stops, restarts,
backtracking, isolated practice of a difficult bar, working out fingerings by
playing fragments.

**Failure mode.** MuQ (and most audio foundation models) has never seen "15
seconds of piano with a 3-second silent break followed by a restart of the
same bar". The model's temporal-dynamics features assume continuity. The
session chunker handles this at the session level; the per-clip scoring model
does not.

**Mitigation.** `practice_synthesis.insert_pauses` + upstream chunking logic
in the session DO (`apps/api/src/do/session-brain.ts`) that splits on
silence/pause boundaries so per-clip scoring never sees a pause embedded in the
middle of a 15s window.

---

## Historical solo-feasible mitigations

1. **Practice augmentation.** Fully solo-executable. No external labelers
   needed, all primitives operate on MIDI or audio signal.
2. **OOD testing harness.** `ood_harness.py` + ~30 phone-captured clips from
   the user's own practice. Chunk B laid the scaffold.
3. **Heteroscedastic σ on the OOD set.** Calibration measured on OOD, not just
   on clean folds. Target: ECE ≤ 0.10 on OOD (looser than the 0.05 clean
   target, because OOD variance is genuinely higher).

---

## Deferred to year 2

### Teacher partnerships

Real practice-domain labeled data — teachers annotating their students'
practice recordings — is the gold standard for measuring distribution shift.
It's deferred because:
- requires external annotators (partnership-dependent)
- requires FERPA-adjacent data handling (student recordings)
- requires teacher-side incentive design (unpaid labor is not sustainable)

Successor #139 moves this work behind ethics approval and an explicitly bounded pilot.

### DANN (Domain-Adversarial Neural Networks)

Train a domain discriminator that tries to classify "studio vs practice", then
train the encoder to fool it. In principle, the encoder learns
domain-invariant features.

Deferred because:
- Practice augmentation covers most of the same ground (synthesizing practice-
  like training data achieves similar effect as adversarial invariance).
- DANN is training-unstable; needs careful λ schedules.
- Marginal gains over augmentation alone are bounded (~2–3pp in comparable
  literature), below our bar for maintenance cost.

Reconsider if augmentation plateaus and the OOD gap is still >10pp.

### Cross-session longitudinal consistency

Beta-user data will give us "same pianist, same piece, weeks apart". A
longitudinal consistency loss (same performer's successive attempts should
embed close) is a potentially strong signal. Deferred because the beta data
doesn't exist yet.

---

## What the harness does in the meantime

Until OOD performance is demonstrably good:

- **σ-gated per-dim feedback.** Dims with σ above the per-dim threshold are
  suppressed from the teacher payload
  (`apps/api/src/services/confidence_gate.ts`). The teacher voice is told in
  the system prompt that some dims were suppressed so it can't
  confabulate specifics.
- **Overall-quality fallback.** When the collapse score is high on a given
  session, the harness surfaces an overall quality observation instead of 6
  per-dim observations. The user doesn't see "dynamics 3, timing 4, phrasing
  4" — they see "this pass has good phrasing energy; try listening for
  dynamics next time."
- **Practice-shape detection.** The session chunker partitions the
  session into clean segments before scoring. Only clean-enough segments
  reach the per-clip model.

---

## Measurement plan

Every sweep emits (wired in #76: `run_ood_test` runs per fold in
`a1_max_sweep` / `ablation_sweep`, aggregated by
`evaluation.summarize_ood_folds` into the validation-gate block):
- Fold pairwise (clean distribution baseline)
- OOD pairwise — reports `SKIPPED` until ~30 phone clips land in
  `data/evals/ood_practice/` (capture protocol in that dir's README)
- OOD-minus-fold gap (clean − OOD; the number that matters)
- Per-dim collapse score on OOD (is the collapse structure domain-dependent?)

Track the gap by learner, piece, device, room, and practice behavior. Report
synthetic and real populations separately; only held-out real audio can open a
teacher-capability gate.
