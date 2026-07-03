# Task 1 — MIREX 2026 Music Performance Difficulty Prediction

> **Status:** LIVING — pre-build, research complete, gate NOT yet cleared. GH issue: #104 (`Jai-Dhiman/crescendAI`).
> **Current state:** Assumption-removal research done (see `research-learnings.md`). No code written (respecting #104 kill/commit gate).
> **Next step:** Phase 0 — email captains + file dataset requests (BLOCKER, needs a human). Then Phase 1 cross-render stability experiment.

## Task spec (verified from MIREX wiki + #104)

- **Input:** solo piano audio (WAV 44.1kHz). **Output:** one real-valued difficulty score per recording, treated as an ordering.
- **Metric:** Kendall's **Tau-c** vs a hidden ordinal pedagogical scale. (Ramoneda's full stack also uses Acc-n + macro-MSE.)
- **Anti-cheat:** per test piece, one human performance + one synthesized soundfont rendering. Exact pair-scoring rule UNDOCUMENTED (ask captains).
- **Constraints:** Docker container, full test set <24h on 1 GPU. Deadline Oct 1 2026; submissions open July 1 2026.
- **Captains:** Huan Zhang (QMUL, `huan.zhang@qmul.ac.uk`), Pedro Ramoneda (Songscription, `pedro@songscription.ai`).
- **Suggested training data:** CIPI / PSyllabus / Mikrokosmos-difficulty (all research-only — see learnings).

## Kill/commit gate (#104)

Beat a trivial (duration / note-count) baseline on Tau-c AND land in a respectable Tau-c band (~0.6+; published cross-dataset 0.56-0.68, audio in-domain 0.77-0.78). ~1 week budget.

## Chosen approach (post-research)

**Reproduce RubricNet's hand-crafted recipe on our AMT output** — NOT adapt our deep pairwise head (killed: A1). Rationale: hand-crafted features match deep encoders (A7), the strong features are pitch/onset-based = AMT's reliable part (A5/A6), and it's cheap + interpretable. MuQ audio fusion is a deferred differentiation probe (A11).

## Plan (phased, each with a verify gate)

- **P0 — Unblock (day 0, no code).** Email both captains (the 5 questions in learnings). File CIPI + PSyllabus (Zenodo) access requests. *Verify:* access granted + gate mechanics confirmed. **BLOCKER — critical path.**
- **P1 — De-risk A4 FIRST (~1 day).** Cross-render feature stability: render N pieces via our fluidsynth harness (+2nd soundfont), run aria-amt on human recording + renders, compute candidate features, measure cross-render correlation. *Verify:* feature rho > ~0.9. *Kill:* if unstable, symbolic path compromised → pivot to MuQ audio (riskier) or kill.
- **P2 — Feature extractor (TDD, deterministic).** pitch-entropy, pitch-set Lempel-Ziv complexity, polyphony, pitch-range; reuse density/velocity/timing/pedal; EXCLUDE duration features. *Verify:* per-feature tau-c on CIPI MIDI reproduces RubricNet Table 2 ballpark.
- **P3 — Ranker + metric + honest splits.** Ordinal ranker (LightGBM LambdaRank / RubricNet-style) -> difficulty scalar. Add Kendall tau-c (rectangular correction) to `apps/evals/model/skill_eval/analyze.py` (absent) + Acc-n + macro-MSE. Two splits: canonical CIPI 60/20/20 AND composer-disjoint. Baselines: note-count, duration, mean-pitch. *Verify/GATE:* feature-ranker tau-c > trivial-baseline tau-c on BOTH splits AND tau-c ~0.6+.
- **P4 — (only if P3 passes) fusion probe.** Frozen MuQ (layers 9-12, 24kHz) as 2nd stream, late-fused with symbolic ranker. *Verify:* fusion > best single stream. The lane to exceed SOTA.

## Cost & odds

- Compute trivial (features+GBM = minutes; MuQ embed = hours, <24h/1GPU OK). Eng ~1 week IF access clears (P1 ~1d, P2 ~2d, P3 ~2d).
- Odds: P1 stability moderate/unknown (why it's first); P3 gate moderate-high conditional on P1 (RubricNet proves hand-crafted hits ~0.6+). Whole spike clears #104: **moderate**, gated on P0 + P1.

## First concrete action for a fresh session

Match/confirm the published symbolic baseline (RubricNet on CIPI) in our environment BEFORE building our AMT-fed version — establishes a trusted anchor + validates our new Kendall tau-c metric. See the pasteable kickoff prompt in the #104 thread / handoff.
