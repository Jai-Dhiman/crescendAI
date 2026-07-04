# Task 1 — MIREX 2026 Music Performance Difficulty Prediction

> **Status:** LIVING — pre-build, research complete, gate NOT yet cleared. GH issue: #104 (`Jai-Dhiman/crescendAI`).
> **Current state:** Assumption-removal research done + task page verified (MIREX wiki, last-modified 2026-07-01). No code written (respecting #104 kill/commit gate). **Nothing blocks starting** — see triage below.
> **Next step:** Fresh session — build feature extractor (TDD) -> run Phase 1 cross-render stability on MAESTRO (real audio vs synth render). Needs NO captain reply, NO CIPI access.

## Task spec (verified from the MIREX 2026 task page, last-modified 2026-07-01)

- **Input:** strictly a WAV file path (44.1kHz solo piano). **Output:** one real-valued difficulty score. **No metadata at inference** (no composer/era/year) — CONFIRMED.
- **Metric:** Kendall's **Tau-c** is the SOLE ranking metric ("agreement between predicted and reference orderings without assuming fixed distances between categories"). Exact formula/scoring-script NOT public.
- **Splits:** test set is **composer-disjoint** from training — CONFIRMED (organizers' own protocol, "to measure true generalization beyond repertoire memorization").
- **Anti-cheat pair:** each test piece has >=1 human performance + 1 synthesized fixed-soundfont rendering, **scored INDEPENDENTLY then aggregated per piece** — CONFIRMED. Exact aggregation fn (mean/median) not specified. => each rendering must rank correctly on its own => cross-render stability is CRITICAL.
- **Suggested (NOT mandatory) training data:** PSyllabus / CIPI / Mikrokosmos-difficulty. External data allowed provided no held-out test pieces leak. No official dev set — self-evaluate on public data.
- **Submission:** Docker container, standardized WAV-in -> score-out interface, full test set <24h on 1 GPU.
- **Dates:** submissions open 2026-07-01, close 2026-10-01, results 2026-10-15.
- **Captains:** Huan Zhang (QMUL, `huan.zhang@qmul.ac.uk`), Pedro Ramoneda (Songscription, `pedro@songscription.ai`).

## Still-open (email captains — NON-blocking, late-stage swaps)

1. Exact Tau-c formula/variant + any official scoring-script link.
2. A published reference/baseline score.
3. Exact test-set size.
4. Exact pair-aggregation function (mean vs median vs rank-then-average).
Defaults until answered: Stuart's tau-c (rectangular correction), our own trivial baselines, mean-of-pair aggregation — all isolated + swappable.

## Kill/commit gate (#104)

Beat a trivial (duration / note-count) baseline on Tau-c AND land in a respectable Tau-c band (~0.6+; published cross-dataset 0.56-0.68, audio in-domain 0.77-0.78). ~1 week budget.

## Chosen approach (post-research)

**Reproduce RubricNet's hand-crafted recipe on our AMT output** — NOT adapt our deep pairwise head (killed: A1). Rationale: hand-crafted features match deep encoders (A7), the strong features are pitch/onset-based = AMT's reliable part (A5/A6), and it's cheap + interpretable. MuQ audio fusion is a deferred differentiation probe (A11).

## Plan (phased, each with a verify gate)

- **P0 — Unblock (background, NON-blocking).** Email captains for the 4 still-open items above; file CIPI/PSyllabus Zenodo requests for later real-data reproduction. Does NOT gate P1/P2. Task mechanics already confirmed from the public page.
- **P1 — De-risk A4 FIRST (~1 day, startable NOW).** Cross-render feature stability using MAESTRO (we have real human audio + aligned MIDI in-repo): treat MAESTRO audio as the "human performance" and a fluidsynth render of the same MIDI as the "synthesized rendering", run aria-amt on BOTH, compute candidate features on each, measure per-feature cross-render correlation. *Verify:* feature rho > ~0.9. *Kill:* if unstable, symbolic path compromised (and this is exactly the independently-scored anti-cheat failure mode) → pivot to MuQ audio or kill. Needs NO captain reply, NO CIPI.
- **P2 — Feature extractor (TDD, deterministic).** pitch-entropy, pitch-set Lempel-Ziv complexity, polyphony, pitch-range; reuse density/velocity/timing/pedal; EXCLUDE duration features. *Verify:* per-feature tau-c on CIPI MIDI reproduces RubricNet Table 2 ballpark.
- **P3 — Ranker + metric + honest splits.** Ordinal ranker (LightGBM LambdaRank / RubricNet-style) -> difficulty scalar. Add Kendall tau-c (rectangular correction) to `apps/evals/model/skill_eval/analyze.py` (absent) + Acc-n + macro-MSE. Two splits: canonical CIPI 60/20/20 AND composer-disjoint. Baselines: note-count, duration, mean-pitch. *Verify/GATE:* feature-ranker tau-c > trivial-baseline tau-c on BOTH splits AND tau-c ~0.6+.
- **P4 — (only if P3 passes) fusion probe.** Frozen MuQ (layers 9-12, 24kHz) as 2nd stream, late-fused with symbolic ranker. *Verify:* fusion > best single stream. The lane to exceed SOTA.

## Cost & odds

- Compute trivial (features+GBM = minutes; MuQ embed = hours, <24h/1GPU OK). Eng ~1 week IF access clears (P1 ~1d, P2 ~2d, P3 ~2d).
- Odds: P1 stability moderate/unknown (why it's first); P3 gate moderate-high conditional on P1 (RubricNet proves hand-crafted hits ~0.6+). Whole spike clears #104: **moderate**, gated on P0 + P1.

## First concrete action for a fresh session (all unblocked — no captain reply, no CIPI)

1. Feature extractor (TDD on sample MIDI): pitch-entropy, pitch-set Lempel-Ziv, polyphony, pitch-range.
2. **Phase 1 cross-render stability** on MAESTRO (real audio vs fluidsynth render) — the highest-value, fully-in-repo experiment; kill-gate for the whole symbolic approach.
3. Kendall Tau-c in `apps/evals/model/skill_eval/analyze.py` (TDD; Stuart's tau-c, swappable).

DEFERRED until CIPI access: reproduce the published RubricNet/CIPI numbers as a trusted anchor. See the pasteable kickoff prompt in the #104 thread / handoff.
