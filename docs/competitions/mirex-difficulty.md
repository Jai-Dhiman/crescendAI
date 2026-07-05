# Task 1 — MIREX 2026 Music Performance Difficulty Prediction

> **Status:** LIVING — Phase 1 kill-gate CLEARED (2026-07-04): symbolic-from-AMT features are render-stable → approach GREENLIT. GH issue: #104 (`Jai-Dhiman/crescendAI`).
> **Current state:** Feature extractor + Kendall tau-c merged (TDD, green). Phase 1 cross-render stability ran on 28 MAESTRO composers → **all 4 features cross-render Spearman > 0.9 → PASS** (refutes the A4 soundfont-fragility fear for onset/pitch features). See "Phase 1 results" below.
> **Next step:** Phase 3 (ordinal ranker on CIPI/PSyllabus, then tau-c gate vs trivial baselines) — BLOCKED on dataset access (email captains). The two components (feature extractor, tau-c) are reusable now.

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

## Phase 1 results — cross-render feature stability (RUN 2026-07-04)

**Verdict: PASS. All 4 candidate features are render-stable → symbolic-from-AMT is GREENLIT for this task.**

Harness: `model/src/claim_measurement/difficulty/phase1_cross_render.py`. N=28 MAESTRO
segments, **one per distinct composer** (Berg → Albéniz, composer-spread selection),
each the first ~30s (aria-amt's hard window). Per segment the difficulty features were
computed three ways: **oracle** (ground-truth MAESTRO MIDI), **AMT-real** (aria-amt on
the real human audio, streamed from the remote MAESTRO zip), **AMT-synth** (aria-amt on
a fluidsynth/MuseScore_General.sf3 render of the *same* GT MIDI). 28/28 segments
succeeded, 0 failures. Correlations are Spearman ρ across the 28 segments.

| Feature | **cross-render ρ** (AMT-real ↔ AMT-synth) | fidelity ρ (oracle ↔ AMT-real) | fidelity ρ (oracle ↔ AMT-synth) | Kept? |
|---|---|---|---|---|
| pitch_entropy | **0.9934** | 0.9973 | 0.9973 | YES |
| pitch_lz_complexity | **0.9929** | 0.9948 | 0.9956 | YES |
| polyphony (onset chord-density) | **0.9885** | 0.9891 | 0.9896 | YES |
| pitch_range | **0.9602** | 0.9986 | 0.9616 | YES (fragile — see below) |

Kill-gate was cross-render ρ > ~0.9. **All four clear it.** This directly **refutes the
pessimistic A4 hypothesis** (that a human recording and a fixed-soundfont render of the
same notes would transcribe inconsistently enough to break the features) — for these
onset/pitch-based features, on this repertoire. It confirms A5/A6: pitch + onset are
AMT's reliable substrate, and difficulty features built only on them survive the
independently-scored anti-cheat design that punishes render-fragility.

**pitch_range is the weakest and should be flagged, not trusted blindly.** Its
oracle↔AMT-real fidelity is near-perfect (0.9986), but cross-render drops to 0.9602 and
oracle↔AMT-**synth** to 0.9616 — because it is a min/max *extremum*: a single spurious
very-high or very-low transcribed note in one render shifts the whole feature. It clears
the gate but is the first candidate for a robustified variant (e.g. 5th–95th-percentile
pitch span) if it ever costs ranking accuracy. Entropy/LZ/polyphony are aggregate
statistics and are correspondingly more robust (all ≥ 0.988 cross-render).

**Caveats (do not over-read this PASS):**
- **Repertoire range is compressed.** MAESTRO is all competition-grade (hard) repertoire,
  so these are *stability* correlations over piece-to-piece variation, NOT difficulty-
  discrimination numbers. Whether these features *rank difficulty* is Phase 3 (needs CIPI).
- **Soundfont mismatch.** Our synth render is MuseScore_General.sf3; MIREX's actual
  anti-cheat rendering soundfont is unpublished and may differ. Stability against one
  soundfont is strong evidence, not proof against theirs.
- **Cleaner-than-MIREX test.** We compare real audio vs a render of that performance's
  OWN MIDI (identical notes), isolating the pure render/acoustic effect. MIREX's human vs
  synth pair may involve different note content; our test removes that confound, which
  makes the near-1.0 fidelity expected and the cross-render number the load-bearing one.

Raw per-segment values: `model/data/results/mirex_phase1_cross_render.json` (primary tree).

## Chosen approach (post-research)

**Reproduce RubricNet's hand-crafted recipe on our AMT output** — NOT adapt our deep pairwise head (killed: A1). Rationale: hand-crafted features match deep encoders (A7), the strong features are pitch/onset-based = AMT's reliable part (A5/A6), and it's cheap + interpretable. MuQ audio fusion is a deferred differentiation probe (A11).

## Plan (phased, each with a verify gate)

- **P0 — Unblock (background, NON-blocking).** Email captains for the 4 still-open items above; file CIPI/PSyllabus Zenodo requests for later real-data reproduction. Does NOT gate P1/P2. Task mechanics already confirmed from the public page.
- **P1 — De-risk A4 FIRST. ✅ DONE 2026-07-04 — PASS (all 4 features cross-render ρ > 0.9).** See "Phase 1 results" above. Note: MAESTRO audio was NOT local (only MIDI); recovered by streaming the first 30s of each WAV from the remote MAESTRO zip via remotezip range reads (`phase1_cross_render.py`).
- **P2 — Feature extractor. ✅ DONE 2026-07-04 (TDD, 21 tests).** `difficulty_features.py`: pitch-entropy, pitch-set Lempel-Ziv (LZ76 integer count), polyphony (onset chord-density — offset-free by design), pitch-range. Duration features EXCLUDED. *Deferred (CIPI-gated):* per-feature tau-c on CIPI MIDI vs RubricNet Table 2. NB polyphony is the onset-cluster proxy, not a true concurrent-voice count, precisely to avoid AMT-unreliable offsets.
- **P3 — Ranker + metric + honest splits.** Kendall tau-c ✅ DONE — Stuart's tau-c (rectangular correction) added to `apps/evals/model/skill_eval/analyze.py` as `kendall_tau_c` (8 TDD tests, swappable for the official scorer). REMAINING (CIPI-gated): ordinal ranker (LightGBM LambdaRank / RubricNet-style) -> difficulty scalar + Acc-n + macro-MSE; two splits (canonical CIPI 60/20/20 AND composer-disjoint); baselines note-count/duration/mean-pitch. *Verify/GATE:* feature-ranker tau-c > trivial-baseline tau-c on BOTH splits AND tau-c ~0.6+.
- **P4 — (only if P3 passes) fusion probe.** Frozen MuQ (layers 9-12, 24kHz) as 2nd stream, late-fused with symbolic ranker. *Verify:* fusion > best single stream. The lane to exceed SOTA.

## Cost & odds

- Compute trivial (features+GBM = minutes; MuQ embed = hours, <24h/1GPU OK). Eng ~1 week IF access clears (P1 ~1d, P2 ~2d, P3 ~2d).
- Odds: P1 stability moderate/unknown (why it's first); P3 gate moderate-high conditional on P1 (RubricNet proves hand-crafted hits ~0.6+). Whole spike clears #104: **moderate**, gated on P0 + P1.

## First concrete action for a fresh session (all unblocked — no captain reply, no CIPI)

1. Feature extractor (TDD on sample MIDI): pitch-entropy, pitch-set Lempel-Ziv, polyphony, pitch-range.
2. **Phase 1 cross-render stability** on MAESTRO (real audio vs fluidsynth render) — the highest-value, fully-in-repo experiment; kill-gate for the whole symbolic approach.
3. Kendall Tau-c in `apps/evals/model/skill_eval/analyze.py` (TDD; Stuart's tau-c, swappable).

DEFERRED until CIPI access: reproduce the published RubricNet/CIPI numbers as a trusted anchor. See the pasteable kickoff prompt in the #104 thread / handoff.
