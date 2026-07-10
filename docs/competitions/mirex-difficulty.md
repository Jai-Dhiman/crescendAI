# Task 1 — MIREX 2026 Music Performance Difficulty Prediction

> **Status:** LIVING — Phase 1 (render-stability) + Phase 3a (signal) + **Phase 3b (trained ranker, the #104 commit-gate)** ALL CLEARED. Symbolic-from-AMT ranks difficulty and beats the strong note_count baseline by a clear margin. GH issue: #104 (`Jai-Dhiman/crescendAI`).
> **Current state:** Feature extractor + tau-c + PSyllabus loader merged (TDD, green). Phase 1: all 4 features cross-render ρ>0.9. Phase 3a: per-feature signal confirmed (pitch_lz 0.70/0.73). **Phase 3b (2026-07-09): trained composer-disjoint 5-fold LGBMRegressor → tau-c 0.7229 ± 0.019, beating note_count-ALONE 0.6703 by +0.053 on every fold (GATE PASS). LambdaRank collapsed to 0.47 (regressor is the head). All 7 features earn their keep.** See "Phase 3b results".
> **Next step:** #104 commit-gate CLEARED. 0.723 is clear-of-baseline but below the ~0.77 audio-SOTA stretch → closing the gap needs richer symbolic features or the P4 MuQ-audio fusion stream, not more tuning. Decide: fork-then-converge into a submission Docker vs pivot to Track B (#105).

## Task spec (verified from the MIREX 2026 task page, last-modified 2026-07-01)

- **Input:** strictly a WAV file path (44.1kHz solo piano), audio-only. **Output:** one **INTEGER** score per recording (captain-confirmed 2026-07-04; scale is arbitrary, metric is relative/rank-based). **No metadata at inference** — CONFIRMED.
- **Metric:** Kendall's **Tau-c** = `scipy.stats.kendalltau(variant='c')` (captain-confirmed 2026-07-04). NB scipy's *default* is tau-b — must pass `variant='c'`. Our `kendall_tau_c` is parity-locked to it.
- **Splits:** No released train set and **no train/test split** — the test set is a **private, expert-annotated** set (captain-confirmed). Suggested public data is only for training/self-eval. Composer-disjointness of the private test vs public data is unconfirmed → keep composer-disjoint self-eval as the safe proxy.
- **Anti-cheat pair:** each test piece has a human performance + a synth soundfont rendering; the two model outputs are **AVERAGED** to derive the piece score and are **NOT required to agree** (captain-confirmed — SOFTER than the earlier "scored independently" reading). => render-fragility partially cancels under averaging; our Phase 1 independent-render PASS is therefore conservative. Bias (systematic human-vs-synth shift) still survives averaging, so Phase 1 remains the right check.
- **Suggested (NOT mandatory) training data:** PSyllabus / CIPI / Mikrokosmos-difficulty — but "**use anything to train, no sanctioned set**" (captain-confirmed). External data allowed provided no private test leak.
- **Submission:** Docker container, standardized WAV-in -> score-out interface, full test set <24h on 1 GPU.
- **Dates:** submissions open 2026-07-01, close 2026-10-01, results 2026-10-15.
- **Captains:** Huan Zhang (QMUL, `huan.zhang@qmul.ac.uk`), Pedro Ramoneda (Songscription, `pedro@songscription.ai`).

## Captain answers (RESOLVED 2026-07-04 — Huan Zhang & Pedro Ramoneda)

All prior open items are now answered. Three change the build:

1. **Datasets:** access GRANTED to CIPI (Zenodo 8037327) + PSyllabus (Zenodo 14794592). **No sanctioned training set — use anything.** => Phase 3 UNBLOCKED.
2. **Scorer:** `scipy.stats` Kendall's tau-c (i.e. `variant='c'`). => our `kendall_tau_c` is parity-locked to it; NO custom variant risk.
3. **Baseline:** none provided. Output must be an **INTEGER**; scale is arbitrary because the metric is relative/rank-based. => emit predictions on a **fine integer grid** (e.g. round(score*1000)) to minimise ties and preserve rank resolution; keep our OWN trivial baselines (note-count, duration) for the kill-gate.
4. **Split:** private expert-annotated test set; no train/test split, no official training data. => train on public data, self-eval composer-disjoint.
5. **Pair scoring:** the human + synth predictions are **AVERAGED**, agreement NOT required (softer than feared). => (a) Phase 1 PASS is conservative; (b) opportunity: **train-time synth-render augmentation** — render PSyllabus/CIPI MIDI to fluidsynth audio (we have the pipeline) and train on both real- and synth-derived features so the model is robust to both test input types.
6. **Metadata:** none at inference, audio-only. => era/composer multitask trick stays dead (A10 confirmed).

## Kill/commit gate (#104)

Beat our OWN trivial (duration / note-count) baseline on tau-c AND land in a respectable tau-c band (~0.6+; published cross-dataset 0.56-0.68, audio in-domain 0.77-0.78). No organizer baseline exists to beat. ~1 week budget.

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
AMT's reliable substrate. (The anti-cheat turned out to be *averaging* the human+synth
predictions, not independent scoring — captain answer #5 — so this render-stability
margin is more than the task strictly needs; averaging only helps.)

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

## Phase 3a results — does the difficulty signal exist? (RUN 2026-07-09)

**Verdict: SIGNAL CONFIRMED. Pitch-sequence complexity ranks difficulty and beats the trivial baselines → build the ranker (Phase 3b). Drop polyphony.**

Harness: `phase3_feature_signal.py`. Each of the 7,901 PSyllabus whole-piece MIDIs →
our features + trivial baselines, Kendall tau-c (scipy variant='c') vs the ground-truth
grade (ps 0..10). 7,899 usable (2 empty/near-empty MIDIs caught + reported, 0 silent
drops). tau-c reported over the full set and on a held-out **composer-disjoint** test
split (1,599 pieces, no composer shared with train).

| Signal | tau-c (full, n=7899) | tau-c (composer-disjoint test, n=1599) | kind |
|---|---|---|---|
| **pitch_lz_complexity** | **0.701** | **0.732** | feature |
| pitch_entropy | 0.614 | 0.629 | feature |
| pitch_range | 0.505 | 0.556 | feature |
| polyphony | 0.010 | 0.009 | feature — **DEAD** |
| note_count | 0.675 | 0.703 | baseline |
| duration | 0.434 | 0.406 | baseline |
| mean_pitch | 0.017 | 0.047 | baseline |

Findings:
- **Signal exists and is strong.** pitch_lz_complexity (0.70 / 0.73) is the best single
  signal and beats the best trivial baseline note_count (0.675 / 0.70). 0.70–0.73 already
  sits in the published audio-SOTA band (tau-c 0.77–0.78, arXiv:2403.03947) — from ONE
  feature on clean MIDI. Confirms A6 (pitch-LZ + entropy are the strong pair).
- **Drop polyphony.** tau-c ≈ 0.01. Verified NOT a bug: its values are well-spread
  (min 1.18, median 1.85, max 3.02, std 0.46) — it just has no monotonic tie to
  difficulty. Musically sensible: vertical chord density ≠ difficulty (a grade-2 chordal
  hymn vs a grade-9 single-line étude invert it). Difficulty lives in the pitch *sequence*,
  not its thickness. Removed from the difficulty feature set (kept in `difficulty_features`
  as a general-purpose feature; just not used for difficulty).
- **note_count is a strong baseline (0.675 / 0.70) — the bar to clear is HIGH, not trivial.**
  Harder pieces have more notes (length × density), so quantity alone ranks difficulty well.
  The single-feature margin over it is thin (LZ 0.70 vs 0.675). **Phase 3b must show the
  combined ranker beats note_count-alone by a clear margin** (target ~0.77+), else the
  features aren't earning their keep over "count the notes." note_count is a legitimate
  inference-time feature (count aria-amt notes), so the ranker uses it too — the honest
  comparison is trained-ranker vs note_count-alone.

Raw table: `model/data/results/mirex_phase3_feature_signal.json`.

## Phase 3b results — trained composer-disjoint ranker, the #104 commit-gate (RUN 2026-07-09)

**Verdict: GATE PASS. A trained model beats note_count-ALONE by a clear, every-fold margin (tau-c 0.723 vs 0.670, +0.053). The symbolic-from-AMT stream is worth building. Caveat: the ~0.77 stretch target is NOT met — 0.723 is a solid clear-of-baseline number, not audio-SOTA.**

Harness: `phase3b_ranker.py`. All 7,899 usable PSyllabus whole-piece MIDIs → 7 features
(the 3 surviving Phase-3a pitch features + note_count + reused velocity/timing proxies from
`expert_anchor/midi_measures.py`) → **composer-disjoint 5-fold** (sklearn `GroupKFold`,
groups=composer; no composer straddles a fold — the A9 generalization protocol). Two model
heads trained per fold; tau-c is scipy `variant='c'`.

| Model | mean tau-c (5-fold) | std | notes |
|---|---|---|---|
| **LGBMRegressor (L2)** | **0.7229** | 0.0192 | the winning head; native grade-scale output |
| LGBMRanker (lambdarank) | 0.4714 | 0.0394 | **collapsed — see below** |
| note_count-ALONE (baseline) | 0.6703 | 0.0244 | the bar to clear |

Per-fold (regressor beats note_count on **every** fold):

| fold | regressor | note_count | margin | n_test |
|---|---|---|---|---|
| 0 | 0.6960 | 0.6312 | +0.065 | 1580 |
| 1 | 0.7335 | 0.6761 | +0.057 | 1580 |
| 2 | 0.7035 | 0.6554 | +0.048 | 1580 |
| 3 | 0.7394 | 0.6907 | +0.049 | 1580 |
| 4 | 0.7420 | 0.6980 | +0.044 | 1579 |

Regressor grade-scale metrics: **Acc±1 = 0.720, macro-MSE = 2.45** (grade output rounded +
clamped to 0..10; macro = averaged over grade classes so common grades don't dominate).

Findings:
- **The regressor wins; LambdaRank collapses (0.47, worse than the baseline).** Cause: one
  ~6.3k-piece query group with relevance gain 2^grade (≈1024 for the rare grade-10 pieces)
  makes NDCG obsess over placing the top handful correctly and ignore the bulk — pathological
  for a *global* all-pairs metric like tau-c. L2 regression spreads error uniformly and ranks
  the whole ordering. **The mission specced LambdaRank; had we built only that head the gate
  would have FAILED spuriously.** Running both heads and picking the winner was load-bearing.
- **All 7 features earn their keep (ablation, composer-disjoint 5-fold):** full 7 → **0.7229**;
  drop dead `vel_mean` → 0.7207; drop all velocity/timing (pitch+note_count only) → 0.7093.
  Dropping features monotonically *lowers* tau-c. This overturned a plausible prior: the
  velocity/timing proxies score badly ALONE (`vel_mean` 0.05, `vel_disp` 0.18, `timing_ioi_cv`
  0.23 — `vel_mean` is as dead as polyphony was) yet LightGBM extracts weak *interaction*
  signal from them across 7,899 pieces, and that data volume prevents the overfit that hurt
  the same features on a 400-piece subset (0.645, margin only +0.011). Keep all 7.
- **0.723 clears the gate but not the 0.77 audio-SOTA band.** The trained multi-feature model
  (0.723) barely exceeds Phase-3a's `pitch_lz`-alone number — the extra features add ~+0.02
  over the single best feature on composer-disjoint generalization. Closing to 0.77 needs
  richer symbolic features (fingering/hand-span/rhythm-complexity as in RubricNet's ~19) or
  the P4 MuQ-audio fusion stream — not more tuning of these 4 real features.
- **Integer-output requirement:** tau-c is rank-based, so the submission emits
  round(regressor_grade_score × fine-grid); the reported tau-c is on the raw float prediction
  (identical ranks, no tie-loss). Confirmed compatible with the captain-required integer format.

Raw table + per-fold + importances: `model/data/results/mirex_phase3b_ranker.json`
(features cached at `mirex_phase3b_features.json` for instant re-runs/ablation).

## Chosen approach (post-research)

**Reproduce RubricNet's hand-crafted recipe on our AMT output** — NOT adapt our deep pairwise head (killed: A1). Rationale: hand-crafted features match deep encoders (A7), the strong features are pitch/onset-based = AMT's reliable part (A5/A6), and it's cheap + interpretable. MuQ audio fusion is a deferred differentiation probe (A11).

## Plan (phased, each with a verify gate)

- **P0 — Unblock (background, NON-blocking).** Email captains for the 4 still-open items above; file CIPI/PSyllabus Zenodo requests for later real-data reproduction. Does NOT gate P1/P2. Task mechanics already confirmed from the public page.
- **P1 — De-risk A4 FIRST. ✅ DONE 2026-07-04 — PASS (all 4 features cross-render ρ > 0.9).** See "Phase 1 results" above. Note: MAESTRO audio was NOT local (only MIDI); recovered by streaming the first 30s of each WAV from the remote MAESTRO zip via remotezip range reads (`phase1_cross_render.py`).
- **P2 — Feature extractor. ✅ DONE 2026-07-04 (TDD, 21 tests).** `difficulty_features.py`: pitch-entropy, pitch-set Lempel-Ziv (LZ76 integer count), polyphony (onset chord-density — offset-free by design), pitch-range. Duration features EXCLUDED. *Deferred (CIPI-gated):* per-feature tau-c on CIPI MIDI vs RubricNet Table 2. NB polyphony is the onset-cluster proxy, not a true concurrent-voice count, precisely to avoid AMT-unreliable offsets.
- **P3a — Feature-signal preview. ✅ DONE 2026-07-09.** tau-c ✅ (scipy variant='c', parity-locked). PSyllabus loader ✅ (`psyllabus.py`, 7 tests). Per-feature signal measured on 7,899 MIDIs (see "Phase 3a results"): pitch_lz/entropy/range have signal, polyphony DROPPED, note_count is a strong baseline.
- **P3b — Trained ranker. ✅ DONE 2026-07-09 — GATE PASS.** `phase3b_ranker.py`: 7 features (pitch_lz/entropy/range + note_count + velocity/timing) → composer-disjoint 5-fold GroupKFold → **LGBMRegressor tau-c 0.7229 ± 0.019** beats note_count-ALONE (0.6703) by +0.053 every fold; Acc±1 0.720, macro-MSE 2.45. LGBMRanker/lambdarank tried and LOST (0.47) — regressor is the head. Integer-output compatible. See "Phase 3b results". *Stretch (0.77) NOT met* → richer features or P4 fusion.
- **P4 — (only if P3 passes) fusion probe.** Frozen MuQ (layers 9-12, 24kHz) as 2nd stream, late-fused with symbolic ranker. *Verify:* fusion > best single stream. The lane to exceed SOTA.

## Cost & odds

- Compute trivial (features+GBM = minutes; MuQ embed = hours, <24h/1GPU OK). Eng ~1 week IF access clears (P1 ~1d, P2 ~2d, P3 ~2d).
- Odds: P1 stability moderate/unknown (why it's first); P3 gate moderate-high conditional on P1 (RubricNet proves hand-crafted hits ~0.6+). Whole spike clears #104: **moderate**, gated on P0 + P1.

## First concrete action for a fresh session (all unblocked — no captain reply, no CIPI)

1. Feature extractor (TDD on sample MIDI): pitch-entropy, pitch-set Lempel-Ziv, polyphony, pitch-range.
2. **Phase 1 cross-render stability** on MAESTRO (real audio vs fluidsynth render) — the highest-value, fully-in-repo experiment; kill-gate for the whole symbolic approach.
3. Kendall Tau-c in `apps/evals/model/skill_eval/analyze.py` (TDD; Stuart's tau-c, swappable).

DEFERRED until CIPI access: reproduce the published RubricNet/CIPI numbers as a trusted anchor. See the pasteable kickoff prompt in the #104 thread / handoff.
