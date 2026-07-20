# Task 1 — MIREX 2026 Music Performance Difficulty Prediction

> **Status:** LIVING — Phase 1 (render-stability) + Phase 3a (signal) + **Phase 3b (trained ranker, the #104 commit-gate)** ALL CLEARED. Symbolic-from-AMT ranks difficulty and beats the strong note_count baseline by a clear margin. GH issue: #104 (`Jai-Dhiman/crescendAI`).
> **Current state:** Feature extractor + tau-c + PSyllabus loader merged (TDD, green). Phase 1: all 4 features cross-render ρ>0.9. Phase 3a: per-feature signal confirmed (pitch_lz 0.70/0.73). **Phase 3b (2026-07-09): trained composer-disjoint 5-fold LGBMRegressor → tau-c 0.7229 ± 0.019, beating note_count-ALONE 0.6703 by +0.053 on every fold (GATE PASS). LambdaRank collapsed to 0.47 (regressor is the head). All 7 features earn their keep.** See "Phase 3b results".
> **Next step (DECIDED 2026-07-13):** run the **P4 MuQ-audio difficulty probe** — the one untested lever that (a) has a higher potential ceiling than symbolic features and (b) operates on the waveform directly, so it BYPASSES the ~0.09 aria-amt gap entirely. It is decisive: if MuQ-audio beats the symbolic 0.764 it is a real edge + dodges the dominant risk; if it also plateaus ~0.76 we have confirmed a genuine ceiling and should ship the simple symbolic version. Timeline: MIREX submissions close **2026-10-01** (~11 weeks) — ample runway for one focused probe before committing. Track B (#105/#107) is the SAFE-submission fallback (system built, ~0.74 head, easy pass, no moat). See "Strategic reassessment".
>
> **Prior state (both walls MEASURED 2026-07-12):** (1) Symbolic ceiling: richer features lifted clean-MIDI tau-c 0.725→**0.764** (Phase 3c), model/data levers saturated. (2) **AMT domain gap ~0.09** (Phase 3e pilot n=43): aria-amt on real audio drops tau-c 0.883→0.792 on-subset, projecting full-task ~0.67 deployed.

## Strategic reassessment (2026-07-13) — the "wall" is softer than Phase 3d first read

Phase 3d called 0.764 "at the label-noise wall (0.767)." That over-read a SINGLE ceiling; three
were conflated:
- inter-**exam-board** agreement (predict one arbitrary syllabus) = **0.767**;
- reliability of the **consensus `ps`** we actually train against (Spearman-Brown) = **~0.857**;
- single-syllabus-vs-`ps` = 0.89 (circular, upper bound only).

We eval against the consensus, whose reliability is ~0.85 — so there is real headroom above 0.764
that Phase 3d under-weighted. AND the 0.76 is a property of OUR proxy (cross-institution syllabus
averaging is a noisy label construction); the MIREX private test is a coherent expert annotation
whose reliability is UNKNOWN and plausibly higher. Counter-evidence it might be a FEATURE (not label)
wall: RubricNet hits the same ~0.77 on CIPI (different, cleaner labels) — if it were pure PSyllabus
label noise, cleaner labels should score higher. Symbolic FEATURES may cap ~0.77 regardless of labels.

We cannot distinguish label-wall from feature-wall with symbolic features alone. The MuQ-audio probe
(P4) is the experiment that separates them AND sidesteps the AMT gap — hence it is now the priority,
not a deferred "fusion" nicety. NB the parallel: Track B independently hit a ~0.73-0.74 frozen-encoder
wall too, hinting a broad frozen-encoder plateau in this band.

## Chosen approach (post-research)

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

## Phase 3c/3d results — pushing past 0.723 and finding the wall (RUN 2026-07-12)

**Verdict: the wall is LABEL NOISE, and richer features took us right up to it. Feature engineering lifted tau-c 0.725 -> 0.764 (+0.039); model/objective changes add nothing; the corpus is not the limit. 0.764 sits at the inter-rater agreement ceiling (split-half 0.749 / pairwise 0.767). Further gains are label-noise-limited and low-yield. The one axis still unmeasured is the aria-amt-on-audio domain gap, which is a DROP risk, not headroom.**

An independent adversarial review (codex/gpt-5.6-sol) converged with our own feature list and
flagged the key blind spot: whole-piece means wash out the hardest passage that often sets the
grade. We built a 37-feature superset (`phase3c_explore.py`) adding, on the AMT-reliable
onset/pitch/velocity substrate only: rhythm/density (notes_per_sec, onset_clusters_per_sec,
median_ioi, IOI entropy/LZ), chord WIDTH (span p90/max + frequency of stretches >7/>12/>16
semitones — distinct from the dead polyphony DENSITY), melodic intervals/leaps, pitch-class
entropy, a robust p5-p95 pitch span, and **sliding 5s-window peaks** (p90/max local density,
complexity, chord-span — codex's "the grade lives in the hardest bar" idea).

**Feature ceiling (composer-disjoint 5-fold, all honest — kitchen-sink uses ALL 37 with no
per-fold selection, so no selection-bias inflation):**

| Feature set | tau-c |
|---|---|
| current-7 (Phase 3b) | 0.7249 |
| **kitchen-sink (all 37)** | **0.7643** |

**Model/objective lever is exhausted** (37 features, 5-fold): L2 0.7638, L1 0.7622, Huber 0.7652,
Poisson 0.7630, 2x-trees 0.7635, deeper 0.7629 — all within +/-0.003 (< the 0.018 fold std). No
loss or capacity change beats ~0.764. (Confirms codex #9: L2 was not leaving tau-c on the table.)

**Not data-gated** (learning curve, current-7, fixed composer-disjoint test): 10% -> 0.684,
25% -> 0.688, 50% -> 0.705, 75% -> 0.712, 100% -> 0.712. Flat past 75% (+0.0007) — doubling the
corpus buys nothing on this feature set.

**Label-noise ceiling** (`phase3d_label_ceiling.py`, inter-syllabus agreement on the SAME piece;
each work's `related_entries` records the grade up to 11 syllabi gave it): only 29% of the 2,501
multi-graded works get the exact same grade across syllabi, 43% disagree by >=2. The tau-c an
independent model can hope to reach:
- pairwise inter-syllabus (predict one arbitrary rater) = **0.767**
- split-half panel reliability (one independent panel vs another) = **0.749**
- Spearman-Brown full-panel step-up (reproducibility of the consensus `ps`) = 0.857
- single-syllabus-vs-`ps` = 0.89, but CIRCULAR (`ps` is derived from that syllabus; AMEB 0.947) — an upper bound only.

**Diagnosis.** We are at the pairwise/split-half ceiling (~0.75-0.77) and at 0.764. The published
"audio-SOTA 0.77-0.78" is NOT a target we're failing to reach — it is this same label ceiling.
Theoretical headroom remains from 0.764 toward the ~0.857 full-panel reproducibility, but that
regime means predicting the consensus BETTER than an individual rater panel does, from a lossy
audio-derived feature proxy — label-noise-dominated and low-yield. **The dominant gate is label
noise, confirmed four ways (features saturate, model saturates, data saturates, labels self-
disagree at ~0.76).** The remaining high-value experiment is NOT more features: it is measuring
the clean-MIDI -> aria-amt-on-audio drop (all 7,901 pieces carry a `youtube_link`; aria-amt weights
+ server are local), because that gap could move the real submission number more than any feature.

Raw: `mirex_phase3c_explore.json` (killed mid-greedy; kitchen-sink + per-feature from log),
`mirex_phase3d_label_ceiling.json`, feature cache `mirex_phase3c_features.json`.

## Phase 3e results — the AMT domain gap (clean MIDI -> aria-amt on audio) (RUN 2026-07-12)

**Verdict: the transcription gap is REAL and ~0.09 tau-c on a 43-piece pilot -- LARGER than the entire feature-engineering gain (+0.041). Deploying on audio is the dominant remaining risk: projected full-task tau-c drops from clean 0.764 toward ~0.67, near the note_count baseline. This is the actual submission number's ceiling, and it is not a feature or label problem -- it is our transcriber.**

Harness: `phase3e_amt_gap.py` (staged: prep -> transcribe -> compare). 44 grade-stratified
PSyllabus pieces (4/grade, 0..10), 43 downloaded (1 YouTube 403), audio -> our **aria-amt**
(the same model the MIREX Docker would run) -> the 37-feature superset, vs the same features
from the clean PSyllabus MIDI. aria-amt caps at 30s/chunk, so each piece was sliced into 30s
WAV chunks (493 total), transcribed, and stitched back with time offsets (the whole-piece
aggregation the mission requires). Ran locally on CPU in a dedicated Python-3.11 aria-amt venv
(the numba-compatible config; 3.13 fails to resolve aria-amt -> librosa -> numba).

| full-clean-trained model, tested on the 43-piece subset | tau-c |
|---|---|
| on CLEAN-MIDI features | 0.883 |
| on our-aria-amt features | 0.792 |
| **GAP (clean - amt)** | **~0.090** |

aria-amt/clean note-count median ratio = 1.04 (no gross over/under-transcription).

Per-feature transcription survival (clean<->amt Spearman across the 43 pieces):
- **survives well** (rho > 0.8): n_pitch_classes 0.92, win_pitch_entropy_p90 0.90, pitch_class_entropy 0.88, pitch_entropy 0.88, notes_per_sec 0.85, onset_clusters_per_sec 0.85, interval_entropy 0.84, fast_note_fraction 0.82, pitch_lz 0.81, ioi_lz 0.80.
- **degrades most** (rho < 0.65): win_interval_mean_p90 0.48, median_ioi 0.53, vel_disp 0.56, vel_range 0.58, frac_chord_span_gt16 0.62, mean_chord_span 0.64. Makes sense: velocity is AMT-*estimated*, and chord-span/fine-window features need precise simultaneity that AMT onset jitter blurs.

Caveats (do NOT over-read the pilot):
- **n=43 is small** -- the gap is noisy (~+/-0.03-0.05) and the subset's absolute tau-c (clean 0.883) is inflated vs the full task's 0.764 because 4-per-grade stratification is a coarser, easier ranking than the full 7,899. The GAP (same pieces, clean vs amt) is the meaningful quantity, not the absolutes.
- **The gap conflates transcription error with performance difference** -- the YouTube recording is a *different* performance than the one PSyllabus transcribed, so 0.09 is an UPPER bound on the pure transcription cost. (It is also the realistic deployment condition: MIREX test audio is likewise arbitrary performances.)
- The robust features (pitch-class entropy, pitch_lz, note density) are the ones to lean on for a submission; the AMT-fragile velocity/chord-span features should be down-weighted or dropped for the audio path.

Implication: a submission built on this stream would likely score ~0.67-0.72, gated FIRST by the aria-amt gap, then by label noise. Closing the gap is a TRANSCRIBER problem (better AMT, train-on-AMT-features augmentation, or AMT-robust feature selection), not a feature-design or model problem.

Raw: `mirex_phase3e_amt_gap.json`. Audio/venv scratch deleted post-run (reproducible via `phase3e_amt_gap.py --stage prep` + the aria-amt venv); results JSON retained.

## Phase 4 results — the MuQ-audio difficulty probe (RUN 2026-07-13)

**Verdict: NO audio edge. A frozen MuQ linear probe TIES note-count and LOSES to symbolic — including the transcribed aria-amt path it was meant to beat. The decisive P4 experiment settles the audio-vs-symbolic question against audio. Ship the symbolic version; do not pursue a MuQ submission.**

Harness: `phase4_muq_probe.py` (staged prep -> embed -> eval). The exact Phase-3e 44-piece
grade-stratified set (so the aria-amt transcriptions line up for a true like-for-like) ->
YouTube audio (24kHz mono, all 44 downloaded) -> **frozen MuQ-large-msd-iter, layers 9-12
mean-aggregated** -> [T, 1024] frame embeddings -> pooled over time to a piece vector (mean, or
mean+std concat = 2048-d). Light head = **RidgeCV, the SAME head for every representation**, so
the comparison isolates the representation, not head capacity. Protocol = in-subset
composer-disjoint grouped 5-fold, repeated over 5 seeds (a stability band is essential at this n).

Because MuQ embeddings only exist for the downloaded subset (embedding all 7,899 YouTube audios
is not cheap), the honest like-for-like is IN-SUBSET 5-fold for ALL THREE representations on the
identical pieces + identical folds. The Phase-3e 0.883/0.792 are a DIFFERENT protocol (head
trained on all 7,899) and are context only, NOT head-to-head with MuQ.

| representation (identical 43 pieces, composer-disjoint 5-fold x 5 seeds, RidgeCV head) | tau-c |
|---|---|
| clean-symbolic (37-d) | **0.783 +/- 0.026** |
| aria-amt-symbolic (37-d) | **0.726 +/- 0.032** |
| **MuQ-audio (meanstd, 2048-d)** | **0.697 +/- 0.021** |
| note_count (floor) | 0.697 |

- MuQ - full-task symbolic 0.764 = **-0.059**. MuQ - clean-symbolic (in-subset) = **-0.081**.
  MuQ - aria-amt-symbolic (in-subset) = **-0.030**.
- **MuQ does not even beat the transcribed symbolic path.** The P4 premise was that reading the
  waveform directly would dodge the ~0.09 aria-amt gap. But symbolic AFTER paying that gap (0.726)
  still beats frozen MuQ (0.697). The audio path has no advantage to trade for the transcription cost.
- **MuQ ties note_count** (0.697 vs 0.697): a frozen MuQ linear probe ranks difficulty no better
  than counting notes.
- `meanstd` (0.697) > `mean` (0.665): per-frame std (dynamic variation) carries signal the time-mean
  discards. Even the better pooling only ties the floor.

**Data-starvation ruled out (PCA disambiguation).** The obvious objection to a null at n=43 is that
a 2048-d probe is starved by ~34 training samples. Tested directly: PCA(k) + RidgeCV under the same
folds, k in {2,4,8,16,32}. Every k, both pools stay 0.64-0.69 -- at or BELOW the note_count floor.
If a difficulty direction existed in frozen MuQ but was starved, a k=2..8 projection (k << n) would
surface it; it does not. Full-dim RidgeCV (0.7045) slightly beats every PCA truncation, so the faint
signal is diffuse, not concentrated -- and never exceeds note-count. The null is REAL, not small-n.
The internal control confirms it: clean/amt symbolic hit 0.72-0.78 under the identical small-n
RidgeCV protocol, so the harness CAN extract difficulty from a good representation -- MuQ is the limit.

**Caveat (stated, not hedged):** this null is specific to FROZEN MuQ + LINEAR probe + layers 9-12 +
whole-piece mean/std pooling. It does NOT prove an end-to-end FINE-TUNED audio encoder cannot beat
symbolic. But fine-tuning is a far larger bet (all 7,899 audios + GPU training), off this cheap
probe's critical path, and the broad frozen-encoder plateau (~0.73-0.74, independently hit by Track B
#105/#107) argues against near-term payoff. The frozen probe was the right cheap decider, and it says no.

**Decision:** ship the simple symbolic version (Phase 3c, clean-MIDI 0.764; deployed ~0.67-0.72
after the aria-amt gap). Do NOT scale the MuQ download/embed (the only reason to -- starvation -- is
refuted). Track B (#105/#107) remains the safe-submission fallback. MIREX submissions close 2026-10-01.

Raw: `mirex_phase4_muq_probe.json`. Audio (875MB wav) + MuQ frame cache deleted post-run
(reproducible via `phase4_muq_probe.py --stage prep` + `--stage embed`); pooled piece vectors
(tiny) + manifest + results JSON retained.

## Phase 5 — continuous-improvement slate (STARTED 2026-07-19)

With ~11 weeks of runway, a 5-experiment slate past the Phase-4 null. Reframe: the SUBMISSION
number (~0.67-0.72) is gated FIRST by the aria-amt gap (~0.09), THEN by label noise -- not by the
clean-MIDI 0.764. So the slate splits into "close the AMT gap" (5a, 5d) and "find new signal / raise
the clean ceiling" (5b Aria probe, 5c consensus labels, 5e CIPI cross-dataset).

### Phase 5a — AMT-robust feature selection (RUN 2026-07-19)

**Verdict: NEAR-NULL on accuracy. Restricting to transcription-robust features does NOT lift the deployed (aria-amt) number -- it stays ~0.79 for all subsets, within n=43 noise. Confirms from the other side that the AMT gap is a TRANSCRIBER problem, not a feature-choice problem (5d attacks it properly).**

Harness: `phase5a_robust_features.py`. Reused the Phase-3e per-feature clean<->amt survival rho to
define subsets, trained LGBM on the full 7,899 clean set (phase3c cache), and measured the deployed
number = full-clean-trained model tested on the 43 aria-amt pieces.

| feature set | clean 5-fold | deploy clean-43 | deploy AMT-43 | gap |
|---|---|---|---|---|
| FULL (37) | 0.764 | 0.883 | 0.792 | 0.090 |
| ROBUST rho>0.80 (11) | 0.746 | 0.842 | 0.792 | 0.050 |
| ROBUST rho>0.75 (18) | 0.758 | 0.883 | 0.800 | 0.083 |

- The 11-feature set HALVES the gap (0.090->0.050) but only because its clean-43 side DROPS to 0.842
  -- the deployed AMT-43 number is unchanged (0.792). Shrinking the gap by lowering the clean side is
  not a deployment win.
- All subsets land at AMT-43 ~0.79-0.80; the +0.007 for ROBUST075 is inside the n=43 +/-0.03-0.05 band.
- One real (non-accuracy) win: the 11-feature robust set matches the full deployed number with 1/3 the
  features -- a parsimony/robustness argument for the submission, not a score gain.
- **Underpowered:** n=43 cannot resolve a ~0.007 effect. Revisit with 5d's larger transcribed set.

Raw: `mirex_phase5a_robust_features.json`.

### Phase 5b — Aria symbolic-embedding probe (RUN 2026-07-19)

**Verdict: NO edge. Frozen Aria (512-d symbolic-FM embedding) ties/slightly-loses to 37 hand-crafted features (0.744 vs 0.749 on the same 600 pieces) and does not beat 0.764. Even the RIGHT-modality foundation model finds no difficulty signal beyond hand-crafted features -> strong evidence for a representation/feature wall, not a bad-feature-choice problem.**

Harness: `phase5b_aria_probe.py`. The right-modality sequel to the MuQ null (Phase 4): Aria is a
symbolic FM (pretrained ~820K MIDIs). 600 grade-strided PSyllabus pieces -> frozen Aria "embedding"
variant (512-d global, EOS-pooled over 300-note chunks) -> RidgeCV probe. SAME RidgeCV head for Aria
and the 37-feature symbolic baseline, composer-disjoint 5-fold x 5 seeds.

| representation (n=600, RidgeCV head, same folds) | tau-c |
|---|---|
| symbolic 37-feat | 0.7493 +/- 0.0044 |
| Aria embedding (512-d) | 0.7435 +/- 0.0078 |
| reference: symbolic full-task (7,899) | 0.764 |

- Aria - symbolic (same sample) = -0.006 (loses, within noise). Aria - 0.764 = -0.020.
- Parallels the MuQ null: neither an audio FM (MuQ) nor a symbolic FM (Aria) beats hand-crafted
  features. The 0.76 band is a property of the TASK/representation, not our feature engineering.
- Caveat: frozen + linear probe + the 512-d "embedding" variant. A fine-tuned Aria or the 1536-d base
  variant / sequence-level features are unrefuted -- but the cheap frozen decider says no edge (same
  logic as Phase 4).

Raw: `mirex_phase5b_aria_probe.json`. Aria embeddings cached at `data/results/aria_probe/emb`
(reproducible via `phase5b_aria_probe.py --stage embed`).

### Phase 5c — consensus / multi-syllabus label modeling (RUN 2026-07-19)

**Verdict: NULL as a submission lever, and the diagnostic points to a FEATURE wall (not a label wall). Reliability-weighting and denoised targets don't beat 0.764; and a cleaner (multi-syllabus-mean) target is HARDER to predict from the 37 features, not easier -- so averaging out label noise does not raise the ceiling. This walks back the "consensus reliability 0.857 = big headroom" hope.**

Harness: `phase5c_consensus_labels.py`. 32% of PSyllabus pieces (2,500) carry >=2 syllabus grades in
`related_entries`; among those 43% disagree by >=2. Composer-disjoint 5-fold LGBM on the 37 features.

| variant | tau-c |
|---|---|
| baseline (ps, unweighted) | 0.7636 |
| weight by n_syllabi | 0.7644 (+0.0008) |
| weight by sqrt(n_syllabi) | 0.7642 (+0.0006) |
| train on mean(related), eval ps | 0.7627 (-0.0009) |
| multi-syllabus subset: train+eval ps | 0.7602 |
| multi-syllabus subset: train+eval mean(related) | 0.7106 |

- Reliability weighting + denoised target = pure noise (all within +/-0.001).
- **Diagnostic:** if the ceiling were fixable consensus noise, the cleaner mean target should rank
  EASIER; instead it ranks HARDER (0.71 vs 0.76). The features already capture what the consensus
  offers -- averaging syllabi adds nothing. (Caveat: integer ps vs continuous mean have different
  tie structure, affecting absolute tau-c; the -0.05 is partly an artifact, but the direction still
  refutes "cleaner labels -> higher score.")
- **Strategic update:** evidence now converges on a **feature wall ~0.76**, consistent with RubricNet
  also capping ~0.77 on CIPI's cleaner labels. Raises the value of 5b (new representation) + 5e (CIPI
  cleaner-label test); lowers the value of further label modeling.

Raw: `mirex_phase5c_consensus.json`. Keyed feature cache: `mirex_phase5c_features.json`.

### Phase 5e — cross-dataset test on Mikrokosmos (RUN 2026-07-19; CIPI-proper blocked)

**Verdict: (1) no label-wall lift -- the higher Mikrokosmos internal tau-c is an EASIER task, not cleaner-label headroom; (2) a real TRANSFER WARNING -- our PSyllabus-trained 37-feature model transfers WORSE than note_count alone to an independent corpus, i.e. it overfits PSyllabus. Argues for a parsimonious, robust submission.**

CIPI proper (Zenodo 8037327) is access-restricted (0 files to anon API -- needs the owner's Zenodo
token). Substituted **Mikrokosmos-difficulty** (Bartok, 147 pieces, expert Henle grades 1-8, open
MusicXML, a MIREX-suggested set). Harness `phase5e_cross_dataset.py`, features via partitura.

| result | tau-c |
|---|---|
| Mikrokosmos internal (37-feat, grade-stratified 5-fold) | 0.804 +/- 0.003 |
| -- its own note_count floor | 0.776 |
| Transfer PSyllabus-trained -> Mikrokosmos (full-37) | 0.705 |
| Transfer (tempo-invariant pitch-only, 15 feat) | 0.687 |
| Transfer note_count baseline | 0.776 |

- **0.804 > 0.764 is a MIRAGE.** Mikrokosmos is single-composer + deliberately grade-ordered, so its
  note_count floor is already 0.776 (vs ~0.70 on PSyllabus). The 37 features add only +0.03 over
  note-count here -- the SAME thin margin as on PSyllabus (+0.06). Cleaner labels did NOT unlock
  feature headroom -> reinforces the 5c feature-wall, does not refute it.
- **Transfer warning (the real finding):** PSyllabus-trained 37-feat (0.705) transfers WORSE than
  note_count alone (0.776) to an independent corpus -> the feature model overfits PSyllabus structure.
  Direct risk for the MIREX private test. Caveat: MusicXML scores have nominal timing (120bpm, no
  performed rubato) + default velocity, handicapping timing/velocity features; note_count is scale-free
  and transfers cleanly. Still, the parsimony signal stands.
- **Actionable:** favour a robust/parsimonious submission (note_count + a few robust pitch features)
  over the full 37 for generalization. Open item: rerun on CIPI proper if the owner supplies a Zenodo token.

Raw: `mirex_phase5e_cross_dataset.json`. Mikrokosmos data: `data/raw/mikrokosmos` (re-download from
github PRamoneda/Mikrokosmos-difficulty). Feature cache: `mirex_phase5e_mikrokosmos_features.json`.

### Phase 5d — train-on-AMT augmentation (IN PROGRESS, started 2026-07-19)

The only remaining lever that moves the DEPLOYED (aria-amt) number, not the clean ceiling. Train ON
aria-amt features so train matches test: render 594 clean MIDIs -> audio (fluidsynth MuseScore_General)
-> slice 30s -> aria-amt -> features; train LGBM; test on the SAME 43 real-audio aria-amt pieces
(Phase 3e). Gate: does amt-trained beat clean-trained (0.792) on the amt-43 test? Also evaluates the
robust-11 and parsimonious-6 subsets (the chosen submission direction, given the 5e overfitting warning).

Harness `phase5d_train_on_amt.py` (prep render+slice / external aria-amt transcribe / eval).
Toolchain verified: aria-amt builds under `uv run --python 3.11 --script extract_amt_midi.py`,
transcribes ~16s/chunk on MPS (sane output: 201 notes, pc_entropy 3.48). 594-piece run ~= 6,000 chunks
~= 27h transcription (resumable). Disk-bounded (full wavs deleted after slicing; 147GB free).
**Results pending the transcription run.**

## Chosen approach (post-research)

**Reproduce RubricNet's hand-crafted recipe on our AMT output** — NOT adapt our deep pairwise head (killed: A1). Rationale: hand-crafted features match deep encoders (A7), the strong features are pitch/onset-based = AMT's reliable part (A5/A6), and it's cheap + interpretable. MuQ audio fusion is a deferred differentiation probe (A11).

## Plan (phased, each with a verify gate)

- **P0 — Unblock (background, NON-blocking).** Email captains for the 4 still-open items above; file CIPI/PSyllabus Zenodo requests for later real-data reproduction. Does NOT gate P1/P2. Task mechanics already confirmed from the public page.
- **P1 — De-risk A4 FIRST. ✅ DONE 2026-07-04 — PASS (all 4 features cross-render ρ > 0.9).** See "Phase 1 results" above. Note: MAESTRO audio was NOT local (only MIDI); recovered by streaming the first 30s of each WAV from the remote MAESTRO zip via remotezip range reads (`phase1_cross_render.py`).
- **P2 — Feature extractor. ✅ DONE 2026-07-04 (TDD, 21 tests).** `difficulty_features.py`: pitch-entropy, pitch-set Lempel-Ziv (LZ76 integer count), polyphony (onset chord-density — offset-free by design), pitch-range. Duration features EXCLUDED. *Deferred (CIPI-gated):* per-feature tau-c on CIPI MIDI vs RubricNet Table 2. NB polyphony is the onset-cluster proxy, not a true concurrent-voice count, precisely to avoid AMT-unreliable offsets.
- **P3a — Feature-signal preview. ✅ DONE 2026-07-09.** tau-c ✅ (scipy variant='c', parity-locked). PSyllabus loader ✅ (`psyllabus.py`, 7 tests). Per-feature signal measured on 7,899 MIDIs (see "Phase 3a results"): pitch_lz/entropy/range have signal, polyphony DROPPED, note_count is a strong baseline.
- **P3b — Trained ranker. ✅ DONE 2026-07-09 — GATE PASS.** `phase3b_ranker.py`: 7 features (pitch_lz/entropy/range + note_count + velocity/timing) → composer-disjoint 5-fold GroupKFold → **LGBMRegressor tau-c 0.7229 ± 0.019** beats note_count-ALONE (0.6703) by +0.053 every fold; Acc±1 0.720, macro-MSE 2.45. LGBMRanker/lambdarank tried and LOST (0.47) — regressor is the head. Integer-output compatible. See "Phase 3b results". *Stretch (0.77) NOT met* → richer features or P4 fusion.
- **P4 — MuQ-audio difficulty probe. ✅ DONE 2026-07-13 — GATE FAIL (no audio edge).** Frozen MuQ-large (layers 9-12, 24kHz) on the same 44-piece audio set, RidgeCV probe, composer-disjoint 5-fold x 5 seeds. **MuQ 0.697 TIES note_count (0.697) and LOSES to clean-symbolic (0.783, -0.081) AND aria-amt-symbolic (0.726, -0.030)** — so it does not even beat the transcribed symbolic path it was meant to bypass. PCA(k=2..32) sweep rules out small-n starvation (all k <= floor). See "Phase 4 results". **Decision: ship symbolic; no MuQ submission; Track B (#105) is the fallback.** Fine-tuned audio (not frozen) is unrefuted but off critical path.

## Cost & odds

- Compute trivial (features+GBM = minutes; MuQ embed = hours, <24h/1GPU OK). Eng ~1 week IF access clears (P1 ~1d, P2 ~2d, P3 ~2d).
- Odds: P1 stability moderate/unknown (why it's first); P3 gate moderate-high conditional on P1 (RubricNet proves hand-crafted hits ~0.6+). Whole spike clears #104: **moderate**, gated on P0 + P1.

## First concrete action for a fresh session (all unblocked — no captain reply, no CIPI)

1. Feature extractor (TDD on sample MIDI): pitch-entropy, pitch-set Lempel-Ziv, polyphony, pitch-range.
2. **Phase 1 cross-render stability** on MAESTRO (real audio vs fluidsynth render) — the highest-value, fully-in-repo experiment; kill-gate for the whole symbolic approach.
3. Kendall Tau-c in `apps/evals/model/skill_eval/analyze.py` (TDD; Stuart's tau-c, swappable).

DEFERRED until CIPI access: reproduce the published RubricNet/CIPI numbers as a trusted anchor. See the pasteable kickoff prompt in the #104 thread / handoff.
