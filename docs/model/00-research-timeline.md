# CrescendAI Research & Product Timeline

> Lean chronology; verdicts only. Detail lives in the linked issues and the
> canonical docs (01-data, 03-encoders, docs/mirex/).

> **Current program:** issue #139 owns the capability-gated Paper #1 benchmark
> and verifier roadmap. This document preserves the chronology and negative
> results; issue state, not old calendar plans, owns active work.

*Core question: "How well is the student playing what the score asks for?"*
Target user: Sarah -- 3 years playing, no teacher, records on her phone, wants
direction on what to work on next. North star: one useful piece of feedback on
one passage she's working on, not a comprehensive review. See
`04-north-star.md` for the full pipeline vision.

---

## Chronology

**2026-03-11 -- Layer 1 Validation (4 experiments).** Competition correlation
PASS (rho=+0.704, mean aggregation; per-dimension dynamics INVERTED at
rho=-0.917 -- model captures "amount" not "appropriateness"). AMT degradation
PASS (0% pairwise drop on studio audio). Dynamic range: diagnostic only
(Cohen's d=0.47), usable for within-student tracking, not absolute
classification. MIDI-as-context: SKIP for raw stats (LLM judge called them
"false precision," 45% B-wins vs 65% gate), but bar-aligned passage-specific
context flagged as the right direction -- became Phase 1 of `04-north-star.md`.

**2026-03-13 -- YouTube AMT validation.** 50 mediocre-quality recordings,
1,225 pairs, 79.9% A1-vs-S2 cross-encoder agreement (all dims > 72%). Doubled
as phone-audio proxy validation.

**2026-03-14 -- Inference cloud-only; phone audio pseudo-validated.** No
on-device ML planned. The 2026-03-13 YouTube AMT validation stood in for
formal phone-audio testing; no longer treated as an existential risk.

**2026-03-18 -- Aria adopted, fold leak fixed, skill-level eval FAILED.**
Decision: replace all custom symbolic encoders (S1/S2/S2H/S3 GNNs) with Aria,
eliminating Phase 3 symbolic-FM research entirely. Same day: piece-stratified
CV folds replaced leaked segment-level folds -- all prior pairwise/R2 numbers
invalidated. Same day, **negative result:** A1-Max showed zero skill-level
discrimination across 5 human-labeled skill buckets (range 0.008 across
beginner-to-professional) -- PercePiano's 100%-advanced-level training data
left the model unable to tell a beginner from Lang Lang. Motivated the (later
abandoned) T5 YouTube Skill Corpus.

**2026-03-19 -- Aria Phase A + clean-fold baseline.** Frozen linear probe:
Aria 59.6% pairwise (marginal but above chance), MuQ 62.2%, error correlation
phi=0.043 (near-zero, down from S2's r=0.738) -- validated dual-encoder
viability. Same day, clean piece-stratified A1-Max baseline: 77.5% pairwise ->
79.85% after loss-weight autoresearch (R2 0.119 -> 0.336). Confirmed the fold
leak had inflated results by only ~1pp, not the feared ~3pp.

**2026-05-27 -- Parallel streams replace gated fusion.** MuQ and Aria each
emit independent 6-dim scores straight to the teacher LLM instead of being
combined via learned gates; stream disagreement becomes teacher-visible signal
instead of being collapsed away. Same decorrelation gate (r < 0.5) as the
retired fusion plan.

**2026-06-19 -- Deterministic claim verifier shipped (#65).**
`claim_taxonomy.json` bumped to v0.1 (dynamics dimension activated).
Measurers, `LocationResolver`, `verify()` orchestrator, and CLI shipped in
`apps/evals/claim_taxonomy/verifier/`. Conventions documented in
`docs/model/claim-verifier-signed-d-conventions.md`.

**2026-06-26 -- Aria Phase C fine-tune (historical, now the #138 bake-off
reference arm).** LoRA rank-32 fine-tune, T1-only, fluidsynth-proxy audio:
mean pairwise 0.6988, R2=0.194 on 4 clean piece-stratified folds. The full
multi-tier run never happened before the program closed (#127).

**2026-07-13 -- Follower baseline + metric shipped (#115, #113).** Monotonic
follower (`follow()`) reproduced the day-0 spike exactly (62/82 matches, 0
teleports). Follower-agnostic scorer (`metric.py`) then **corrected #115's own
characterization**: the monotonic follower actually recovers ~5-7s after a
backward repeat/restart; only a forward jump yields infinite relock latency --
that is the true never-recovers pathology.

**2026-07-19 -- Jump-aware DP shipped (#118).** Bar-boundary jump transitions
cut repeat/restart median relock 50.6s -> 5.74s. **Negative result:** no
single global jump-penalty pair can fix jump relock without also breaking the
repeat-cliff case -- the two failure modes sit on opposite sides of one
penalty knob. Motivated #119 (HMM / state-dependent costs).

**2026-07-20 -- HMM follower shipped (#119).** Opt-in Viterbi-HMM decoder
beside the untouched additive DP, plus calibrated per-note confidence via
forward-backward posteriors. The A8 capstone test proves the HMM relocks both
a backward repeat and a forward skip in one clip where no single additive
penalty pair can (the #118 repeat-cliff negative result).

**2026-07-22 -- HMM autoresearch tuning.** Lowering `p_jump_back` cut clean
false-jumps 62 -> 9 while improving jump relock 0.222 -> 1.0. **Negative
result (a measured pitch-only ceiling):** the remaining 9 false-jumps are
unreachable by any flat or distance-scaled jump penalty -- they trace to clips
where a late passage coincidentally matches an early one, and pitch alone
cannot distinguish replay from coincidence. Resolvable only by a timing/IOI
signal (deferred; follower P3).

**2026-07-23 -- Transkun migration (#128).** `/transcribe` moved from
Aria-AMT to Transkun (MIT, ISMIR 2024). Chroma pseudo-truth recall improved
40% -> 45%; **negative result:** the cost-vs-error guard regressed 0.667 ->
0.586 (reported, not silently ratcheted). Piece-ID stability under the new
substrate could not yet be measured.

**2026-07-25 to 07-27 -- Audio-native teacher pivot; MuQ/Aria program closed
(#127/#129/#130).** Qwen and MuQ/Aria training epics closed in favor of an
audio-native probe. **Negative result:** Gate 0 failed its original all-axis
rule -- dynamics showed strong signal, pedaling was weak and inconsistent,
phrasing remained untested (axis-dependent capability, neither a clean pass
nor flat inability). Successor #139 starts with a modular real-audio
benchmark and independent verifier.

---

## Other negative results worth keeping

- **Score alignment via MuQ embeddings failed** (pre-2026-03): standard DTW on
  MuQ embeddings gave ~18s onset error (MuQ encodes semantic content, not
  temporal features); a learned MLP projection collapsed to a degenerate
  representation. Conclusion: use the right representation per sub-problem --
  MuQ for quality, spectral/symbolic features for alignment.
- **Gated audio-symbolic fusion did not help** with S2 (ISMIR paper: fusion
  R2 0.524 < audio-only 0.537, error correlation r=0.738 -- both streams
  failed on the same samples). Aria's much lower error correlation
  (phi=0.043) made the fusion-vs-no-fusion question moot by replacing fusion
  with parallel streams (2026-05-27) rather than answering it head-on.
- **A from-scratch symbolic foundation model was unnecessary.** Aria (SOTA on
  6 MIR benchmarks, 820K MIDI pretraining) covered the need until superseded
  by MoonBeam-839M (#138, 2026-08-03) -- see `03-encoders.md`.

---

## Superseded / relocated sections

- **Data inventory:** canonical home is `01-data.md`.
- **Encoder results** (A1-Max, Aria, legacy S1-S3 leaked-fold tables):
  canonical home is `03-encoders.md`.
- **Model v2 training plan and parallel-stream architecture:** canonical home
  is `03-encoders.md` (encoder detail) and `04-north-star.md` (pipeline
  framing).
- **Forward-looking phase roadmap** (Phase 2 temporal reasoning, Phase 4 real
  audio, T5 skill-corpus collection): superseded by GitHub issue tracking; see
  `04-north-star.md` for the still-current pipeline vision and
  `docs/mirex/track-a-difficulty-prediction.md` for the active MIREX Track A
  program.
