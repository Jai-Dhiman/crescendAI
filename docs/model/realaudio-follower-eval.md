# Real-Audio Score-Follower Eval — SOURCE OF TRUTH

**Status:** v1 proxy track live (2026-07-26, #133). Gold (accuracy) track pending. This eval is the **source of truth** for score-follower quality; the synthetic `follower_bench` (below) is superseded and slated for removal once the gold track is trusted.

## Why this exists

The follower's job is production: track where an amateur is in the score from **real phone-recorded audio**. The prior benchmark (`model/src/follower_bench/`) measured the follower on *pristine ASAP MIDI with hand-spliced pathologies* — no audio, no AMT, no amateurs. It could not answer "does this survive real inputs." This eval closes that gap by running the follower on real YouTube recordings of amateurs practicing rep pieces, transcribed by the **exact production transcriber**. No synthetic clips, no augmentation.

## Corpus

- **279 real clips, 16 rep pieces** (all of `DEFAULT_SCORE_BY_PIECE`), sourced from the pre-approved practice-video review `data/evals/practice_eval/*/candidates.yaml` (`approved: true`; 299 approved, 93% transcribed, 20 link-rot/download fails).
- Pipeline (matches production): approved `video_id` → `piece_id_eval.acquire.acquire_audio` (yt-dlp, 16 kHz mono WAV) → **Transkun** (`apps/inference/amt/transkun_cli.transcribe_wav`, the production `/transcribe` engine, #128) → bundle JSON.
- Bundles live at `data/evals/realaudio_bundles/<piece>/<video_id>.json` — **gitignored and rebuildable** (see "Rebuild"). `substrate_versions.transcriber` stamps the engine so a transcriber swap is detectable.

## Two metric tracks (hybrid ground truth)

Real audio has no synthetic answer key, so the eval measures on two tracks:

**Proxy track (live) — automatic, all clips, anchor-free.** Answers "does the follower behave sanely." Per clip via the #119 HMM (`follow_hmm` + `TUNED_HMM_PARAMS`):
- `coverage` — fraction of transcribed notes placed on the score.
- `score_span_frac` — min→max score-seconds reached / score duration (did it traverse the whole piece?).
- `monotonicity` — backward steps beyond a 0.5 s chord-noise tolerance (real repeats/restarts OR alignment slips — reported, not judged).
- `confidence` — forward-backward posterior mass on each decoded column.
- `conf_vs_monotone` calibration — does confidence drop at backward steps (the "knows when it's lost" signal)?

**Gold track (planned, #133 S3) — human bar-tap, ~20–30 clip subset.** The one **non-circular accuracy** number: a human taps bar onsets → `(audio_sec → score bar)` reference; the follower's decoded score-position at each tap is compared to the true bar → bar-localization error, % within tolerance, relock latency after stops. Also adjudicates whether low-confidence proxy clips are genuine hard cases (correct to abstain) or follower failures.

## v1 proxy results (279 clips, 0 harness failures)

| Signal | Median | Spread |
|---|---|---|
| score span | 1.00 | 12% of clips <0.5 |
| coverage | 0.74 | p10 0.48 → p90 0.86 |
| confidence | 0.89 | p10 0.18 → p90 0.96; 21% of clips <0.5 |
| clips with ≥1 repeat/restart | — | 72% |

Reading: the follower traverses the full score on the large majority of real amateur clips, is robust to real AMT + phone-audio noise, and raises a self-doubt signal on ~21% (the beginner-heavy, hesitation-heavy pieces — `fur_elise`, `pathetique`). **Accuracy remains unverified** — high coverage/confidence/span is *consistent with* good following but is not proof; a confidently-wrong alignment yields the same proxies. That is the gap the gold track fills.

**PASS bars are deliberately not yet set** — they will be built on per-clip distributions, not medians (the median `backward_frac` of 0.00 hid that 72% of clips contain repeats; aggregates lie about exactly the behavior we care about).

## Rebuild

```bash
cd model    # PRIMARY checkout (data/ is gitignored, absent in worktrees)
WT=<path-to-issue-133-worktree>/model
# 1. build the corpus (resumable; ~3.3 min/clip CPU; parallelize by disjoint --pieces groups)
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.build_corpus \
  --bundles-root data/evals/realaudio_bundles
# 2. run the proxy eval
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.realaudio \
  --bundles-root data/evals/realaudio_bundles --scores-root data/scores --out /tmp/report.json
```

## Relationship to synthetic `follower_bench` (superseded)

`follower_bench/` supplied the matcher development history (#115 monotonic DP → #118 jump DP → #119 HMM) on synthetic clips. The **matchers survive** (`follower.py`, `hmm.py`), as do `metric.py`'s scorer core and `score_notes.py` — this eval reuses them. The **synthetic clip/pathology machinery** (`clip_generator`, `pathologies`, `segments`, `trajectory`, `asap_alignment`, `gap_report`) and the `claim_measurement/gate1/` corruption layer are slated for removal **once the gold track validates this eval as the trusted source of truth** (#133 S5).
