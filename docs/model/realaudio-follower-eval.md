# Real-Audio Score-Follower Eval — SOURCE OF TRUTH

**Status:** v1 proxy track live (2026-07-26, #133). Gold (accuracy) track **tooling built** (2026-07-27, #133 S3); awaiting human tapping to produce the real accuracy numbers + final PASS bars. This eval is the **source of truth** for score-follower quality; the synthetic `follower_bench` (below) is superseded and slated for removal once the gold track is trusted.

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

**Gold track (tooling built, #133 S3) — human bar-tap, 32-clip subset.** The one **non-circular accuracy** number: a human taps bar downbeats → `(bar_number → audio_sec)` reference; the follower's decoded score-position at each tap is compared to the true bar's score-second → bar-localization error, % within tolerance, relock latency after restarts. Also adjudicates whether low-confidence proxy clips are genuine hard cases (correct to abstain) or follower failures.

*Why the two clocks compare:* a tap's `audio_sec` is WAV playback time; the follower's `MatchedNote.perf_time` is the Transkun onset in that **same** WAV clock (the tap tool serves the transcription's source WAV, not a YouTube re-encode). Interpolating the follower's `perf_time → score_position` staircase at each tapped `audio_sec` yields a decoded score-second directly comparable to `measure_table[bar_number].start_sec`. Both live in score-render seconds, so the error is a real cursor offset, not a proxy. Error is also reported in **bars** (error ÷ local bar duration) so it is tempo-invariant across the rep.

## v1 proxy results (279 clips, 0 harness failures)

| Signal | Median | Spread |
|---|---|---|
| score span | 1.00 | 12% of clips <0.5 |
| coverage | 0.74 | p10 0.48 → p90 0.86 |
| confidence | 0.89 | p10 0.18 → p90 0.96; 21% of clips <0.5 |
| clips with ≥1 repeat/restart | — | 72% |

Reading: the follower traverses the full score on the large majority of real amateur clips, is robust to real AMT + phone-audio noise, and raises a self-doubt signal on ~21% (the beginner-heavy, hesitation-heavy pieces — `fur_elise`, `pathetique`). **Accuracy remains unverified** — high coverage/confidence/span is *consistent with* good following but is not proof; a confidently-wrong alignment yields the same proxies. That is the gap the gold track fills.

**PASS bars are deliberately not yet finalized** — they will be built on the observed **tap-level distribution** across the labeled clips, not per-clip medians (the median `backward_frac` of 0.00 hid that 72% of clips contain repeats; aggregates lie about exactly the behavior we care about). `gold_report.py` pools every tap, never medians-of-medians.

The verdict currently runs against **provisional placeholder bars** (`PROVISIONAL_PASS` in `gold_report.py`): `within_1bar_frac ≥ 0.85` and `median_abs_err_bars ≤ 0.5` — the minimum a live cursor needs (land on the right measure on the large majority of downbeats). These are advisory and printed with a `PROVISIONAL` flag until the human tapping is done; then re-derive the bars from the actual distribution and lock them.

## Gold track — how to label and score (#133 S3)

Subset: `model/src/follower_eval/gold_subset.json` — 32 clips (per rep piece, the lowest- and highest-confidence v1 clip; spans confidence 0.03→0.97 across all 16 pieces).

```bash
cd model    # PRIMARY checkout (data/ + the WAVs are gitignored, absent in worktrees)
WT=<path-to-issue-133-worktree>/model
# 1. label: opens a local page that serves each clip's real WAV; tap Space on each
#    bar downbeat (edit "Next bar" back on a repeat/restart), Save per clip.
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.tap_tool --serve   # http://localhost:8766
# 2. score: runs the follower on each labeled clip, prints bar-localization
#    accuracy + a (provisional) PASS/FAIL verdict.
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.gold_report \
  --bundles-root data/evals/realaudio_bundles --scores-root data/scores --out /tmp/gold_report.json
```

Gold labels are written to `data/evals/realaudio_bundles/<piece>/<vid>.gold.json` (`{bar_taps:[{bar_number,audio_sec}]}`) — gitignored, resumable (the tool re-loads existing taps). Modules: `tap_tool.py` (labeler), `accuracy.py` (metric), `gold_report.py` (report + verdict); tests in `tests/follower_eval/test_accuracy.py`.

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
