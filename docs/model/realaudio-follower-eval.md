# Real-Audio Score-Follower Eval — SOURCE OF TRUTH

**Status:** v1 proxy track live (2026-07-26). Accuracy is measured two ways (2026-07-27, #133 S3): **Track A** — automatic, against ASAP's human-verified beat alignment (live; first numbers in) — and **Track B** — a light-touch human validator for the real amateur clips (tool built, browser-verified; awaiting a labeling pass). The **bar-tap tool is superseded**: labeling by ear needs bar numbers against scores the labeler doesn't have — Track A gets ground truth from ASAP instead, Track B needs only "watch and flag." This eval is the **source of truth** for score-follower quality; the synthetic `follower_bench` (below) is superseded and slated for removal once the accuracy tracks are trusted.

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

**Track A — automatic, against ASAP beat alignment (`asap_eval.py`).** ASAP ships, per real performance, a human-verified alignment between the performance timeline and the score timeline (`performance_beats[i]` ↔ `midi_score_beats[i]`). That pairing *is* the answer key: at performance-time `p_i` the true score position is `s_i`. We run the follower on the ASAP performance and compare its decoded score-position at each `p_i` to `s_i` → localization error in seconds and tempo-invariant **beats**, over 1000+ real pianists in the rep idiom, no labeling. Two modes: **full-follow**, and **random mid-performance cold-start** (feed only notes from a random `t_start` — the real interactive case, where the user is somewhere in the middle with no lead-in). First result on clean ASAP MIDI: near-perfect (full ~0.001 beat median; cold-start ~0.001 beat, pooled within-1-beat 0.98) — an isolation result showing the *matcher* is not the weak link, so real-audio degradation will come from transcription noise.

**Through real audio (`asap_audio.py`, `--perf-source audio`).** The same performances are supplied as **MAESTRO audio → Transkun** (the production transcriber) with ASAP's alignment still the answer key, so the only thing that changes between the two modes is the note source and the numbers are directly comparable. A missing bundle is an error, never a fall back to MIDI — that would report a clean-MIDI number as if it had come through audio.

> **Do not use ASAP's metadata `start` column as the audio time offset.** ASAP performances are excerpts cut from longer MAESTRO recordings, so a transcription is in MAESTRO's clock while the beat truth is in ASAP's. `start` looks like that offset and is wrong: for `Bach/Fugue/bwv_858/Zhang01M` it reads **90.331 s while the true offset is 89.831 s**. Half a second is a large systematic error at beat resolution, and it corrupts the number silently instead of failing. `derive_shift()` recovers the offset from the notes themselves (both MIDIs are on disk) and raises below 98% note agreement rather than guessing.

Only performances whose MAESTRO audio is present locally can run in audio mode (`asap_audio --list`); 519 of the 1066 ASAP rows carry a MAESTRO link, and the rest of that audio is a download.

**Track B — light-touch human validation of the amateur clips (`validate_tool.py`).** The amateur phone-audio clips have no independent ground truth and ASAP has neither phone audio nor amateur restarts, so the real target still needs a human — but not bar numbers. The tool draws two note-strips on one score-time axis: the played notes *where the follower placed them* (`decode_at` of each onset) over the score reference, with a playhead driven by the follower's decoded position. When the follower is right the rows line up; when wrong they clash (visibly and against the audio). The human holds SPACE over wrong spans and picks one verdict (`tracked` / `recovered` / `wrong` / `junk`). `validate_report.py` aggregates these into a success fraction and, crucially, the **low-confidence adjudication**: a low-conf clip marked `junk` = the follower was right to abstain; marked `wrong` = a real failure. Follower views are cached to disk (`--precompute`) because `follow_hmm` is O(perf×score) — big clips take minutes to compute but then load instantly.

**The validator follows the piece-ID'd score, not the folder label.** The corpus labels are known-wrong, so `validate_tool` resolves each clip's score through `piece_id.py` (`load_piece_id_map` → `resolve_score_id`). A clip the ID stage abstained on still gets validated, but against the folder label and flagged **SCORE UNVERIFIED** in the UI, so a labeling failure is never recorded as a follower failure. A missing piece-ID map is a loud error naming the command to run; `--trust-labels` is the explicit opt-in to the old wrong-score behavior. The view cache is keyed by clip **and score**, so a re-labeled clip is never served a view computed against its old score.

## v1 proxy results (279 clips, 0 harness failures)

| Signal | Median | Spread |
|---|---|---|
| score span | 1.00 | 12% of clips <0.5 |
| coverage | 0.74 | p10 0.48 → p90 0.86 |
| confidence | 0.89 | p10 0.18 → p90 0.96; 21% of clips <0.5 |
| clips with ≥1 repeat/restart | — | 72% |

Reading: the follower traverses the full score on the large majority of real amateur clips, is robust to real AMT + phone-audio noise, and raises a self-doubt signal on ~21% (the beginner-heavy, hesitation-heavy pieces — `fur_elise`, `pathetique`). **Accuracy remains unverified** — high coverage/confidence/span is *consistent with* good following but is not proof; a confidently-wrong alignment yields the same proxies. That is the gap the gold track fills.

**PASS bars are deliberately not yet finalized.** Track A gives a per-beat error distribution over real performances; Track B gives the amateur-clip success fraction and the low-confidence adjudication. Bars get set from the observed distributions once the MAESTRO-audio Track A run and a Track B labeling pass are in — not from the clean-MIDI numbers, which are the easy end.

## How to run the accuracy tracks (#133 S3)

```bash
cd model    # PRIMARY checkout (data/raw/asap + the WAVs are gitignored/absent in worktrees)
WT=<path-to-issue-133-worktree>/model

# TRACK A — automatic, ASAP ground truth (no labeling). --limit caps the corpus.
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.asap_eval \
  --limit 40 --random-starts 8 --window-sec 20 --out /tmp/asap_trackA.json

# TRACK A through real audio: build MAESTRO->Transkun bundles, then re-run.
# (--list first: only performances whose MAESTRO audio is local can run.)
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.asap_audio --list
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.asap_audio          # transcribe (minutes/clip)
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.asap_eval --perf-source audio \
  --out /tmp/asap_trackA_audio.json

# PIECE-ID — required before Track B; the corpus labels are wrong.
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.piece_id \
  --clips <piece/vid ...> --out data/evals/realaudio_bundles/_piece_id.json   # resumable

# TRACK B — light-touch validation of the amateur clips.
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.validate_tool --precompute   # once (minutes; big clips)
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.validate_tool --serve        # http://localhost:8767 — watch & flag
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.validate_report \
  --bundles-root data/evals/realaudio_bundles                                       # aggregate + adjudicate low-conf
```

Track B outputs `data/evals/realaudio_bundles/<piece>/<vid>.validate.json` (verdict + wrong-spans; gitignored, resumable). Modules: `asap_eval.py` (Track A), `validate_tool.py` + `validate_report.py` (Track B); tests in `tests/follower_eval/test_asap_eval.py`.

**Superseded:** the bar-tap gold tool (`tap_tool.py` / `accuracy.py` / `gold_report.py`, `gold_subset.json`) — labeling by ear needs bar numbers the labeler can't produce without scores. Kept for now (the `decode_at` + error core is reused by Track A/B); removable once the two tracks are trusted.

## Piece-ID — the corpus is mislabeled (`piece_id.py`)

**Finding (2026-07-27):** the practice corpus is labeled by the folder a YouTube video was curated into, and those labels are unreliable — VERIFIED cases: `fantaisie_impromptu/JbYGHXsQiqk` is someone sightreading **Chopin Op.25/5** (title confirms); `nocturne_op9no2/rNkfVVKbICk` is **op.9 no.1**, not the no.2 score shown; `fur_elise/BShLXl02VvQ` is a 13-min mixed session (5534 notes vs 905, pitch-class cosine 0.32). The follower was handed the **wrong score** — its low confidence on those clips was *correct* (a mismatch detector), not failure. This reframes the v1 "21% low-confidence" set as largely **score mismatch**, not follower difficulty. (Track A on ASAP is unaffected — its score↔performance pairings are verified.)

**Stage:** per clip, identify the score actually played against the 10,494-score catalog — ngram trigram shortlist (`data/fingerprints/ngram_index.json`) UNION the folder label translated via `SCORE_FILENAME_BY_PIECE`, then follower-verify each candidate on a 60 s window, decide by **coverage × confidence** with an abstain floor (confidence is the arbiter; a wrong score can cover a tonal window but never earns high posterior). Catalog scores are all `load_score`-compatible, so any candidate is followable. VERIFIED: fantaisie → RE-LABELED `chopin.etudes_op_25.5` (cov 0.62/conf 0.84 vs 0.51/0.06); bach_prelude → CONFIRMED `bach.prelude.bwv_846` (cov 0.99/conf 0.97 via the label channel, since ngram is blind to its arpeggios).

```bash
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.piece_id \
  --clips fantaisie_impromptu/JbYGHXsQiqk bach_prelude_c_wtc1/w03EKJjOTJE --k 6 --window-sec 30
```

Limits: verify is a ~25-way transpose search × K candidates (~30–60 s/clip — background the corpus pass); the ngram+label shortlist misses clips that are *both* mislabeled *and* arpeggiated (neither channel surfaces the truth → abstain), which a pitch-histogram/chroma shortlist channel would recover. NOT YET DONE: corpus-wide relabel pass; wiring the identified score into `validate_tool`.

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
