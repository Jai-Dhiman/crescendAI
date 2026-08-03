# Real-Audio Score-Follower Eval — Provisional

**Status:** Track A is complete on 55 competition-grade recordings. Track B's human labeling pass is **complete on the 32-clip subset** (2026-08-03) — no high-confidence failure observed, 21/21 tracked where the piece is verified — but that subset is extreme-sampled by construction, and PASS bars are still unset. This eval is therefore **not yet the score-follower source of truth**: it needs a representative-sample pass and agreed bars. The bar-tap approach is superseded: labeling by ear requires bar numbers against scores the labeler does not have. Track A uses ASAP's alignment, and Track B asks the listener only to watch and flag disagreements.

## Why this exists

The follower's product job is to track an amateur from ordinary room or phone audio. The prior benchmark (`model/src/follower_bench/`) used pristine ASAP MIDI with hand-spliced pathologies. Track A now measures real performed audio through the production transcriber, but on competition-grade recordings. Track B covers amateur YouTube practice, but lacks independent position truth until the human pass is complete. Together they narrow the gap; neither alone proves phone-recorded, non-linear practice performance.

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

Only performances whose MAESTRO audio is present locally can run in audio mode (`asap_audio --list`); 519 of the 1066 ASAP rows carry a MAESTRO link, `--fetch` pulls the rest per-file from `ddPn08/maestro-v3.0.0`.

**Track A through-audio result (2026-07-27, 55 performances, 13 composers).** 56 ASAP performances transcribed from MAESTRO audio through Transkun; 55 have a usable ASAP alignment (1 excluded: `Scriabin/Sonatas/5/ChernovA06M` is not `score_and_performance_aligned`).

Paired over the **same 55 performances and the same windows**, so the only variable is the note source:

| Track A note source | full-follow median | cold-start median | cold-start pooled within-1-beat |
|---|---|---|---|
| ASAP performance MIDI (matcher isolation) | 0.000 beats | 0.000 beats | 0.9242 (440 windows) |
| **MAESTRO audio → Transkun** | **0.005 beats** | **0.005 beats** | **0.9205** (440 windows) |
| transcription cost | +0.005 beats | +0.005 beats | **−0.37 pp** |

Transcription fidelity: AMT note count vs the reference MIDI is a **median 0.991** of the reference (worst: dense virtuoso writing — Liszt Ballade 2 0.91, Islamey 0.92). Transcription cost is ~0.27× realtime on CPU.

Reading: going through the real transcriber costs **0.005 beats of median localization and 0.37 percentage points of within-1-beat**, on a fully paired 55-performance comparison (`data/evals/trackA_audio_8.json` vs `trackA_midi_paired.json`). The follower is not transcription-limited on competition-grade recordings. Scope limit: these are competition recordings from MAESTRO, not phone or room audio, and not amateur playing — Track B covers that population and has no independent position truth. The weakest clips are `Schumann/Arabeske/Min09M` (within-1-beat 0.76) and `Schubert/Impromptu_op.90_D.899/4_no_repeat` (0.66), both of which are also the weakest on clean MIDI, so they are matcher/repeat-structure cases rather than audio cases.

**Track B — light-touch human validation of the amateur clips (`validate_tool.py`).** The amateur clips have no independent position truth, and ASAP has neither phone audio nor amateur restarts. The tool draws two note strips on one score-time axis: played notes at the follower's inferred positions over the score reference. The human holds SPACE over wrong spans and chooses `tracked`, `recovered`, `wrong`, or `junk`. `validate_report.py` keeps those outcomes separate and crosses them with confidence computed against the resolved score; it does not collapse `junk`, `recovered`, and `tracked` into one success number. Follower views are cached because `follow_hmm` is O(performance notes × score notes).

**Piece-ID over the 32-clip subset (2026-07-27): 16 label-confirmed, 5 RE-LABELED, 11 abstain.** 16% of the subset had been validated against the wrong score:

| filed as | actually playing | conf |
|---|---|---|
| `moonlight_sonata_mvt1/_KGpW2ROcwA` | `beethoven.piano_sonatas.1-1` (Op. 2/1 — the video title confirms it) | 0.91 |
| `rachmaninoff_prelude_csm/v80RecqrtJ8` | `rachmaninoff.preludes_op_23.4` | 0.97 |
| `bach_prelude_c_wtc1/A_wTcQZoyxM` | `bach.prelude.bwv_870` (WTC **II**) | 0.94 |
| `pathetique_mvt2/IqbxUz9xi1c` | `beethoven.piano_sonatas.8-1` (mvt **1**) | 0.76 |
| `fantaisie_impromptu/JbYGHXsQiqk` | `chopin.etudes_op_25.5` | 0.84 |

**This resolves the v1 "21% low-confidence" question: confidence is a working score-mismatch detector.** Re-running the follower on the corrected score moves every re-labeled clip from **0.03–0.07 to 0.67–0.88** confidence, while the 11 abstained clips stay at **0.11–0.39** on their (probably correct) label score. So low confidence has two distinct causes and they separate cleanly: *wrong score* recovers when corrected, *hard/hesitant playing* does not. The residual low-confidence set is performance difficulty, not mislabeling. Of the 11 abstains, 4 already rank the folder-label piece first (at low confidence), so a wider `--k` shortlist would not help — only a longer verify window might.

**The validator follows the piece-ID'd score, not the folder label.** `validate_tool` resolves each clip through `piece_id.py`. When piece-ID abstains, the UI labels the folder-score fallback **SCORE UNVERIFIED**; reports must stratify those rows and must not present them as verified follower accuracy. Each saved validation records the resolved score, score source, and confidence from that exact follower view. A missing piece-ID map is a loud error, and `--trust-labels` is an explicit diagnostic override. The view cache is keyed by clip and score.

## v1 proxy results (279 clips, 0 harness failures)

| Signal | Median | Spread |
|---|---|---|
| score span | 1.00 | 12% of clips <0.5 |
| coverage | 0.74 | p10 0.48 → p90 0.86 |
| confidence | 0.89 | p10 0.18 → p90 0.96; 21% of clips <0.5 |
| clips with ≥1 repeat/restart | — | 72% |

Reading: the follower traverses the full score on most amateur clips and raises a low-confidence signal on about 21%. These are behavior proxies, not accuracy. A confidently wrong alignment can produce the same coverage and span, and the recordings do not establish a controlled phone-audio domain. Track B must adjudicate the clips before this result can license a product claim.

## Track B human pass — result (2026-08-03, 32 clips)

All 32 subset clips are labeled. Outcomes stratified by resolved-score confidence (threshold 0.5) and by score source:

| stratum | tracked | recovered | wrong | junk |
|---|---|---|---|---|
| high confidence (≥0.5) | 21 | 1 | **0** | **0** |
| low confidence | 1 | 0 | 2 | 7 |
| score **verified** by piece-ID | 20 | 1 | **0** | **0** |
| score **unverified** (piece-ID abstained) | 2 | 0 | 2 | 7 |

Median fraction of playback flagged wrong: 0.0 (p90 0.0).

**Two findings, both stated at the strength the sample supports.**

1. **No high-confidence failure was observed.** Every `wrong` and `junk` clip sits in the low-confidence stratum; no clip was confidently mistracked. This is the human-adjudicated form of the "knows when it's lost" property the proxy track could only suggest. It is an observation of zero failures in 22 high-confidence clips, not a measured failure *rate* — the upper bound is loose at this n.
2. **Where the piece is verified, the follower tracked it — 21 of 21** (20 `tracked`, 1 `recovered`). The 11 unverified-label rows cannot support a verified accuracy claim in either direction: 7 were judged `junk`, which is consistent with the follower correctly declining an unusable or unidentifiable clip, and the 2 `wrong` rows are ambiguous between follower failure and a wrong score on screen.

### Abstain-resolution retry (2026-08-03) — mostly a negative result

The 11 abstained clips were re-run at a 4× verify window (`--window-sec 120 --k 12`, vs the original 30 s/k6) to de-confound the two `wrong` verdicts, both of which sat in the unverified-score stratum. Results in `_piece_id_retry120.json` (kept **separate** from the trusted `_piece_id.json`).

**Only 1 of 11 resolved.** A longer window is not the lever:

- Label-piece confidence rose in 9 of 11 clips but *fell* in 2 (`mozart_k545_mvt1/znDedazaZ6Q` 0.47 → 0.20), so it is not monotone in evidence.
- 6 of 11 still rank the label at position ≥2, several far down (`fur_elise/BShLXl02VvQ` 11th, `nocturne_op9no2/rNkfVVKbICk` 12th). `fur_elise/BShLXl02VvQ` is a known 13-minute mixed practice session. **The abstain set is dominated by clips that are not a single piece**, and a longer window makes multi-piece clips worse. The lever is segmentation (detect piece changes within a clip), not more evidence per window.
- **Abstain agrees with the human `junk` verdict on 7 of 11 clips** — two independent routes to "this clip is unusable", which corroborates that abstaining is well-calibrated rather than merely conservative.

**What it settled about the two `wrong` verdicts:**

| clip | label rank @120 s | label conf 30 s → 120 s | reading |
|---|---|---|---|
| `liszt_liebestraum_3/KBsZuxQLp9k` | 1 (ngram + label agree) | 0.11 → 0.24 | score probably correct → **candidate genuine follower failure** |
| `schumann_traumerei/A9zEB2mWbrI` | 7 | 0.12 → 0.10 | score probably wrong → **mislabeled clip, not a follower failure** |

**This does NOT satisfy #108's resume trigger.** That trigger requires *trusted* evidence of a measured follower failure; the liszt clip is at confidence 0.24 against a 0.50 accept floor. Treating "label ranked first, below the floor" as verification would convert a null result into a pass. It is one strong candidate failure, and one failure removed from the pool — not a confirmation.

**Pending decision:** `chopin_waltz_csm/cAo5RtmpFVU` resolved to `chopin.waltzes.64-2` (0.46 → 0.88), which is the same score it was already validated against — only its verification status changed. Merging it into `_piece_id.json` moves it from the unverified to the verified stratum (verified successes 21 → 22). Not merged yet, because it changes a reported number.

**The subset is extreme-sampled, not representative.** `gold_subset.json` takes the lowest- and highest-confidence clip per piece by design, so these counts are not corpus rates and must not be reported as such. A representative rate needs a random-sample pass over the 279-clip corpus.

**PASS bars remain unset — deliberately, and this is a human-lit call.** The distribution needed to set them now exists (above, plus Track A's per-beat errors), but choosing thresholds is research-gate interpretation, not a derivation. Candidate shape, for a decision rather than as a decision: gate on *no high-confidence failures* plus a floor on verified-score success, and keep `recovered` separate from `tracked` since relocking is partial evidence. Do not gate on the pooled 32-clip success fraction — it mixes strata that mean different things.

**Provenance note.** Six records (the 2026-08-01 labeling session) predate `validate_tool` recording score provenance and were **backfilled**, not re-labeled: `score_id` / `score_source` recomputed via `resolve_score_id`, `follower_confidence` read from that clip's cached view. Those three fields are derived and were verified byte-identical to what the validator writes on all 26 natively-saved records; the human verdicts and wrong spans are untouched. The migrated records carry `provenance: "backfilled"`. Five of the six are the re-labeled clips, so if any result hinges on them, re-label rather than trust the migration.

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

Track B outputs `data/evals/realaudio_bundles/<piece>/<vid>.validate.json` with the verdict, wrong spans, resolved score, score source, and resolved-score confidence. Modules: `asap_eval.py` (Track A), and `validate_tool.py` plus `validate_report.py` (Track B). Tests live under `tests/follower_eval/`.

**Superseded:** `tap_tool.py` and `gold_report.py`. `accuracy.py` remains because Track A and Track B reuse its decoding and error core. `gold_subset.json` remains the committed clip selection. Delete the superseded tools only in the separately approved cleanup batch.

## Piece-ID — the corpus is mislabeled (`piece_id.py`)

**Finding (2026-07-27):** corpus folder labels are unreliable. The 32-clip pass confirmed 16 labels, relabeled five clips, and abstained on 11. Correcting the five scores raised their confidence from 0.03–0.07 to 0.67–0.88. The 11 abstentions remained at 0.11–0.39 on their likely labels. Mislabeling explains the five relabeled clips; it does not explain the residual low-confidence group.

**Stage:** per clip, identify the score actually played against the 10,494-score catalog — ngram trigram shortlist (`data/fingerprints/ngram_index.json`) UNION the folder label translated via `SCORE_FILENAME_BY_PIECE`, then follower-verify each candidate on a 60 s window, decide by **coverage × confidence** with an abstain floor (confidence is the arbiter; a wrong score can cover a tonal window but never earns high posterior). Catalog scores are all `load_score`-compatible, so any candidate is followable. VERIFIED: fantaisie → RE-LABELED `chopin.etudes_op_25.5` (cov 0.62/conf 0.84 vs 0.51/0.06); bach_prelude → CONFIRMED `bach.prelude.bwv_846` (cov 0.99/conf 0.97 via the label channel, since ngram is blind to its arpeggios).

```bash
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.piece_id \
  --clips fantaisie_impromptu/JbYGHXsQiqk bach_prelude_c_wtc1/w03EKJjOTJE --k 6 --window-sec 30
```

Limits: verification is a transpose search across K candidates and costs about 30–60 seconds per clip. The n-gram-plus-label shortlist can miss clips that are both mislabeled and arpeggiated. The validator wiring is complete; a corpus-wide relabel pass is not.

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

`follower_bench/` supplied the matcher development history (#115 monotonic DP → #118 jump DP → #119 HMM) on synthetic clips. The matchers, metric core, and score-note loader remain. The synthetic clip/pathology machinery and `claim_measurement/gate1/` corruption layer remain until Track B is labeled, PASS bars are fixed, and #133 closes. Their later removal requires the approved destructive-cleanup batch.
