# Real-Audio Score-Follower Eval — Provisional

**Status:** Track A is complete on 55 competition-grade recordings. Track B's human labeling pass is **complete on the 32-clip subset** (2026-08-03) — no high-confidence failure observed, 22/22 tracked or recovered where the piece is verified — and **complete on a representative 40-clip random sample** (2026-08-05) — 31/31 verified-score success, 0 `wrong` verdicts, 0 high-confidence failures, and a corpus mislabel rate of **2.5%** rather than the subset's 16%. PASS bars are now set: **bar 1 (zero high-confidence failures) passes; bar 2 (verified success ≥0.90 at the 95% Wilson lower bound) is at 0.890 and needs a few more verified clips.** This eval is therefore **not yet the score-follower source of truth** — it is one labeling round away. The bar-tap approach is superseded: labeling by ear requires bar numbers against scores the labeler does not have. Track A uses ASAP's alignment, and Track B asks the listener only to watch and flag disagreements.

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

**Piece-ID over the 32-clip subset (2026-07-27): 16 label-confirmed, 5 RE-LABELED, 11 abstain** (`_piece_id.json` now reads 17/5/10 after the waltz clip was merged from the wider-window retry). 16% of *this subset* had been validated against the wrong score — **that 16% is a subset rate, not a corpus rate**; the corpus figure is 2.5%, see [Piece-ID contamination is a subset artifact](#piece-id-contamination-is-a-subset-artifact-random-sample-2026-08-04):

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
| score **verified** by piece-ID | 21 | 1 | **0** | **0** |
| score **unverified** (piece-ID abstained) | 1 | 0 | 2 | 7 |

Median fraction of playback flagged wrong: 0.0 (p90 0.0).

**Two findings, both stated at the strength the sample supports.**

1. **No high-confidence failure was observed.** Every `wrong` and `junk` clip sits in the low-confidence stratum; no clip was confidently mistracked. This is the human-adjudicated form of the "knows when it's lost" property the proxy track could only suggest. It is an observation of zero failures in 22 high-confidence clips, not a measured failure *rate* — the upper bound is loose at this n.
2. **Where the piece is verified, the follower tracked it — 22 of 22** (21 `tracked`, 1 `recovered`). The 10 unverified-label rows cannot support a verified accuracy claim in either direction: 7 were judged `junk`, which is consistent with the follower correctly declining an unusable or unidentifiable clip, and the 2 `wrong` rows are ambiguous between follower failure and a wrong score on screen.

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

**Merged (owner-approved):** `chopin_waltz_csm/cAo5RtmpFVU` resolved to `chopin.waltzes.64-2` (0.46 → 0.88) — the same score it was already validated against, so only its verification status changed. It is merged into `_piece_id.json`, moving it from the unverified to the verified stratum (verified successes 21 → 22); its validation record carries `score_source_note` recording that it was verified on the second, wider-window pass.

**The subset is extreme-sampled, not representative.** `gold_subset.json` takes the lowest- and highest-confidence clip per piece by design, so these counts are not corpus rates and must not be reported as such. A representative rate needs a random-sample pass over the 279-clip corpus — the piece-ID half of that pass is now done (next section); the human follower-accuracy half is not.

### Piece-ID contamination is a subset artifact (random sample, 2026-08-04)

`_random_sample.json` draws **40 clips uniformly at random** (`random.Random(133).sample`) from the 247 clips of the 279-clip corpus that `gold_subset.json` had not already taken. It touches all 16 pieces and is *not* selected on confidence, so its rates are corpus rates. Piece-ID was run over all 40 (`_piece_id_sample40.json`):

| piece-ID outcome | random sample (n=40) | 95% Wilson | `gold_subset` (n=32) |
|---|---|---|---|
| label confirmed | 30 — **75.0%** | 59.8–85.8% | 17 — 53.1% |
| **RE-LABELED** (validated against the wrong score) | 1 — **2.5%** | 0.4–12.9% | 5 — 15.6% |
| abstain | 9 — **22.5%** | 12.3–37.5% | 10 — 31.2% |

**Use 2.5%, not 16%, as the corpus mislabel rate.** The two are not in conflict: `gold_subset` selects the lowest-confidence clip per piece by construction, and low confidence is exactly the signal that detects a wrong score, so the subset is enriched for mislabeling *by design*. Its 16% was never a corpus estimate and must not be quoted as one.

Stated at the strength the data supports: the design argument is what settles this, not the test. Fisher's exact on the re-label counts is **p = 0.082** and on abstain **p = 0.43** — at these n the observed gap is only marginally distinguishable from sampling noise, so the honest claim is "the corpus rate is 2.5% [0.4%, 12.9%] and the subset rate cannot stand in for it", not "the subset is proven inflated".

The single corpus re-label is `rachmaninoff_prelude_csm/t0i7L6kE5k4` → `rachmaninoff.preludes_op_23.4` (conf 0.95) — the **same folder and same wrong target** as the subset's `v80RecqrtJ8`. That folder holds four sampled clips: two genuine Op. 3/2, one Op. 23/4, one abstain. Contamination clusters by folder rather than spreading uniformly, so a corpus-wide rate averages over folders that are clean and folders that are partly mis-filed; treat per-folder purity, not the global rate, as the thing to check before trusting any single piece's clips.

**PASS bars are set — see [PASS bars](#pass-bars-owner-set-2026-08-05).** They were deliberately left unset until a corpus-representative distribution existed, because the 32-clip subset could not support any meaningful threshold: at n=22 verified clips, even a perfect 22/22 yields a 95% Wilson lower bound of 0.851, so a "≥0.90" bar was unmeetable regardless of how well the follower performed. Setting bars from that sample would have meant either an unclearable gate or a gate quietly lowered to fit the evidence.

**Provenance note.** Six records (the 2026-08-01 labeling session) predate `validate_tool` recording score provenance and were **backfilled**, not re-labeled: `score_id` / `score_source` recomputed via `resolve_score_id`, `follower_confidence` read from that clip's cached view. Those three fields are derived and were verified byte-identical to what the validator writes on all 26 natively-saved records; the human verdicts and wrong spans are untouched. The migrated records carry `provenance: "backfilled"`. Five of the six are the re-labeled clips, so if any result hinges on them, re-label rather than trust the migration.

## Track B corpus pass — result (2026-08-05, random 40 clips)

The 40 clips of `_random_sample.json` are all labeled. **These are corpus rates; the 32-clip table above is not.** Never pool the two for rate estimation.

| stratum | tracked | recovered | wrong | junk |
|---|---|---|---|---|
| high confidence (≥0.5) | 31 | 2 | **0** | **0** |
| low confidence | 3 | 0 | **0** | 4 |
| score **verified** by piece-ID | 30 | 1 | **0** | **0** |
| score **unverified** (abstained) | 4 | 1 | **0** | 4 |

- **Verified-score success (tracked + recovered): 31/31 = 1.000**, 95% Wilson [0.890, 1.000].
- **High-confidence failures: 0/33**, 95% Wilson upper bound 10.4% (the subset alone gave 14.9%).
- **Zero `wrong` verdicts anywhere in the corpus sample.** The subset's two `wrong` rows were both products of its selection: one was a mislabeled clip, one a candidate genuine failure at confidence 0.24.

**Confidence is conservative on the corpus, not merely calibrated — and the subset hid this.** In `gold_subset` the low-confidence stratum was 1 tracked against 9 wrong/junk; in the corpus sample it is **3 tracked against 4 junk**, with tracked clips going down to confidence **0.283** (`mozart_k545_mvt1/anaTIyEI9vI`, also `chopin_etude_op10no4/inatg1mSCqE` 0.317, `liszt_liebestraum_3/MnZRLzeplJg` 0.433). Selecting the lowest-confidence clip per piece selects the cases where low confidence was *right*; the corpus also contains hard playing the follower follows correctly anyway. Reading: a low-confidence corpus clip is roughly a coin flip between "unusable clip" and "tracked fine", so low confidence should suppress a claim, not be reported as a follower failure.

**`fraction_wrong` is UNUSABLE for this pass — the spans were lost, not absent.** It reads 0.000 on all 40 clips, but the labeler marked confusing passages with SPACE and saw the red spans render on the timeline; they did not reach disk. Do not interpret 0.000 as "no mistracked playback was observed": it is missing data. **The verdicts are unaffected** — they are a separate field, were entered and saved normally, and are what both PASS bars are computed from — so the bar results stand.

Root cause is not fully established. The end-to-end path (real key hold → real verdict click → real Save) was driven under Playwright and writes spans correctly, including under the labeler's reported conditions (audio playing throughout, never pausing). Four defects were found and fixed; the leading candidate is the third, which silently overwrote saved spans with `[]` on any re-visit-and-re-save:

| defect | fix |
|---|---|
| SPACE also toggled native play/pause when the `<audio>` element had focus, freezing `currentTime` under the hold | the player is blurred on focus, so SPACE only ever marks |
| a hold shorter than 0.05 s was discarded with no feedback (which is what a paused hold produces) | the tool now says the mark was NOT recorded and why |
| `selectClip` blanked marks and verdict, and re-selecting a saved clip restored nothing — so a second Save posted `[]` over real spans | saved spans and verdict are round-tripped from disk and restored on re-select; leaving with unsaved work prompts; reload prompts |
| Save could post spans that disagreed with the red bar, and accepted `wrong`/`junk` with no marks silently | Save refuses on bar/payload divergence, confirms an unmarked `wrong`/`junk`, shows a live mark count, and reports the number saved |

Regression tests cover the round-trip (`test_list_clips_carries_already_saved_spans_and_verdict`, `test_list_clips_unlabeled_clip_has_no_saved_state`).

## PASS bars (owner-set, 2026-08-05)

Two bars, both on Track B, chosen by the owner from the observed distribution. `recovered` counts toward success but is always reported separately, since relocking is partial evidence.

| # | bar | measured on | status |
|---|---|---|---|
| 1 | **Zero `wrong` or `junk` verdicts among clips with resolved-score confidence ≥ 0.5** | all labeled clips | **PASS** — 0/33 corpus, 0/55 pooled |
| 2 | **Verified-score success ≥ 0.90 as the 95% Wilson *lower* bound** | corpus-representative clips only (`_random_sample*.json`), never `gold_subset` | **PENDING** — 31/31 gives 0.890, short by 0.010 |

Bar 1 is zero-tolerance rather than a rate because the consequence is asymmetric: a confidently-wrong alignment is the one the product acts on, so a single instance is qualitatively different from a rate. Bar 1 can never be *proven*, only left unrefuted at the current n; quote the sample size with it.

Bar 2 is stated on the lower bound, not the point estimate, so that it cannot be passed by a small sample that happens to be clean. It is **not yet met**: 31 verified clips at 100% success bound out at 0.890. Thirty-five perfect verified clips reach 0.9011. `_random_sample2.json` (15 more clips, `random.Random(1330)` over the 207 still unlabeled) was drawn to close that gap; it is poolable with `_random_sample.json` because both are uniform draws, and poolable with neither `gold_subset.json` nor any confidence-selected set.

**Do not report the pooled 72-clip success fraction as a rate.** Pooling is defensible only for the *existence* claim behind bar 1 — a counterexample found in either sample refutes it — and even there the pooled bound (0.065 upper) describes a mixture that is harder than the corpus, not the corpus itself.

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

## Piece-ID — folder labels need verifying (`piece_id.py`)

**Finding (2026-07-27, subset):** the 32-clip pass confirmed 16 labels, relabeled five clips, and abstained on 11. Correcting the five scores raised their confidence from 0.03–0.07 to 0.67–0.88. The 11 abstentions remained at 0.11–0.39 on their likely labels. Mislabeling explains the five relabeled clips; it does not explain the residual low-confidence group.

**Corrected at corpus scale (2026-08-04):** the random 40-clip sample re-labels **1 in 40 (2.5%)**, not 1 in 6. "The corpus is mislabeled" was an artifact of reading an extreme-sampled subset as a corpus; the accurate statement is that folder labels are *unverified* and contamination clusters in a few folders. Piece-ID is still mandatory before any clip is used as evidence — a 2.5% base rate is fatal at n=1, which is exactly how these clips get consumed. Details and CIs in [Piece-ID contamination is a subset artifact](#piece-id-contamination-is-a-subset-artifact-random-sample-2026-08-04).

**Stage:** per clip, identify the score actually played against the 10,494-score catalog — ngram trigram shortlist (`data/fingerprints/ngram_index.json`) UNION the folder label translated via `SCORE_FILENAME_BY_PIECE`, then follower-verify each candidate on a 60 s window, decide by **coverage × confidence** with an abstain floor (confidence is the arbiter; a wrong score can cover a tonal window but never earns high posterior). Catalog scores are all `load_score`-compatible, so any candidate is followable. VERIFIED: fantaisie → RE-LABELED `chopin.etudes_op_25.5` (cov 0.62/conf 0.84 vs 0.51/0.06); bach_prelude → CONFIRMED `bach.prelude.bwv_846` (cov 0.99/conf 0.97 via the label channel, since ngram is blind to its arpeggios).

```bash
PYTHONPATH="$WT/src" .venv/bin/python -m follower_eval.piece_id \
  --clips fantaisie_impromptu/JbYGHXsQiqk bach_prelude_c_wtc1/w03EKJjOTJE --k 6 --window-sec 30
```

Limits: verification is a transpose search across K candidates and costs about 30–60 seconds per clip. The n-gram-plus-label shortlist can miss clips that are both mislabeled and arpeggiated. The validator wiring is complete. Piece-ID now covers 72 of 279 clips (32 subset + 40 random sample); a full corpus-wide relabel pass is not done.

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
