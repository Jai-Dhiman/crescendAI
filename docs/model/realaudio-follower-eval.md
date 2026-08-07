# Real-Audio Score-Follower Eval

**Status:** Track A is complete on 55 competition-grade recordings. Track B's human labeling pass is **complete on the 32-clip subset** (2026-08-03) — no high-confidence failure observed, 22/22 tracked or recovered where the piece is verified — and **complete on a corpus-representative 55-clip random pool** (2026-08-05). **Both PASS bars are MET**: zero failures among 46 high-confidence clips, and 42/42 verified-score success with a 95% Wilson lower bound of **0.9162**. Zero `wrong` verdicts across all 55 corpus clips; corpus mislabel rate **1.8%**, not the extreme-sampled subset's 16%. This eval is now the score-follower source of truth for amateur real audio, within its stated scope. The bar-tap approach is superseded: labeling by ear requires bar numbers against scores the labeler does not have. Track A uses ASAP's alignment, and Track B asks the listener only to watch and flag disagreements.

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

**`fraction_wrong` is UNUSABLE for the 40-clip pass — the spans were lost, not absent.** (The fix below is confirmed working: the 15-clip pass captured real spans, including 5 spans covering 13.6% of playback on `fantaisie_impromptu/V4a_F21W-bo`. Only the 40 are affected.) It reads 0.000 on all 40 clips, but the labeler marked confusing passages with SPACE and saw the red spans render on the timeline; they did not reach disk. Do not interpret 0.000 as "no mistracked playback was observed": it is missing data. **The verdicts are unaffected** — they are a separate field, were entered and saved normally, and are what both PASS bars are computed from — so the bar results stand.

Root cause is not fully established. The end-to-end path (real key hold → real verdict click → real Save) was driven under Playwright and writes spans correctly, including under the labeler's reported conditions (audio playing throughout, never pausing). Four defects were found and fixed; the leading candidate is the third, which silently overwrote saved spans with `[]` on any re-visit-and-re-save:

| defect | fix |
|---|---|
| SPACE also toggled native play/pause when the `<audio>` element had focus, freezing `currentTime` under the hold | the player is blurred on focus, so SPACE only ever marks |
| a hold shorter than 0.05 s was discarded with no feedback (which is what a paused hold produces) | the tool now says the mark was NOT recorded and why |
| `selectClip` blanked marks and verdict, and re-selecting a saved clip restored nothing — so a second Save posted `[]` over real spans | saved spans and verdict are round-tripped from disk and restored on re-select; leaving with unsaved work prompts; reload prompts |
| Save could post spans that disagreed with the red bar, and accepted `wrong`/`junk` with no marks silently | Save refuses on bar/payload divergence, confirms an unmarked `wrong`/`junk`, shows a live mark count, and reports the number saved |

Regression tests cover the round-trip (`test_list_clips_carries_already_saved_spans_and_verdict`, `test_list_clips_unlabeled_clip_has_no_saved_state`).

## PASS bars — both MET (owner-set 2026-08-05, measured 2026-08-05)

Two bars, both on Track B, chosen by the owner from the observed distribution. `recovered` counts toward success but is always reported separately, since relocking is partial evidence.

| # | bar | measured on | result |
|---|---|---|---|
| 1 | **Zero `wrong` or `junk` verdicts among clips with resolved-score confidence ≥ 0.5** | corpus-representative pool | **PASS** — 0/46 |
| 2 | **Verified-score success ≥ 0.90 as the 95% Wilson *lower* bound** | corpus-representative pool only (`_random_sample*.json`), never `gold_subset` | **PASS** — 42/42 = 1.000, lower **0.9162** |

Bar 1 is zero-tolerance rather than a rate because the consequence is asymmetric: a confidently-wrong alignment is the one the product acts on, so a single instance is qualitatively different from a rate. It can never be *proven*, only left unrefuted at the current n — always quote the sample size with it.

Bar 2 is stated on the lower bound, not the point estimate, so that it cannot be passed by a small sample that happens to be clean. That distinction did real work: at 31 verified clips a **perfect** 31/31 still bounded out at 0.890 and failed. `_random_sample2.json` (15 clips, `random.Random(1330)` over the 207 then-unlabeled) closed the gap to 42 verified clips. Note that at n=42 bar 2 tolerates **zero** failures — 41/42 bounds at 0.877 and would fail — so it is not yet an independent accuracy floor; that needs 53 verified clips (1 failure) or 69 (2).

**Do not report the pooled 87-clip success fraction as a rate.** Only `_random_sample.json` + `_random_sample2.json` (both uniform draws, 55 clips) are corpus-representative. `gold_subset.json` is confidence-selected and belongs in no rate estimate.

## Corpus-representative result (55 clips, 2026-08-05)

| stratum | tracked | recovered | wrong | junk |
|---|---|---|---|---|
| high confidence (≥0.5) | 43 | 3 | **0** | **0** |
| low confidence | 3 | 0 | **0** | 6 |
| score **verified** by piece-ID | 41 | 1 | **0** | **0** |
| score **unverified** (abstained) | 5 | 2 | **0** | 6 |

**Zero `wrong` verdicts across all 55 corpus clips.** Combined corpus piece-ID over both draws: 74.5% confirmed, **1.8% re-labeled** [0.3%, 9.6%], 23.6% abstain.

**Abstain is a sensitive, low-precision detector of unusable clips: 6/6 recall, 6/13 precision.** Every human-judged `junk` clip in the corpus pool was one piece-ID had abstained on, and no `junk` clip reached the verified stratum. The two `junk` clips from the second draw show why the mechanism works and what it is actually detecting:

| clip | why it is unusable | piece-ID |
|---|---|---|
| `fantaisie_impromptu/V4a_F21W-bo` | a **left-hand-only arrangement** — the catalog score genuinely is not what is being played | ABSTAIN, conf 0.164 |
| `fur_elise/gJ9x1vXYijU` | performed on a **steel pan**, not piano; its harmonics defeat the transcription front-end | ABSTAIN, conf 0.372 |

Both are "this audio is not this score played on a piano" — a *score-resolution or transcription-domain* failure, not a follower failure. That is exactly why bar 2 excludes the unverified stratum: with the wrong score on screen the clip cannot test follower accuracy in either direction. The other 7 abstained clips tracked or recovered normally, so abstain means "do not trust a claim here", not "this clip is bad".

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

`follower_bench/` supplied the matcher development history (#115 monotonic DP → #118 jump DP → #119 HMM) on synthetic clips. **The synthetic machinery was pruned on 2026-08-05, owner-approved, once both PASS bars were met** (S5): `clip_generator.py`, `pathologies.py`, `gap_report.py`, `calibration.py`, `trajectory.py`, `metric.py` and their tests are deleted. Re-running the #115/#118/#119 synthetic comparisons now requires a git revert.

**The package itself is load-bearing and must not be deleted.** `follower_eval` imports `hmm` (`follow_hmm`, `TUNED_HMM_PARAMS`), `follower` (`bar_boundary_columns`), `score_notes`, `segments`, and `asap_alignment` from it across `validate_tool`, `piece_id`, `realaudio`, `accuracy`, `asap_eval`, and `asap_audio`. Those five modules, `__init__.py`, and their tests stay.

**`claim_measurement/gate1/` was NOT pruned.** Earlier revisions of this doc grouped it with the #133 cleanup; that was wrong. It is the live GATE 1 localization harness cited by `docs/model/claim-verifier-signed-d-conventions.md` and covered by two active test files.

> Verifying the prune: run the suite from the PRIMARY checkout. `data/` is gitignored, so `__file__`-anchored ASAP paths resolve into a worktree that has no `data/raw/asap-dataset` and those tests fail environmentally. The prune took the worktree run from 33 failed / 113 passed to 5 failed / 98 passed — it removed pre-existing environmental failures and introduced none; the 5 survivors pass 7/7 when run where the data lives.

## Clean-audio baselines beyond score position (#108 / #148, 2026-08-07)

Track A's baseline covers score **position** only. Two quantities #108 lists as
its remaining scope had no clean-audio baseline at all, so #148's per-factor
degradation table had nothing to subtract from. Both are now measured on the
same ASAP performances, with the **same rule: truth is ASAP's human-verified
beat alignment and never any aligner** — #101's gate1 scored parangonar against
parangonar, so its residuals measured agreement, not accuracy.

Truth provenance is stated at the strength it has, not more: the beat anchors
are human-verified by ASAP's authors; the piecewise-linear interpolation between
anchors and the same-pitch nearest-neighbour assignment rule are **ours and
unverified**. The correct label is "human-verified beat alignment + deterministic
assignment rule", never "human-verified note correspondence".

### Per-note correspondence (`follower_eval/note_correspondence.py`)

n=60 ASAP performances, 72,406 performed notes, 0 failures. Cluster bootstrap
over **performances**, not notes — notes inside one performance share a pianist,
tempo and score, so a note-level bootstrap over 72k correlated observations
returns an interval several times too tight.

| metric | value | 95% CI |
|---|---|---|
| precision | **0.9807** | 0.9774 – 0.9836 |
| recall | **0.7705** | 0.7558 – 0.7845 |
| F1 | **0.8630** | 0.8526 – 0.8727 |

**Reading: the follower is precise but incomplete.** When it pairs a note it is
right 98% of the time, and it declines to pair ~23% of notes that have a correct
answer. Position accuracy does not reveal this — a follower can sit at the right
score time while pairing the wrong notes. For a cursor, recall of 0.77 is fine;
for wrong-note flagging or per-note timing feedback, **recall is the binding
constraint**.

Counted, never hidden: 890 notes (1.2%) have no same-pitch score note in
tolerance (wrong notes, ornaments — not charged to the follower); 1,556 (2.1%)
have two indistinguishable same-pitch candidates (trills) and are excluded,
because choosing one manufactures truth. Result: `note_correspondence_baseline.json`.

### Directional onset timing (`follower_eval/onset_direction.py`)

Same 60 performances, 9,647 scored notes, 0 failures.

| | value | 95% CI |
|---|---|---|
| **sign accuracy** | **0.8468** | 0.8202 – 0.8729 |
| null: shuffled | 0.5876 | 0.5689 – 0.6076 |
| null: majority | 0.7612 | — |
| median magnitude error | **13.6 ms** | — |

INFORMATIVE: the CI lower bound clears the majority null with margin.

**Direction is measured against a LOCAL reference tempo, and that choice is the
substance.** Notated tempo was rejected — an amateur at 60% of marked tempo would
have every note labelled late, encoding tempo choice rather than error. A global
affine fit was rejected — a genuine ritardando makes a whole closing section read
late. Only a local reference supports "you rushed *this* note". Truth side uses
ASAP's per-beat anchors; system side uses the follower's own local fit over its
matches within ±2 s.

Two readings this metric invites and must not receive:

- **The shuffled null's expectation is not 0.5.** It is `p*q + (1-p)(1-q)` ≈
  0.59 here, so an arm carrying *zero* per-note information still agrees well
  above chance. A majority-class null ships alongside it for this reason, and a
  run beating neither is reported as uninformative rather than as a number.
- **The deadband is a population statement, not a trim.** It removes 82.5% of
  matched notes, because ASAP's anchors predict most onsets to within 20 ms and
  an on-time note has no meaningful direction. The headline describes genuinely
  displaced notes — the population the early/late call is *for* — and is not a
  statement about all notes.

Result: `onset_direction_baseline.json`.

### Shared metric core (`follower_eval/ood_eval.py`)

#148's table must be subtractable from Track A's, so `ood_eval` imports
`asap_eval`'s `follow_window` / `_beat_errors` / `_summarize` and recomputes
nothing. Two tests pin it: an identity assertion that
`ood_eval.follow_window is asap_eval.follow_window`, and a source check that no
metric arithmetic has accreted locally.

Exercised on the one factor ASAP supplies at two levels today, reproducing the
**exact 55-performance paired set — 55 paired, 0 dropped, 0 failures**:

| level | median beats | Δ median | within-1-beat | Δ pp |
|---|---|---|---|---|
| midi | 0.0025 | — | 0.9505 | — |
| audio | 0.0064 | **+0.0039** | 0.9488 | −0.17 |

**Not a byte-identical reproduction of the Track A row above, and must not be
presented as one.** Track A's transcription cost is +0.005 beats against +0.0039
here, the difference being aggregation (mean of per-take medians vs Track A's
pooled figure). More importantly Track A's within-1-beat 0.9242 → 0.9205
(−0.37 pp) is a **cold-start** number while this table's within-1-beat row is
**full-follow** — the two are not comparable. What is reproduced is the paired
set, the direction, and the magnitude of the transcription cost to within 0.001
beat. Result: `ood_note_source_table.json`.

### G-OOD-6 behavior statistics are NON-DISCRIMINATING

`follower_eval/behavior_stats.py` defines the six statistics #148's G-OOD-6 was
written against but never enumerated, and computes median/IQR/n over the
279-clip corpus (`corpus_behavior_iqr.json`). All six are functions of the
transcribed note stream alone — reusing this doc's proxy-track metrics
(coverage, `backward_frac`, confidence) was rejected because those are *follower
outputs*, so matching on them would show the follower reacts similarly to both
arms rather than that the playing is similar.

**The corpus denominator is the build manifests (`ok|skip` rows), not a
directory glob.** `data/evals/realaudio_bundles/` holds 366 bundle files against
a 279-clip corpus; a glob silently moves every quantile.

| statistic | n | median | IQR |
|---|---|---|---|
| `active_duration_s` | 278 | 256.8 | 158.5 – 350.9 |
| `note_rate_per_min` | 278 | 175.3 | 121.2 – 248.6 |
| `pause_rate_per_min` | 278 | 0.715 | 0.165 – 1.735 |
| `longest_pause_s` | 278 | 3.54 | 2.09 – 5.53 |
| `repeat_event_frac` | 278 | 0.439 | 0.276 – 0.643 |
| `local_tempo_jitter` | 277 | 0.270 | 0.169 – 0.357 |

**Negative control, and it is a negative result: the 56 ASAP competition
performances — professional, linear, zero practice behavior — pass G-OOD-6 5 of
6 against its ≥4 bar**, with AUC 0.42–0.58 on five of the six statistics. The
cause is structural: "median inside the IQR" asks whether an arm is *typical*,
and an IQR is the middle 50% of a heterogeneous population, i.e. a wide target.
This is the failure mode G-OOD-3 pre-registered against ("an invented threshold
is the mechanism by which a null becomes a pass") surfacing in G-OOD-6 instead.

Owner-approved amendment (2026-08-06): the six statistics and the ≥4/6 bar
stand, plus a pre-registered clause — **the gate counts as evidence only if the
ASAP control FAILS the same test**. It passes, so G-OOD-6 is recorded
NON-DISCRIMINATING: its numbers are descriptive, and a pass must not be quoted
as evidence of representativeness. Re-run the control with
`behavior_stats.py --control data/evals/asap_audio_bundles`.

Wrong-note rate is absent and cannot be added — detecting a wrong note requires
the score, which is truth. G-OOD-6 bounds session shape, stopping, repetition
and steadiness only.

### Multi-channel take sync (`follower_eval/take_capture.py`)

Validated by **recovery against injected ground truth**, not against its own
output: synthetic channels are resampled onto a known clock and band-shaped and
gain-modulated so a correlator needing identical waveforms fails. Over six
conditions (offset −1.25…12 s, drift −150…500 ppm): offset error ≤ 0.13 ms
(one sample at 8 kHz, i.e. sample-limited), drift error ≤ 0.3 ppm.

**Two slates define the drift fit but cannot test it** — a line through two
points fits both exactly, so residual non-linearity is unidentifiable, and phone
AGC making drift non-constant is precisely the assumption two slates cannot
check. `sync_channel` therefore accepts optional **mid-take slates**, held out
of the fit, and reports their residual; `max_mid_residual_s` is `None` without
one, meaning UNTESTED rather than fine. **Record three claps per take.**

**`MIN_SLATE_CORR` alone is not a sufficient guard.** A channel that stops
before the tail clap mis-locks onto an unrelated transient at corr 0.512 —
above the 0.50 floor — giving an offset 47 s from the head offset (genuine
matches score 0.82–0.85). It is caught instead by a **physical** bound: the
implied 274,000 ppm drift is three orders of magnitude past any real crystal.
A physical bound needs no tuning and does not soften on real room audio, where a
correlation floor tuned on synthetic fixtures would not transfer.

A missing channel raises and never falls back to the reference channel — that
fallback would report a clean-channel number as a phone-channel one, which is
the exact quantity #148 measures. Same precedent as `asap_audio.py` refusing to
fall back to MIDI.

### G-OOD-0, the blocking gate (`calibration_recall.py`)

Reference-channel Transkun note recall against the known score, bar ≥ 0.95.
Scored **only** on takes the manifest declares `"behavior": "calibration"`; a
practice take contains deliberate wrong notes, and scoring the gate there would
fail it on the performance rather than on the channel.

**A miss has three possible causes and they do not share a remedy:** Transkun
failed to transcribe a played note (what the gate is for), the performer did not
play it, or the *matcher* failed to pair a correctly-transcribed note. The third
is the dangerous one — Phase 2's provenance-B truth is parangonar output, so
scoring this gate with parangonar alone would let a matcher weakness either
cancel the session for the wrong reason or pass the gate and then silently
degrade the truth it qualified. That is the soft form of #101's gate1 mistake.

Recall is therefore reported as **two arms**, and combining them with the single
0.95 bar — no second threshold is introduced — yields an actionable verdict:

| verdict | condition | what it means |
|---|---|---|
| `PASS` | parangonar arm ≥ bar | scored through the matcher Phase 2 will actually use |
| `FAIL_MATCHER` | parangonar < bar, timing-free ≥ bar | notes are in the transcription; the aligner is losing them. **Mic placement will not help** |
| `FAIL_CHANNEL` | both < bar | notes are absent from the transcription. This is the failure #148's remedy is written for; Phase 2 stops |
| `UNINFORMATIVE` | matcher floor < bar | the matcher scores the score against *itself* below the bar, so neither arm carries information. Same precedent as G-OOD-6 |

The **timing-free arm** is the longest common subsequence of the two pitch
streams. It never consults a clock, so tempo, rubato and drift cost it nothing.
It is not a proof-grade upper bound — parangonar can emit a non-monotone pair
LCS forbids — so it localizes a failure rather than replacing the other arm.
The **matcher floor** runs the score against itself through the identical
parangonar call, isolating matcher loss with no audio, piano or performer
involved; it is reported unconditionally.

**Chord ordering is load-bearing, and was measured, not assumed.** Transcribed
onsets carry ~20 ms of jitter, which desynchronises a chord's notes and makes a
plain `(onset, pitch)` sort return them in arrival order. Measured, that put the
timing-free arm at **0.75 on a perfect transcription** — permanently under the
bar, making `FAIL_MATCHER` unreachable and misattributing every failure to the
channel. Grouping notes within `CHORD_WINDOW_S = 0.05` and ordering by pitch
lifts it to 0.9833. On injected drop rates of 0/2/10/30 %, the parangonar arm
reads 1.0000 / 0.9833 / 0.9167 / 0.7000.

**`FAIL_MATCHER` has not been observed live.** It is pinned by unit test, but
parangonar held at 0.9969 even on deliberately repetitive material (a scale
exercise at 1.7× tempo), so no synthetic case triggered it. Treat it as a
diagnostic that fires if the matcher degrades on real room audio, not as a
branch validated end to end.

Recall here is a **lower bound** on Transkun's true note recall: a note the
performer did not play counts as a Transkun miss, and separating the two needs
the symbolic retrofit #148 deferred. The bias direction is safe — a `PASS` is
conservative, and only a `FAIL` is ambiguous, which is what the two arms exist
to disambiguate.

Audio goes in through the same ffmpeg path the 279-clip corpus used (16 kHz
mono); a different path would introduce a channel difference that is not the one
being measured. A transcriber returning zero notes **raises** rather than
reporting a recall of 0.0. No bootstrap interval: a calibration session is a
handful of takes, and a resampling CI over three of them would be decoration.

### Not built yet (blocking #148's recording session)

- **`take_capture` intake** — rename/convert/completeness-check raw exports.
- **`rig_hash` enforcement** — `ood_eval.paired_table` does not read it, so
  "a subtraction across two rig hashes fails loudly" is not yet true.
- **`align_truth`** — parangonar invocation, human-verification round-trip,
  G-OOD-1 A-vs-B bookkeeping.
