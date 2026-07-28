# Track A (Task 1) — Music Performance Difficulty Prediction

**Status:** SPEC CAPTURED, research PENDING · **Issue:** [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) · **Updated:** 2026-07-03 · **Wheelhouse fit: HIGH**

> ⚠️ This doc has the task spec + our-asset mapping, but has NOT had the deep source-pinned research that Track B got. The "Research checklist" below is the work to do (mirror the Track B method). Treat numbers here as first-pass, not verified.

## Why this is the stronger track (thesis-aligned)
The task is **difficulty scoring of solo-piano audio** — the same modality, encoder, and ordinal-ranking machinery CrescendAI already runs. Unlike Track B, our assets map directly:
- **MuQ (frozen, on piano)** — our production audio encoder is *already* solo-piano-oriented; here that's an advantage, not a liability.
- **Ordinal ranking head** — our deployed A1-Max head already trains BCE-pairwise + **ListMLE** ranking + CCC regression on frozen MuQ embeddings (`model/src/model_improvement/audio_encoders.py`). The task's metric is **Kendall Tau-c** (ordinal agreement) — a near-exact match to what we already optimize.
- **PercePiano + T2/T5 ordinal data** — we already curate piano ordinal/skill data.
- **Piece-ID / score library (#21, #96)** — score-side difficulty signal if a score/audio hybrid helps.

## Task spec (from the MIREX task page)
- **Goal:** predict a **single real-valued difficulty score per solo-piano recording** (ordinal pedagogical scale, hidden granularity). Each test piece has ≥2 recordings (one human performance + one synthesized rendering), scored independently, aggregated per piece.
- **Input:** **WAV, 44.1 kHz, mono or stereo** (path to a WAV file).
- **Output:** a real-valued difficulty score per recording (treated as an ordering).
- **Metric:** **Kendall's Tau-c** (official ranking). Dev-time supplementary: MSE, Accuracy±1, balanced accuracy, Spearman ρ.
- **Submission:** **Docker container** with standardized inference interface; input = WAV path, output = single score. Must finish the full test set in **24h on one GPU**.
- **Rules:** train on any data (public/proprietary/internal); **no held-out eval repertoire** used directly/indirectly for training.
- **Timeline:** opens Jul 1, closes **Oct 1**, results Oct 15, 2026.
- **Captains:** Pedro Ramoneda (Songscription) pedro@songscription.ai · Huan Zhang (Clefer) huan@clefer.com.

## Named datasets
| Dataset | What | Size / labels | Link |
|---|---|---|---|
| **PSyllabus** (Ramoneda 2025) | audio recordings, classical piano | 7,901 recordings, 11-level pedagogical grades, 13 grading systems | zenodo.org/records/14794592 |
| **CIPI** (Ramoneda 2024) | MusicXML piano pieces | 652 scores, 9-level Henle | zenodo.org/records/8037327 |
| **Mikrokosmos-difficulty** (Ramoneda 2022) | piano difficulty | (Bartók Mikrokosmos) | github.com/PRamoneda/Mikrokosmos-difficulty |

Note: **PSyllabus is audio** (matches the task's audio input) — the primary training set. CIPI is score-based (MusicXML) — useful only if we go audio+score hybrid or transcribe.

## Provisional approach (to pressure-test in /brainstorm)
- **Spine:** frozen MuQ (piano) embeddings → our existing ordinal-ranking head, retargeted to a single difficulty scalar, trained on **PSyllabus audio**, optimized for Kendall Tau-c ordering. This is almost entirely reuse of the A1-Max stack.
- **Baseline to beat:** Ramoneda et al.'s own published difficulty models (audio + symbolic). *Research needed — find their reported Tau/accuracy.*
- **Edge hypothesis:** our piano-specialized frozen encoder + ordinal head may be genuinely competitive here (unlike Track B). This is the transferable-asset story worth testing.

## Research checklist (DO THIS — mirror Track B's method)
- [ ] Fetch/read the PSyllabus, CIPI, Mikrokosmos papers (Ramoneda et al.) — exact schemas, label semantics, licenses, train/test protocol.
- [ ] Find the **published SOTA** difficulty-prediction numbers (Tau-c / Acc±1) and architectures — what's the bar?
- [ ] Confirm the reference Docker template / inference wrapper once released; nail the exact I/O contract.
- [ ] Determine whether audio-only (PSyllabus) or audio+score hybrid is stronger; does transcription help or hurt?
- [ ] Map our A1-Max head → single-scalar difficulty regression/ranking; what changes?
- [ ] Licenses of PSyllabus/CIPI (commercial-use fork question, as with Track B).
- [ ] The "human vs synthesized rendering" aggregation — does it bias toward performance-quality vs score-difficulty? (This is subtle: the task says *difficulty*, but a human *performance* encodes execution quality too.)
- [ ] Open questions for the captains (Ramoneda/Zhang).

## Must-confirm / open questions
1. Reference Docker template details (not yet released as of research date).
2. Whether difficulty is meant as **score difficulty** (composition) or **performance difficulty** (execution) — the two-recordings-per-piece design suggests they want a piece-level score robust to performer, i.e. score difficulty. Confirm.
3. PSyllabus/CIPI licenses.

## Stage 2 — Transkun difficulty-head re-fit (#135, in progress)

**Why:** aria-amt was replaced repo-wide by Transkun (#128). Swapping the transcriber *alone* is tau-c-neutral because the deployed LightGBM head was trained on **clean** score note-counts and overfit the transcriber's biases. The fix is to **re-fit the head on Transkun-transcribed features**. The Stage-2 learning-curve gate confirmed this is worth it:

| eval N | arm B (transkun-trained) | deployed 7.9k-clean on transkun | B − A (matched-N) | verdict |
|---|---|---|---|---|
| 39 | — | — | +0.035 CI[−0.058,+0.129] | underpowered (looked dead) |
| 161 | — | 0.812 | +0.043 CI[+0.007,+0.081] | marginally significant |
| **604** | **0.807** | **0.804** | **+0.050 CI[+0.029,+0.071]** | **STRONG green** |

At n=604 a head trained on just 604 Transkun pieces already matches the deployed head trained on all 7,899 clean pieces (reading Transkun), and is still climbing → the full re-fit should beat deployed by ~**+0.04–0.05 tau-c** (full task).

**Harness:** `model/src/claim_measurement/transcription_bench/stage2_refit_curve.py` (stages: `prep` | `transcribe` | `pipeline` | `curve`). Data lands in `model/data/results/amt_gap_curve/` (gitignored). Imports the #104 difficulty feature code by absolute path (the `.worktrees/issue-104-mirex-difficulty` worktree must exist).

### Runbook (full 7.9k transcription)

Run the pipeline (per piece: download → Transkun → drop wav; disk-safe, resumable, skips pieces that already have a MIDI). It auto-resumes on any non-zero exit and `caffeinate` blocks system sleep — **lid open, on AC**:

```
caffeinate -is bash -c 'until uv run --script /Users/jdhiman/Documents/crescendai/.worktrees/issue-135-transkun-refit/model/src/claim_measurement/transcription_bench/stage2_refit_curve.py --stage pipeline --all --workers 4; do echo "[$(date)] stopped; resuming in 15s"; sleep 15; done'
```

Watch progress in a second terminal (climbs toward ~7,100–7,300; ~8.5% of pieces are unavailable on YouTube):

```
watch -n 30 'ls /Users/jdhiman/Documents/crescendai/model/data/results/amt_gap_curve/transkun_mid/*.mid | wc -l; df -h /Users/jdhiman/Documents/crescendai | tail -1'
```

When the count plateaus, re-run the gate curve at full N:

```
uv run --script /Users/jdhiman/Documents/crescendai/.worktrees/issue-135-transkun-refit/model/src/claim_measurement/transcription_bench/stage2_refit_curve.py --stage curve
```

**Still to build** (after the data lands): the retrain+eval stage — extract the 37 difficulty features from all Transkun MIDIs, retrain the LightGBM head (`DEPLOYED_PARAMS`) on Transkun features, CV-eval vs the clean-trained deployed head; if it wins, that head is the MIREX Track A submission.

## Decision log (append-only)
- **2026-07-03** — Track A spec captured from MIREX task page; assets mapped (strong fit). Deep research NOT yet done — next session should run the Research checklist before /brainstorm. Provisional lean: this is the higher-value track (real asset transfer vs Track B's no-moat).
- **2026-07-27** — Stage-2 Transkun re-fit gate = STRONG green at n=604 (#135); full 7.9k transcription launched (user-driven, see Runbook). Earlier #104 "closed research line / feature wall ~0.76" conclusions predate the head-re-fit lever, which the n=39 probe was too underpowered to detect.
