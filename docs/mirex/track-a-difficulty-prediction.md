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

## Decision log (append-only)
- **2026-07-03** — Track A spec captured from MIREX task page; assets mapped (strong fit). Deep research NOT yet done — next session should run the Research checklist before /brainstorm. Provisional lean: this is the higher-value track (real asset transfer vs Track B's no-moat).
