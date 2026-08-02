# Track A (Task 1) — Music Performance Difficulty Prediction

**Status:** PIPELINE BUILT AND MEASURED; on the feature wall · **Issue:** [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) · **Updated:** 2026-08-01 · **Deadline:** 2026-10-01

> The spec below (task, datasets, I/O contract) is current. The **"Provisional approach" section is SUPERSEDED** — the shipped pipeline is not frozen MuQ + the A1-Max head. See "Measured state" for what actually runs and what it scores.

## Measured state (read this before proposing work)

**Shipped pipeline:** audio → **Transkun** transcription → **37 hand-crafted symbolic features** (`candidate_features`, onset/pitch/velocity only) → **LightGBM regression** head. Not MuQ, not the ordinal head.

**Measurement convention — composer-disjoint 5-fold CV Kendall tau-c.** A random split lets a model memorize "Czerny pieces are grade 4" and score well without learning difficulty, so folds are `GroupKFold` over composer (1,066 composers across 5,798 pieces). Absolute tau-c also moves with the grade mix and the subset, so **compare gaps under identical folds, never levels across harnesses**.

**⚠️ The 0.824 "deployed-on-transkun" figure is CONTAMINATED — do not use it as a target.** In `stage2_refit_curve.py` the "deployed" head is fit on *all* PSyllabus records and then scored on a subset of those same records, i.e. train-on-test. No honestly cross-validated number can beat it. The valid anchor is the 37-feature arm measured under the same folds as whatever it is being compared to.

**Honest anchor (2026-08-01, #137, 5,798 Transkun MIDIs):**

| Arm | mean-fold tau-c | pooled OOF | vs anchor |
|---|---|---|---|
| 37 base features, regression | **0.7934 ± 0.0165** | 0.7971 | — (anchor) |
| 37 base + 32 Transkun-unlocked | 0.7987 ± 0.0166 | 0.8028 | **+0.0058** CI [+0.0032, +0.0083] SIG |
| 32 Transkun-unlocked only | 0.7577 ± 0.0199 | 0.7619 | −0.0352 |
| 37 base, rank-transformed target | 0.7927 ± 0.0164 | 0.7971 | +0.0001 (tie) |

**The feature wall is real and the frontier is close to exhausted.** #137 tested the last untested feature premise — that every prior null was measured on offset-blind features — and the new offset/pedal family delivered a significant but negligible **+0.0058**. It takes only **8.6% of LightGBM gain** while `pitch_lz_complexity` alone takes **66.9%**. Hand-crafted symbolic features have hit their limit; the remaining unrefuted lever is an end-to-end encoder fine-tune (#138).

**Transcriber constraint worth remembering: Transkun's pedal head is BINARY.** It emits CC64 as only 0 and 127, alternating. Pedal *depth* features (depth mean, value entropy, half-pedal fraction) are constant across the entire corpus and are therefore not expressible from this transcriber — recovering them needs a different model, not a different feature. Pedal *timing* (change rate, on-fraction, segment length) does survive and is the strongest single contribution of the new family.

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

## Provisional approach — SUPERSEDED 2026-08-01 (kept for decision history)

> What follows was the 2026-07-03 plan. It was **not** what shipped: the pipeline went symbolic
> (Transkun → hand-crafted features → LightGBM), not frozen-MuQ → A1-Max head. The MuQ probe is
> one of the nulls recorded in the decision log. See "Measured state" above for current reality.

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
- **2026-07-22 (#125 Stage-1)** — Transkun adopted over aria-amt for the symbolic stream (offset F1 0.79 vs 0.37, MIT licence). Swap measured neutral on difficulty tau-c on its own; the adopt decision stood on transcription quality, not on a difficulty win.
- **2026-07-31 (#135)** — Full-scale Stage-2 re-fit gate came back **MARGINAL**: re-training the head on Transkun features rather than clean MIDI gave B−A = +0.016 at n=5,798. Diagnosis: the 37 features exclude offsets *by design*, so they are transcriber-robust and clean-vs-Transkun barely differs — the gate tested the wrong lever. Motivated the #137/#138 split.
- **2026-08-01 (#137) — FEATURE FRONTIER CLOSED.** Built 32 Transkun-unlocked features (articulation via duration/IOI ratio, duration entropy/LZ, true time-weighted voicing, chord-release dispersion, pedal timing) and a composer-disjoint CV tau-c ablation with fixed folds shared by all arms and a paired bootstrap over pieces. Result: **+0.0058 tau-c** (0.7934 → 0.7987), significant but negligible; the new family is ~90% redundant with the 37 and takes 8.6% of model gain. **This was the last untested feature premise** — every prior null was measured on offset-blind features, and removing that blindness did not move the wall. Two by-products: (a) established that the **0.824 anchor is train-on-test contaminated**; (b) established that **Transkun cannot express pedal depth** (binary CC64). Harness: `model/src/claim_measurement/difficulty/{transkun_features,tk_ablation}.py`, 26 unit tests.
  - *Not refuted, still open:* the lambdarank arm scored −0.139 but used **one query group per fold**, and NDCG is top-heavy while tau-c scores the whole list — a formulation bug, not a verdict on rank-native objectives. Multi-scale / per-section aggregation of the new features remains untried.
- **Standing implication** — with hand-crafted symbolic features exhausted, **#138 (end-to-end encoder fine-tune) is the remaining unrefuted lever** before the 2026-10-01 deadline.
