# Layer 1 Validation Experiments Design

**Date:** 2026-03-09
**Status:** Approved
**Context:** `docs/model/04-training-results.md` (Layered Roadmap, Layer 1)

## Goal

Validate four assumptions before investing in model improvements (Layer 2) or data collection (Layer 3):

1. Does the model's quality signal generalize beyond PercePiano?
2. Does symbolic input survive automatic music transcription?
3. Does the model have dynamic range at intermediate skill levels?
4. Does MIDI-as-context improve teacher feedback quality?

## Background: Per-Dimension Findings

Audio and symbolic encoders have complementary strengths (fold 3 evaluation):

| Dimension | A1 (Audio) | S2 (GNN) | Winner |
|---|---|---|---|
| dynamics | ~0.70 | ~0.77 | S2 |
| timing | ~0.77 | ~0.65 | A1 |
| pedaling | ~0.72 | ~0.72 | Tie |
| articulation | ~0.66 | ~0.70 | S2 |
| phrasing | ~0.63 | ~0.63 | Tie |
| interpretation | ~0.74 | ~0.77 | S2 |

All symbolic results use ground-truth MIDI. In production, symbolic input comes from AMT of phone audio. The symbolic advantage may not survive this bridge.

## Artifacts

| Artifact | Scope |
|---|---|
| `model/notebooks/model_improvement/04_layer1_validation.ipynb` | Experiments 1-3 (model evaluation) |
| `model/src/model_improvement/feedback_eval.py` | Experiment 4 (LLM feedback evaluation) |
| `model/src/model_improvement/midi_comparison.py` | Structured MIDI comparison (new) |

## Experiment 1: Competition Correlation

**Question:** Does A1's quality signal predict expert competition placement?

**Data:** Chopin 2021 International Piano Competition. Audio downloaded from the Chopin Institute's official YouTube channel via yt-dlp. Collection pipeline exists in `competition.py`.

**Pipeline:**

1. Run `competition.py` to download audio, segment into 30s chunks, extract MuQ embeddings (locally on M4)
2. Load A1 checkpoint (fold 3), score all segments across 6 dimensions
3. Aggregate per-performer using three methods: mean, median, and min scores across segments
4. Compute Spearman rho of each aggregation vs competition placement (per round and overall)
5. Compute per-dimension correlations to identify which dimensions carry the strongest quality signal

**Output:** Correlation table + scatter plots.

**Decision gate:**
- rho > 0.3 on at least one aggregation: model quality signal is real, proceed to Layer 2
- rho < 0.2: model may be fitting PercePiano artifacts, investigate before proceeding

## Experiment 2: AMT Degradation Test

**Question:** How much does S2's per-dimension pairwise accuracy drop when fed transcribed MIDI instead of ground-truth MIDI?

**Data:** 50 MAESTRO recordings from 10-15 pieces with multiple performers. Ground-truth MIDI already available locally (1,276 files). MuQ embeddings already on GDrive (24,321 segments). Audio to be downloaded for the selected subset (~5 GB).

**Transcription models:**
- **YourMT3+** (ceiling): MAESTRO onset F1 ~96.7%, ~370M params, PyTorch. GitHub: mimbres/YourMT3.
- **ByteDance Piano Transcription** (floor): High-resolution with pedal regression, lighter weight, PyTorch. GitHub: bytedance/piano_transcription.

**Pipeline:**

1. Select 50 recordings from contrastive_mapping.json (pieces with multiple performers)
2. Download corresponding MAESTRO audio files
3. Transcribe each with YourMT3+ and ByteDance
4. Build S2 score graphs from: (a) ground-truth MIDI, (b) YourMT3+ MIDI, (c) ByteDance MIDI
5. Run S2 on all three, compute pairwise accuracy per dimension
6. Diagnostic metrics: velocity MAE, note F1, pedal F1 between transcribed and ground-truth

**Performance on M4:** YourMT3+ transcription ~2-5x realtime on CPU. S2 inference is lightweight. Total ~4-8 hours for 50 recordings.

**Decision gates:**
- Per-dimension pairwise drop < 10%: symbolic path viable in production, include in fusion plans
- Per-dimension pairwise drop > 20%: symbolic path compromised, focus on audio-only + MIDI-as-context
- Velocity MAE specifically measures whether the dynamics advantage (S2's strongest dimension) survives

## Experiment 3: Dynamic Range at Intermediate Level

**Question:** Does A1 produce meaningful score variance on intermediate-level recordings, or does everything collapse to uniformly low scores?

**Data:** 10-20 YouTube recordings of intermediate pianists playing PercePiano/MAESTRO repertoire. Target:
- Schubert D960 student recital performances, diploma exam recordings
- Common MAESTRO repertoire (Chopin, Beethoven) from student/amateur channels

**Pipeline:**

1. Search YouTube for intermediate performances, download via yt-dlp
2. Segment into 30s chunks, extract MuQ embeddings locally
3. Run A1 on all chunks, record per-dimension scores
4. Compare distributions: intermediate vs PercePiano (advanced) vs MAESTRO (professional)

**Key measurements:**
- Score separation between intermediate and advanced
- Per-dimension variance within intermediate performances (does the model differentiate dimensions, or output uniformly low scores?)
- Within-session variance across chunks (does the model detect that some passages went better than others?)

**Performance on M4:** MuQ extraction ~1-2 min per 30s chunk. ~200 chunks total = ~3-4 hours.

**No hard gate.** This is diagnostic. If the model shows no dynamic range at intermediate level, it informs Layer 3 data collection priorities (need intermediate training data) but does not block Layer 2.

## Experiment 4: MIDI-as-Context Feedback Test

**Question:** Does structured MIDI comparison alongside A1 scores produce more specific, actionable teacher feedback than A1 scores alone?

**Data:** 20 PercePiano segments from Schubert D960 mvt 3 (37 available performances, Score reference MIDI exists). Select segments spanning the quality range: top 5, middle 10, bottom 5 by A1 overall score.

**MIDI comparison features** (new code in `midi_comparison.py`):
- Velocity curve comparison (performance vs score dynamic markings)
- Onset deviation profile (timing differences per note/beat)
- Note accuracy (missed/added notes, pitch errors)
- Pedal event comparison (timing and duration vs score)
- Phrasing structure (legato/staccato patterns vs score articulation)

**Pipeline:**

1. Run A1 on 20 selected segments, get per-dimension scores
2. Build structured MIDI comparison for each segment against Score MIDI
3. Generate teacher observations using the subagent prompt (`docs/apps/06a-subagent-architecture.md`):
   - **Condition A:** A1 dimension scores + student context only
   - **Condition B:** A1 dimension scores + student context + structured MIDI comparison
4. **LLM judge (20 pairs):** Present both observations in randomized order to Claude with rubric:
   - Specificity: does it reference particular passages/bars?
   - Actionability: does it tell the student what to do?
   - Accuracy: does it match what the MIDI analysis reveals?
5. **Manual review (10 pairs):** Read both observations blind, rate on same criteria. Calibrate LLM judge agreement.

**Cost:** ~40 LLM API calls (20 pairs x 2 conditions + 20 judge calls).

**Decision gates:**
- MIDI-context wins > 65% of pairs (LLM judge): build score comparison pipeline for teacher subagent
- MIDI-context wins < 55%: extra context doesn't justify the piece library requirement
- Manual review agrees with LLM judge > 70%: judge is trustworthy for future evals

## Dependencies

**Python packages (add to model/pyproject.toml):**
- `yourmt3` -- YourMT3+ transcription (Experiment 2)
- `piano-transcription-inference` -- ByteDance transcription (Experiment 2)
- `yt-dlp` -- audio download (Experiments 1, 3)

**Existing infrastructure reused:**
- `model/src/model_improvement/competition.py` -- competition data collection
- `model/src/model_improvement/evaluation.py` -- pairwise accuracy evaluation
- `model/src/model_improvement/graph.py` -- S2 score graph construction
- `model/src/audio_experiments/extractors/muq.py` -- MuQ embedding extraction
- A1 and S2 checkpoints in `data/checkpoints/model_improvement/`

**Data on GDrive:**
- `maestro_cache/muq_embeddings/` -- 24,321 pre-extracted embeddings
- `maestro_cache/contrastive_mapping.json` -- 204 pieces with multi-performer mapping

**Data locally:**
- `maestro_cache/` -- 1,276 ground-truth MIDI files
- `percepiano_cache/` -- embeddings, labels, folds, piece mapping

## Execution

All experiments run locally on M4 MacBook (32GB RAM). No GPU required. Estimated total compute: ~1 week (dominated by MuQ extraction and AMT transcription).

Experiments are independent and can run in any order. The notebook (Experiments 1-3) can execute cells independently. The feedback eval script (Experiment 4) runs standalone.

## Outputs

| Experiment | Output | Feeds Into |
|---|---|---|
| 1. Competition correlation | Spearman rho table (overall + per-dim + per-round) | Layer 2 go/no-go |
| 2. AMT degradation | Per-dimension pairwise table (GT vs YourMT3+ vs ByteDance) | Fusion vs MIDI-as-context decision |
| 3. Dynamic range | Box plots (intermediate vs advanced distributions, per dim) | Layer 3 data collection priorities |
| 4. Feedback quality | Win rate table + qualitative examples | Score comparison pipeline decision |

Results update `docs/model/04-training-results.md` Layer 1 section with actual numbers and gate decisions.
