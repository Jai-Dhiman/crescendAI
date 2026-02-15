# Priority Signal Validation Experiment

## Goal

Validate whether masterclass teaching moments (STOP/CONTINUE) can be predicted from audio. Using 65 extracted moments from 5 Chopin masterclass videos, determine if there is a learnable signal that distinguishes passages where a teacher intervenes from passages where the teacher lets the student continue.

This is a Phase 1 validation experiment -- prove the concept works before investing in data scaling.

## Background

The masterclass pipeline (feat/transcribe-first-pipeline) extracts structured teaching moments from YouTube masterclass videos. Each moment records when a teacher stopped a student, what feedback they gave, and the estimated timestamps of student playing before the stop.

The existing PercePiano quality model predicts 19 performance dimensions from MuQ audio embeddings (R²=0.537). The hypothesis is that teacher stopping decisions may correlate with these quality dimensions, or with raw audio features directly.

## Data

### Source

- 65 STOP moments from `tools/masterclass-pipeline/all_moments.jsonl`
- 5 masterclass videos (WAV files already downloaded)
- Teachers: Arie Vardi, Krystian Zimerman, Murray Perahia, Benjamin Zander, Garrick Ohlsson
- Pieces: Chopin Ballades and Etudes

### Segment Extraction

**STOP segments:** Audio between `playing_before_start` and `playing_before_end` for each moment -- the passage the student was playing when the teacher decided to intervene.

**CONTINUE segments:** Audio gaps between consecutive moments in the same video where the student was playing but the teacher did not stop. Inferred from the space between one moment's `feedback_end` and the next moment's `playing_before_start`.

ASAP dataset integration deferred -- start with masterclass-only data to keep acoustic conditions consistent.

## Architecture

### Project Structure

```
model/
  src/masterclass_experiments/
    __init__.py
    data.py          -- moment parsing, segment extraction
    features.py      -- MuQ embeddings, PercePiano quality scores
    models.py        -- classifiers (Model A and Model B)
    evaluation.py    -- leave-one-video-out CV, metrics, qualitative analysis
  notebooks/masterclass_experiments/
    01_priority_signal_validation.ipynb  -- orchestrator
```

Notebook acts as orchestrator and visualization layer. All logic lives in importable modules under `src/masterclass_experiments/`.

### Module Details

#### data.py

- `load_moments(jsonl_path) -> list[Moment]`: Parse JSONL, extract STOP segment boundaries.
- `extract_segments(moments, wav_dir, output_dir) -> list[Segment]`: Slice WAV files into STOP and CONTINUE clips. Each segment saved as a separate WAV with metadata (video_id, label, timestamps).

#### features.py

Two extraction paths reusing existing infrastructure:

- **MuQ embeddings:** Reuse `MuQExtractor` from `audio_experiments/extractors/muq.py`. Stats pooling (mean + std) produces a 2048-dim vector per segment.
- **PercePiano quality scores:** Load trained PercePiano checkpoint, run inference on each segment to produce a 19-dim quality score vector.

Cache structure:

```
model/data/masterclass_cache/
  segments/          -- WAV clips
  muq_embeddings/    -- .pt files
  quality_scores/    -- .pt files (19-dim vectors)
```

#### models.py

**Model B (stacked, recommended starting point):** Logistic regression (sklearn) on 19 PercePiano quality scores. Tests whether teacher stopping decisions can be explained by the quality dimensions we already measure. Simple, interpretable, appropriate for 65 data points.

**Model A (direct):** Logistic regression on 2048-dim MuQ stats-pooled embeddings. If signal exists, optionally add a small MLP (2048 -> 128 -> 1). Tests whether raw audio features capture something beyond the 19 quality dimensions.

#### evaluation.py

- **Cross-validation:** Leave-one-video-out (5 folds, one per video). Avoids data leakage from shared acoustic conditions within a video.
- **Quantitative metrics:** AUC-ROC, accuracy, precision, recall.
- **Qualitative analysis:** For each held-out video, list top false positives (model said STOP, teacher didn't) and false negatives (teacher stopped, model said CONTINUE) with timestamps for manual listening.

### Notebook Flow

1. Setup -- imports, paths, config
2. Data prep -- load moments, extract segments, show counts and duration stats
3. Feature extraction -- MuQ embeddings + PercePiano inference, cache results
4. Model B -- train logistic regression on quality scores, evaluate, confusion matrix
5. Model A -- train on MuQ embeddings, evaluate, confusion matrix
6. Comparison -- side-by-side AUC/accuracy, qualitative inspection
7. Conclusions -- signal assessment, next steps

## Key Decisions

- **Simple models first.** With 65 positive examples, sklearn logistic regression before neural nets. Complexity only if signal warrants it.
- **Leave-one-video-out CV.** Prevents leakage from shared acoustic conditions, recording quality, and teacher style within a single video.
- **Masterclass-only negatives.** CONTINUE segments from the same videos as STOP segments, keeping acoustic conditions matched. ASAP augmentation deferred.
- **Reuse existing infrastructure.** MuQ extractor, PercePiano model, caching patterns all come from `audio_experiments/`.

## Success Criteria

- **Signal exists:** Either model achieves AUC > 0.6 (meaningfully above 0.5 chance)
- **Interpretable patterns:** Model B coefficients reveal which quality dimensions predict stops
- **Qualitative plausibility:** False positives/negatives make sense when listening to the audio

## What This Does NOT Cover

- Scaling to more videos (next step if signal exists)
- ASAP dataset integration
- Production model training
- Real-time inference
