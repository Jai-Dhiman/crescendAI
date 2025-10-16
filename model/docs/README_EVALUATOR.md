# Evaluator Model: Data Labeling and Training Guide (PyTorch)

This guide covers only labeling and training for the evaluator model that predicts performance dimensions from short audio segments.

1) Scope and Target
- Goal: Train an evaluator that maps 10–20s piano segments to 12–16 continuous dimensions.
- Strategy: Label only the dimensions each dataset is strong for; don’t force all dims everywhere. Use an active-learning loop to reduce manual labeling.

2) Dimension Set (lean, high-signal)
- Execution: timing_stability, tempo_control, rhythmic_accuracy, articulation_length, articulation_hardness, pedal_density, pedal_clarity, dynamic_range, dynamic_control, balance_melody_vs_accomp
- Shaping: phrasing_continuity, expressiveness_intensity, energy_level
- Timbre: timbre_brightness, timbre_richness, timbre_color_variety

Dataset targeting
- MAESTRO/ASAP/MAPS: execution + phrasing (+ some expressiveness/energy)
- CCMusic/MusicNet/YouTube curated: timbre + interpretation anchors

3) Environment (Python via uv + PyTorch)
- Use uv for environments and dependencies.
- Core deps: torch, torchaudio, pytorch-lightning, hydra-core/omegaconf, numpy, pandas, librosa, soundfile, scikit-learn, einops, tqdm, wandb (optional for tracking).

4) Repo Layout (labeling + training only)
- data/
  - manifests/: JSONL manifests per dataset and split
  - anchors/: anchors.json with low/mid/high exemplars per dimension
  - splits/: consolidated train/valid/test JSONL
- labeling/
  - quick_labeler.py (Streamlit/Gradio)
- scorer/
  - dataset.py (mel extraction, labels/masks)
  - model.py (backbone + heads + masked loss)
  - train.py (Lightning training loop)
  - infer.py (MC-dropout uncertainty for active learning)
  - calibrate.py (optional temperature/affine calibration)
- configs/ (Hydra YAMLs for model, data, training)

5) Segmenting and Manifests
- Segment length: 10–20s; align to 8–16 bars where possible (MAESTRO/ASAP/MAPS). For CCMusic/MusicNet/YouTube, use stable timbre sections or phrase windows.
- Do not redistribute audio; keep only URIs/IDs and time spans.

Manifest schema (JSONL, one row per segment)
{
  "segment_id": "MAESTRO_2018_1234_m3_bars_17_24",
  "dataset": "MAESTRO",
  "audio_uri": "file:///abs/path/to/clip.wav",  // or s3://...
  "sr": 22050,
  "t0": 12.34,
  "t1": 24.56,
  "bars": [17, 24],
  "dims": ["timing_stability", "tempo_control", ...],
  "labels": {"timing_stability": 0.62, "pedal_clarity": 0.41},
  "label_mask": {"timing_stability": 1, "tempo_control": 0, ...},
  "rater_id": "expert_01",
  "source": "human",  // or "model" / "model_verified"
  "provenance": {"piece": "Beethoven WoO 80", "license": "CC BY-NC-SA"},
  "anchors": ["timing_stability_low_v1", "timing_stability_high_v1"]
}

6) Rubric and Anchors
- For each dimension, define 3 anchors: low, mid, high (10–20s clips) with 1–2 sentence descriptions.
- Example anchors.json entry:
{
  "timing_stability": {
    "low":  {"clip_uri":"s3://.../ts_low.wav",  "desc":"Noticeable pulse jitter at the beat level"},
    "mid":  {"clip_uri":"s3://.../ts_mid.wav",  "desc":"Mostly steady with minor fluctuations"},
    "high": {"clip_uri":"s3://.../ts_high.wav", "desc":"Consistently steady pulse across bars"}
  }
}

7) Quick Labeling Workflow
- Streamlit/Gradio UI that:
  - Loads a manifest row, plays segment between t0–t1
  - Shows sliders only for dims in dims list
  - Shows 3 anchors per dimension
  - Writes labels + label_mask back to a labeled JSONL (or DB)
- Start solo; later add multiple raters for reliability.

8) Data Loader and Mel Extraction (PyTorch)
- torchaudio load → resample to 22050 Hz → 128-mel power → AmplitudeToDB (clip [-80,0]) → map to [0,1]
- Fix time axis to 128 frames via center-crop/pad (consistent input to model)
- Yield: mel [B, 1, 128, 128], y [B, D], mask [B, D], ds_id [B]

9) Model and Loss
- Backbone: small CNN front-end projecting to 256D embedding (AST-like is fine later)
- Head: concatenate dataset embedding; 2-layer MLP to D outputs
- Loss: masked MAE across labeled dimensions only

10) Training Loop (Lightning)
- Optimizer: AdamW (lr≈3e-4), batch_size≈64, epochs≈30
- Logs: train/val masked MAE; optionally per-dimension/ per-dataset metrics
- Checkpoints: save best val loss

11) Calibration (recommended)
- Fit per-dimension temperature/affine calibration on validation predictions to align to rater scales
- Report MAE before/after calibration; optionally ECE bins

12) Active Learning (model-assisted labeling)
- Uncertainty: MC dropout (enable dropout at eval; K≈8 passes; variance per dimension)
- Selection: choose top-uncertain segments per target dimension, balanced across datasets
- UI pre-fill: load model predictions as slider defaults; human verifies/adjusts → faster labeling
- Iterate: label → retrain → calibrate → resample

13) Evaluation
- Per-dimension MAE + Spearman per dataset (MAESTRO/ASAP/MAPS/CCMusic/MusicNet/YouTube curated)
- Calibration quality (MAE delta); inter-rater vs model-to-human if multiple raters

14) Licensing Hygiene
- Maintain a license matrix; treat MAESTRO/ASAP/MusicNet as non-commercial unless rights secured
- Distribute labels as segment IDs + numeric values only; never redistribute audio
- For YouTube-curated anchors/segments, get creator permission or keep internal

15) Suggested Initial Timeline
- Week 1: Create manifests + anchors; implement quick labeler; label 1–2k mixed segments
- Week 2: Train v0 (30 epochs); calibrate; evaluate by dataset/dimension
- Week 3: Active learning round 1 (1–2k); model-assisted labeling; retrain v1
- Week 4: Second AL round or freeze v1; export scorer for downstream tutor

Appendix: Implementation Pointers
- Keep dimension values in [0,1] with clear anchors to stabilize training
- Be conservative with audio augs for execution dims; avoid heavy time-stretch for timing labels
- Seed RNG; log with W&B or MLflow; use Hydra configs for reproducibility

16) Integrating VirtuosoNet and MidiBERT-Piano

Overview
- VirtuosoNet: use its alignment pipeline (ScoreXML ↔ Score MIDI ↔ Performance MIDI) to produce bar/beat-aligned segments and expressive features.
- MidiBERT-Piano: use a pretrained symbolic encoder to provide structural/phrase embeddings for distillation (optional, training-time only).

16.1 Alignment and Segmentation (VirtuosoNet)
- Run VirtuosoNet's pyScoreParser alignment for MAESTRO/ASAP/MAPS to obtain score↔performance alignment.
- Export 8–16 bar windows (≈10–20s) as segments; populate manifest fields: bars, t0, t1.
- Keep alignment provenance so you can reference bar numbers in tutor feedback.

16.2 Weak/Pseudo Labels from Symbolic Proxies
- Derive execution proxies per segment from aligned performance MIDI:
  - timing_stability/rhythmic_accuracy: onset deviation stats (mean/var), tempo jitter vs. score grid.
  - articulation_length/hardness: realized duration vs. notated; onset velocity patterns.
  - pedal_density/clarity: CC64 usage proportion, release timing vs. note overlaps.
  - dynamic_range/control: velocity min–max and smoothness across beats.
  - balance_melody_vs_accomp: voice separation (melody track vs. accompaniment energy).
- Map each metric to [0,1] using robust min–max with clipping (e.g., 5th–95th percentile) and store as pseudo labels.
- Calibrate pseudo→human on a small human-labeled calibration set (per dimension affine or isotonic). Store confidence/weight.

Manifest extensions (example)
{
  "pseudo_labels": {"timing_stability": 0.58, "articulation_length": 0.31},
  "pseudo_conf":   {"timing_stability": 0.6,  "articulation_length": 0.5},
  "symbolic_feats_uri": "s3://.../features/seg123.npz",
  "alignment": {"beats": [ ... ], "bars": [17,24]}
}

16.3 Multi-Modal Training and Losses
- Masked multi-task loss (as before) for human labels.
- Pseudo-label loss: include pseudo_labels with reduced weight; per-dimension weights from pseudo_conf or a global α (e.g., 0.3–0.5) increased after calibration improves.
- Distillation (optional):
  - Compute MidiBERT-Piano embedding e_sym for the same symbolic segment.
  - Add a projection head on the audio embedding e_aud and minimize L_distill = 1 - cos_sim(Proj(e_aud), StopGrad(e_sym)).
  - Use only during training; no symbolic path at inference.

Overall loss (per batch)
L = L_human(masked_mae) + α · L_pseudo(masked_mae on pseudo dims) + β · L_distill(cosine)
Typical starting weights: α=0.3, β=0.1

16.4 Active Learning with Symbolic Guidance
- Uncertainty: MC dropout variance per dimension (as before).
- Disagreement: |pred_audio - calibrated_pseudo| per dimension; high disagreement suggests informative samples.
- Selection: pick top-K by a mixture of uncertainty and disagreement, balanced across datasets and dimensions.

16.5 Where Not to Use Proxies
- Timbre dims (timbre_brightness/richness/color_variety) require audio labels; symbolic has no acoustics.
- Pedal "clarity" in acoustic sense still benefits from audio labels beyond CC64 proxies (resonance/room effects).

16.6 Practical Steps to Wire This Up
- Add a symbolic processing script to compute per-segment features and pseudo labels, plus a small calibration fitter.
- Extend the dataloader to read pseudo_labels and weights (optional) and expose them in the batch.
- Add loss weighting in the Lightning module; gate losses by availability.
- For distillation, cache MidiBERT-Piano embeddings for segments (symbolic-only) and load them at train time.
- Log per-dimension metrics separately for human vs pseudo supervision to track calibration quality.

16.7 Timeline Addendum
- Before Week 1 labeling: run alignment (VirtuosoNet) and generate pseudo labels for execution dims.
- Week 2 training: include α-weighted pseudo loss; if using distillation, add β-weighted cosine loss.
- Week 3 AL: incorporate disagreement with proxies when selecting segments to label.
