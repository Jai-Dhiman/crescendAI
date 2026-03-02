# Slice 1: Phone Audio Validation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether MuQ produces meaningful quality assessments on phone-recorded piano audio from a real instrument.

**Architecture:** Record real piano performances on a phone under varying conditions. Run through the existing HF inference endpoint. Analyze whether scores are meaningful, differentiated, and stable.

**Tech Stack:** Existing MuQ HF inference endpoint, Python/Jupyter for analysis, phone for recording

---

## Context

All MuQ training data comes from synthetic Pianoteq audio. The practice companion product requires MuQ to work on phone recordings of real pianos. If scores degenerate on phone audio, the product direction needs to change. This is an existential gate.

## Experiment Design

### Recording Protocol

Record yourself playing on your real piano using your phone. Capture at least 15 recordings across these axes:

**Controlled variables (same piece, different conditions):**
1. Phone on music stand (~1 foot from piano)
2. Phone 3 feet from piano
3. Phone across the room (~10 feet)
4. Quiet room vs. some ambient noise (fan, window open)

**Controlled variables (same conditions, different playing):**
5. Same passage played with deliberately flat dynamics vs. exaggerated dynamics
6. Same passage played with clean pedaling vs. muddy pedaling
7. Same passage played with good timing vs. deliberately rushed
8. A piece you know well vs. one you're sight-reading (quality difference)

**Repertoire variety:**
9. Chopin (pedal-heavy, rubato-heavy)
10. Bach (articulation-heavy, voicing-heavy)
11. A loud dramatic passage vs. a quiet lyrical passage

### What to Measure Per Recording

Run each recording through the existing HF inference endpoint (`POST` with audio URL or direct upload). Collect:

- 6-dimension scores (dynamics, timing, pedaling, articulation, phrasing, interpretation)
- Processing time
- Any inference errors or failures

### Analysis (Python notebook)

**Test 1: Non-degenerate scores**
- Are scores distributed across a reasonable range? Or do they collapse to the same value for all phone recordings?
- Plot histograms per dimension across all recordings
- FAIL if variance per dimension < 0.01 (all scores same)

**Test 2: Differentiation**
- Does the flat-dynamics recording score lower on dynamics than the exaggerated-dynamics one?
- Does the muddy-pedal recording score lower on pedaling than the clean one?
- Does the rushed recording score lower on timing than the controlled one?
- Does the well-known piece score higher overall than the sight-reading?
- PASS if directionally correct on at least 4/5 paired comparisons

**Test 3: Distance robustness**
- Same passage recorded at 1ft, 3ft, 10ft. Do scores stay within a reasonable band?
- Compute max absolute difference per dimension across distances
- PASS if max difference < 0.15 per dimension (scores shift but don't wildly change)

**Test 4: Noise robustness**
- Same passage in quiet room vs. ambient noise. Do scores stay reasonable?
- PASS if max difference < 0.15 per dimension

**Test 5: Known weakness detection**
- You know your own playing. Do the scores reflect what you know about yourself?
- This is subjective but critical. Does the system surface things that feel right?

### Decision Matrix

| Result | Action |
|---|---|
| Tests 1-4 pass, Test 5 feels right | Proceed with confidence. Phone audio works. |
| Tests 1-2 pass, 3-4 marginal | Scores are meaningful but distance/noise sensitive. Add guidance: "Place phone on music stand." Proceed cautiously. |
| Test 2 fails (no differentiation) | MuQ embeddings collapse on phone audio. Investigate: run phone audio through MuQ embedding extraction, visualize t-SNE. Is the embedding space degenerate? |
| Test 1 fails (degenerate scores) | Inference pipeline problem. Check audio preprocessing (resampling, format). May need phone-audio-specific preprocessing. |
| Multiple failures | Existential risk confirmed. Options: (a) fine-tune MuQ with phone audio augmentation, (b) switch to a more robust feature extractor for phone audio, (c) require high-quality recording setup. |

### Deliverables

1. A set of 15+ phone recordings stored in `model/data/phone_validation/`
2. Inference results as a JSON/CSV file
3. Analysis notebook at `model/notebooks/model_improvement/03_phone_validation.ipynb`
4. A one-paragraph conclusion written at the top of the notebook: proceed / proceed with caveats / blocked

### Tasks

**Task 1: Record audio samples**
- Record 15+ samples per the protocol above
- Save as WAV or M4A files
- Name files descriptively: `chopin_nocturne_1ft_quiet.m4a`, `bach_invention_muddy_pedal.m4a`, etc.
- Store in `model/data/phone_validation/`

**Task 2: Run inference**
- Write a simple Python script or notebook cell that:
  1. Uploads each file to R2 (or a temporary public URL)
  2. Calls the HF inference endpoint
  3. Collects 6-dimension scores
  4. Saves results to `model/data/phone_validation/results.json`

**Task 3: Analyze results**
- Create `model/notebooks/model_improvement/03_phone_validation.ipynb`
- Run Tests 1-5 as described above
- Generate plots: dimension histograms, paired comparisons, distance/noise robustness
- Write conclusion at the top

**Task 4: Decide and document**
- Based on results, update the practice companion design doc with findings
- If blocked, document what needs to change before proceeding
