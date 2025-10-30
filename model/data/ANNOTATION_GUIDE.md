# MAESTRO Annotation Guide

## Overview

You'll be annotating **50 diverse MAESTRO pieces** across **12 dimensions** with a **dual-annotation strategy** (annotate twice for consistency).

## Quick Start (Today - 1 hour)

### Step 1: Install Label Studio (5 min)

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start

# Open browser: http://localhost:8080
# Create account (local only, no signup needed)
```

### Step 2: Create Project (5 min)

1. Click "Create Project"
2. Project Name: "MAESTRO Piano Performance Annotation"
3. Go to "Labeling Setup" tab
4. Click "Code" view
5. Copy-paste contents of `label_studio_config.xml`
6. Click "Save"

### Step 3: Prepare Audio Segments (20 min)

You need to:

1. Download selected MAESTRO pieces (from `data/maestro_selected.csv`)
2. Segment into 20-30 second clips
3. Import into Label Studio

**Option A: Manual segmentation (quick start)**

```bash
# Use ffmpeg to extract 20-30 second segments
# Example: Extract 3 segments from each piece

# Segment 1: 0-25 seconds
ffmpeg -i input.wav -ss 0 -t 25 output_segment1.wav

# Segment 2: middle (calculate based on duration)
ffmpeg -i input.wav -ss 120 -t 25 output_segment2.wav

# Segment 3: climax/ending
ffmpeg -i input.wav -ss 240 -t 25 output_segment3.wav
```

**Option B: Automated segmentation script (I can create this)**

### Step 4: Import Data (5 min)

1. In Label Studio, go to your project
2. Click "Import"
3. Upload audio files OR provide file paths
4. Label Studio will create tasks automatically

### Step 5: Start Annotating! (25 min)

1. Open first task
2. Listen 2-3 times
3. Rate all 12 dimensions (use hotkeys 1-9)
4. Add optional notes
5. Submit
6. **Target: 5-10 segments in first session**

---

## Annotation Protocol

### Time Investment

**Per segment:**

- Listen 2-3 times: 2-3 minutes
- Rate 12 dimensions: 3-4 minutes
- Add notes (optional): 1 minute
- **Total: 6-8 minutes per segment**

**Weekly goal:**

- 10 segments/day × 5 days = 50 segments/week
- Time: 1 hour/day

### Dual-Annotation Strategy

**Round 1 (Week 1):**

- Annotate all 50 segments
- Refine your rubric as you go
- Take notes on edge cases

**Cooldown (Week 2):**

- Wait 7-14 days (prevents memory bias)

**Round 2 (Week 3):**

- Re-annotate same segments WITHOUT looking at Round 1
- Faster (6-8 min → 5-6 min per segment)

**Consistency Analysis:**

- Compare Round 1 vs Round 2
- Calculate MAE per dimension
- Target: MAE < 10 points

---

## Rating Scale Interpretation

### 0-100 Scale Anchors

**90-100: Excellent**

- Professional/virtuoso level
- Few if any flaws
- Musically compelling

**75-89: Very Good**

- Advanced level
- Minor imperfections
- Generally convincing

**60-74: Good**

- Intermediate level
- Some noticeable issues
- Recognizable and competent

**40-59: Fair**

- Early intermediate
- Multiple issues
- Needs improvement

**20-39: Poor**

- Beginner level
- Significant problems
- Struggles with basics

**0-19: Very Poor**

- Cannot execute properly
- Unplayable or unmusical

### Per-Dimension Guidelines

#### 1. Note Accuracy

- **High (85-100)**: Virtually all notes correct
- **Medium (60-84)**: Mostly correct, few wrong notes
- **Low (0-59)**: Frequent pitch errors

#### 2. Rhythmic Precision

- **High (85-100)**: Steady tempo, accurate rhythms
- **Medium (60-84)**: Generally steady with occasional lapses
- **Low (0-59)**: Unsteady tempo, frequent timing errors

#### 3. Dynamics Control

- **High (85-100)**: Wide range, nuanced control
- **Medium (60-84)**: Some variety, adequate contrast
- **Low (0-59)**: Flat/monotonous, poor control

#### 4. Articulation Quality

- **High (85-100)**: Appropriate legato/staccato, clear
- **Medium (60-84)**: Generally appropriate, some blur
- **Low (0-59)**: Muddy or choppy, inappropriate

#### 5. Pedaling Technique

- **High (85-100)**: Clean, enhances music
- **Medium (60-84)**: Adequate, some blur
- **Low (0-59)**: Excessive or insufficient, muddy

#### 6. Tone Quality

- **High (85-100)**: Beautiful, controlled, varied color
- **Medium (60-84)**: Pleasant, adequate control
- **Low (0-59)**: Harsh, uncontrolled, poor color

#### 7. Phrasing

- **High (85-100)**: Natural, convincing musical sentences
- **Medium (60-84)**: Adequate shape, some awkwardness
- **Low (0-59)**: Choppy, unclear direction

#### 8. Expressiveness

- **High (85-100)**: Emotionally engaging, communicative
- **Medium (60-84)**: Some emotion, adequate expression
- **Low (0-59)**: Flat, unemotional, mechanical

#### 9. Musical Interpretation

- **High (85-100)**: Thoughtful, coherent artistic choices
- **Medium (60-84)**: Competent, safe choices
- **Low (0-59)**: Unclear or inappropriate choices

#### 10. Stylistic Appropriateness

- **High (85-100)**: Idiomatic, period-appropriate
- **Medium (60-84)**: Generally appropriate
- **Low (0-59)**: Anachronistic or inappropriate

#### 11. Overall Musicality

- **High (85-100)**: Compelling musical performance
- **Medium (60-84)**: Pleasant, adequate musicality
- **Low (0-59)**: Unmusical, mechanical

#### 12. Overall Technical Quality

- **High (85-100)**: Clean, confident execution
- **Medium (60-84)**: Competent with some issues
- **Low (0-59)**: Many technical problems

---

## Tips for Consistency

### Before Each Session

- Listen to 2-3 "anchor" segments you've already rated
- Recalibrate your sense of the scale

### During Rating

- Use the full 0-100 range (don't cluster around 50-70)
- Be consistent with your internal anchors
- When uncertain, mark in notes

### Quality Checks

- Every 10 segments, review your last 3 ratings
- Check: Am I being too harsh/generous?
- Adjust if drifting

---

## Segment Selection Strategy

From each of the 50 pieces, extract **3-5 segments**:

1. **Beginning (0-25s)**: Exposition, often easier
2. **Middle (varies)**: Development, moderate difficulty
3. **Climax**: Most technically demanding passage
4. **Quiet passage**: Test dynamics, tone quality
5. **Ending**: Resolution, often slower

**Total: 150-250 segments** (3-5 per piece × 50 pieces)

---

## Next Steps After Annotation

### Week 1-3: Annotation Complete

- 50 pieces × 3-5 segments = 150-250 segments
- Dual-annotated (Round 1 + Round 2)

### Week 4: Consistency Analysis

```bash
# I'll create a script to analyze Round 1 vs Round 2
python scripts/analyze_annotation_consistency.py \
  --round1 data/annotations/round1.json \
  --round2 data/annotations/round2.json
```

### Week 5: Dataset Preparation

```bash
# Convert Label Studio export to training format
python scripts/prepare_expert_annotations.py \
  --input data/label_studio_export.json \
  --output data/annotations/
```

### Week 6: Training

- Train model on averaged annotations (Round 1 + Round 2) / 2
- Evaluate per-dimension performance
- Compare to your annotation consistency

---

## Troubleshooting

### "This is taking too long!"

- Start with fewer dimensions (just 6 technical)
- Increase to 12 once comfortable
- Use keyboard shortcuts (1-9 hotkeys)

### "I'm unsure about a dimension"

- Mark uncertainty in notes
- Come back later with fresh ears
- Consistency matters more than perfection

### "Segments sound similar"

- MAESTRO is competition-level (all ~advanced)
- Look for subtle differences
- Use full scale range (even 80 vs 85 matters)

---

## Files Created for You

1. `data/maestro_selected.csv` - 50 selected pieces
2. `label_studio_config.xml` - Label Studio template
3. `scripts/analyze_maestro_diversity.py` - Diversity analysis
4. `scripts/calculate_note_accuracy.py` - Note accuracy from MIDI (not needed for MAESTRO)

## What I Can Help With Next

1. **Create automated segmentation script** (extract 3-5 clips per piece)
2. **Build consistency analysis script** (Round 1 vs Round 2 comparison)
3. **Create annotation export/conversion script** (Label Studio → training format)
4. **Build Dataset.py** (while you annotate)

Let me know what you'd like me to build next!
