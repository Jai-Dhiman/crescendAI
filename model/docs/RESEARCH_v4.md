# Designing an Audio-Based PercePiano System for Piano Performance Evaluation

The fusion of audio modality with the existing MIDI-based PercePiano framework offers significant potential for improving piano performance evaluation, particularly for timbral and emotional dimensions where the current system achieves only **R² = 0.397** (piece-split). This report synthesizes research on the PercePiano methodology, audio encoders, and multimodal fusion strategies to provide a complete design blueprint for an enhanced audio-based system.

---

## PercePiano methodology reveals both strengths and critical limitations

The PercePiano dataset (Park et al., Nature Scientific Reports 2024) represents the first comprehensive expert-annotated collection for multi-level perceptual piano evaluation. **53 expert annotators** (graduate-level pianists and music theory specialists) evaluated **1,202 eight-bar segments** on 19 perceptual dimensions using a 7-point Likert scale, yielding **12,652 total annotations** with an average of 10.52 annotations per segment.

The inter-rater reliability reveals a crucial insight: individual ICC(1,1) values are uniformly "poor" (0.16–0.42), yet averaged ICC(1,k) scores are "excellent" (0.92–0.98). This confirms that while perceptual evaluation is inherently subjective at the individual level, **collective expert judgment converges toward meaningful perceptual ground truth**. The best agreement appears on pedal features (ICC = 0.42 for pedal saturation, 0.33 for pedal clarity), while dynamic range shows the weakest reliability (ICC = 0.16).

The current best-performing model—**Bi-LSTM with Score Alignment and Hierarchical Attention Network**—extracts features via VirtuosoNet's pyScoreParser, encoding aligned score-performance pairs through a three-level hierarchy (note → beat → measure). Key hyperparameters include hidden sizes from {64, 128, 256}, 1–2 layers, batch size 8, and learning rates of 5e-5 or 2.5e-5. The score alignment contributes **+11.9% absolute R² improvement**, while hierarchical attention adds another **+9.3%**, for a total of 21.2% improvement over baseline Bi-LSTM.

However, MIDI-only processing fundamentally limits evaluation of timbral dimensions. The original annotations were collected using **Logic Pro with a Yamaha Grand Piano VST**—audio stimuli that the model never sees. This modality gap explains why dimensions like timbre_variety, timbre_depth, and timbre_brightness likely underperform despite their high ICC values.

---

## The 19 perceptual dimensions span four hierarchical levels

Understanding the theoretical structure of PercePiano's dimensions is essential for modality mapping:

| Level | Category | Dimensions | Perceptual Scope |
|-------|----------|------------|------------------|
| **Low-level** | Timing, Articulation | timing (stable/unstable), articulation_length, articulation_touch | Detectable with a few notes |
| **Mid-low** | Pedal, Timbre | pedal_amount, pedal_clarity, timbre_variety, timbre_depth, timbre_brightness | Judged in 2–4 bars |
| **Mid-high** | Dynamic, Music Making | dynamic_loudness, dynamic_sophistication, dynamic_range, tempo, space, balance, drama | Requires phrase context |
| **High-level** | Emotion, Interpretation | mood_valence, mood_energy, mood_imagination, interpretation | Needs full performance context |

The dimensions derive from piano performance criticism (Gerig, Schonberg), historical pedagogy (Czerny), and MIR literature (Friberg et al.). Selection criteria required dimensions to be widely used, balanced in connotation, and requiring human perception—not simple computation.

---

## MERT-330M emerges as the optimal audio encoder for this task

A systematic comparison of music audio encoders reveals **MERT-330M** as the strongest candidate for piano performance evaluation:

| Model | Parameters | Music Pretraining | Layer Interpretability | Practical Viability |
|-------|-----------|-------------------|----------------------|-------------------|
| **MERT-330M** | 330M | 160K hours, CQT teacher | Excellent | ✓ Fast inference, ~8GB VRAM |
| MERT-95M | 95M | 1K hours | Good | ✓ Lightweight option |
| Jukebox-5B | 5B | Music generation | Good | ✗ Hours per minute, impractical |
| Wav2Vec2 | 95–317M | Speech (Librispeech) | Moderate | ✓ Fast, but not music-optimized |
| HuBERT | 95–317M | Speech | Moderate | ✓ Better than Wav2Vec2 for music |
| CLAP | ~200M | Audio-text pairs | Limited | ✓ Good for semantic matching |
| MusicGen | 300M–3.3B | Generation only | N/A | ✗ Not designed for understanding |

MERT's dual-teacher architecture—combining **RVQ-VAE (EnCodec) for acoustic features** and **Constant-Q Transform for pitch/harmonic structure**—directly addresses the timbral and harmonic dimensions that MIDI lacks. On the MARBLE benchmark, MERT-330M achieves **91.3% on music tagging**, **87.9 F1 on beat tracking**, and **94.4% on pitch detection**—all relevant to piano performance assessment.

Layer-wise analysis suggests **layers 18–24 capture high-level performance quality** while **layers 12–18 encode note-level features**. For optimal feature extraction, a learned weighted sum across layers (as in standard SUPERB/MARBLE protocols) allows the model to adaptively emphasize relevant layers per dimension.

### Recommended MERT configuration

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", output_hidden_states=True)
# Input: 24kHz audio, 5-second segments (native), or up to 30s with relative position embeddings
# Output: 75Hz frame rate, extract weighted sum of layers 12-24 for downstream regression
```

Segment length recommendation: **5–10 seconds** aligns with both MERT's native 5-second window and PercePiano's 8-bar segments (approximately 8–30 seconds depending on tempo).

---

## Dimension-modality mapping reveals where audio adds critical value

Synthesizing evidence from Bruno Repp's work on expressive timing, Chowdhury & Widmer's ISMIR 2021 study on mid-level features, and acoustic correlate research, each PercePiano dimension can be mapped to its optimal modality:

| Dimension | MIDI Strength | Audio Strength | Recommended Modality | Confidence |
|-----------|--------------|----------------|---------------------|------------|
| **timing** | Precise IOI encoding | — | MIDI primary | High |
| **articulation_length** | Duration ratios | Onset sharpness | Both (MIDI primary) | High |
| **articulation_touch** | Velocity patterns | Attack spectral shape | Both | Medium |
| **pedal_amount** | CC64 events | Sympathetic resonance, decay | **Audio primary** | High |
| **pedal_clarity** | Pedal on/off timing | Spectral smearing, part-pedal | **Audio primary** | High |
| **timbre_variety** | — | Spectral flux, MFCC variance | **Audio only** | High |
| **timbre_depth** | — | Spectral richness, harmonics | **Audio only** | High |
| **timbre_brightness** | — | Spectral centroid | **Audio only** | High |
| **dynamic_loudness** | Velocity | RMS energy | Both | High |
| **dynamic_sophistication** | Velocity contour | Loudness micro-dynamics | Both | Medium |
| **dynamic_range** | Velocity range | Actual dB range | Both | High |
| **tempo** | IOIs | — | MIDI primary | High |
| **space** | Note density, rests | Reverb/decay characteristics | Both | Medium |
| **balance** | Velocity between hands | Spectral balance | Both | Medium |
| **drama** | Dynamic contrast | Energy contours | Both | Medium |
| **mood_valence** | Mode (major/minor) | Spectral features | Both (score advantage) | Medium |
| **mood_energy** | Attack rate, tempo | RMS, spectral centroid | Both | High |
| **mood_imagination** | — | Timbral variety, phrasing | **Audio primary** | Low |
| **interpretation** | All symbolic | All acoustic | **Both required** | High |

**Key finding**: The four timbre dimensions and two pedal dimensions are audio-critical—they fundamentally require acoustic information unavailable in MIDI. Research confirms spectral centroid correlates strongly (r > 0.5) with perceived brightness, and pedal effects manifest only as acoustic phenomena (sympathetic resonance, decay modification).

---

## Cross-attention with gated fusion offers the most promising architecture

Survey of multimodal fusion literature reveals several effective approaches for MIDI+audio integration:

### Fusion architecture options

**Cross-Attention Fusion** (recommended primary approach): The MIAO framework and Nested Music Transformer both demonstrate that cross-attention between modalities effectively captures complementary information. For PercePiano, the MIDI encoder can use self-attention while the audio encoder queries MIDI representations via cross-attention—allowing audio features to be contextualized by symbolic structure.

**Gated Multimodal Unit (GMU)**: Research shows GMU outperforms mixture-of-experts for controlling modality contribution. Gates can learn per-dimension weights, naturally implementing dimension-specific modality preferences discovered in the mapping table.

**FiLM (Feature-wise Linear Modulation)**: Apply MIDI-conditioned affine transformations to audio features at decoder layers. Research indicates decoder-side FiLM outperforms encoder-side, with L2 regularization critical for preventing overfitting.

### Recommended hybrid architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Proposed Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│  MIDI Stream:                 Audio Stream:                     │
│  ┌──────────────┐            ┌──────────────┐                   │
│  │ Score-Aligned │            │  MERT-330M   │                   │
│  │ VirtuosoNet   │            │ (frozen or   │                   │
│  │ Features      │            │  fine-tuned) │                   │
│  └──────┬───────┘            └──────┬───────┘                   │
│         │                           │                            │
│         ▼                           ▼                            │
│  ┌──────────────┐            ┌──────────────┐                   │
│  │  Bi-LSTM +   │◄──────────►│ Bi-LSTM +    │                   │
│  │  HAN Encoder │  (cross-   │ Projection   │                   │
│  └──────┬───────┘  attention)└──────┬───────┘                   │
│         │                           │                            │
│         └──────────┬────────────────┘                            │
│                    ▼                                             │
│         ┌─────────────────────┐                                  │
│         │ Gated Fusion Layer  │ (dimension-specific gates)       │
│         └──────────┬──────────┘                                  │
│                    ▼                                             │
│         ┌─────────────────────┐                                  │
│         │ 19-Dim Regression   │ (separate heads per dimension)   │
│         └─────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Alignment strategy between modalities

Beat-synchronous timestamp mapping aligns MIDI note onsets with corresponding audio frames. Research shows that combining neural alignment (CRNN) with DTW achieves **20% improvement** over DTW alone. For PercePiano:

1. Use the existing score-to-performance alignment from VirtuosoNet
2. Map aligned note positions to audio frame indices based on onset times
3. Apply cross-attention at beat or measure granularity to handle minor timing discrepancies

### Expected improvement estimates

Based on similar multimodal music tasks:

- Music emotion recognition: **+8–16% absolute** improvement with audio over symbolic alone
- MT3 transcription with audio encoder: **+16.67%** improvement
- Genre classification with multimodal fusion: reaches **94.58%** vs 70–80% unimodal

For PercePiano, conservative estimates suggest:

- **Timbre dimensions**: +15–25% R² improvement (currently severely underserved)
- **Pedal dimensions**: +10–20% R² improvement
- **Emotional dimensions**: +5–15% R² improvement
- **Overall average**: +8–12% R² improvement (from 0.397 to ~0.47–0.52)

---

## MAESTRO's original recordings eliminate synthesizer concerns

The MIDI-to-audio rendering question has a straightforward solution: **use MAESTRO's original Disklavier recordings**. The dataset includes:

- **CD-quality audio** (44.1–48 kHz, 16-bit stereo) from Yamaha CFIIIS Disklaviers
- **~3ms alignment accuracy** between MIDI and audio
- **Real acoustic piano** with natural room acoustics, string resonance, and pedal effects
- All three pedal types preserved (sustain CC64, sostenuto CC66, una corda CC67)

This approach eliminates domain gap concerns entirely. For any MIDI segments without MAESTRO audio (e.g., e-Competition files), **Pianoteq PRO** offers the best rendering option:

- Physical modeling captures sympathetic resonance and half-pedaling
- JSON-RPC API enables batch processing
- Matches Yamaha piano characteristics reasonably well
- Parametric control allows generating multiple timbral variants for data augmentation

### Audio pipeline specification

```
Input: MAESTRO MIDI + Audio (or Pianoteq-rendered audio for non-MAESTRO data)
Format: WAV, 44.1kHz stereo, 16-bit
Segment: 5-10 second clips aligned to phrase boundaries
Normalization: -14 LUFS at piece level (preserves within-performance dynamics)
Augmentation: Time stretching ±5%, pitch shift ±50 cents, room impulse convolutions
```

---

## Implementation roadmap prioritizes ablation-ready experiments

### Phase 1: Baseline reproduction and audio feature extraction (Weeks 1–3)

1. **Reproduce PercePiano baseline**: Implement Bi-LSTM+SA+HAN architecture, verify R² = 0.397 on piece-split
2. **Extract MERT features**: Process all MAESTRO audio segments through MERT-330M, store layer-wise embeddings
3. **Build audio-only baseline**: Train regression heads on MERT features alone to establish audio ceiling

### Phase 2: Unimodal audio model (Weeks 4–5)

1. Train MERT → Bi-LSTM → 19-dim regression
2. Evaluate per-dimension R² to identify audio-advantaged dimensions
3. Compare against PercePiano baseline per dimension

### Phase 3: Multimodal fusion implementation (Weeks 6–8)

1. Implement cross-attention fusion between MIDI and audio streams
2. Add gated fusion layer with dimension-specific learned gates
3. Train end-to-end with MSE loss (matching PercePiano)
4. Conduct ablation: freeze MIDI encoder, vary audio encoder layers

### Phase 4: Ablation studies and analysis (Weeks 9–10)

| Ablation | Purpose |
|----------|---------|
| MIDI only (reproduction) | Baseline verification |
| Audio only (MERT) | Audio ceiling per dimension |
| Audio only (Wav2Vec2) | Non-music-pretrained comparison |
| Audio only (spectrogram CNN) | Learned vs. SSL comparison |
| MIDI + Audio (early fusion) | Concatenation baseline |
| MIDI + Audio (cross-attention) | Proposed architecture |
| MIDI + Audio (late ensemble) | Simple combination |
| Dimension-specific gates | Gate weight analysis |

### Cross-validation strategy

- **Piece-split** (primary): Tests generalization to unseen pieces
- **Performer-split** (secondary): Tests generalization to unseen performers
- **4-fold cross-validation** following PercePiano protocol

### Evaluation metrics

- **R²** (primary): Coefficient of determination per dimension
- **Correlation** (Pearson, Spearman): Per-dimension correlations
- **Range Accuracy RA@k**: Accounts for annotator variance (prediction within μ ± kσ)
- **MSE**: For direct comparison with PercePiano

### Computational requirements

| Component | GPU Memory | Training Time (est.) |
|-----------|-----------|---------------------|
| MERT-330M feature extraction | ~8GB | ~2 hours for full dataset |
| Multimodal training | ~12GB | ~6 hours per fold |
| Full experimental pipeline | A100-40GB | ~1 week |

---

## Risk mitigation addresses key uncertainties

**Risk 1: Synthesizer mismatch for annotations**
The original annotators heard Logic Pro + Yamaha VST, not MAESTRO Disklavier recordings. Mitigation: Compare model performance on MAESTRO audio vs. Pianoteq-rendered audio; if significant gap exists, consider rendering with Pianoteq for training.

**Risk 2: MERT pretraining bias toward pop music**
MERT was trained on 160K hours of general music, not classical piano. Mitigation: Include spectrogram CNN baseline and consider light fine-tuning of MERT on piano data.

**Risk 3: Alignment errors between modalities**
Misaligned MIDI and audio could corrupt cross-attention learning. Mitigation: Use PercePiano's pre-verified score alignment; add alignment verification step using onset detection.

**Risk 4: Overfitting on small dataset**
1,202 segments with 19 dimensions creates high-dimensional target space. Mitigation: Regularization (dropout 0.3, weight decay 1e-4), dimension grouping (low/mid/high heads), and early stopping.

---

## Conclusion: A multimodal system addresses PercePiano's fundamental limitations

The current MIDI-only PercePiano approach cannot capture timbral nuances that human annotators evaluated from audio stimuli. Adding MERT-330M audio encoding with cross-attention fusion directly addresses this modality gap, with evidence suggesting **10–25% R² improvements** on timbre and pedal dimensions.

The implementation path is clear: leverage MAESTRO's existing audio, extract MERT embeddings, fuse via cross-attention with learned dimension-specific gates, and validate through systematic ablations. The resulting system would represent the first audio-based perceptual piano performance evaluation framework, bridging a critical gap between symbolic music understanding and acoustic perception.

**Key technical recommendations**:

- **Audio encoder**: MERT-330M with weighted layer summation (layers 12–24)
- **Fusion**: Cross-attention between MIDI HAN and audio Bi-LSTM, followed by GMU gating
- **Data**: MAESTRO original audio (no re-rendering needed for core dataset)
- **Training**: Staged approach—pretrain unimodal, then joint fine-tuning

This framework directly connects computational predictions to the actual perceptual judgments that annotators made, yielding a more valid and interpretable piano performance evaluation system.
