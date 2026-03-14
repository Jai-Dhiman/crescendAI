# North Star: The Perfect Model

Vision document for the ideal piano performance evaluation model. Captures long-term research direction and the rationale behind current architectural decisions.

> **Status (2026-03-14):** Vision document. No items here are actively in progress. See `03-encoders.md` for current work.

---

## Why Fusion Failed (and When It Becomes Viable)

The ISMIR paper (`paper/ismir_v2/main.tex`) tested audio-symbolic fusion on our data:

| Model | R2 |
|-------|-----|
| Audio only (MuQ + LoRA) | 0.537 |
| **Audio-symbolic fusion (concat)** | **0.524** |
| Symbolic only (baseline) | 0.347 |

Fusion *underperformed* audio-only. Error correlation between modalities: **r = 0.738** -- both fail on the same samples.

### Root Cause: Pretraining Asymmetry

Both modalities derive from identical MIDI source data (PercePiano uses Pianoteq-rendered audio). The audio signal contains no information absent from the symbolic input. The advantage stems from:

1. **Pretrained inductive bias:** MuQ was pretrained on 160K hours of music. The symbolic encoder was trained from scratch on ~14K-24K graphs.
2. **Representation accessibility:** MuQ's pretrained features already encode musical quality patterns. The symbolic encoder must learn these from limited labeled data.

This framing has an important implication: **the gap may narrow as symbolic foundation models (pretrained on large MIDI corpora) become available.** The current comparison is between a pretrained audio model and a from-scratch symbolic model -- the advantage reflects pretraining scale as much as modality choice.

### When to Revisit Fusion

Fusion becomes viable when:
- A symbolic foundation model (pretrained on millions of MIDI files) exists and matches MuQ's representation quality
- Error correlation between modalities drops below ~0.5 (indicating genuinely different failure modes)
- Gated per-dimension fusion (F3) can exploit real complementarity: audio for timing/resonance, symbolic for structure/dynamics

---

## The Perfect Model (Unlimited Resources)

### Layer 1: Symbolic Foundation Model

Pretrain a symbolic model on millions of MIDI performances with self-supervised objectives:
- Masked note prediction (BERT-style)
- Next-bar prediction (autoregressive)
- Contrastive learning across performances of the same piece

With comparable pretraining scale to MuQ, the r=0.738 error correlation should drop -- the models would fail on genuinely different aspects of the signal. At that point, per-dimension gated fusion becomes meaningful:
- Symbolic excels at structure: voice leading, harmonic analysis, motivic development
- Audio excels at timbre: pedal resonance, hammer voicing, room acoustics

### Layer 2: Score-Conditioned Quality

The current model evaluates performance quality in isolation. The ideal model evaluates quality *relative to what was written.*

```
quality = f(z_performance_audio, z_performance_midi, z_score_midi)
```

**What this enables:**
- "Your dynamics match the crescendo Chopin marked" vs "dynamics are flat" (absolute)
- Rubato detection: systematic deviation from score timing + compensatory return = intentional expression (not just "timing issues")
- Difficulty-aware feedback: "this passage is technically demanding, your articulation is impressive given the difficulty"
- Open-ended piece support: any piece with available MIDI score

**Prerequisites:** Score alignment pipeline (chroma DTW for measure-level), 2K+ labeled (performance, score, quality) triples.

### Layer 3: Real Audio Training

All current training uses Pianoteq-rendered MIDI (synthesized, studio-quality). Neither modality captures:
- Pedal resonance subtlety (sympathetic vibration, half-pedaling)
- Hammer voicing differences across pianos
- Real room acoustics
- Timbral warmth of a well-voiced Steinway vs a digital keyboard

Expert-annotated real piano recordings (acoustic grands, different venues, different recording setups) would unlock dimensions the current system literally can't hear. At this point, audio and symbolic capture genuinely different things -- real complementarity.

### Layer 4: Temporal Understanding

Currently: independent 15-second chunks evaluated in isolation.

Perfect model: understands musical arc across an entire piece or movement.
- Phrasing scores become meaningful at the passage/section level
- Interpretation captures how a performer shapes the overall narrative
- Requires attention across chunks or hierarchical architecture (local phrase-level -> section-level -> movement-level)

### Layer 5: Expert Labels at Scale

Current ceiling: crowdsourced IRT (PercePiano). Inter-rater agreement ~80%.
- 3-5 expert piano teachers annotating 5K-10K segments pushes the label quality ceiling
- Active learning: prioritize segments where model is most uncertain
- May reveal current model is already near the label quality ceiling

---

## North Star Architecture

```
Score MIDI ---------> [Symbolic Foundation Model] -------> z_score
                                                              |
Performance Audio --> [Audio Foundation Model (MuQ)] -----> z_audio   --> [Score-Conditioned
Performance MIDI --> [Symbolic Foundation Model] ---------> z_perf        Gated Fusion] --> 6 dims
                                                              |
                                                    [Temporal Attention
                                                     across chunks]
```

Each component addresses a specific limitation of the current system:

| Component | Addresses | Current Limitation |
|-----------|----------|--------------------|
| Symbolic foundation model | Pretraining asymmetry | S2 trained from scratch, error correlation r=0.738 |
| Score conditioning | Quality vs appropriateness | Dynamics inversion (rho=-0.917 in competition) |
| Real audio training | Pianoteq domain gap | All training on synthesized audio |
| Temporal attention | Chunk independence | 15s chunks evaluated in isolation |
| Expert labels | Annotation noise ceiling | Crowdsourced IRT labels |

---

## Intermediate Milestones (Between Now and North Star)

### Timing Direction (Cheap Win, No Dependencies)

Extract systematic rushing/dragging from onset deviations. Single flag in teacher prompt. The LLM judge found timing direction consistently valuable even when other MIDI stats were rejected. Implementable without score alignment.

### MIDI as LLM Context (Alternative to Fusion)

Instead of model-level fusion, feed structured MIDI comparison to the teacher subagent alongside A1 scores. The LLM reasons about complementary signals -- more robust to AMT noise, fully interpretable, buildable in days.

### Bar-Aligned Passage Context (Wave 3 Prerequisite)

Not raw MIDI statistics ("velocity MAE = 15") but musically grounded context ("in bars 12-16, velocity drops to pp but the score asks for mf crescendo"). Requires score alignment pipeline. This is the correct version of Experiment 4's concept.
