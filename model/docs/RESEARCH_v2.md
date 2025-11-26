# Multi-Modal Piano Performance Evaluation: A Technical Research Report

Your system achieves Pearson r ≈ 0.2, which explains only **4% of variance** and falls substantially below both human agreement (r ≈ 0.5-0.7) and published ML baselines (r ≈ 0.3-0.5). The core issues are: cross-attention fusion destroying information from misaligned encoders, synthetic labels measuring piece difficulty rather than performance quality, and an untrained MIDI encoder failing to learn aligned representations. This report identifies concrete solutions across eight research areas, prioritized by expected impact and implementation cost.

## The fusion problem stems from representation misalignment, not architecture

Your cross-attention fusion degradation (-19%) is a documented failure mode when combining pre-trained and untrained encoders with different embedding geometries. MERT's 768-dim representations encode acoustic information through self-supervised pre-training on 160K hours of music, while your 256-dim MIDIBert learns from scratch on a much narrower task. Cross-attention assumes aligned semantic spaces—without this, attention weights become essentially random, injecting noise rather than combining information.

**Replace cross-attention with simpler, more robust fusion methods.** Low-Rank Multimodal Fusion (LMF) from Liu et al. (2018) captures multiplicative interactions between modalities without requiring aligned spaces, decomposing the tensor fusion weight into modality-specific low-rank factors. Gated Multimodal Units (GMU) from Arevalo et al. (2017) learn which modality to trust per-sample via sigmoid gating: `fused = gate * audio + (1-gate) * midi`. FiLM conditioning (Perez et al., 2018) allows MIDI to modulate audio processing via learned scale and shift parameters, which is particularly appropriate when MIDI should guide what audio features matter.

**Add projection heads before any fusion.** Project both modalities to a shared 512-dimensional space using 2-layer MLPs with LayerNorm and GELU: `Linear(768→512) → LayerNorm → GELU → Linear(512→512)` for audio, `Linear(256→512) → LayerNorm → GELU → Linear(512→512)` for MIDI. L2-normalize outputs before fusion. This architectural pattern comes from CLIP and MuLan, which successfully align vision-language and audio-language representations.

**Freeze MERT initially with staged unfreezing.** Train only projection heads and fusion module for 10 epochs, then unfreeze MERT's top 4 layers with 10x lower learning rate. This prevents catastrophic forgetting of pre-trained features while allowing alignment to develop. Add **modality dropout** (15-20%) during training to prevent one modality from dominating and ensure both encoders learn useful representations independently.

PianoBind (2024) directly addresses your use case—combining audio, MIDI, and text for piano music using MidiBERT as the symbolic encoder. Their key finding: averaged-loss training clearly outperforms staged alignment approaches. CLaMP3 (2025) demonstrates that contrastive pre-training enables cross-modal alignment even between very different modalities.

## PercePiano provides the expert annotations you need—and it's public

The **PercePiano dataset** (Park et al., 2024, Nature Scientific Reports) is publicly available at github.com/JonghoKimSNU/PercePiano with **1,202 segments** annotated across **19 perceptual features** by **53 music experts** (graduate-level pianists or higher). Each segment received 5-12 annotations on a 7-point Likert scale covering timing, articulation, dynamics, pedaling, timbre, music making, emotion/mood, and interpretation. This is derived from the same MAESTRO performances you're using but with actual quality annotations rather than synthetic heuristics.

**Their baseline architecture achieves R² = 0.397** using Bi-LSTM with score-aligned features and hierarchical attention networks (HAN). The score-alignment component alone provides a **21.2% absolute improvement** over Bi-LSTM—incorporating tempo deviation, onset deviation, dynamics deviation, and articulation deviation from the reference score. This finding is critical: performance evaluation inherently requires a reference. Without score alignment, models cannot distinguish "deliberately rubato" from "accidentally wrong timing."

Human inter-rater reliability in music assessment is inherently limited. PercePiano reports ICC(1,1) "poor" for individual annotators but ICC(1,k) "excellent" when averaged—individual experts disagree substantially, but **5-12 annotators converge toward reliable consensus**. This sets a realistic upper bound: even with perfect labels, your model cannot exceed average human agreement of roughly r ≈ 0.6-0.7 for most dimensions.

Other relevant datasets include **ATEPP** (1,000 hours of piano from 49 world-renowned pianists with expressive performance data) and **GiantMIDI-Piano** (10,855 MIDI files for pre-training). The **FBA dataset** (Florida Bandmasters Association student auditions) provides multi-instrument assessment with 4 criteria but is not publicly released.

## Your label problem requires multi-pronged solutions

Synthetic labels measuring MIDI complexity cannot teach performance quality—piece difficulty correlates weakly with how well someone played it. This is your most fundamental issue. Three complementary approaches address this:

**Use Snorkel to aggregate multiple weak labeling functions.** Your existing MIDI complexity heuristics (note density, velocity variance, rhythmic complexity) can each become separate labeling functions. Snorkel's generative model learns which functions to trust based on agreement/disagreement patterns, outputting probabilistic labels that are more robust than any single heuristic. Key insight: Snorkel can model correlations between LFs without requiring ground truth.

**Co-training exploits your dual audio-MIDI views.** Train separate models on audio and MIDI features, then have each provide pseudo-labels for samples where the other is confident. This works because the views are conditionally independent given quality—audio captures tone and expression while MIDI captures timing and note accuracy. DivideMix (Li et al., 2020) extends this by fitting a Gaussian Mixture Model to per-sample losses, dividing samples into "clean" (labeled) and "noisy" (treated as unlabeled), then applying semi-supervised learning.

**Controlled degradation creates real quality variance.** Your planned approach is sound: take MAESTRO performances and systematically inject timing jitter (20-100ms standard deviation), wrong notes (5-15% substitution rate), dynamics compression (reduce velocity range by 30-70%), and audio noise. Create **4 clearly distinct tiers** with known quality ordering. This provides ranking supervision even without absolute scores. Key: ensure degradations correlate with the specific dimensions you're evaluating—timing perturbation affects rhythm accuracy, velocity randomization affects dynamics, note deletion affects note accuracy.

For loss functions, **switch from MSE to Huber loss** (robust to outliers) or MAE (inherently noise-tolerant). Consider **bootstrap loss** (Reed et al., 2014): `L = β * MSE(y, pred) + (1-β) * MSE(pred, pred)` which softens potentially noisy labels using the model's own predictions.

## Ranking and ordinal approaches address the narrow distribution problem

Your MAE of 4.2 on a 0-100 scale indicates predictions clustering in a ~4-point range—the model learned to predict the mean because MSE incentivizes this when target variance is low. All MAESTRO performances are virtuoso-level, so there's genuinely little quality variance in your training data.

**Add pairwise ranking loss alongside regression.** For pairs (A, B) where A should score higher:

```
L_rank = max(0, margin - (score_A - score_B))
L_total = α * MSE + (1-α) * L_rank
```

Start with α=0.5, margin=5.0. This forces the model to distinguish between performances even when both are high-quality. The **allRank** library (github.com/allegro/allRank) provides ListNet, LambdaRank, and other learning-to-rank losses in PyTorch.

**Convert to ordinal regression using CORAL or CORN.** Discretize your 0-100 scale into 20 bins (5-point resolution). CORAL (Cao et al., 2020) transforms this into K-1 binary classification problems ("Is score > threshold_k?") with weight-sharing to guarantee rank monotonicity. Available as `pip install coral-pytorch`. This approach maintains ordering guarantees while avoiding the mean-regression problem of MSE.

**Apply Label Distribution Smoothing (LDS) and Feature Distribution Smoothing (FDS)** from Yang et al.'s "Delving into Deep Imbalanced Regression" (ICML 2021). LDS re-weights loss by inverse of smoothed label density, upweighting rare score ranges. FDS calibrates feature statistics across nearby target bins. Both are specifically designed for deep learning with narrow target distributions and validated on age estimation (a similar problem). Official implementation at dir.csail.mit.edu.

**Contrastive learning for quality-aware representations.** CONTRIQUE (Madhusudana et al., 2021) uses contrastive learning to cluster by quality without labeled data. Create positive pairs (same piece, similar quality tier) and negative pairs (same piece, different quality tiers). This can leverage unlabeled performance data and learns to distinguish subtle quality differences.

## The MIDI encoder needs pre-training and alignment

Your MIDIBert trained from scratch cannot match MERT's representation quality. MidiBERT-Piano (official: github.com/wazenmai/MIDI-BERT) uses 12-layer transformers with 768-dim hidden size, pre-trained on ~4,166 pieces using masked language modeling (15% token masking). It converges in 2.5 days on 4x GTX 1080 Ti and outperforms Bi-LSTM in 1-2 epochs of fine-tuning.

**Pre-train your 256-dim encoder on GiantMIDI-Piano** (10,855 MIDI files, 38.7M notes). Scale down to 6 layers, 4 attention heads. Use CP (Compound Word) representation: [Bar, Position, Pitch, Duration]. Expected compute: 12-24 hours on single A100. This is the highest-impact technical change with clear implementation path.

**Align to MERT's space via contrastive learning on paired data.** MAESTRO provides perfectly aligned audio-MIDI pairs. Use InfoNCE loss:

```python
logits = normalize(audio_proj(mert_out)) @ normalize(midi_proj(midi_out)).T / τ
loss = cross_entropy(logits, labels) + cross_entropy(logits.T, labels)
```

Temperature τ=0.07 works well. This is your planned "contrastive pre-training" step—the research strongly supports its effectiveness.

**Consider knowledge distillation from MERT to MIDI encoder.** Train the MIDI projection to match MERT embeddings for aligned pairs: `L_distill = MSE(midi_proj(midi_out), mert_out.detach())`. This directly transfers MERT's learned acoustic semantics to the MIDI representation space.

## Data augmentation requires careful design to preserve or intentionally degrade quality

**MIDI augmentations that preserve quality labels:** Pitch shifting (transposition ±6 semitones), tempo scaling (0.8-1.2x), small velocity jitter (σ=5-10), small timing perturbation (σ=10-20ms as humanization). MidiTok (github.com/Natooz/MidiTok) provides comprehensive tokenization with built-in augmentation.

**MIDI augmentations that degrade quality (for creating training variance):** Large timing errors (σ>50ms), wrong notes (pitch substitution 5-15%), velocity randomization (destroying dynamics), note deletion (10-20%). These create negative samples with known quality deficits.

**Audio augmentations that preserve quality:** Small pitch shift (±2 semitones), room impulse response application (recording condition invariance), light background noise (SNR >20dB), SpecAugment (time/frequency masking for regularization). Use audiomentations (github.com/iver56/audiomentations) for CPU or torch-audiomentations (asteroid-team) for GPU acceleration.

**Audio augmentations that degrade quality (for training variance):** Heavy distortion/clipping, severe noise (SNR <10dB), codec artifacts, extreme filtering. These affect the "tone quality" and "recording quality" dimensions.

## Evaluation requires proper baselines and statistical rigor

Your r ≈ 0.2 is barely above simple feature-based baselines (r ≈ 0.1-0.3) and significantly below published systems (PercePiano achieves R² = 0.397, roughly r ≈ 0.63). Report the following baselines:

- **Predicting global mean**: r = 0.0 by definition (sanity check)
- **Linear regression on simple features**: velocity variance, timing deviation, dynamic range
- **Pre-trained symbolic model**: MidiBERT or similar
- **Your current multimodal system**
- **With score alignment**: if you add this capability

Use **leave-one-performer-out** cross-validation as primary protocol, **leave-one-piece-out** as secondary. Never allow the same performer or piece in both train and test—this prevents models from learning identity rather than quality.

Report **Pearson r with 95% bootstrap CI** (1000+ iterations, BCa method), MSE, and Spearman ρ. For comparing models, use **Williams' test** for dependent correlations since both models predict on the same data. Report Cohen's q effect sizes (small ≈ 0.1, medium ≈ 0.3, large ≈ 0.5).

## Implementation roadmap prioritized by expected impact

| Priority | Action | Expected Δr | Effort | Timeline |
|----------|--------|------------|--------|----------|
| **1** | Pre-train MIDI encoder (MidiBERT-style on GiantMIDI-Piano) | +0.10-0.15 | Medium | Week 1-2 |
| **2** | Replace cross-attention with LMF or gated fusion | +0.05-0.10 | Low | Days |
| **3** | Add projection heads + freeze-then-unfreeze training | +0.03-0.07 | Low | Days |
| **4** | Train on PercePiano expert annotations | +0.15-0.25 | Low | Week 2 |
| **5** | Contrastive alignment (MERT↔MIDI) on MAESTRO pairs | +0.05-0.10 | Medium | Week 2-3 |
| **6** | Add pairwise ranking loss (α=0.5) | +0.05-0.08 | Low | Days |
| **7** | Implement controlled degradation pipeline | +0.10-0.15 | Medium | Week 3-4 |
| **8** | Apply LDS/FDS for imbalanced regression | +0.03-0.05 | Low | Days |
| **9** | Add score alignment features | +0.10-0.20 | High | Week 4+ |

**Quick experiments to validate before full implementation:**

1. Train audio-only model on PercePiano—if this reaches r>0.3, expert labels are the primary issue
2. Run fusion ablation: concatenation→gated→LMF→cross-attention to identify best approach
3. Add ranking loss to current model and measure impact on prediction spread
4. Compute baseline: linear regression on {velocity_std, onset_deviation, dynamic_range}

**Key repositories:** MidiBERT (github.com/wazenmai/MIDI-BERT), MERT (github.com/m-a-p/MERT), CLaMP (github.com/sanderwood/clamp3), PercePiano (github.com/JonghoKimSNU/PercePiano), allRank (github.com/allegro/allRank), coral-pytorch (github.com/Raschka-research-group/coral-pytorch), DIR/LDS/FDS (dir.csail.mit.edu), audiomentations (github.com/iver56/audiomentations), MidiTok (github.com/Natooz/MidiTok).

## The path from r=0.2 to r>0.5 is achievable

Your target of r>0.5 on technical dimensions is realistic given that PercePiano achieves R²=0.397 (r≈0.63) and human inter-rater agreement reaches r≈0.6-0.7. The critical insight from this research is that your **labels are the primary bottleneck**, not your architecture. Expert annotations from PercePiano combined with controlled synthetic degradation to create training variance should yield the largest improvements. The fusion and encoder improvements—while important—are secondary to having labels that actually measure what you're trying to predict.

Switching from synthetic complexity metrics to expert perceptual annotations, pre-training the MIDI encoder, replacing cross-attention with simpler fusion, and adding ranking losses should collectively move you from r≈0.2 to r≈0.4-0.5 within the first month of implementation. Further gains toward r≈0.6 will likely require score alignment (computationally expensive but adds ~20% improvement in PercePiano's experiments) and larger-scale expert annotation collection.
