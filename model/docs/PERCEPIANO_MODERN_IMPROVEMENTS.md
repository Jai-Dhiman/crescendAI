# Modernizing PercePiano: A comprehensive technical blueprint

PercePiano's hierarchical attention network achieves an R² of **0.56** on piano performance assessment, but modern architectural advances could push this significantly higher. The most impactful improvements involve replacing Bi-LSTM with **xLSTM or Conformer architectures**, adopting **graph neural networks** for score encoding, leveraging **MERT and MusicBERT** pretrained models to address limited labeled data, and replacing one-hot encoding with **learned factorized embeddings**. This report systematically analyzes every architectural component and provides concrete recommendations based on 2023-2025 research.

---

## Sequence models have fundamentally advanced beyond Bi-LSTM

The Bi-LSTM at PercePiano's core can be replaced by several architecturally superior alternatives. **xLSTM** (Extended LSTM), published at NeurIPS 2024 by Sepp Hochreiter's team, introduces matrix memory cells and exponential gating that outperform both Transformers and Mamba on language modeling benchmarks. The mLSTM variant is fully parallelizable during training while maintaining the familiar LSTM interface, making it a natural upgrade path. On speech enhancement tasks, xLSTM-SENet achieves PESQ and STOI scores on par with Conformer and Mamba architectures.

**Mamba** (Gu & Dao, 2023) offers linear-time sequence modeling with selective state spaces, achieving **5× higher inference throughput** than Transformers. Audio Mamba (AuM) demonstrated comparable performance to Audio Spectrogram Transformers with only one-third the parameters across six audio classification benchmarks. For piano specifically, AEROMamba achieved **15% improvement** over baselines on piano datasets with 14× inference speedup. The linear complexity makes Mamba particularly attractive for processing complete performances of several minutes.

**Conformer architecture** combines convolutional local pattern extraction with transformer global attention—exactly the balance needed for music, where both local motifs and long-range structure matter. ChordFormer (February 2025) applied Conformers to chord recognition, achieving 6% improvement in class-wise accuracy. NVIDIA's Fast Conformer variant provides 2.8× speedup through Longformer-style attention patterns.

The **Music Transformer** introduced relative positional attention that captures repeating patterns regardless of absolute position—crucial for musical structure. Combined with efficient attention variants (Longformer, BigBird), this enables processing sequences 8× longer than standard transformers with linear complexity.

For PercePiano specifically, xLSTM represents the lowest-friction upgrade since it maintains LSTM semantics while gaining parallel training and matrix memory. For maximum performance on long sequences, Mamba or Conformer should be benchmarked.

---

## Graph neural networks transform score encoding

PercePiano's Hierarchical Attention Network can be substantially improved by representing musical scores as graphs rather than sequences. **AnalysisGNN** (CMMR 2025) demonstrated state-of-the-art results across 20+ note-level music analysis tasks including cadence detection, Roman numeral analysis, and phrase segmentation by treating notes as nodes connected by temporal, simultaneous, and voice edges.

**ChordGNN** (ISMIR 2023) established a standardized graph representation: notes become nodes with pitch, duration, onset, and offset features, while edges encode temporal relationships (consecutive notes), simultaneous relationships (chords), voice connections (melodic lines), and onset relationships (rhythmic alignment). The **GraphMuse library** (ISMIR 2024) provides open-source implementations of these graph representations.

Jeong et al. (ICML 2019) specifically applied gated graph neural networks to **expressive piano performance modeling**, achieving more human-like rendered performances than sequence-based models. Their architecture used GNNs at the note level followed by hierarchical attention at the measure level—a hybrid approach that could directly replace PercePiano's encoder.

The key advantage of GNNs for performance assessment is their ability to capture polyphonic relationships that sequential models struggle with. When multiple voices interact, message passing between note nodes naturally encodes how the performer balances them, whether inner voices are properly voiced, and whether chord voicings support the melodic line.

---

## Modern attention mechanisms address alignment and long sequences

The score-performance alignment problem benefits tremendously from recent attention advances. **Soft-DTW** (Cuturi & Blondel, ICML 2017) provides a fully differentiable version of dynamic time warping that can be used as a training loss, enabling end-to-end learning of alignment without a separate DTW stage. The temperature parameter γ controls the smoothness-accuracy tradeoff.

**Monotonic attention** mechanisms (Raffel et al., ICML 2017) offer O(1) attention at inference time by exploiting the inherently sequential nature of score-performance alignment. Since performances generally follow the score order (with local deviations for ornaments), monotonic chunkwise attention (MoChA) provides an efficient, naturally-constrained alternative to full attention.

For handling the multi-modal nature of score-performance pairs, **Perceiver** architectures excel. PerceiverS (IEEE TASLP 2025) introduced multi-scale cross-attention with separate long-range (32,768 token) and short-range (1,024 token) context windows, achieving state-of-the-art on MAESTRO for expressive music generation. The Perceiver's latent bottleneck naturally forces learned alignment between modalities.

**FlashAttention-3** should be considered essential infrastructure, providing **2-4× speedup** and **10-20× memory savings** while computing exact attention. On H100 GPUs, FlashAttention-3 achieves 75% of theoretical maximum FLOPs compared to 35% for the previous version. Combined with sparse patterns (global tokens at measure boundaries, local windows for temporal context), this enables processing complete performances efficiently.

---

## Pretrained models and semi-supervised learning address data scarcity

PercePiano's 1,202 annotated segments represent a critical bottleneck. Modern approaches can leverage millions of unlabeled MIDI files through self-supervised pretraining.

**MERT** (NeurIPS 2023) provides the strongest foundation model for music understanding, achieving state-of-the-art on 4 of 14 MIR tasks in the MARBLE benchmark while using only **6.6% of the parameters** of Jukebox-5B (330M vs 5B). MERT's dual-teacher approach—combining RVQ-VAE for acoustic learning with Constant-Q Transform for pitch/harmonic inductive bias—creates representations that transfer well to downstream tasks. For audio-based performance assessment, MERT embeddings provide a strong starting point.

For symbolic MIDI, **MusicBERT** introduced OctupleMIDI encoding representing each note with 8 attributes (time signature, tempo, bar, position, instrument, pitch, duration, velocity), achieving **4× sequence compression** compared to REMI encoding. Bar-level masking prevents information leakage during pretraining. On PercePiano's own benchmark, MusicBERT-base achieved **R² = 0.57**, slightly outperforming the custom Percept-HAN architecture.

**Semi-supervised methods** can extract value from unlabeled student performances. FixMatch and MixMatch with appropriate audio augmentations achieve baseline performance with **less than 5% labeled training data** on industrial sound and music datasets. The key is selecting appropriate augmentations: SpecAugment (frequency and time masking) works better for audio than image-derived techniques like Cutout.

**GiantMIDI-Piano** (ByteDance) provides 10,855 piano works with 38.7 million notes from 2,786 composers for pretraining. Combined with MAESTRO's 200+ hours of aligned audio-MIDI performance data, a two-stage pretraining pipeline (first on GiantMIDI for scale, then fine-tuning on MAESTRO for expression) can substantially improve performance on downstream assessment tasks.

**Active learning** strategies can maximize annotation efficiency. Top-K entropy sampling—selecting the most uncertain segments within performances—reduced annotation requirements by **up to 92%** while maintaining sound event detection performance in recent work. For performance assessment, this means strategically labeling technically challenging or expressively ambiguous passages rather than random sampling.

---

## Learned embeddings dramatically outperform one-hot encoding

One-hot pitch encoding wastes dimensionality and fails to capture musical relationships. **Note2Vec** approaches using skip-gram training on chord progressions learn embeddings where cosine similarity reflects music-theoretic relationships. Stanford's Chord2Vec achieved **100% accuracy** on major/minor classification versus 81.1% for one-hot baselines.

**Factorized embeddings** separately encode pitch (64-128 dimensions), velocity (32-64 dimensions), duration (32-64 dimensions), and position (32-64 dimensions), then concatenate and project to a unified representation. This mirrors MusicBERT's OctupleMIDI but with continuous rather than discrete velocity and timing, better capturing expressive nuances.

**Hierarchical position encodings** (HRPE, 2024) modify attention with relative position biases at note, beat, bar, and phrase levels simultaneously, enabling joint learning of short and long-term dependencies. The **Hyperbolic Music Transformer** (IEEE Access 2023) embeds beat-bar-phrase hierarchy in hyperbolic space, naturally capturing the tree-like structure of musical meter.

For transposition invariance, **relative pitch encoding** based on intervals rather than absolute pitches helps models generalize across keys. PESTO (ISMIR 2023) demonstrated transposition-equivariant pitch estimation using Toeplitz matrices in the architecture, ensuring predictions shift proportionally to input transposition.

**Rotary Position Embeddings (RoPE)**, now standard in LLaMA and modern transformers, encode absolute position through rotation matrices while incorporating explicit relative position dependency in attention. For music, where relative timing matters more than absolute position, RoPE provides significant advantages over learned absolute position embeddings.

---

## Every architecture component can be systematically upgraded

Beyond the major modules, systematic improvements to every component compound into substantial gains.

**Normalization** should shift from LayerNorm to **Pre-RMSNorm**, which provides 7-64% speedup by eliminating redundant mean subtraction. Pre-norm placement (before attention and FFN rather than after) enables more stable training without warmup schedules. LLaMA, Llama 2/3, and PaLM all use Pre-RMSNorm.

**Activation functions** should upgrade from ReLU or GELU to **SwiGLU** (SiLU with gating), which combines the benefits of smooth activation with multiplicative gating. LLaMA and PaLM demonstrate consistent improvements from SwiGLU in FFN layers. This requires three linear projections instead of two but provides meaningful quality improvements.

**Pooling strategies** should move from mean pooling to **attention pooling**, where learned attention weights adaptively aggregate token representations. For music assessment, different passages matter differently for different criteria—attention pooling naturally handles this by learning to focus on technically challenging or expressively significant moments.

**Loss functions** should consider **Huber loss** (MSE for small errors, MAE for large) for robustness to outlier annotations. For ordinal rating scales, the **CORAL framework** (Consistent Rank Logits) preserves ordinal structure through binary classifiers with rank consistency constraints. **Uncertainty weighting** (Kendall et al.) automatically balances multi-task losses by learning task-specific uncertainties.

**Optimizers** should use **AdamW** with decoupled weight decay (0.01-0.05), cosine learning rate schedule with 5-10% warmup, and separate learning rates for pretrained (1e-5) versus new (1e-4) parameters. For very large batch sizes in distributed training, LAMB provides layer-wise adaptive learning.

**Regularization** should include standard dropout (0.1-0.2) after attention and FFN, attention dropout on attention weights, and potentially **DropHead** (dropping entire attention heads at 0.1 rate with scheduling) to prevent head dominance. Label smoothing (0.1) prevents overconfident predictions on subjective assessment scores.

---

## Recent papers establish new performance benchmarks

The PercePiano paper itself (Park et al., Scientific Reports 2024) established the first comprehensive benchmark for expert-guided piano performance evaluation with 19 perceptual features across 4 hierarchical levels. Their Percept-HAN baseline achieved **R² = 0.56** and **Range Accuracy@0.5 = 72.74%**.

Critically, their results showed that **hierarchical attention outperforms flat transformers** on this task. Music Transformer achieved only R² = 0.43, substantially worse than models explicitly encoding musical hierarchy (note → voice → beat → bar). This finding should inform architectural choices: any transformer-based replacement must incorporate hierarchical structure, whether through explicit encoding or through GNN preprocessing.

**Identifying Critical Segments Affecting Piano Performance Evaluation** (CIKM 2024) introduced SHAP-based feature importance analysis combined with change-point detection to identify musically critical passages, then used LLMs to generate interpretable feedback. When performers applied the generated feedback, their actual performance scores improved—demonstrating the practical value of explainable assessment.

The **MARBLE benchmark** (NeurIPS 2023) provides standardized evaluation across 18 MIR tasks on 12 datasets, with constrained, semi-constrained, and unconstrained tracks. MERT-330M achieves ROC-AUC up to 0.89 on music tagging tasks, suggesting substantial headroom for improvement on assessment tasks through transfer learning.

No papers have yet reported beating PercePiano's R² = 0.57 (MusicBERT-base) on their specific benchmark, but this reflects the benchmark's novelty rather than optimality of current approaches. The systematic architectural improvements outlined in this report—GNN encoding, modern sequence models, pretrained representations, learned embeddings—have each demonstrated substantial gains on related tasks and should compound when properly integrated.

---

## Implementation roadmap prioritizes highest-impact changes

For practitioners modernizing PercePiano, the following priority order maximizes impact while managing implementation complexity:

**Phase 1 (Highest impact, moderate effort)**: Replace Bi-LSTM with Conformer or xLSTM encoder; integrate MERT or MusicBERT pretrained weights; add FlashAttention for efficiency. Expected improvement: 10-20% relative R² gain based on related task improvements.

**Phase 2 (High impact, higher effort)**: Implement GNN-based score encoding with ChordGNN-style graph construction; add Perceiver-style cross-attention for score-performance alignment; replace one-hot with factorized learned embeddings and RoPE position encoding. Expected improvement: 15-25% additional relative gain.

**Phase 3 (Optimization and scaling)**: Apply semi-supervised learning with FixMatch on unlabeled performances; implement active learning for efficient annotation; add Soft-DTW alignment loss; optimize all components (SwiGLU, Pre-RMSNorm, attention pooling). Expected improvement: 10-15% additional relative gain.

**Phase 4 (Research frontiers)**: Explore Mamba for very long sequence processing; investigate diffusion models for assessment; develop LLM-based interpretable feedback generation; pursue multimodal fusion of audio and MIDI representations.

The total estimated improvement potential is substantial—possibly pushing R² from 0.56-0.57 toward 0.70-0.75 range—though this requires empirical validation. Each component has demonstrated significant gains on related music understanding tasks; the research question is how these improvements compound in the specific context of performance assessment.

---

## Conclusion: A synthesis of modern music AI for assessment

The PercePiano system, while pioneering, was designed before several transformative advances in sequence modeling, self-supervised learning, and music representation. The path forward combines **architectural upgrades** (xLSTM/Conformer replacing Bi-LSTM, GNNs for score encoding, Perceiver-style cross-attention), **representation improvements** (factorized embeddings, hierarchical position encoding, RoPE), **data efficiency techniques** (MERT/MusicBERT transfer learning, semi-supervised methods, active learning), and **systematic component optimization** (Pre-RMSNorm, SwiGLU, attention pooling, Huber loss).

The key insight from PercePiano's own evaluation—that hierarchical attention outperforms flat transformers—should guide all replacements: any new architecture must explicitly model musical hierarchy, whether through graph structure, hierarchical attention levels, or position encodings that encode metric structure. The goal is not merely to apply the latest models, but to apply them in ways that respect the unique hierarchical, polyphonic, and expressive nature of musical performance.

These improvements address every component identified in the original query: Bi-LSTM alternatives that excel on music (xLSTM, Conformer, Mamba), HAN replacements with superior alignment capabilities (GNN + Perceiver), techniques for limited annotations (MERT pretraining, semi-supervised learning, active learning), modern attention mechanisms (FlashAttention, monotonic attention, Soft-DTW), learned representations (factorized embeddings, HRPE, RoPE), and systematic component analysis (normalization, activation, pooling, loss functions, optimizers, regularization). Together, they provide a comprehensive blueprint for next-generation music performance assessment systems.
