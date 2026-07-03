# Competition research learnings (shared, task-agnostic)

> **Status:** LIVING. Distilled from two research passes (external assumption-targeted deep-research sweep, 102 agents / 24 confirmed; + file-grounded internal repo audit) run 2026-07-02/03. Confidence tags: [CONFIRMED] survived adversarial verification; [repo] read from our own code; [UNKNOWN] genuinely open.
> **Scope:** Reusable findings for any CrescendAI performance-analysis work. Task-specific plans live in `mirex-difficulty.md` and `rewardbench.md`.

## The assumption audit (what we believed vs what is true)

| # | Assumption | Verdict | Reality (source) |
|---|---|---|---|
| A1 | Our pairwise-ranking head is a 1-dim swap for difficulty | **KILLED** [repo] | Head is 6-dim, coupled across 4 losses (BCE-rank + ListMLE + InfoNCE + CCC-reg), and pairs **within-piece** for quality ranking. Difficulty is **piece-level → needs cross-piece pairs**. `aria_encoder.py:243`, `losses.py:405` |
| A2 | Reuse our deterministic difficulty features | **HALF** [repo] | Have: note-density, velocity mean/dispersion, timing CV / IOI-CV, pedal fraction, per-frame pitch. **Missing the strong ones:** pitch-entropy, pitch-set complexity, polyphony, pitch-range. `midi_measures.py`, `tokenizer.py:110` |
| A3 | MuQ has an internal difficulty prior (rho~0.62) | **KILLED** [repo] | No such experiment exists. R2=0.537 is *perceptual*, *synthetic audio*, *leaked-fold* (clean = 0.336), from a preprint our own README says not to lean on. **No validated internal difficulty signal.** |
| A4 | Symbolic-from-AMT is soundfont-invariant | **REFUTED** [CONFIRMED] | Disklavier re-recordings of identical MAESTRO notes transcribe differently by acoustic environment → a human recording and a soundfont render **may transcribe inconsistently**. arXiv:2402.01424 |
| A5 | AMT is accurate enough on hard repertoire | **SPLIT** [CONFIRMED + repo] | Onsets/pitches robust (0.97 F1) even at 180+ notes/30s. Durations fragile (offset F1 0.85; dur-ordering rho 0.41-0.57 on pedaled). Degrades OOD/dense ("entanglement problem"). Only <=30s validated (aria-amt hard-truncates at 30s). aria-amt paper; arXiv:2408.04737; internal #64 |
| A6 | Note-count/density is the difficulty signal | **CORRECTED** [CONFIRMED] | Density weak; avg-pitch weakest (0.017-0.088). **Strongest hand-crafted features: pitch-entropy + pitch-set Lempel-Ziv complexity (~0.58 tau-c).** Removing pitch-set-LZ costs -5.2 Acc-9 (largest single ablation). RubricNet Table 2/4 |
| A7 | Hand-crafted features << deep encoders | **REFUTED** [CONFIRMED] | RubricNet (hand-crafted, interpretable) matches/beats the deep ensemble on CIPI Acc-9 (41.4 vs 39.5). Cheaper, interpretable, competitive. arXiv:2408.00473 |
| A8 | We know the gate numerically | **UNKNOWN** | Official tau-c variant, scoring script, reference baseline, split protocol, and **how the human-vs-soundfont pair is scored** are all undocumented publicly. Hard dependency on captains. |
| A9 | CIPI has a clean split | **CORRECTED** [CONFIRMED] | CIPI split is 60/20/20 stratified by difficulty+length, **not composer-disjoint** → build our own composer-disjoint split to avoid self-deception. arXiv:2306.08480 |
| A10 | Era/composer metadata usable at inference | **UNKNOWN** | The multi-task feature that lifted tau-c 0.77->0.78 may be train-only. Ask captains. |
| A11 | Fused audio+symbolic difficulty is a known win | **UNKNOWN** | No published fused audio+symbolic difficulty model. Unexplored — opportunity + risk. |

## AMT reliability profile (load-bearing for any symbolic-from-audio work)

- **Trust:** onset timing (3-8 ms noise), pitch, note-count/density, note recall (0.93-1.0). Build difficulty features from these.
- **Distrust:** note durations/offsets (ratio up to 3.2x on pedaled textures; offset F1 0.85). Exclude duration-dependent features.
- **Untested / risky:** cross-render consistency (human vs soundfont) — REFUTES the "soundfont-invariant" hope; must be measured. Whole-piece fidelity (>30s) — aria-amt truncates at 30s. Dense/OOD repertoire — degrades.
- Pipeline: `WAV -> 16kHz -> aria-amt -> MIDI` ; `MIDI tokens -> Aria (512-dim)` ; `WAV -> 24kHz -> MuQ (1024-dim)`. Two independent audio front-ends. [repo]

## Datasets & licensing (difficulty task)

- **CIPI** ("Can I Play It?"): 652 scores, Henle 1-9 ordinal (annotated Prof. Rolf Koenen, expert-refined), **CC BY-NC-SA 4.0, request-gated, NON-COMMERCIAL**. Zenodo 8037327. github.com/PRamoneda/difficulty-prediction-CIPI
- **PianoSyllabus (PSyllabus):** 7,901 pieces, 11 levels, 1,233 composers, **CC-BY-NC-4.0 NON-COMMERCIAL, no audio redistribution** (ships transcribed MIDI + CQT), Zenodo-on-request, DOI 10.5281/zenodo.14794592.
- **Mikrokosmos-difficulty (MKD):** ~147 pieces, 3 levels.
- **Licensing consequence:** ALL captain-endorsed difficulty data is research-only. Valid for a MIREX submission; a **shipped** difficulty head needs a separately-licensed or self-labeled corpus (distinct later workstream).
- **Contamination:** CIPI/PSyllabus repertoire overlaps MAESTRO (trains the transcriber) — uncontrolled, unquantified. Split held-out by composer, never by recording.

## SOTA reference numbers (difficulty)

- Symbolic (CIPI, Ramoneda ensemble = ArGNN+VirtuosoNet+Pitch, LATE-FUSION not MoE): 39.47% Acc-9 / 87.27% Acc+/-1 / MSE 1.13. arXiv:2306.08480
- Symbolic interpretable (RubricNet): 41.4% Acc-9 / MSE 1.7 on CIPI; 79.6% on MKD. arXiv:2408.00473
- Audio (PSyllabus CRNN over CQT + transcribed piano-roll): tau-c 0.77 single-task -> 0.78 +era-multitask, ~37% acc, MSE ~1.85. arXiv:2403.03947 (IEEE/ACM TASLP 2025)
- Audio benchmark (Zhang PianoJudges, 4 encoders, Audio-MAE best): difficulty 33.7% << expertise-ranking 93.6%, technique 46.7%. arXiv:2407.04518
- Published tau-c (image multi-task bootleg): 0.68 CIPI / 0.59 PianoStreet / 0.56 FreeScores. arXiv:2309.16287
- **Metric stack (Ramoneda canonical):** Acc-n (Acc0/Acc1) + macro-MSE + Kendall tau-c. NOT tau-c-only.

## Open questions for the captains (blockers)

Email Huan Zhang `huan.zhang@qmul.ac.uk` + Pedro Ramoneda `pedro@songscription.ai`, cite `music-ir.org/mirex/wiki/2026:Music_Performance_Difficulty_Prediction`:
1. CIPI + PSyllabus access + which is the **sanctioned** training set?
2. Official **scoring script** + exact **tau-c variant** + reference **baseline**?
3. **Train/test split** protocol?
4. How is the **anti-cheat pair** (human performance + synthesized rendering) scored — averaged / both-must-agree / separate?
5. Is **era/composer metadata** available at inference, or train-only?

## Primary sources
arXiv:2306.08480 (CIPI), 2403.03947 (PSyllabus), 2408.00473 (RubricNet), 2407.04518 (PianoJudges), 2309.16287 (image/ordinal), 2402.01424 (AMT acoustic overfit), 2408.04737 (AMT entanglement), aria-amt paper (alexander-spangher.com/papers/aria_amt.pdf); MIREX 2026 wiki; Zenodo 8037327 + 14794592.
