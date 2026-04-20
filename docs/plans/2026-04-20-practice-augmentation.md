# Practice-Distribution Augmentation — MIDI Corruption + Room IR

**Goal:** Close the distribution gap between PercePiano / T5 (studio-recorded,
finished performances) and the actual distribution CrescendAI runs on in
production (practice-room phone recordings of in-progress playing). Measured by
OOD pairwise-accuracy gap (Chunk B's harness).

**North star metric:** OOD-vs-clean pairwise gap. Baseline (no augmentation):
unknown, but expected to be large — 15–25pp based on literature on domain
shift in audio perception models. Target after this plan: gap ≤ 10pp.

## Why

- **The harness will tell the user things about their practice that are wrong.**
  If the model was trained on only finished performances, the first page-turn
  thump, the first stop-and-restart, the first wrong-note-recovered-fast, it
  produces a confidently-wrong 6-vector. The harness can't veto that
  downstream because the inputs to the veto (σ from heteroscedastic heads) are
  calibrated on the same skewed distribution.
- **Practice ≠ performance, not just in content but in acoustic.** Phone mic
  + untreated room ≠ studio mic + treated room. Even the same pianist playing
  the same piece will score differently. Audio-domain augmentation can't fix
  content skew; MIDI-domain augmentation can.
- **Combined beats either alone.** MIDI corruptions make the *content* look
  more like practice (wrong notes, pauses). Room-IR convolution makes the
  *acoustic* look more like practice (reverb, noise). Both needed.

## Scope

**In scope.** MIDI corruption synthesis, room IR convolution, augmented
training datasets, per-sample corruption probability schedule. Integration
with existing `AudioAugmentor`.

**Out of scope.** Real practice-room data collection — that's the year-2
roadmap item. This plan only builds *synthetic* practice distribution.

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `model/src/model_improvement/practice_synthesis.py` | Pure MIDI corruption module. Functions: `drop_notes(midi, rate)`, `substitute_wrong_notes(midi, rate)`, `jitter_tempo(midi, std)`, `compress_velocity(midi, factor)`, `insert_pauses(midi, rate, max_dur)`. All return a new `PrettyMIDI` object. |
| `model/data/raw/room_irs/README.md` | Documents how to curate ~15–20 real practice-room impulse responses. Sources: student-room recordings, church acoustics for contrast, a few bad-phone-mic IRs. |
| `model/scripts/render_corrupted_audio.py` | Takes a MAESTRO MIDI, applies a `practice_synthesis` pipeline, renders to audio via FluidSynth, convolves with a random IR from `room_irs/`. Output: new WAV in `data/raw/maestro_corrupted/`. |
| `model/src/model_improvement/data.py:PracticeAugmentedDataset` | Wraps an existing audio dataset. With probability `p_corrupt` per sample, substitutes a pre-rendered corrupted version. Non-corrupted samples pass through unchanged. |

### Modified Files

| File | Change |
|---|---|
| `model/src/model_improvement/augmentation.py:AudioAugmentor` | Add `room_ir_convolve(audio, ir_bank, p)` and `add_practice_noise(audio, snr_range)`. These run on top of the existing audio augmentations. |
| `model/src/model_improvement/a1_max_sweep.py` | New config dimension: `corrupt_prob ∈ {0.0, 0.25, 0.5}`. Gated on having corrupted audio pre-rendered. |

## Phases

### P0 — MIDI corruption primitives (1.5 days)

- [ ] Implement the five corruption functions in `practice_synthesis.py`.
  Parameters chosen to match observed practice-distribution statistics:
  - `drop_notes`: rate 2–8% (occasional missed notes)
  - `substitute_wrong_notes`: rate 5–15% (pick a neighbor key, preserve timing)
  - `jitter_tempo`: std 3–8% of local tempo (unsteady pulse)
  - `compress_velocity`: factor 0.6–0.85 (dynamic flattening on cheap mics)
  - `insert_pauses`: rate 1 per 20s, duration 0.5–2s (stop-and-restart)
- [ ] Unit tests on a fixture MIDI: each corruption produces the expected
  number of changes.

### P1 — IR curation + rendering pipeline (2 days)

- [ ] Curate 15–20 real IRs. Phone-captured ones from user's own practice room,
  church-reverb ones from freesound.org (CC-licensed), a few deliberately bad
  phone-mic IRs (clipped, noisy).
- [ ] Write `render_corrupted_audio.py`. Pipeline: MIDI → corruptions →
  FluidSynth render → IR convolve → output WAV. Should process 1 MAESTRO clip
  per ~5s on M4.
- [ ] Render a seeded subset: 500 MAESTRO clips × 3 random corruption profiles
  = 1500 synthetic practice clips. Cache to `data/raw/maestro_corrupted/`.
- [ ] Extract MuQ embeddings on the corrupted clips and cache to
  `data/embeddings/maestro_corrupted/`. Uses existing `extract_muq_embeddings.py`.

### P2 — Dataset wrapper + training integration (1 day)

- [ ] Write `PracticeAugmentedDataset`. Takes a base dataset and a
  corruption probability. On `__getitem__`, with probability `p` returns the
  corrupted version instead.
- [ ] Wire into `a1_max_sweep.py`: wrap the T3 MAESTRO dataset only — we do
  *not* want to corrupt PercePiano or T5 because they are already valid signal.

### P3 — Sweep and OOD measurement (1.5 days)

- [ ] Run mix sweep with `corrupt_prob ∈ {0, 0.25, 0.5}`. Report:
  - fold pairwise (should hold or modestly improve with `p=0.25`)
  - OOD pairwise against Chunk B's real practice clips (expected to jump)
  - collapse (should not regress)
- [ ] `p=0.5` is a test of the "too much corruption → model learns practice-
  only" failure mode. Expect fold pairwise to drop here; it's the regression
  signal for the ablation.

### P4 — Document the corruption distribution (0.5 day)

- [ ] Write `docs/model/07-distribution-shift.md` (this plan feeds into that
  concept doc).
- [ ] Log the final corruption-parameter distribution used for the winning
  config into `data/manifests/practice_synthesis_v1.json`.

## Exit Criteria

- OOD pairwise gap drops by ≥5pp vs no-augmentation baseline.
- Fold pairwise does not regress by more than 1pp at the winning corrupt_prob.
- All corruption primitives have unit tests.
- IR bank is versioned (filename includes hash) so reruns are deterministic.

## Risks

- **IR curation is slow and manual.** Mitigation: start with 8 IRs, expand to
  20 only if the 8-IR baseline is promising.
- **FluidSynth soundfont mismatch.** Different soundfonts produce different
  timbres; our corrupted audio must sound plausibly like a real piano under
  phone mic. Mitigation: match MAESTRO's soundfont or use a high-quality free
  one and validate by listening to ~20 random renders before committing to the
  full 1500.
- **Corrupted data leaks into test folds.** `folds.json` is piece-stratified —
  corrupted MAESTRO clips must inherit the same piece IDs so piece-stratified
  CV still holds.
