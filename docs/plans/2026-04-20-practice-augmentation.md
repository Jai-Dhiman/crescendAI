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

        See `docs/model/08-uncertainty-and-diagnostics.md` § Baseline snapshot (pre-Wave 1).


## Exit Criteria

- OOD pairwise gap drops by ≥5pp vs no-augmentation baseline.
- Fold pairwise does not regress by more than 1pp at the winning corrupt_prob.
- All corruption primitives have unit tests.
- IR bank is versioned (filename includes hash) so reruns are deterministic.

## Execution Timing

**Infrastructure:** merged (`practice_synthesis.py`, `render_corrupted_audio.py`,
`PracticeAugmentedDataset`, unit tests, `corrupt_prob` sweep dimension).

**Experimental runs are gated on T5 labeling completion AND IR curation.** Two
gates, not one:

1. The `data/raw/room_irs/` bank is empty — 8–20 real practice-room IRs must be
   curated before `render_corrupted_audio.py` produces usable output. This can
   start in parallel with T5 labeling since it's a data-collection task, not a
   model-training task.
2. The `corrupt_prob ∈ {0.0, 0.25, 0.5}` sweep and OOD gap measurement happen
   at training time, after T5 labeling completes and the PercePiano mix
   winning ratio is locked (see `2026-04-20-percepiano-anchor-emphasis.md`).

The OOD harness baseline (no augmentation) should be captured first — that
number is the reference point for the ≥5pp gap-reduction exit criterion.

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
