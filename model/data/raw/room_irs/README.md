# Practice-Room Impulse Responses (IRs)

Real room impulse responses used by `AudioAugmentor` to convolve studio-clean
training audio into practice-room acoustics (distribution-shift mitigation,
issue #76 / `docs/model/07-distribution-shift.md` Skew 2).

## Status

Scaffold only — no IRs committed. `AudioAugmentor.room_ir_convolve()` falls back
to a synthetic exponential-decay room IR when this directory is empty, which is
adequate for training but should be replaced with real IRs before any OOD eval
claim. Curate ~15-20 IRs.

## Format

- `.wav` files, any sample rate (resampled to the model rate on load).
- Mono or stereo (stereo is mono-mixed on load).
- Short — a room IR is typically 0.3-2.0 s. Trim silence before the direct
  impulse so pre-delay is realistic.

## How to capture

The goal is the acoustic the model is deployed in, not concert-hall reverb.

1. **Practice rooms (most important).** Record a balloon pop, a hand clap, or a
   sine-sweep deconvolution in your actual practice space(s). Untreated small
   rooms with a real piano are the target distribution.
2. **Bad-phone-mic chains.** A few IRs captured *through a phone mic* fold the
   mic coloration into the IR — valuable, since production input is phone audio.
3. **Contrast rooms (a few).** A bright/live room and a dead/carpeted room widen
   the augmentation coverage so the model does not overfit one room.

## Public-domain / CC sources (verify license before committing)

- OpenAIR (openairlib.net) — many CC-BY room IRs.
- MIT Acoustical Reverberation Scene Statistics Survey IRs.

Record the source + license of each committed IR in a sibling `SOURCES.md` so a
later license audit can verify provenance.

## What this is NOT

- Not concert-hall / cathedral reverb — that is the wrong distribution.
- Not training data on its own — IRs only shape existing audio via convolution.
