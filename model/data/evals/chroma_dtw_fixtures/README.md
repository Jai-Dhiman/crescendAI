# chroma_dtw_fixtures

Tiny committed fixture (3 chunks) used by the smoke test to verify the verify-CLI contract end-to-end without depending on MAESTRO/skill_eval/practice_eval downloads.

- fix_001: gold-truth chunk (synthetic, frame 0)
- fix_002: amateur-style chunk (no ms-truth)
- fix_003: silence chunk (zeros)

Real corpora live under model/data/evals/{skill_eval,practice_eval} and model/data/raw/asap; the harness reaches them via the chunk_sampler module, not this directory.
