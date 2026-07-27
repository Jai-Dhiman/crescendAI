# Model and research scope

`CLAUDE.md` is the canonical Claude Code context for this scope. Keep the
sibling `AGENTS.md` byte-identical as a compatibility mirror. Read the
repository-root `CLAUDE.md` first.

- Use `uv` for every Python environment and command.
- Track experiments with Trackio. Record the split, seed, dataset provenance,
  gate criteria, and artifact location before interpreting a result.
- PercePiano is evaluation-only. Never train or gate on it.
- Create split manifests before training and share them across contenders.
- Report synthetic-rendered and real-audio populations separately; never pool
  them into one headline result.
- Preserve ex-ante gate criteria and negative results. Architecture choices,
  ratchet promotions, research conclusions, and paid/cloud runs are human-lit.
- Use explicit exceptions for missing data or failed transcription. Never
  substitute clean MIDI, synthetic audio, cached output, or a weaker model
  silently.

## Data and tools

- Paths are centralized in `src/paths.py`.
- Offloaded-data behavior is defined by
  `data/manifests/r2_offload.json`; follow the exact rehydration error from
  `paths.ensure_local()`.
- Use `partitura` for MusicXML/MIDI score parsing and conversion; do not add
  `music21`.
- Keep notebooks as thin orchestration over importable `src/` modules.
- Common deterministic verification: `just test-model`. Run heavier evaluations
  only when their required real data and truth artifacts are present.

## Routing

- Research chronology and current decisions: `docs/model/`.
- Data provenance: `docs/model/01-data.md`.
- Encoder status: `docs/model/03-encoders.md`.
- Distribution shift and uncertainty:
  `docs/model/07-distribution-shift.md`,
  `docs/model/08-uncertainty-and-diagnostics.md`.
- Score library: `docs/model/10-score-library-catalog.md`.
- Claim-verifier conventions and active gates:
  `docs/model/claim-verifier-signed-d-conventions.md`.

Formatting, lint, unit tests, and deterministic inventory may run dark. Dataset
selection, research-gate interpretation, model promotion, and remote compute
spend remain human-lit.
