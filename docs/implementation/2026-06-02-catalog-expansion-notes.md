# Implementation Notes — Catalog Expansion

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Pre-build environment findings (controller)

- `model/data/scores/*.json` is GITIGNORED (`model/data/.gitignore: /scores/*.json`, only `titles.json` tracked). The 244 existing score JSONs are NOT in git; they live only in the developer's main working tree and are regenerated locally. The fresh worktree therefore starts with ZERO ingested score JSONs in `model/data/scores/`.
- `model/data/fingerprints/ngram_index.json` and `rerank_features.json` ARE tracked.
- Consequence for Group D: `git add model/data/scores/` (D3 commit) is a no-op for ignored files. The committable artifacts of Group D are the manifest, lockfile, eval_piece_map.json, fingerprints, and justfile. The 11 score JSONs remain gitignored by design (consistent with the existing 244).
- Consequence for fingerprint regen: a correct 255-piece fingerprint requires the 244 existing scores to be present in the worktree's `model/data/scores/`. They are not present in a fresh worktree. This is a Group-D execution dependency surfaced for the executor.
- Baseline: 100 passed, 24 skipped in score_library suite.
- Verified dependency surfaces: schema fields (ScoreNote/Bar/ScoreData), Scores.root, DATA_ROOT, parse_score_midi(midi_path, piece_id, composer, title) all match the plan.
