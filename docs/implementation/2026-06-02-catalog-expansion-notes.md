# Implementation Notes — Catalog Expansion

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Pre-build environment findings (controller)

- `model/data/scores/*.json` is GITIGNORED (`model/data/.gitignore: /scores/*.json`, only `titles.json` tracked). The 244 existing score JSONs are NOT in git; they live only in the developer's main working tree and are regenerated locally. The fresh worktree therefore starts with ZERO ingested score JSONs in `model/data/scores/`.
- `model/data/fingerprints/ngram_index.json` and `rerank_features.json` ARE tracked.
- Consequence for Group D: `git add model/data/scores/` (D3 commit) is a no-op for ignored files. The committable artifacts of Group D are the manifest, lockfile, eval_piece_map.json, fingerprints, and justfile. The 11 score JSONs remain gitignored by design (consistent with the existing 244).
- Consequence for fingerprint regen: a correct 255-piece fingerprint requires the 244 existing scores to be present in the worktree's `model/data/scores/`. They are not present in a fresh worktree. This is a Group-D execution dependency surfaced for the executor.
- Baseline: 100 passed, 24 skipped in score_library suite.
- Verified dependency surfaces: schema fields (ScoreNote/Bar/ScoreData), Scores.root, DATA_ROOT, parse_score_midi(midi_path, piece_id, composer, title) all match the plan.

## Build-controller deviation: no subagent-dispatch tool available

The build skill assumes a `Task` tool to dispatch fresh per-task implementer/reviewer subagents. This environment exposes no such dispatcher (only the TaskList task-tracking tools). The controller therefore executed each task directly while strictly enforcing the build discipline per task: (1) write the failing test, (2) run and watch it FAIL for the right reason, (3) implement exactly the plan's code, (4) run and watch it PASS, (5) commit with the plan's exact message. After each task, a spec-compliance check (re-read diff vs plan requirements) and a code-quality self-review were performed before moving on. Sonnet-4.6-subagent convention is moot since no subagents were spawned.

## Task G0-1 (catalog_coverage.py) — DONE
Implemented exactly as planned. 6 tests pass. CANONICAL_MAP = 16 entries. No deviations.

## Group A (validate.py A1-A5) — DONE
All 5 checks implemented exactly as planned: min_notes/total_bars/monotonic_onsets, pitch_range, bar_count, quantization (16th-grid via start_tick deltas), key_agreement (Krumhansl-Schmuckler, enharmonic tonic map). 19 tests pass. No chroma. The +60-tick fixed-offset quant fixture yields median 0.125 > 0.10 (violation); clean grid 0.0 (pass) — matches challenge re-review. No deviations.

## Group B (manual.py B1-B4) — DONE
ingest_manifest with temp-staging all-or-nothing atomicity (CONCERN 2 fix): staged JSONs only moved into scores_dir after ALL pieces resolve; lockfile written last; HALT leaves scores_dir + lockfile untouched. B2-B4 are behavior-locking tests over B1's impl (passed on first run, as the plan predicted; watch-it-fail satisfied at B1 module-absence). 4 tests pass. Semgrep flagged urllib dynamic-URL (WARNING) on _http_fetch — accepted: plan mandates stdlib urllib (requests/httpx not deps), URLs from pinned manifest, line carries noqa S310. No deviations.
