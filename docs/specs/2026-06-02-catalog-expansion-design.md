# Catalog-Expansion Subsystem Design

**Goal:** Add schema-valid score JSONs for the 11 missing `practice_eval` pieces so all 16 labeled evaluation pieces have a catalog entry, regenerate the fingerprint index (244 → 255), and commit a `slug → piece_id` map that the #21 chroma harness consumes.
**Not in scope:** the #21 chroma feasibility harness itself; production R2/D1 upload (`cmd_upload`); any Rust/WASM; generalizing `parse.py` (it is already source-agnostic).

## Problem

The 244-piece ASAP-derived score catalog (`model/data/scores/*.json`) overlaps only 5 of the 16 labeled `practice_eval` pieces (`model/data/evals/practice_eval/<slug>/`). The other 11 — Für Elise, Clair de Lune, Moonlight mvt1, K545 mvt1, Träumerei, etc. — are absent because they are not in the ASAP dataset, and ASAP is the only ingestion path (`cli.py cmd_parse` → `discover.py`). Consequences:

1. **Blocks #21.** The chroma-identification feasibility harness cannot measure recall on a meaningful sample: 11 of its 16 query pieces have no catalog target to match against.
2. **Strategic gap.** The catalog omits the most common beginner/intermediate repertoire, so *no* piece-ID algorithm can identify those pieces today.

The naïve fix (download 11 MIDIs, drop them in) is unsafe: a wrong edition, a performance-timed MIDI (rubato scrambles bar segmentation → poisons the per-bar trigram index in `fingerprint.py`), or a transposed file silently corrupts the catalog. The DoD minimums (≥20 notes, total_bars≥1, monotonic onsets) certify "not empty," not "correct."

## Solution (from the user's perspective)

A new offline command, `uv run python -m score_library.cli parse-manual --manifest data/manifests/manual_scores.json`, fetches each piece's MIDI from a ranked list of pinned public-domain URLs, parses it through the existing `parse_score_midi`, and runs it through an independent validation gate. The first candidate source that passes the gate wins and is pinned (URL + sha256) into a build-written lockfile. If every candidate for a piece fails, the command halts with a per-candidate failure table — no partial catalog is committed. After all 11 pass, `just fingerprint` regenerates the index over 255 pieces and `eval_piece_map.json` records all 16 `slug → piece_id` pairs.

## Design

**Approach:** Engraved MIDI → reuse `parse.py` UNCHANGED → independent validation gate → `just fingerprint`. Driven by a ranked-URL manifest plus a build-written lockfile (uv.lock semantics: manifest = intent, lockfile = resolution).

**Key decisions and trade-offs:**

- **MIDI over MusicXML.** The engraved PDF is ground truth; every MIDI and MusicXML is a third-party transcription of it, so format does not determine note-correctness. MIDI feeds the entire tested `parse.py` → `fingerprint` path unchanged and carries real velocities (keeps the rerank velocity channel healthy). MusicXML's only structural advantage — authoritative bar boundaries — is recovered by the validation gate instead of a new ingestion module. Trade-off accepted: bars are re-derived from the tick grid, so a *performance-timed* MIDI is the failure mode. The quantization check is a COARSE backstop against grossly-unquantized / fixed-large-offset MIDIs (its median-deviation metric has a 0.125-beat ceiling and cannot reliably catch smooth rubato); the PRIMARY mitigations are key-agreement + bar-count plausibility + human source vetting in Group D.

- **`parse.py` and the JSON schema are untouched.** `parse_score_midi(path, piece_id, composer, title)` has zero ASAP-specific logic; ASAP-specificity lives only in `discover.py`. The new driver feeds it `(path, piece_id, composer, title)` tuples. The quantization check recovers the 16th-note grid from `bar.start_tick` deltas + `time_signature` (no `ticks_per_beat` field needed), so no schema change.

- **Validation is independent of chroma.** The gate must NOT use chroma self-recognition: gating acceptance on "does the chroma matcher rank this score #1 for its own audio" pre-selects scores the matcher likes, which is exactly what #21 measures — it would rig #21's result. The gate uses only chroma-independent signals: Krumhansl key agreement and bar-count plausibility (the primary discriminators), plus pitch range and a coarse metric-quantization backstop (low-power; catches gross/fixed-offset mis-timing, not smooth rubato).

- **Ranked sources + lockfile.** Manifest carries a ranked `sources` list per piece; the build tries them in order and pins the winner's sha256 to the lockfile. This gives reproducibility (content pin defeats a URL silently swapping content) and resilience (auto-fallback) without committing binary MIDI blobs to git.

## Modules

**`score_library/validate.py`** — DEEP.
- Interface: `validate_score(score: ScoreData, expected: ExpectedMeta) -> list[Violation]` (empty list = pass).
- Hides: Krumhansl-Schmuckler key-profile correlation (primary), bar-count tolerance band (primary), DoD-minimum and pitch-range logic, and a coarse 16th-grid recovery + median-deviation backstop (metric is meter-independent: deviation in sixteenth-note units, ceiling 0.5, threshold 0.4 sixteenths; triplet textures yield ~0.333 and pass, a fixed half-sixteenth offset yields 0.5 and fails; it does NOT reliably detect smooth rubato).
- Tested through: `validate_score` only, with synthetic `ScoreData` fixtures. Never internals.

**`score_library/manual.py`** — DEEP.
- Interface: `ingest_manifest(manifest_path, scores_dir, lock_path, fetch_fn=_http_fetch) -> IngestReport`.
- Hides: URL fetch (stdlib `urllib`), sha256 hashing, ranked-source iteration, lockfile read/write, temp-file lifecycle, temp-staging of winning JSONs (moved into `scores_dir` only after ALL pieces resolve, so a HALT is all-or-nothing at the filesystem boundary), wiring `parse_score_midi` → `validate_score`, and the HALT-on-all-fail logic with per-candidate failure tables.
- Tested through: `ingest_manifest` with an injected `fetch_fn` returning fixture bytes (DI at the public boundary — `fetch` is an external dependency, not an internal collaborator). Never internals.

**`score_library/catalog_coverage.py`** — DEEP-ish (the verification harness).
- Interface: `check_coverage(scores_dir: Path, mapping: dict[str, str]) -> list[str]` (empty = all 16 present, non-trivial, monotonic).
- Hides: per-piece existence + DoD-minimum checks (the brainstorm's DoD logic).
- Tested through: `check_coverage` with tmp_path fixtures.

**`score_library/cli.py`** — shallow glue (matches existing `cmd_*` pattern): `cmd_parse_manual` → `ingest_manifest`. Acceptable as established-pattern glue, not a new abstraction.

## Verification Architecture

- **Canonical success state:** 11 new `data/scores/{piece_id}.json` that parse to schema-valid `ScoreData` and pass `validate_score`; `data/fingerprints/ngram_index.json` regenerated over 255 pieces; `data/evals/piece_id/eval_piece_map.json` with 16 entries all resolving to existing files.
- **Automated check:** `check_coverage` against the canonical 16-entry mapping (exposed as `just catalog-verify`), plus `validate.py` and `manual.py` unit suites.
- **Harness (Task Group 0):** `catalog_coverage.py` is buildable before the feature — its logic is TDD'd against tmp_path fixtures, then run against the real catalog at the end. It is RED until the 11 scores land and goes GREEN as the acceptance gate. No test asserts chroma rank.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/score_library/validate.py` | Validation gate (`validate_score`, `ExpectedMeta`, `Violation`) | New |
| `model/src/score_library/manual.py` | Ranked-source ingestion driver (`ingest_manifest`) | New |
| `model/src/score_library/catalog_coverage.py` | Coverage/quality harness (`check_coverage`) | New |
| `model/src/score_library/cli.py` | Add `parse-manual` subcommand | Modify |
| `model/data/manifests/manual_scores.json` | 11-piece ranked-URL manifest (intent) | New |
| `model/data/manifests/manual_scores.lock.json` | Build-written resolution (URL + sha256) | New |
| `model/data/evals/piece_id/eval_piece_map.json` | 16-entry `slug → piece_id` (the #21 contract) | New |
| `model/data/scores/{11 piece_id}.json` | Ingested score JSONs | New |
| `model/data/fingerprints/ngram_index.json` + `rerank_features.json` | Regenerated over 255 pieces | Modify |
| `model/tests/score_library/test_validate.py` | Tests for the gate | New |
| `model/tests/score_library/test_manual.py` | Tests for the driver | New |
| `model/tests/score_library/test_catalog_coverage.py` | Tests for the harness | New |
| `justfile` | `catalog-verify` recipe | Modify |

## Open Questions

- Q: Do reputable Mutopia/MuseScore-PD MIDIs exist for all 11 pieces with metric (non-performance) timing? Default: the ranked-`sources` + auto-fallback + HALT design surfaces any piece that has no passing source during build, at which point a replacement URL is sourced; the build does not commit a partial catalog.
- Q: Bar-count plausibility band width, given some source MIDIs unfold repeats (inflating `total_bars` since `parse.py` tiles bars across `total_ticks`). Default: a wide band `[0.7×expected, 2.2×expected]` — catches truncation/wrong-piece while tolerating repeat-unfolding; key-agreement carries the real discrimination (quantization is only a coarse backstop). A `bar_count` violation during ingestion means "re-check `expected_bars` against the actual source," not "reject the source" — repeats can inflate `total_bars` past `2.2×` on a correct MIDI.
