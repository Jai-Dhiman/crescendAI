# Dimension-Tag Retrieval for the Exercise Matcher

**Goal:** Make the exercise matcher retrieve candidates by the diagnosed weakness *dimension* (via curated technique tags) instead of by cosine similarity to a score-passage embedding — so the matched exercise is dimension-relevant, and dimensions the corpus cannot serve raise an explicit error instead of returning an off-dimension drill.
**Not in scope:** The score-MIDI → query-embedding bridge (rejected — see Problem); embedding-based within-bucket ranking (deferred until #17 lands corpus breadth + a real query signal); the api-side pgvector index and real student-output wiring (#29); any change outside `model/src/exercise_corpus/`.

## Problem
Slices B/C/D shipped on `issue-36-exercise-matcher-transforms-briefing`. `build_briefing` (`model/src/exercise_corpus/briefing.py`) requires a 512-dim Aria `query_embedding`. The proposed way to produce it was a "score-MIDI → query-embedding bridge": take the diagnosis's `piece_id` + `bar_range`, slice the score to those bars, Aria-embed the slice, and feed `build_briefing`.

That bridge is wrong by construction:
- A `DiagnosisArtifact` (`apps/api/src/harness/artifacts/diagnosis.ts`) is a claim about a **performance-quality dimension** (timing / dynamics / pedaling / articulation / phrasing / interpretation). A *score* MIDI encodes the written notes (pitch + notated rhythm + flat velocity); it is identical regardless of how the student played, so it carries **zero** information about the diagnosed dimension.
- `aria.embedding.get_global_embedding_from_midi` is a contrastive **content** embedding ("which piece is this"), not a quality encoder. Embedding a score slice retrieves exercises with similar *texture/key*, on an axis orthogonal to the weakness.
- `build_briefing` already routes `exercise_type`/`transform`/`instruction` on `diagnosis.dimension` (the `_DIMENSION_PLAN` dict). The retrieved match only supplies `matched_title` and which primitive gets materialized — so the cosine query was barely load-bearing, and the corpus is Hanon-degenerate (#17: 20/22 in a 0.006–0.017 cosine ball) so it cannot discriminate anyway.

## Solution (from the user's perspective)
A diagnosis with `dimension="timing"` returns a ranked list of *timing-relevant* exercises (Hanon finger-independence drills, Czerny velocity), never the one lyrical Burgmüller study. A diagnosis with `dimension="pedaling"` raises `NoPrimitiveForDimensionError` — because the corpus genuinely contains no pedaling exercise — instead of silently prescribing a finger drill. The raised error *is* the #17 corpus-breadth signal, made legible.

## Design
Route retrieval on the one signal we trust — the categorical `dimension` — via a hand-authored, version-controlled tag table.

- **`technique_tags.toml`** (sibling to `sources.toml`) maps each primitive_id to `{dimensions, techniques}`. Conservative authoring: only high-confidence claims. Hanon 1–20 + Czerny → `{articulation, timing}`; Burgmüller → `{phrasing, interpretation}`. `dynamics` and `pedaling` are intentionally untagged → they raise.
- **`load_tags`** reads the TOML and validates it against the catalog: unknown dimension label or a tag referencing a primitive_id absent from the catalog → `ValueError` (keeps tags ↔ catalog in lockstep, catches editorial drift early).
- **`match_by_dimension`** filters catalog rows to those whose tagged dimensions include the requested dimension, raises `NoPrimitiveForDimensionError` on an empty bucket (no off-dimension fallback), and ranks **deterministically** by `(source_exercise_number, primitive_id)`. There is no cosine query in tag mode, so `Match.score` is `nan` (documented sentinel).
- **`build_briefing`** drops `query_embedding`, takes a `tags` map, and calls `match_by_dimension(diagnosis.dimension, tags, ...)`. Everything downstream (`_DIMENSION_PLAN`, severity validation, `should_prescribe` cooldown, slice-C transform, templates, `record_prescription`) is unchanged.

**Trade-offs chosen.** (1) Tags live in a tracked TOML, not a column on the regenerated SQLite catalog — the catalog is machine-regenerated and would silently drop hand-authored tags on every rebuild. (2) The existing cosine `match_exercises` is kept intact but no longer the entry point: it is the validated within-bucket *ranker* for when #17 supplies breadth + a real query signal. Not deleted — a documented deferred tiebreaker. (3) `build_briefing`'s `query_embedding` param is removed outright (not kept for back-compat): pre-beta, the only callers are in-repo tests.

## Modules
**`tags.py`** — editorial tag layer.
- Interface: `TagSet(dimensions: frozenset[str], techniques: frozenset[str])`; `load_tags(path, known_primitive_ids) -> dict[str, TagSet]`.
- Hides: TOML parsing, dimension-vocabulary validation, tags↔catalog lockstep validation.
- Tested through: `load_tags` on tmp_path TOML fixtures (happy path + two rejection paths).
- Depth: DEEP — one call returns a validated map; all parsing/validation hidden.

**`match.py::match_by_dimension`** — dimension-filtered deterministic retrieval.
- Interface: `match_by_dimension(dimension, tags, db_path=None, index=None, top_k=5) -> list[Match]`; raises `NoPrimitiveForDimensionError`.
- Hides: catalog load, dimension filtering, empty-bucket exception, deterministic ranking, `nan`-score construction.
- Tested through: synthetic catalogs + tag dicts via `match_by_dimension`.
- Depth: DEEP — trivial interface (dimension + tags → ranked matches) absorbing filter + rank + corpus-gap exception.

**`briefing.py::build_briefing`** (rewire) — unchanged interface depth; the matcher dependency swaps from cosine to dimension-tag. Tested through `build_briefing` end-to-end.

## Verification Architecture
- **Canonical success state:** `build_briefing` returns an `ExerciseBriefing` whose `matched_primitive_id` is tagged for the diagnosis's dimension; an untagged dimension (`pedaling`, `dynamics`) raises `NoPrimitiveForDimensionError`; the full B→C→D loop still produces a valid transformed variant for a tagged dimension.
- **Automated check:** `cd model && uv run pytest tests/exercise_corpus/ -q` — all green.
- **Harness (Task Group C):** an **offline integration test** builds a synthetic catalog using the 22 *real* primitive_ids, loads the *shipped* `technique_tags.toml`, and asserts the real editorial buckets (`timing` = Hanon ∪ Czerny excluding Burgmüller; `phrasing` = Burgmüller only; `pedaling`/`dynamics` raise). This validates the actual shipped tags without depending on the gitignored real catalog DB. The `REAL_DB`-backed E2E test remains skip-if-absent (the DB is gitignored / not present in a fresh worktree).

## File Changes
| File | Change | Type |
|------|--------|------|
| `model/src/exercise_corpus/tags.py` | `TagSet` + `load_tags` | New |
| `model/src/exercise_corpus/technique_tags.toml` | conservative tag table for the 22 primitives | New |
| `model/src/exercise_corpus/match.py` | add `NoPrimitiveForDimensionError` + `match_by_dimension`; keep `match_exercises` | Modify |
| `model/src/exercise_corpus/briefing.py` | `build_briefing`: drop `query_embedding`, add `tags`, call `match_by_dimension` | Modify |
| `model/tests/exercise_corpus/test_tags.py` | tests for `load_tags` | New |
| `model/tests/exercise_corpus/test_match.py` | add `match_by_dimension` tests + offline shipped-TOML integration test | Modify |
| `model/tests/exercise_corpus/test_briefing.py` | migrate call sites to `tags=`; add dimension-retrieval + raises tests | Modify |

## Open Questions
- Q: Should `load_tags` reject a primitive with an empty `dimensions` list? Default: allow it (empty bucket → simply never matched); no extra rule.
- Q: Should `Match.score` in tag mode be `nan` or `0.0`? Default: `nan` (unambiguous "no cosine computed"; `0.0` is a valid orthogonal cosine).
