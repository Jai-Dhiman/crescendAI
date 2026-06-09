# ADAPT Layer: Passage-Aware Transform Parameters Design

**Goal:** `build_briefing` computes transpose interval and excerpt span from the student's actual diagnosed passage instead of hardcoded constants, so every prescribed drill is automatically transposed into the student's key and sized to their specific bar range.

**Not in scope:**
- Mode matching (major/minor distinction) — tonic shift only
- Difficulty or technique discrimination (#42/#43)
- API wiring or ExerciseArtifact changes (#29)
- Dynamic selection among multiple simultaneous transforms (transform remains singular)

---

## Problem

`build_briefing` currently emits two hardcoded values regardless of the student's actual piece:

1. `_EXCERPT_BARS = (1, 4)` — every excerpt-based exercise always covers bars 1–4 of the primitive, ignoring how many bars the student actually struggled over (`diagnosis.bar_range`).
2. No transposition — every drill is delivered in C major even when the student is playing in Eb, so the "drill the fingering pattern in your piece's key" affordance is completely absent.

The `ExerciseBriefing` struct also has no field to communicate the key shift downstream, so even a downstream renderer has no way to present "play this 3 semitones up."

---

## Solution (from the user's perspective)

After this change:

- A student struggling with timing in bars 5–12 of a Chopin nocturne in Eb minor receives: an 8-bar Hanon drill segment (matching the bar count), transposed +3 semitones, with instruction text that says "...in the key of Eb."
- A student struggling with phrasing in bars 3–6 of a Bach prelude in C major receives: a 4-bar Burgmuller drill with transpose_semitones=0 (no shift needed), and the instruction says nothing about a key (C is the default).
- If the piece's key signature is unavailable, `transpose_semitones` and `target_key` are `None`; the drill is still emitted without transposition, and no error is raised (the key being absent is legitimately possible for piece IDs without a score JSON on disk).

---

## Design

### Option chosen: separate `transpose_semitones` field (Option A)

The `transform` / `transform_params` pair remains singular and represents the score-domain transform applied to the exercise primitive (excerpt / tempo / None). Transposition is orthogonal — it applies universally regardless of which dimension transform is selected — so it lives in its own fields `transpose_semitones: int | None` and `target_key: str | None`. A downstream renderer that wants to call `transforms.transpose()` reads `transpose_semitones`; one that wants to print "in the key of Eb" reads `target_key`.

### Key resolution pipeline (three pure helpers in `keys.py`)

`parse_key_to_pc(key_signature: str) -> int` maps a human-readable key string ("C major", "Am", "Eb", "C#m", "F#", "Gb") to a pitch-class integer 0–11 by stripping mode suffix and resolving enharmonic equivalents. This is a pure function with no I/O — testable in complete isolation.

`load_passage_key(piece_id: str, scores_dir: Path | None) -> str | None` resolves `scores_dir/<piece_id>.json` and returns its `key_signature` field (which may be null in the JSON). Raises `FileNotFoundError` if the JSON is absent. The `scores_dir` default is anchored to `Path(__file__).resolve().parents[N]` pointing at `model/data/scores/` — never relative to CWD, which changes under `just` recipes.

`transpose_interval(from_pc: int, to_pc: int) -> int` computes nearest-octave transposition in [-5, +6] semitones. Tritone (d=6) resolves to +6 by convention.

### Excerpt span from `bar_range`

When `transform == "excerpt"`, `_transform_params` now computes `length = bar_range[1] - bar_range[0] + 1` and returns `{"start_bar": 1, "end_bar": length}`. This maps the student's bar count onto the primitive starting at bar 1 — the primitive's total length is not consulted here; that bounds check belongs to `transforms.excerpt()` at materialization time.

### TagSet gains a required `key: str` field

Every exercise primitive is in a specific key. The tag layer already knows which corpus primitives exist (validated against the catalog). Adding `key` to `TagSet` makes the key information available to `build_briefing` without an extra file read. All 22 entries in `technique_tags.toml` receive `key = "C"` (all are C major: Hanon 1–20, Czerny op.299 no.1, Burgmuller op.100 no.1). `load_tags` raises `ValueError` if any entry lacks `key`.

### Error handling

| Condition | Behavior |
|-----------|----------|
| `diagnosis.piece_id` score JSON absent | `FileNotFoundError` from `load_passage_key` |
| `key_signature` null in JSON | `transpose_semitones=None`, `target_key=None` — not an error |
| `key_signature` non-null but unparseable | `ValueError` from `parse_key_to_pc` |
| TOML entry lacks `key` | `ValueError` from `load_tags` |
| Transpose pushes pitch off 88-key range | `ValueError` from `transforms.transpose()` (pre-existing, unchanged) |

---

## Modules

### `model/src/exercise_corpus/keys.py` (NEW)

**Interface:**
```python
def parse_key_to_pc(key_signature: str) -> int: ...
def load_passage_key(piece_id: str, scores_dir: Path | None = None) -> str | None: ...
def transpose_interval(from_pc: int, to_pc: int) -> int: ...
```

**Hides:** enharmonic normalization table, path anchoring to `__file__`, JSON parsing, nearest-octave arithmetic.

**Depth verdict: DEEP** — three callers (`build_briefing` only) get a three-function surface hiding 40+ lines of lookup tables and file I/O.

---

### `model/src/exercise_corpus/tags.py` (MODIFY)

**Interface:** `TagSet(dimensions, techniques, key)` and `load_tags(path, known_primitive_ids) -> dict[str, TagSet]`.

**Hides:** TOML parsing, validation of dimension labels, `key` presence enforcement.

**Depth verdict: DEEP** — `key` field addition widens the struct by one required attribute; interface remains narrow.

---

### `model/src/exercise_corpus/briefing.py` (MODIFY)

**Interface:** `build_briefing(diagnosis, tags, history, now, db_path, index, top_k, scores_dir) -> ExerciseBriefing`.

**Hides:** key resolution pipeline (delegates to `keys`), excerpt-length derivation, instruction text assembly, cooldown logic.

**Depth verdict: DEEP** — one function + one dataclass hide the entire prescription logic.

---

## Verification Architecture

- **Canonical success state:** `cd model && uv run pytest tests/exercise_corpus/ -q` passes all tests except the 8 pre-existing `test_transforms.py` failures (gitignored MIDI data absent). Specifically: `test_keys.py` all green, `test_briefing.py` all green (including updated `_tags` helper and new transpose/excerpt assertions), `test_tags.py` all green with updated TOML fixtures.
- **Automated check:** `cd model && uv run pytest tests/exercise_corpus/test_keys.py tests/exercise_corpus/test_briefing.py tests/exercise_corpus/test_tags.py -q`
- **Harness:** Task Group 0 is not needed — pure-function tests for `keys.py` serve as the verification harness. The E2E fixture for `build_briefing` (a committed JSON in `model/tests/exercise_corpus/fixtures/scores/`) is committed in Task 3 before any build_briefing changes.

---

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/src/exercise_corpus/keys.py` | New module: `parse_key_to_pc`, `load_passage_key`, `transpose_interval` | New |
| `model/src/exercise_corpus/tags.py` | Add `key: str` to `TagSet`; enforce presence in `load_tags` | Modify |
| `model/src/exercise_corpus/technique_tags.toml` | Add `key = "C"` to all 22 entries | Modify |
| `model/src/exercise_corpus/briefing.py` | Add `transpose_semitones`/`target_key` to `ExerciseBriefing`; add `scores_dir` param to `build_briefing`; compute transpose and excerpt-from-bar_range | Modify |
| `model/tests/exercise_corpus/test_keys.py` | New test file for `keys.py` helpers | New |
| `model/tests/exercise_corpus/test_briefing.py` | Update `_tags` helper to pass `key=`; add tests for transpose fields and excerpt span | Modify |
| `model/tests/exercise_corpus/test_tags.py` | Update TOML fixture strings to include `key = "C"` | Modify |
| `model/tests/exercise_corpus/fixtures/scores/test_piece_eb.json` | Minimal fixture JSON with `key_signature: "Eb"` for E2E build_briefing test | New |

---

## Open Questions

- Q: Should `load_passage_key` accept `scores_dir=None` and fall back to the anchored default, or should callers always pass an explicit path?
  Default: accept `None` and resolve to the anchored default when `None` is passed — keeps the API clean for production callers while remaining injectable for tests.
