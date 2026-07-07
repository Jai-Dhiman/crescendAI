# Synthetic Clip Generator for the Score-Follower Benchmark Design

**Goal:** Given an ASAP piece and a practice-pathology type, deterministically produce a spliced performance note stream with an exact, ground-truth score-position-over-time trajectory, so every other Phase-0 component of EPIC #108 (score follower) has labeled data to consume.
**Not in scope:** Audio rendering (#112), Aria AMT transcription (#112), the trajectory-accuracy metric (#113), the follower harness (#114), the baseline follower itself (#115), the visualizer (#116). No MIDI-file serialization to disk (returns an in-memory note stream — file I/O is added by #112 when it needs a renderable `.mid`). No batch/corpus-generation driver script (the single-clip `generate()` call is the full deliverable; iterating it over many pieces is the harness's job in #114).

## Problem

EPIC #108 wants to build a note-level score follower that stays correct through real practice pathologies — repeats, skips, restarts, hesitations, wrong notes, tempo rubato. Before any follower or metric can be built, there must be performance data where the TRUE score position at every instant is known exactly, including through repeats/jumps/restarts where a naive "performance time is monotonically related to score position" assumption breaks. No such labeled dataset exists in `model/data/`. Real recordings of practice pathologies exist nowhere with ground truth; the only tractable path is to manufacture them from ASAP's already-verified beat-level performance↔score alignment (`model/data/raw/asap-dataset/asap_annotations.json`, 1066 aligned performances), by literally cutting and re-splicing that verified alignment.

## Solution (from the user's perspective)

There is no end user for this issue — the "user" is issue #114 (follower harness) and later. That caller can request:

```python
from follower_bench.clip_generator import generate
clip = generate(asap_piece="Bach/Fugue/bwv_846/Shi05M.mid", pathology_type="repeat", seed=42)
clip.notes              # tuple[PerfNote, ...] — the pathology-injected note stream
clip.true_trajectory    # TrueTrajectory — exact score-position(perf_time) ground truth
clip.event_labels       # tuple[PathologyEvent, ...] — what was injected, and where
```

Same `asap_piece` + `pathology_type` + `seed` always produces the identical clip (deterministic). Requesting a piece with no usable ASAP alignment raises `AsapAlignmentMissingError` — the caller is expected to catch it and skip that piece with a logged reason; nothing is ever fabricated.

## Design

**Truth substrate.** Each ASAP annotation entry provides `performance_beats` (seconds, in the human recording) paired index-for-index with `midi_score_beats` (score-beat position). Zipped together, these are a piecewise-linear map `perf_time -> score_position` that is exact by construction for a clean, unmodified performance. This is `trajectory.from_alignment()`.

**Hard-splice via Segments, not a fabricated pathology model.** Every pathology (repeat/jump/restart/hesitation/tempo_swing) is expressed as an ordered list of `Segment(src_start, src_end, dst_start, time_scale)` objects: "replay this span of the CLEAN performance's time axis, starting at this point in the NEW clip, optionally time-scaled." Applying a segment list to the clean note stream (`segments.apply_segments`) mechanically produces the spliced note stream. Applying the *same* segment list to the clean trajectory's own beat anchors (`trajectory.build_trajectory_from_segments`) mechanically produces the new ground-truth trajectory. Because both are driven off the identical segment list, the two artifacts cannot drift apart — the trajectory is exact by construction, never independently estimated or fabricated.

This is the key design decision: rather than modeling each pathology as a bespoke trajectory-shape generator (e.g., "a repeat produces a sawtooth"), every pathology reduces to picking splice points and combinations of `Segment`s. `wrong_note` is the only pathology that isn't a timeline rearrangement — it's a single `NoteMutation` (pitch substitution near a chosen time) layered on top of an identity segment, since the score position genuinely does not change.

**Where the pathology's tunable "where does it happen" comes from `random.Random(seed)`**, using fixed, non-overlapping fractional bands of the piece's own beat-anchored duration (e.g. one splice point drawn from the first-third of the piece, the other from the back-third) — this guarantees valid splice geometry for any piece with `duration > 0`, with no piece-specific tuning and no failure mode besides a literal zero-duration alignment.

**Alternatives considered:**
- *Synthetic hesitation ramps / probabilistic tempo curves instead of hard splices* — rejected per the approved design: "no synthetic hesitation ramps in v1." A continuous tempo curve is approximated instead as N=4 piecewise-linear `Segment`s with linearly-varying `time_scale`, which stays exact within each piece (no fabricated interpolation of ground truth) while still producing a nonlinear-looking overall time warp.
- *Serializing spliced MIDI to disk inside `generate()`* — rejected as speculative; #112 (audio rendering) is the first consumer that needs a file, and it can call `partitura.save_performance_midi` itself from the returned in-memory notes.
- *A batch corpus-generation script inside #111* — rejected; the design's public interface is single-clip `generate()`, and no downstream issue in this design needs a driver until #114.

## Modules

All new, under `model/src/follower_bench/`.

- **`asap_alignment.py`** — Interface: `load_alignment(asap_piece, asap_root=DEFAULT_ASAP_ROOT, annotations_path=DEFAULT_ANNOTATIONS_PATH) -> ClipAlignment`, exception `AsapAlignmentMissingError`. Hides: JSON parsing of the 42MB annotations file, path resolution (performance MIDI + sibling `midi_score.mid`), and the validity checks (`score_and_performance_aligned`, minimum beat count, matched-length arrays). Tested through: `load_alignment()` only.
- **`trajectory.py`** — Interface: `TrueTrajectory` (frozen dataclass with `anchors` and `.score_position_at(t)` / `.is_monotonic_non_decreasing()`), `from_alignment(alignment) -> TrueTrajectory`, `build_trajectory_from_segments(clean_traj, segments) -> TrueTrajectory`. Hides: piecewise-linear interpolation/clamping, and the anchor-carryover arithmetic that makes spliced trajectories exact. Tested through: the three public functions/methods only.
- **`segments.py`** — Interface: `PerfNote`, `Segment` (with `.dst_end`), `NoteMutation`, `apply_segments(notes, segments) -> list[PerfNote]`, `apply_note_mutations(notes, mutations) -> list[PerfNote]`. Hides: the note-filtering/time-shift/re-sort arithmetic of the splice engine. Tested through: `apply_segments` / `apply_note_mutations` only.
- **`pathologies.py`** — Interface: `PATHOLOGY_TYPES`, `PathologyEvent`, `ClipPlan`, `build_plan(alignment, pathology_type, rng) -> ClipPlan`. Hides: the seeded splice-point selection and the per-pathology-type `Segment`/event construction (all 7 branches, including the tempo-swing sub-segment ramp). Tested through: `build_plan` only.
- **`clip_generator.py`** — Interface: `SynthClip`, `generate(asap_piece, pathology_type, seed) -> SynthClip`. Hides: loading the performance MIDI via `partitura`, and composing the other four modules into one call. Tested through: `generate` only.

## Verification Architecture

- **Canonical success state:** for a given `(asap_piece, pathology_type, seed)`, `generate()` returns a `SynthClip` whose `true_trajectory` has the injected discontinuity/behavior at the correct performance-time, and whose non-injected regions reproduce the exact ASAP beat-alignment values (not an approximation of them).
- **Automated check:** unit + property tests calling `generate()` and the four module functions directly through their public interfaces (no mocking of internals) — described task-by-task in the implementation plan. Two real, already-committed ASAP fixtures anchor the integration-level tests: `Liszt/Transcendental_Etudes/1/LuoJ05M.mid` (40.7s, 92 aligned beats, `score_and_performance_aligned: true` — used for all "happy path" generate() tests) and `Beethoven/Piano_Sonatas/16-1/LuoJ03M.mid` (`score_and_performance_aligned: false` — used for the missing-alignment skip test).
- **Harness:** no separate harness is buildable or needed beyond the test suite itself — the "canonical correct output" for this issue *is* the test suite's assertions (exact trajectory values at exact times), and the ASAP dataset itself is the fixture data (already committed, no acquisition step). No Task Group 0 harness task is added; Task 1 in the plan is ordinary package scaffolding, not a harness.

## File Changes

| File | Change | Type |
|------|--------|------|
| `model/pyproject.toml` | add `src/follower_bench` to `[tool.hatch.build.targets.wheel].packages` | Modify |
| `model/src/follower_bench/__init__.py` | package marker | New |
| `model/src/follower_bench/asap_alignment.py` | `ClipAlignment`, `load_alignment`, `AsapAlignmentMissingError` | New |
| `model/src/follower_bench/trajectory.py` | `TrueTrajectory`, `from_alignment`, `build_trajectory_from_segments` | New |
| `model/src/follower_bench/segments.py` | `PerfNote`, `Segment`, `NoteMutation`, `apply_segments`, `apply_note_mutations` | New |
| `model/src/follower_bench/pathologies.py` | `PATHOLOGY_TYPES`, `PathologyEvent`, `ClipPlan`, `build_plan` | New |
| `model/src/follower_bench/clip_generator.py` | `SynthClip`, `generate` | New |
| `model/tests/follower_bench/test_package.py` | package import smoke test | New |
| `model/tests/follower_bench/test_asap_alignment.py` | `load_alignment` behavior tests | New |
| `model/tests/follower_bench/test_trajectory.py` | `TrueTrajectory` / `from_alignment` / `build_trajectory_from_segments` tests | New |
| `model/tests/follower_bench/test_segments.py` | `apply_segments` / `apply_note_mutations` tests | New |
| `model/tests/follower_bench/test_pathologies.py` | `build_plan` tests, all 7 pathology types + error cases | New |
| `model/tests/follower_bench/test_clip_generator.py` | `generate()` end-to-end + property test | New |

## Open Questions

- Q: Should `wrong_note` mutate exactly one note, or a random span of several?  Default: exactly one note (nearest to a seeded time point) in v1 — simplest, still exercises "local-error tolerance" for the follower; multi-note corruption is a trivial future extension (loop the same `NoteMutation` construction) if #114/#115 need it.
- Q: Should segment note-selection be half-open `[src_start, src_end)` or closed at both ends?  Default: half-open, to prevent a note landing exactly on a shared boundary between two adjacent segments (e.g. `repeat`'s seg1/seg3 boundary at `y`) from being duplicated.
