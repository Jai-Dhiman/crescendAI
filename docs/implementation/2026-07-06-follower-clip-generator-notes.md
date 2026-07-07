# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 0: ASAP dataset symlink
Symlinked `model/data/raw/asap-dataset` from primary checkout (resolved via `git rev-parse --git-common-dir`). All three fixtures verified present. Not committed (data/raw gitignored).

## Task 1: Package scaffold
Created `follower_bench` package (docstring-only __init__), empty test __init__, importability test, appended `"src/follower_bench"` to pyproject wheel packages. Pre-edit list matched plan prediction exactly. `uv sync` run to register editable install. Commit ba78f68a.

## Tasks 2-4: asap_alignment.py (Group B)
`load_alignment` + `ClipAlignment` + `AsapAlignmentMissingError` implemented per plan; 4 validation branches (unknown key, not-aligned, insufficient/mismatched beats, missing MIDI files). Real-fixture happy path (Liszt, 92 beats), real not-aligned rejection (Beethoven 16-1), and unknown-key rejection all covered. No deviations.

## Tasks 5-8: segments.py (Group D) + watch-item
`PerfNote`, `Segment` (+`.dst_end`), `apply_segments` (source-range filter + destination affine remap, sorted by onset), `NoteMutation`, `apply_note_mutations` (nearest-onset pitch substitution, clamp [0,127], ValueError on empty notes). All "should already pass" tasks passed as predicted — the generic splice engine handled repeat-duplication and jump-omission with no changes. Added an EXTRA watch-item test (challenge review): `test_apply_segments_jump_splice_omits_notes_in_the_skipped_range` asserts note-level omission of the dropped middle span AND destination-timeline remap of the tail, at the apply_segments level (not just via downstream trajectory tests).

## Tasks 9-13: trajectory.py (Group C) — ONE DEVIATION
`TrueTrajectory` (piecewise-linear `score_position_at` with clamping, `is_monotonic_non_decreasing`), `from_alignment`, `build_trajectory_from_segments` (anchor carryover + discontinuity epsilon for non-contiguous splices).
DEVIATION: `DISCONTINUITY_EPS_S` reduced from the plan's `1e-3` to `1e-6`. The plan claimed Task 13's jump test would pass against `1e-3`, but it did not — the epsilon shifts a discontinuous segment's start anchor forward, stretching the remaining interpolation interval and injecting ~2.5e-4 relative slope error at points far from the jump, exceeding `pytest.approx`'s 1e-6 default rel tol (`score_position_at(1.5)` returned 1.74975 vs expected 1.75). Fix: `1e-6`, which keeps the jump landing an EXACT anchor hit (epsilon-invariant, so the boundary assertions are untouched) and shrinks the slope error to ~2.5e-7. Reviewed and judged SOUND: no package code hard-codes the literal (Task 24 references the symbol), and 1e-6 is still a "sharp near-instant transition" per the module docstring. Test unchanged (it correctly encodes the intended semantic).

## Tasks 14-22: pathologies.py (Group E) + 2 watch-items
`build_plan` dispatches 7 pathology types → `ClipPlan(segments, events, note_mutations)`: clean (identity), repeat (3-seg back-jump), jump (2-seg forward skip), restart (2-seg back to earlier point), hesitation (2 contiguous-source segs + destination pause gap), wrong_note (identity seg + NoteMutation), tempo_swing (lead-in + 4 varying-time_scale sub-segments + tail, fully contiguous). `_pick_two_points` uses disjoint uniform bands (0.15-0.35, 0.55-0.75) guaranteeing ordered interior points. ValueError on unknown type and on zero-duration alignment.
WATCH-ITEM 1 (hesitation mid-pause flatness): Task 18 test additionally builds the trajectory and asserts `score_position_at` at the MIDDLE of the destination-time pause gap equals the paused score position (not just at the gap boundaries) — confirms the ground truth stays flat through the whole pause.
WATCH-ITEM 2 (explicit tempo_swing guard): tempo_swing has its own explicit `if` branch with early return, followed by a final `raise NotImplementedError(...)` — no bare fallthrough. An 8th pathology type added to PATHOLOGY_TYPES without a matching branch would raise loudly rather than silently build a tempo_swing plan.

## Tasks 23-26: clip_generator.py (Group F)
`SynthClip` + `generate(asap_piece, pathology_type, seed)`: load_alignment → seeded `random.Random(seed)` → build_plan → from_alignment → `_load_perf_notes` (partitura.load_performance_midi) → apply_segments → apply_note_mutations → build_trajectory_from_segments → in-memory SynthClip. No MIDI serialization / batch driver (deferred to #112+). Errors propagate uncaught (AsapAlignmentMissingError / FileNotFoundError / ValueError) — the spec's "never fabricate, always raise/skip" contract, regression-tested at the public boundary via the real unaligned Beethoven fixture. Determinism verified (two same-seed calls byte-identical). Task 26 is a parametrized property test across all 6 non-clean pathologies asserting the pre-event region matches the clean ASAP correspondence exactly. `apply_note_mutations(spliced, [])` on non-empty notes with empty mutations correctly does not raise. No deviations.

## Final state
Full suite: `cd model && uv run pytest tests/follower_bench/ -q --no-cov` → 33 passed. 5 modules: asap_alignment, segments, trajectory, pathologies, clip_generator. All 6 task-group reviews (spec + quality) returned PASS/APPROVED with zero CRITICAL/IMPORTANT findings.
