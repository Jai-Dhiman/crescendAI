# Implementation Notes — Corpus Drill Renderable Assets (#46)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline note
8 pre-existing failures in `model/tests/exercise_corpus/test_transforms.py` are ENVIRONMENTAL,
not regressions: they load gitignored `data/midi/exercise_primitives/*.mid` which are absent in
any clean checkout. Confirmed the same `.mid` absence on the issue-46 base tree. Our tasks use the
committed `.xml` primitives instead. These 8 are out of scope for this build.

## Task 1: Verovio↔partitura faithful-shift oracle
- Created test_render_assets_oracle.py exactly per plan; added verovio>=4.0.0 to model dev deps (resolved 6.2.1).
- Result: 44 passed, 2 xfailed (burgmuller_001, czerny_001 cross-engine baseline). Commit ea49d080.
- No production code touched. Spec review PASS, code review APPROVED (2 MINOR cosmetic notes only:
  no parts-guard on load_score; string-typed semitone API is undocumented inline — both non-blocking).

## Task 2: off-keyboard rejection symmetry
- Appended _max_in_range_up helper + test_partitura_rejects_off_keyboard_transpose to the oracle file.
- Watch-it-fail proven via bogus-match (real msg: "transpose by 25 puts pitch 109 outside piano range [21, 108]").
- Result: 45 passed, 2 xfailed. Commit 4c6d9d50. Group 0 GREEN — A/B/C unblocked.
- Code review MINOR notes (non-blocking): no highest<=108 guard in helper; test name says "partitura" but reject lives in transforms.py.
