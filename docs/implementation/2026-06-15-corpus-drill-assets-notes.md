# Implementation Notes — Corpus Drill Renderable Assets (#46)

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline note
8 pre-existing failures in `model/tests/exercise_corpus/test_transforms.py` are ENVIRONMENTAL,
not regressions: they load gitignored `data/midi/exercise_primitives/*.mid` which are absent in
any clean checkout. Confirmed the same `.mid` absence on the issue-46 base tree. Our tasks use the
committed `.xml` primitives instead. These 8 are out of scope for this build.
