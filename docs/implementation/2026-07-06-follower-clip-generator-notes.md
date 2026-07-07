# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 0: ASAP dataset symlink
Symlinked `model/data/raw/asap-dataset` from primary checkout (resolved via `git rev-parse --git-common-dir`). All three fixtures verified present. Not committed (data/raw gitignored).

## Task 1: Package scaffold
Created `follower_bench` package (docstring-only __init__), empty test __init__, importability test, appended `"src/follower_bench"` to pyproject wheel packages. Pre-edit list matched plan prediction exactly. `uv sync` run to register editable install. Commit ba78f68a.
