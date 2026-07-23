# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Baseline

Baseline suite (`cd model && uv run python -m pytest tests/`): 748 passed, 47 failed, 31 skipped, 2 xfailed.
All 47 failures are pre-existing data-availability failures in this fresh worktree (missing gitignored
`model/data/raw/asap-dataset/` and `model/data/weights/aria-medium-base/` -- R2-offloaded datasets,
the documented worktree gotcha). None relate to audio_teacher. Accepted as baseline; final suite must
show the identical failure set plus all audio_teacher tests green.
