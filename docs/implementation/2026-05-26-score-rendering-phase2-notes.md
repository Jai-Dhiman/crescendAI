# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## BUILD HALTED — Task 0 Spike Gate Failure

**Date:** 2026-05-26

**Reason:** The Task 0 hard gate (Ballade load() under 200ms) FAILED. Actual measured timings:

- **Ballade** (56.7KB, 16 pages): 3803ms total
  - `loadZipDataBuffer`: 1908ms
  - `renderToSVG` x16 pages: 1152ms
  - `renderToTimemap`: 743ms
- **Nocturne** (20.2KB, 3 pages): 550ms total

These numbers make eager IR build at load time non-viable within the 200ms constraint. The spec's 25ms extrapolation was incorrect — the actual bottleneck is `loadZipDataBuffer` (the MXL parse + layout pass), not the SVG rendering.

**Also discovered during implementation:**

1. **Verovio SVG format mismatch**: `<use>` elements use `transform="translate(x, y)"` not `x="..." y="..."` attributes. The plan's `extractNotePositions` regex needed to be rewritten with a linear scan approach.

2. **Timemap field name mismatch**: The plan specified `{ includeNotes: true }` and expected `{ notes: string[], qon: number }` per entry. Actual Verovio API: `{ includeMeasures: true }` produces `{ on: string[], qstamp: number }`. Both `notes` and `qon` are wrong field names.

3. **SVG `width` attribute format**: Verovio emits `width="960px"` (with `px` suffix), not `width="960"`. The regression assertion `/width="(\d+)"/` would not match and needs `/width="([\d.]+)px"/` or similar.

**Tasks completed before halt:**
- Task 0: Test written and confirmed failing correctly (entry.ir undefined) ✓
- Task 1: `parseScoreIR` + types implemented and tested ✓ (needed fixes for Verovio's actual SVG format)
- Task 2: `processGetPageRequest` export + CacheEntry shape ✓
- Task 3: `loadPiece` partially updated — halted at spike gate failure

**Action required:** Revise spec toward lazy-IR strategy before resuming build.

Options:
1. **Lazy IR**: Build IR on first `get_ir` request, not during `load`. Load only does loadZipDataBuffer + buildMeasureIndex. Pages and IR built on demand.
2. **Raise threshold**: Accept 4-6 seconds as load time, hide behind a progress indicator. Change 200ms gate to 10000ms.
3. **Background IR**: Resolve `load()` immediately after loadZipDataBuffer+buildMeasureIndex, then build IR in the background.

## Task 1: IR Types and parseScoreIR pure function

**Deviations from plan:**
- Fixed ReDoS semgrep finding: replaced `new RegExp(name)` with hardcoded `extractTranslateXY` / fallback x/y attribute parsing.
- Rewrote SVG extraction from regex-based to linear tag-scan to handle nested SVG structure.
- `extractTranslateXY` parses `transform="translate(x,y)"` (Verovio's format) with fallback to `x="..."` (test fixtures).
- `extractMeasureNoteMap` uses linear scan (not nested `</g>` regex) to correctly attribute notes to measures.

## Task 2: Worker get_page and get_ir message handlers + updated CacheEntry

**Deviations from plan:**
- Made `ir` and `pageSvgs` optional during Task 2 transition, then required in Task 3.

## Task 3: Update loadPiece (HALTED at spike gate)

**Timemap field discovery:** Verovio's `renderToTimemap({ includeMeasures: true })` returns entries with `on: string[]` (note ids) and `qstamp: number`. Plan specified `notes: string[]` and `qon: number` — both wrong field names. Corrected in implementation.
