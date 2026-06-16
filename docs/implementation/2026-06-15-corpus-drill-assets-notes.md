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

## Task 3: build() produces valid MXL assets
- Created build_render_assets.py (build()) + test_build_render_assets.py per plan. DOCTYPE-strip + wrap_as_mxl_zip only; idempotent; fail-loud naming the file on partitura load failure. Commit ef51e6ea. 1 passed.
- Spec PASS. Code review APPROVED (3 MINOR non-blocking): double DOCTYPE-strip (harmless, build pre-strips for idempotency probe + wrap re-strips); hard-coded ==22 in test (redundant w/ len(produced)==len(xml_files)); imports private _strip_doctype across package boundary (coupling smell). None fixed (MINOR).

## Task 6: loadPiece transpose param
- loadPiece gained transpose?: number (4th param) + applyOpts(t) helper (VEROVIO_OPTS then transpose only if defined && !=0); 3 loadPiece-internal toolkit sites use applyOpts. Commit 7692b0f5.
- DEVIATION (justified): plan's stripIds (single id="..." regex) was too narrow — Verovio randomizes IDs in 4 places (id attrs, xlink:href suffix, <style> CSS, class id- tokens). Implementer wrote a multi-step normalizer so the transpose:0==omitted structural lock is real.
- REVIEW FIX (commit 95cd63cf): code review found the class id- strip was global (over-strip risk masking real engraving diffs). Scoped it to inside class="..." values via capture-group replace. Re-review APPROVED.
- Tests: 2/2 transpose + 16/16 existing worker tests green. Test 1 (transpose:2 != transpose:0) is a weak predicate alone (raw SVGs always differ via random IDs); test 2 is the real no-op lock — both plan + challenge accepted test 1.

## Task 4: idempotency + fail-loud + 22 committed assets
- Appended test_build_is_idempotent + test_build_raises_naming_bad_xml; generated + committed 22 .mxl (real ZIPs, not gitignored). Commit b33e1553 (23 files). build_render_assets.py unchanged from Task 3.
- Watch-it-fail proven for bad-xml guard (temp revert → DID NOT RAISE). Plan's <not-musicxml> junk sufficed (no fallback needed).
- Spec PASS. Code review APPROVED (1 MINOR: local `import pytest` inside test fn — cosmetic, not fixed).

## Task 7: ScoreRenderer.load composite cache key
- Wired transpose through worker load message + ScoreRenderer. Composite key ${pieceId}:${transpose ?? 0} on toolkitCache (worker), sentPieceIds + irCache + getIR (renderer); bytesCache/ensureBytes stay bare-pieceId (bytes transpose-independent). Commit a2c122a4.
- Caller audit: all 6 scoreRenderer.load callers + getIR caller pass no transpose → resolve to :0 (byte-identical back-compat). Verified.
- REVIEW FIX (commit c9778537): code review flagged CRITICAL — worker-error cleanup paths (onmessage-error, onerror) deleted BARE pieceId from sentPieceIds (now composite-keyed) → claimed stale-key blocks retry. HONEST FINDING: bug was SHADOWED — load()'s own catch already clears the composite key on both error paths, so it never manifested. Fix threads sentKey?:string into PendingRequest so worker-error paths delete the correct composite key (defense-in-depth, internally correct independent of load's catch), and removed the now-dead sentPieceIds.has(pieceId) clause in ensureBytes (always false post-refactor; bytesCache.has dedupes). Added a regression test (error-then-retry re-sends bytes). Re-review APPROVED.
- All 24 web score tests green.

## Task 5: just build-exercise-assets + seed-exercise-assets recipes
- Appended both recipes to Justfile (NOTE: file is tracked as `Justfile` capital-J on this macOS repo). Commit 42c3bb25.
- build-exercise-assets prints "built 22 assets"; rebuild is byte-deterministic (no .mxl git changes). seed-exercise-assets succeeded live (wrangler available, 22 seeded to local R2 scores/v1/{id}.mxl).
- Spec PASS, code review APPROVED (1 MINOR: reviewer suspected a comment source-path mismatch, but the comment's "model/data/scores/exercise_primitives/*.xml" EXACTLY matches build()'s _DEFAULT_XML_DIR — comment is correct, no fix needed). nullglob + zero-count guard is a correctness improvement over the seed-scores recipe it mirrors; correct musicxml+zip content-type.
