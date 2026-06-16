# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: Surface prescribed_exercise in eval_context
- Added `receivedRealInferenceChunk: z.boolean().default(false)` to session-brain schema and createInitialState
- V6 gate changed from `!state.isEvalSession` to `(!state.isEvalSession || state.receivedRealInferenceChunk)` — preserving #22 legacy path invariant
- `wsPayloadWithEval` injects `prescribed_exercise: artifact.prescribed_exercise ?? null` into eval_context (not the full artifact — redundancy removed per code review)
- `SynthesisResult.__post_init__` approach was rejected: removed in favor of explicit construction-site population at `run_recording`
- TS Test 2 ("remains false for eval_chunk-only sessions") is a Zod-default shape/invariant test (plan explicitly allowed this — locks the #22 preservation contract, not setter behavior)
- Commits: 804cb505 (initial), 1bc2e6b1 (fixes)

## Task 2: score.py pure scoring module
- `SessionCapture` defined canonically in score.py; `local_session.py` will import from here
- `_score_tempo` initially had semantic inconsistency (absent tempo_factor returned False for weak_prior flag without checking timing-dominance); fixed to use `ex.get("tempo_factor", 1.0)` consistently
- Added 2 tests for absent tempo_factor path (20 total after fix)
- bar_range_grounding returns None for corpus_drill exercises (only scored for own_passage_loop)
- Commits: cee56dfd (initial), 05673ee1 (fixes)

## Task 3: shared/local_session.py deep driver
- SessionCapture re-exported from score.py (not redefined) — canonical definition stays in score.py
- dominant_dimension extracted from teaching_moments[0].get("dimension"), NOT from artifact.get("dominant_dimension") — artifact is not embedded in eval_context (removed in Task 1 fix)
- Two CRITICALs caught and fixed: (1) missing session pre-creation via POST /api/practice/start before WS connect (client-generated UUID not registered server-side); (2) missing per-chunk ack gate (must await chunk_processed before sending next chunk_ready, mirroring pipeline_client.py protocol)
- ConnectionClosed now caught in synthesis recv loop; check_services catches requests.Timeout
- Commits: 8ec52365 (initial), ff5b9efa (protocol fixes)

## Task 4: eval_routing.py orchestrator + baseline.json + justfile
- ratchet_baseline.py extracted as standalone script (justfile heredoc with embedded Python and `.` chars causes just parse error)
- Smoke exits 0 on missing practice_eval audio (warning printed, wiring still validated) — fix commit aca3eb22
- `_check_no_universal_piece_id_failure` variable `kind_correct_count` is a proxy for "piece ID succeeded" — name is misleading but logic is correct; error message may blame seed-fingerprint when bar_range bug is the cause
- `test_baseline_floors_are_floats_in_0_1` iterates REQUIRED_AXES not data.keys() — avoids false failure on `notes` string key
- parents[3] depth correction applied (plan had off-by-one for BASELINE_PATH in test)
- Commits: b2efc025 (initial), aca3eb22 (smoke fix)

## Task 5: Delete dead scaffolding
- All 4 exercise_data occurrences removed (declaration + checkpoint-restore + checkpoint-save + details dict)
- Line 262 (checkpoint-restore) was the critical one identified in 3rd challenge pass — confirmed removed
- Zero dangling references; import parse OK
- Commit: a1509d23
