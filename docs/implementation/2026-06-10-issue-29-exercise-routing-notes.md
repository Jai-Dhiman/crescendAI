# Implementation Notes

Decisions, deviations, and tradeoffs made during build. Read this before running /review.

## Task 1: ExerciseRoutingDecision contract schema
- Created ExerciseRoutingDecisionSchema as discriminated union on "kind" field
- Added .strict() to OwnPassageLoopSchema after code review found it was needed to reject extraneous fields (e.g. primitive_id on own_passage_loop)
- DIMS_6 cast required: `as unknown as [string, ...string[]]` for Zod enum non-empty-tuple constraint

## Task 2: SynthesisArtifact schema migration
- Removed proposed_exercises field; added prescribed_exercise: ExerciseRoutingDecisionSchema.nullable().default(null)
- .nullable().default(null) is the correct Zod chain: nullable=accepts null, default=omitted→null

## Task 3: Phase 2 prompt update
- Replaced exerciseInstruction string only; surgical change
- VALID_ARTIFACT cast removed post-Task-2 landing (b293c751 fix commit)

## Task 4: DB schema migration + pending-exercise service rewrite
- exerciseId column dropped; title/instruction/routing_json/piece_id added (all nullable)
- drizzle-kit generate was interactive (rename detection) — SQL written manually to apps/api/drizzle/0004_pending_exercises_routing.sql
- Migration applied locally via psql; journal entry committed
- stageDominantExercise now single INSERT into pendingExercises only

## Task 5: DO wiring (SessionBrain)
- pieceCtx was already in scope at the call site (line 1569) as {composer, title, pieceId}|null
- WASM pkg/ dirs missing from worktree (gitignored) — copied from main before tests could run

## Task 6: assignPendingExercise rewrite
- No exercises catalog join; reads routingJson + pieceId from pending row
- ExerciseSetPayload.exercises[].exerciseId made optional (exerciseId?: string)
- web/src/lib/types.ts ExerciseSetConfig already had exerciseId optional — no change needed
- TODO(S3) comment documents deferred studentExercises tracking

## Task 7: Delete dead code
- readme-molecules.test.ts and docs/harness/skills/molecules/README.md also needed updating (found in quality review) — fixed in 4096b122
- docs/harness/skills/molecules/exercise-proposal.md doc still exists (out of scope per plan)

## Task 8: prescribe_exercise chat tool
- ExerciseRoutingDecisionSchema.and() incompatible with .strict() OwnPassageLoopSchema — built fresh discriminated union for prescribeExerciseSchema instead
- piece_id field added to both variants (needed for scoreClip construction at chat-tool boundary)
- piece_id divergence from canonical ExerciseRoutingDecisionSchema documented as forward risk (no current bug)

## Task 9: ExerciseSetCard test
- No component changes needed — scoreClip and exerciseId rendering already conditional
- Existing test file had 3 tests; added 2 new corpus_drill stub tests

## Task 10: Eval render update
- No changes needed — run_eval.py has zero proposed_exercises/prescribed_exercise field access
- Empty commit for verification documentation

## Task 11: Fixture sweep
- 6 files updated: runHook.test.ts, phase2-schema.test.ts, integration.test.ts, teacher-synthesize-v6.test.ts, teacher.test.ts, phase2.test.ts (line 116)
- Negative assertions (not.toContain("proposed_exercises")) in phase2.test.ts correctly left intact
