# Compounds

High-level orchestrators. One compound per hook (event or schedule). Compounds call molecules (and may call atoms for utility reads). See `../README.md` for three-tier overview.

**Final size:** 4 compounds.
**Reliability ceiling:** compounds spanning more than 8-10 molecules hit a failure mode. Keep them tight.
**Single write constraint:** a compound dispatches many molecules for analysis; the compound writes one teacher-facing artifact. Skills contribute intelligence, not parallel speech.

## Compound catalog
- `session-synthesis` -- triggered_by: `OnSessionEnd`. Runs 7 diagnosis molecules in parallel, calls `prioritize-diagnoses`, runs `exercise-proposal` per top-N diagnoses, writes one SynthesisArtifact (synthesis_scope=session). Replaces the current monolithic synthesis prompt. Reads live-practice-companion's stop_moment DiagnosisArtifacts as inputs (Option B compound-overlap policy).
- `live-practice-companion` -- **UNIMPLEMENTED / HISTORICAL** (no `OnRecordingActive` hook exists; depends on the removed `classify-stop-moment` atom -- STOP classifier deleted 2026-05-27). Original design: continuous during recording, dispatching `cross-modal-contradiction-check` and (where phrase boundaries exist) `rubato-coaching` per flagged chunk. If revived, gate on worst-dimension deviation, not STOP. See `docs/model/09-stop-classifier-removed.md`.
- `weekly-review` -- triggered_by: `OnWeeklyReview` (scheduled). Calls `fetch-session-history`, re-aggregates diagnoses across sessions, writes one SynthesisArtifact (synthesis_scope=weekly, recurring_pattern mandatory).
- `piece-onboarding` -- triggered_by: `OnPieceDetected` (first time). Dispatches `phrasing-arc-analysis` and `dynamic-range-audit` against reference cohort percentiles; writes one SynthesisArtifact (synthesis_scope=piece_onboarding, all focus_areas severity=minor).

Each compound declares its molecule dependencies in YAML `depends_on` and its hook in `triggered_by`. Compounds never call other compounds.
