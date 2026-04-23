# Compounds

High-level orchestrators. One compound per hook (event or schedule). Compounds call molecules (and may call atoms for utility reads). See `../README.md` for three-tier overview.

**Target size:** 3-5 compounds.
**Reliability ceiling:** compounds spanning more than 8-10 molecules hit a failure mode. Keep them tight.
**Single write constraint:** a compound dispatches many molecules for analysis; the compound writes one teacher-facing artifact. Skills contribute intelligence, not parallel speech.

Populated by V5 work. Initial candidate list:

- `session-synthesis` -- OnSessionEnd. Runs ~5-7 molecules, produces one teacher synthesis artifact. Replaces current monolithic synthesis prompt.
- `live-practice-companion` -- continuous during recording. Dispatches cross-modal-contradiction-check and rubato-coaching in real time; writes observation artifacts on STOP.
- `weekly-review` -- OnWeeklyReview (scheduled). Runs many molecules across many sessions; produces a longitudinal synthesis artifact.
- `piece-onboarding` -- OnPieceDetected (first time a student plays a new piece). Runs phrasing-arc-analysis and dynamic-range-audit against reference performances; produces a piece-orientation artifact.

Each compound declares its molecule dependencies in YAML `depends_on` and its hook in `triggered_by`.
