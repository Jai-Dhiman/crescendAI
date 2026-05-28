# Exercise System Rebuild — Brainstorm Index

**Status:** Slice A SHIPPED — slice B ready for design (pending slice A manual acceptance gate)
**Created:** 2026-05-27
**Owner:** Jai

This file is the navigation index for a multi-session brainstorm. Each slice is brainstormed → /plan → /challenge → /build → /review → /ship independently. This file tracks status and dependencies so a new session can pick up cleanly. It contains no designs — designs live in `docs/specs/` and `docs/plans/` once approved.

## Background

Current exercise system has shipped schema + endpoints + UI but the proposal logic is a hardcoded rule-based template engine in `apps/api/src/harness/skills/molecules/exercise-proposal.ts` (6 dims × 3 severities = ~18 string templates). No real exercise corpus, no retrieval, no piece-awareness, no transformation.

Rebuild target: scrape public-domain piano pedagogy → segment into primitives → auto-tag with music21 + Aria embeddings → retrieval-based matcher with deterministic music21 transformations (transpose/tempo/span) → inject pre-retrieved candidates into teacher briefing → close the loop via score-follower's per-bar dimension deltas across sessions. Zero budget, no timeline pressure.

Prior conversation context: ChatGPT discussion (2026-05-27) covered intent, applicability of the RNCM MEC 2026 sight-reading paper, hybrid retrieval-vs-generation reasoning, automation strategy, the 2-minute teacher data-flow example, and memory-loop closure.

## Current state (shipped, do not break)

- **Schema:** `exercises` / `exercise_dimensions` / `student_exercises` Postgres tables (Drizzle, `apps/api/src/db/schema/exercises.ts`)
- **API:** `GET /api/exercises`, `POST /api/exercises/assign`, `POST /api/exercises/complete`
- **Harness:** `exercise-proposal` molecule at `apps/api/src/harness/skills/molecules/exercise-proposal.ts` (rule-based, no DB)
- **Artifact:** `ExerciseArtifact` Zod schema at `apps/api/src/harness/artifacts/exercise.ts` (stable interface — UI consumes this)
- **UI:** `ExerciseSetCard` / `ExerciseSetExpanded` on web; `ExerciseSetCard.swift` on iOS
- **Docs:** `docs/apps/04-exercises.md` (V1 vision; describes 20–30 curated, LLM-text-only custom, focus mode deferred)

## Target pipeline (high-level, not a design)

```
[1] Corpus construction (Python, offline, slice A)
      IMSLP / Mutopia / KernScores
        → MusicXML normalize
        → segment into primitives
        → auto-tag (music21 rules + Aria embeddings)
        → write to Postgres
                ↓
[2] Schema + data model (slice B)
      Add primitive provenance, license, pgvector embeddings,
      piece-bar linkage, transformation params
                ↓
[3] Matcher + transformer (runtime, slice C)
      Aria-embedding similarity retrieval (pgvector)
      → deterministic music21 transformations
      → ExerciseArtifact output
                ↓
[4] Briefing + memory integration (slice D)
      Inject candidate_exercises into teacher briefing
      Close loop via score-follower per-bar dim deltas
```

## Slices

| ID | Name | Status | Spec | Plan | Blocked on |
|---|---|---|---|---|---|
| **A** | Corpus embedding validation | **SHIPPED** (code on main; manual acceptance gate pending: acquire Hanon/Czerny op.299/Burgmuller op.100 MusicXML, run `python -m exercise_corpus.run`, confirm k-NN source purity >= 0.70 @ k=5 AND >= 11/15 within-source neighbor pairs pedagogically sensible; slices B/C/D stay blocked until gate passes) | deleted | deleted | — |
| B | Schema + data model evolution | UNBLOCKED FOR DESIGN (pending slice A manual acceptance gate) | — | — | A manual acceptance gate |
| C | Matcher + transformer service | NOT STARTED | — | — | B |
| D | Briefing + memory integration | NOT STARTED | — | — | C, chroma-DTW score follower |

## Approval workflow per slice

`/brainstorm` (this skill) → design approved → `/plan` writes spec + TDD plan → `/challenge` reviews plan → `/build` executes → `/review` → `/ship`. After each `/ship`, update this index's slice table with links to the shipped artifacts and flip the next slice to DESIGN IN PROGRESS.

## Resume instructions

1. Read this file.
2. Find the slice marked `DESIGN IN PROGRESS`.
3. Read its draft notes if present below; otherwise resume `/brainstorm` grilling from the next unresolved question.

### Slice A notes (SHIPPED — code on main, manual acceptance gate pending)

Decisions locked so far:

1. **Scope:** Narrow validation (~125-250 primitives). Goal = verify Aria embeddings cluster pedagogically before investing in scraper/license/multi-format plumbing.
2. **Sources:** Hanon Virtuoso Pianist + Czerny op.299 + Burgmüller op.100. All unambiguously public domain globally. Acquired as manual MusicXML URLs (no scraper in slice A).
3. **Blast radius / storage:** Throwaway-style, local-only. SQLite catalog + segmented MusicXML/MIDI files. Touches no production DB or code paths. Slice B later designs the production schema + migration that ingests this output.
4. **Format:** MusicXML canonical + derived MIDI. NO MEI (extra info is musicology-oriented, not pedagogically actionable; our sources are MusicXML-native so MusicXML->MEI conversion manufactures nothing). MEI reconsidered only for a future repertoire-score-library surface.
5. **Library:** partitura (CPJKU, v1.8.0) as canonical. music21 available narrowly for harmonic analysis only if the tagger needs it. Rationale: ML-native note-array representation, best-in-class MIDI/performance handling, MEC-community standard, strategic alignment with CPJKU score-following/performance-modeling lineage.
6. **Location (REVISED from apps/corpus/):** `model/src/exercise_corpus/`. Reason: the entire Aria dependency chain (weights, ariautils, aria pkg, paths.py, embedding code) lives in `model/`. Corpus *construction* is model-side; corpus *serving* (slice C) will be api-side.
7. **Aria embedding reuse:** Reuse `model/src/model_improvement/aria_embeddings.py`. Use `variant="embedding"` (512-dim, EOS-pooled global embedding). Fits our short primitives (tens of notes, < 300-note chunk threshold) exactly as it already handles PercePiano segments. Outputs to `model/data/` following existing conventions.

Open questions remaining for slice A: (a) segmentation granularity (what counts as one primitive), (b) auto-tagger scope (which tags in slice A vs deferred), (c) success criteria (concrete pass/fail bar for "embeddings cluster pedagogically").
