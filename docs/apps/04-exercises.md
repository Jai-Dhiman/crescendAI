# Exercises and Focused Practice

> **Status (2026-03-19):** Exercise DB schema DEFINED. Endpoints IMPLEMENTED (`GET /api/exercises`, exercise tracking). CEO review (2026-03-19): Exercises ship as artifacts in the unified container system (see `05-ui-system.md`). Exercise artifact is the only artifact type in the web beta. Focus mode DEFERRED to Phase 3.
>
> **Status update (2026-06-10, S1 shipped — #29):** The legacy `proposed_exercises` (synthesis) and `create_exercise` (chat tool) paths have been removed and replaced by the `ExerciseRoutingDecision` contract. All exercise prescriptions — whether emitted during post-session synthesis or via the `prescribe_exercise` chat tool — now produce a typed routing decision that persists to `pending_exercises`. See [S1 Contract](#s1-exercise-routing-contract-shipped-29) below.
>
> **Status update (2026-06-11, own-passage loop playback shipped — #45):** `ExerciseSetCard` redesigned with score-first layout and `LoopTransport` interactive playback. `LoopPlayer` (smplr piano + metronome + `LoopClock`) drives audio. `useLoopPlayer` hook manages countdown/playback state. `score-worker` gained `get_clip_playback` message for IR + playback notes. `tempoFactor` flows from the prescription routing decision into the transport slider.
>
> **Status update (2026-06-16, corpus-drill renderable assets + Verovio transpose-on-demand shipped — #46, epic #44 S3):** The 22 exercise primitives now ship as committed `.mxl` assets and render in the web score panel at any in-range key. (1) `model/src/exercise_corpus/build_render_assets.py` `build()` turns each committed `model/data/scores/exercise_primitives/*.xml` into a committed `model/data/exercise_primitives/mxl/{id}.mxl` (partitura-validate -> DOCTYPE-strip -> `wrap_as_mxl_zip`, idempotent, fail-loud on bad XML); `just build-exercise-assets` runs it and `just seed-exercise-assets` puts the `.mxl` to local R2 `scores/v1/{id}.mxl` (served by the unchanged `GET /api/scores/:pieceId/data`). (2) `ScoreRenderer.load(pieceId, transpose?)` and the worker's `loadPiece(..., transpose?)` thread an optional semitone integer to Verovio's `transpose` (stringified only at the `tk.setOptions` boundary); toolkit/IR caches now key on a composite `${pieceId}:${transpose ?? 0}` — this also fixed a pre-existing P0 where all clip rendering shared a non-composite cache key. Byte fetch stays keyed by `pieceId`; `transpose: 0` is byte-identical to the legacy path. (3) Hermetic Verovio-vs-partitura faithful-shift pitch-class oracle (`model/tests/exercise_corpus/test_render_assets_oracle.py`): intra-engine faithful-shift invariant across all 22 primitives (48 passed) + cross-engine baseline witness (2 documented `xfail` divergences: `burgmuller_001` repeat-expansion, `czerny_001` accidental-realization). Runtime selection/display (Worker picks primitive + key) is S4. (`corpus_drill` card still stub-renders until S4 wires selection.)
>
> **Status update (2026-06-23, corpus-drill runtime selection + transposed/excerpted display shipped — #47, epic #44 S4):** `corpus_drill` now selects a renderable primitive and emits a real `scoreClip` instead of stub text — the epic is architecturally complete on the 22-primitive corpus. (1) `buildCorpusDrillClip` (`apps/api/src/services/corpus-drill.ts`) picks a primitive by `target_dimension` using a faithful `match_by_dimension` sort (mirrors the model's `(source_exercise_number, primitive_id)` ascending ordering) over `exercise_primitives_manifest.json` (an opt-in emit from the model's `build_render_assets.py`); on no dimension match it WIDENs to an EXPLICIT neutral default (`hanon_001`), raising rather than silently surfacing the sort-first primitive. (2) Transpose is computed from the student's passage key vs the primitive's key via a TS port of the model's key parser (`apps/api/src/services/keys.ts` — `parseKeyToPc` superset + `transposeInterval`, parity-fixture-locked); the resolve step is best-effort (any unresolvable input -> `transpose: 0` + a structured warn, never a silent null). (3) The decision is wired into BOTH prescription paths — `prescribe_exercise` (chat, `tool-processor.ts`) and `assignPendingExercise` (`exercises.ts`) — and emits `scoreClip{pieceId: primitiveId, bars: [1, totalBars], tempoFactor, transpose}`. (4) `ExerciseSetCard` (`apps/web`) threads `scoreClip.transpose` into the existing Verovio render path (S3/#46), so the card now engraves the chosen primitive transposed + excerpted + playable. Local green: web 163/163, api relevant 43/43, model 4/4. Remaining (post-S4): live in-browser click-through of a prescribed `corpus_drill` (real Verovio transpose + smplr audio) before any deploy; corpus breadth (S5/#17) and RANK/difficulty (S6/#42/#43).
>
> **Status update (2026-06-28, relevance@1 eval + cosine RANK built — #103, epic #44 S6):** Two things landed as tested infrastructure (production selection behavior UNCHANGED — the cosine ranker is committed but not yet wired into the serve path). (1) **Relevance@1 LLM-judge** (`apps/evals/pipeline/exercise_routing/relevance.py`): given a diagnosed weakness (dimension + teaching-moment context) and the chosen drill's `title`+`techniques`, a judge rates 0-3 pedagogical fit; `selector_relevance_at_1` = fraction ≥2, wired into `exercise-routing-eval` over the dominant-weakness selection of every invoked session (`build_selector_case`, a counterfactual that is powered even when most routes are `own_passage_loop`). The judge runs through the **authenticated `crescendai` AI Gateway** (`gateway_judge.py`, `cf-aig-authorization`; the shared `LLMClient`'s legacy `crescendai-background` gateway now 401s). Validated against the real judge: it flags the existing defect — `pedaling → chopin_001` = 3 (appropriate), `timing → chopin_etude_001` = 0 (an arpeggio etude for a timing weakness). This is the gap `dimension_match` (label-equality) is structurally blind to. (2) **Cosine RANK** (`cosine-select.ts` + Python mirror `cosine_select.py`): FILTER-then-RANK — filter the catalog to `target_dimension`, then cosine-rank within it against an Aria embedding of the student's weak passage, over a committed L2-normalized 154-drill catalog asset (`exercise_embeddings.json`, exported by `export_embeddings.py`). `embed_server.py` serves `aria-medium-embedding` for the serve-time query (the Worker has no torch). The manifest now carries `title`+`techniques`+`source` so the judge and ranker see WHICH drill was picked. **NOT yet locked / not yet live:** the baseline number and the cosine A/B re-measure are deferred (local disk full → R2 can't persist chunks → eval yields 0 chunks; AMT — the weak-passage MIDI source for the cosine query — only runs via the now-fixed local container, which needs disk + Docker). `selectPrimitive` already honors an explicit `primitive_id`, so wiring is synthesis → embed → cosine → `routing.primitive_id`. See FILTER→RANK→ADAPT below.
>
> **Status update (2026-06-27, 154-drill corpus reaches the product — #102, builds on #49):** The corpus the LLM can route to grew **22 → 154** primitives, closing the **pedaling=0 gap** (0 → 37 pedaling-routable drills; all 6 dimensions now routable). (1) `build_render_assets.build()` was reworked to enumerate the roster from `model/data/embed_ready_manifest.json` + `technique_tags.toml` (the authoritative per-primitive dimensions/key) instead of the 22 committed `*.xml`, and regenerates `apps/api/src/services/exercise_primitives_manifest.json` (154 entries `{dimensions, key, totalBars}`, the durable tracked record `corpus-drill.ts` routes over). (2) Each drill gets ONE render-gated asset, tiered by source fidelity: **A** committed MusicXML → MXL (20 Hanon + czerny_001 + burgmuller_001); **B** Chopin etudes → Verovio-native **MEI** from the original public-domain `.krn` (cleanest; `getPieceData` prefers `.mei`, 24 drills); **C** Mutopia prebuilt MIDI → partitura MusicXML → MXL (legible PoC engraving, 108 drills). `seed-exercise-assets` seeds both `.mei` + `.mxl` from the unified `model/data/exercise_primitives/assets/` dir to local R2 `scores/v1/{id}`. (3) Selection logic in `corpus-drill.ts` is unchanged; the stable-first per dimension simply resolves over a 7× larger library (pedaling → `chopin_001`, timing → `chopin_etude_001`). VERIFIED in the real browser Verovio worker (`/app/sandbox`, `just dev-light`): a tier-C pedaling drill and a tier-B etude MEI both render real sheet music. Local-first, NOT deployed — prod R2 seeding of the 154 assets is a separate deliberate step (tier-C MXL is legible-not-publication-grade; etude `.mei` is PD-clean).

---

## S1 Exercise Routing Contract (shipped #29)

### ExerciseRoutingDecision union

Every exercise prescription the teacher emits is now a discriminated union keyed on `kind`:

```typescript
type ExerciseRoutingDecision =
  | {
      kind: "own_passage_loop";
      target_dimension: "dynamics" | "timing" | "pedaling" | "articulation" | "phrasing" | "interpretation";
      bar_range: [number, number];   // [start_bar, end_bar] from the student's score
      tempo_factor: number;          // 0 < tempo_factor <= 1.0 (e.g. 0.75 = 75% tempo)
    }
  | {
      kind: "corpus_drill";
      target_dimension: "dynamics" | "timing" | "pedaling" | "articulation" | "phrasing" | "interpretation";
      bar_range: [number, number];
      tempo_factor: number;
      primitive_id?: string | null;  // corpus exercise primitive; null until S3/S4
    };
```

`own_passage_loop` directs the student to loop a specific passage from their own piece at a reduced tempo — the primary vehicle when no catalog drill is needed. `corpus_drill` reserves a slot for a matched catalog primitive (S3/S4 fill this in; until then it stub-renders as text).

### What replaced what

| Removed | Replaced by |
|---------|-------------|
| `proposed_exercises` field on `SynthesisArtifact` | `prescribed_exercise: ExerciseRoutingDecision \| null` |
| `create_exercise` chat tool (free-text, unstructured) | `prescribe_exercise` chat tool (emits `ExerciseRoutingDecision`) |

### Rendering

- **`own_passage_loop`** — rendered by `ExerciseSetCard` with interactive loop playback (shipped #45). Score-first card layout: score clip at the top, `LoopTransport` bar below (play/stop, tempo slider, bar-range label), animated `ScoreCursor` tracking playback. `LoopPlayer` audio orchestrator drives smplr piano + metronome aligned to `LoopClock`. `tempoFactor` from the prescription drives the initial tempo slider value.
- **`corpus_drill`** — fully rendered as of S4 (#47). `buildCorpusDrillClip` selects a primitive by `target_dimension` (faithful `match_by_dimension` sort; WIDEN -> explicit `hanon_001`), computes transpose from the student passage key vs the primitive key, and emits a `scoreClip{pieceId, bars:[1,totalBars], tempoFactor, transpose}`. `ExerciseSetCard` engraves it transposed + excerpted + playable via the existing Verovio path (S3/#46). Wired into both the `prescribe_exercise` chat tool and `assignPendingExercise`.

### Persistence

Prescriptions go to `pending_exercises` (migration `0004_pending_exercises_routing.sql`, applied locally to `crescendai_dev`). The `exercises` catalog table is NOT written to by the prescription path — `pending_exercises` is the staging layer between the teacher decision and any future catalog promotion.

Schema change: `pending_exercises` dropped `exercise_id` (FK to catalog) and gained `title`, `instruction`, `routing_json` (the serialized `ExerciseRoutingDecision`), and `piece_id`.

### Deferred follow-ons

- **Eval ASCF baseline re-lock:** `run_eval.py` now renders `prescribed_exercise` into prose so the eval pipeline runs green, but the locked baseline number is NOT re-locked (credit-gated — deferred to a dedicated eval session).
- **`exercise-proposal.md` catalog cleanup:** the harness skill catalog entry and `depends_on` narrative are out of date (the `exercise-proposal` molecule is deleted). Cleanup deferred; the validators that catch it are excluded from the default runner (matches prior precedent from other removed skills).

---

## Why Exercises Matter

The core feedback loop in CrescendAI is: observe weakness, prescribe exercise, evaluate improvement, close the loop. Without exercises, the system can only describe problems. With exercises, it can fix them.

Today, the teacher pipeline (02-pipeline.md) detects when something sounds off and explains what to work on. But "your pedaling is blurring the harmonic changes" is only half the job. The other half is "here's a 5-minute exercise to fix it, and I'll listen to tell you if it worked."

Exercises connect two otherwise separate capabilities:

1. **The exercise database** provides the content -- curated method exercises (Hanon, Czerny), repertoire-specific drills, and LLM-generated custom exercises grounded in the student's actual music.
2. **Focus mode** provides the interaction pattern -- a guided mini-session that isolates a weak dimension, walks through targeted exercises, evaluates each attempt, and measures improvement.

The 6 target dimensions are defined in `model/02-teacher-grounded-taxonomy.md`: dynamics, timing, pedaling, articulation, phrasing, interpretation. Every exercise targets one or more of these dimensions.

---

## Exercise Database

### Schema

Two D1 tables store exercises and student-exercise tracking.

**`exercises` table:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique exercise identifier |
| `title` | TEXT NOT NULL | Exercise name |
| `description` | TEXT NOT NULL | What this exercise trains, why it matters |
| `instructions` | TEXT NOT NULL | Step-by-step how to practice it |
| `difficulty` | TEXT NOT NULL | `beginner` / `intermediate` / `advanced` |
| `category` | TEXT NOT NULL | `technique` / `musicality` / `ear-training` / `warmup` |
| `repertoire_tags` | TEXT | JSON array, e.g. `["Chopin", "nocturne", "romantic"]` |
| `notation_content` | TEXT | MusicXML or Lilypond source (nullable for text-only exercises) |
| `notation_format` | TEXT | `musicxml` / `lilypond` / null |
| `midi_content` | BLOB | MIDI bytes for playback (nullable) |
| `source` | TEXT | `curated` / `method:Hanon` / `method:Czerny` / `generated` |
| `variants_json` | TEXT | JSON: alternate versions (different keys, tempos, dynamics) |
| `created_at` | TEXT NOT NULL | ISO 8601 timestamp |

Index: `idx_exercises_difficulty` on `difficulty`.

**`exercise_dimensions` junction table:**

| Column | Type | Description |
|--------|------|-------------|
| `exercise_id` | TEXT NOT NULL | FK to `exercises(id)` |
| `dimension` | TEXT NOT NULL | One of: dynamics, timing, pedaling, articulation, phrasing, interpretation |

Primary key: `(exercise_id, dimension)`.
Index: `idx_exercise_dimensions_dim` on `dimension`.

```sql
CREATE TABLE exercise_dimensions (
    exercise_id TEXT NOT NULL REFERENCES exercises(id),
    dimension TEXT NOT NULL,
    PRIMARY KEY (exercise_id, dimension)
);
CREATE INDEX idx_exercise_dimensions_dim ON exercise_dimensions(dimension);
```

### Exercise Properties

Each exercise carries enough metadata to support filtering and pedagogical context:

- **target dimensions** -- stored in the `exercise_dimensions` junction table. Most exercises target 1-2 dimensions.
- **difficulty** -- coarse difficulty level. Maps roughly to student proficiency, not piece difficulty.
- **category** -- the type of practice activity. `technique` covers mechanical drills (scales, arpeggios, Hanon). `musicality` covers expressive work (voicing, phrasing). `ear-training` covers listening exercises. `warmup` covers session openers.
- **notation** -- optional MusicXML or Lilypond source. Many exercises are text-only (especially LLM-generated ones). Notation rendering is handled by `05-ui-system.md`.
- **variants** -- JSON describing alternate versions. A dynamics exercise might have variants in different keys or at different tempos.
- **source** -- provenance. `curated` = human-authored by Jai. `method:Hanon` = adapted from a published method. `generated` = created by the LLM for a specific student context.

### Curated Seed Data

V1 targets 20-30 curated exercises across all 6 dimensions and 3 difficulty levels, covering common repertoire contexts. Examples:

| Title | Dimension(s) | Difficulty | Category | Source |
|-------|-------------|------------|----------|--------|
| Dynamic Contrast Scales | dynamics | intermediate | technique | curated |
| Legato Pedal Harmonic Changes | pedaling | intermediate | technique | curated |
| Melody Extraction | articulation, dynamics | advanced | musicality | curated |

**Dynamic Contrast Scales (intermediate, dynamics):**
> Play a C major scale ascending over 2 octaves. Start at pp and arrive at ff by the top. Descend from ff back to pp. Focus on making the gradient as smooth as possible -- no sudden jumps. Repeat in the key of your current piece.

**Legato Pedal Harmonic Changes (intermediate, pedaling, Chopin-specific):**
> Take a 4-bar phrase from your Chopin piece where the harmony changes every bar. Play with full sustain pedal. Now: lift the pedal exactly on each new harmony and re-engage immediately. Listen for any overlap or gap. The goal is a seamless legato with zero harmonic bleed.

**Melody Extraction (advanced, articulation + dynamics):**
> Choose a passage where the RH has melody over LH chords. Play the melody alone at mf. Now add the LH at pp -- the melody should remain just as clear. If the LH starts to overpower, reduce it further. Record and listen back: can you hear every note of the melody clearly?

### LLM-Generated Custom Exercises

When the curated DB has no close match, or when an exercise should reference the student's specific passage, the LLM generates a custom exercise.

**Constraints:**
- Text instructions only. The LLM does NOT generate notation (hallucination risk for musical notation is too high).
- Instructions reference the student's existing score: "Take bars 20-24. Play the left hand alone..."
- Post-processing rejects exercises that exceed 1000 characters or attempt to include notation.

**LLM prompt template:**

```
The student is working on {piece} by {composer}. Their {dimension} needs attention,
specifically around {chunk_time_description}.

Generate a practice exercise that:
1. Uses their actual music (reference specific bars/passages if possible)
2. Isolates the {dimension} skill
3. Provides 3-4 concrete steps
4. Describes what "improved" sounds like

Keep it practical -- something they can do in 5 minutes right now.
Do NOT generate musical notation -- reference the score they already have.
```

**Storage:** Generated exercises go into the same `exercises` table with `source = "generated"`. If a student responds positively, the exercise can be promoted to curated status after human review.

### Query Logic

Exercise selection filters by dimension, difficulty, and novelty (not previously assigned to this student), with repertoire-matching as a soft preference:

```sql
SELECT e.* FROM exercises e
JOIN exercise_dimensions ed ON ed.exercise_id = e.id
LEFT JOIN student_exercises se
    ON se.exercise_id = e.id AND se.student_id = ?
WHERE ed.dimension = 'dynamics'
    AND e.difficulty = 'intermediate'
    AND se.id IS NULL  -- not previously assigned
ORDER BY
    CASE WHEN e.repertoire_tags LIKE '%Chopin%' THEN 0 ELSE 1 END,
    RANDOM()
LIMIT 3;
```

The API exposes this as `GET /api/exercises?dimension=dynamics&level=intermediate&student_id=...`, returning the top 3 candidates. On iOS, SwiftData mirrors the same filter logic locally, querying cached exercises first and falling back to the API if the cache is stale. New exercises reach iOS devices via the sync protocol (`POST /api/sync` response includes `exerciseUpdates`).

---

## Focus Mode (Guided Practice)

> **Deferred to Phase 3.** The CEO review (2026-03-19) scoped focus mode out of the web beta. Beta ships with inline exercise artifacts in chat. Focus mode (multi-exercise guided sequences) ships after the core observe-prescribe loop is validated with real users.

Focus mode transforms the app from a passive listener into an active teacher. It is the teaching loop: observe, identify, diagnose, prescribe, evaluate.

### Entry Points

**System-initiated** -- after teaching-moment selection (02-pipeline.md) identifies a dimension as the top teaching moment in 3+ of the last 5 sessions:

> "I've noticed pedaling keeps coming up. Want to do a focused session on it next time?"

The student can accept, dismiss, or say "later."

**Student-initiated** -- the student says or taps "I want to work on my dynamics" or "Let's focus on pedaling." The system enters focus mode for that dimension immediately.

### Session Flow

A focus session has 5 stages:

1. **Introduction.** The teacher LLM sets context:
   > "Let's work on your pedaling. I've been hearing some harmonic bleed in your Chopin -- the sustain is carrying across chord changes. We'll do 3 exercises to sharpen that."

2. **Curated exercise.** Query the exercise DB for a matching curated exercise (target dimension, student level, repertoire match). Present the title and instructions. Student plays. MuQ evaluates the attempt, focused on the target dimension. Teacher gives brief feedback:
   > "Better -- the chord change at the top was cleaner. Try it once more and really listen for the bass note ringing over."

3. **Custom exercise.** The LLM generates an exercise using the student's actual passage where the issue was detected:
   > "Now take bars 20-24 of your Nocturne. Play just the LH with pedal. Lift and re-engage on each new harmony. Then add the RH and keep the same pedal discipline."

   Student plays. MuQ evaluates. Teacher feedback.

4. **Integration.** Student plays the full passage with everything from the previous exercises applied. MuQ evaluates. Compare the target dimension score to the original teaching moment that triggered focus mode:
   > "That's markedly cleaner. The harmonic changes in bars 22-23 aren't bleeding anymore. Keep this awareness next time you run through the whole piece."

5. **Wrap-up.** Record before/after scores in `student_exercises`. Update the student model: "Focused on pedaling in session N." Suggest next steps:
   > "Try running through the piece now and I'll listen for whether the pedaling holds in context."

The student can skip an exercise or end the session early at any point.

### Focused Evaluation

MuQ inference runs via the cloud HF endpoint on each exercise attempt, using the same inference path as regular practice. The key difference is how feedback is delivered:

- **Weight to target dimension only.** If the student is working on pedaling, suppress observations about timing even if timing is worse than usual.
- **Non-target exception.** Only surface a non-target observation if that dimension is severely below baseline (a large negative deviation) -- something is badly off.
- **Before/after comparison.** Each exercise attempt records the target dimension score. The wrap-up computes overall improvement.

This keeps the session focused and avoids overwhelming the student with unrelated feedback.

### Session State Machine

```json
{
    "mode": "focus",
    "target_dimension": "pedaling",
    "trigger": "system",
    "trigger_context": "Pedaling was the top teaching moment in 3 of last 5 sessions",
    "exercises": [
        {
            "type": "curated",
            "exercise_id": "ex-ped-003",
            "status": "completed",
            "attempts": 2,
            "target_dim_before": 0.35,
            "target_dim_after": 0.52
        },
        {
            "type": "generated",
            "exercise_id": "ex-gen-001",
            "status": "in_progress",
            "attempts": 1
        },
        {
            "type": "integration",
            "status": "pending"
        }
    ],
    "overall_improvement": null
}
```

States per exercise: `pending` -> `in_progress` -> `completed` (or `skipped`). The session itself progresses linearly through the exercise list, with the student able to skip forward or exit.

---

## Student Tracking

### `student_exercises` Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique record identifier |
| `student_id` | TEXT NOT NULL | FK to `students` |
| `exercise_id` | TEXT NOT NULL | FK to `exercises` |
| `session_id` | TEXT | FK to `sessions` (nullable for standalone assignments) |
| `assigned_at` | TEXT NOT NULL | ISO 8601 timestamp |
| `completed` | BOOLEAN | Default FALSE |
| `response` | TEXT | `positive` / `neutral` / `negative` / `skipped` |
| `dimension_before_json` | TEXT | JSON: 6-dim scores before exercise |
| `dimension_after_json` | TEXT | JSON: 6-dim scores after exercise |
| `notes` | TEXT | Student feedback on the exercise |
| `times_assigned` | INTEGER | Default 1 |

Unique constraint: `(student_id, exercise_id, session_id)`.

Index: `idx_student_exercises` on `(student_id, exercise_id)`.

### How Exercise History Feeds the Subagent

The two-stage subagent pipeline (02-pipeline.md) uses exercise history as context when selecting teaching moments and generating observations:

- **Exercise effectiveness:** If a student completed an exercise targeting pedaling and their before/after scores improved, the subagent knows that dimension is responding to intervention. It may raise the bar for flagging pedaling in future sessions.
- **Negative responses:** If a student marked an exercise as `negative` or `skipped` it, the subagent avoids prescribing similar exercises and may try a different approach to the same dimension.
- **Repeat assignments:** The `times_assigned` counter and query logic (exclude previously-assigned exercises) prevent the system from recycling the same exercises.

---

## Dependencies

| Dependency | Location | Relationship |
|------------|----------|-------------|
| Teaching moment selection | `02-pipeline.md` | A dimension persisting as the top teaching moment across sessions triggers focus mode (deviation-magnitude gate) |
| Exercise UI components (`exercise_set`, notation rendering) | `05-ui-system.md` | Frontend rendering of exercise cards and notation |
| 6-dimension taxonomy | `model/02-teacher-grounded-taxonomy.md` | All exercises and evaluations use these 6 dimensions |
| D1 sync protocol | `docs/architecture.md` (Sync section) | Exercise data syncs to iOS via `POST /api/sync` |
| MuQ cloud inference | `02-pipeline.md` | Focus mode evaluation uses the same HF endpoint inference path |
| Unified artifact container | `05-ui-system.md` | Exercise artifact renders via the `<Artifact>` container system (inline in chat, expandable to viewport) |

---

## Open Questions

1. **Curated exercise count for V1.** 20-30 is the starting target. At what point does the LLM-generated path become more important than expanding the curated set?
2. **Difficulty progressions.** Should exercises link to successor exercises (exercise A leads to exercise B)? Useful for building skills incrementally, but adds schema complexity.
3. **Notation content.** Defer entirely to V2, or include Lilypond for a few key exercises in V1?
4. **Exercises per focus session.** 3 (curated + custom + integration) feels right but may be too many for a student who has already been practicing for 45 minutes. Should the system adapt based on session length?
5. **Focus mode timing.** Should focus mode interrupt a regular practice session mid-stream, or always be a separate mini-session?
6. **No improvement path.** What happens if the student doesn't improve during focus mode? Encouraging message + suggest trying again tomorrow? Adapt exercises on the fly? Lower expectations?
7. **Multi-session focus plans.** "Work on pedaling for the next 3 sessions" is not in scope for V1. When does it become worth building?

---

## Recommendation Model: FILTER → RANK → ADAPT (added 2026-06-08, #36)

> Status (2026-06-09): `FILTER` stage shipped (#36 — dimension gating via technique_tags + match_by_dimension). `ADAPT` stage shipped (#41 — passage-driven transpose/excerpt/tempo params in `build_briefing`). `RANK` and full `FILTER`/difficulty discrimination are follow-on issues gated on #17 corpus breadth. Supersedes the "embedding-similarity retrieval" framing in the rebuild index for the *runtime selection* step.

Recommending an exercise is not one decision keyed on one signal. It is a pipeline of three decisions, each failing differently, each tuned independently:

| Stage | Question | Signals | Failure mode if wrong |
|---|---|---|---|
| **1. FILTER** (hard, AND'd) | Which exercises are *eligible*? | dimension (shipped), difficulty-vs-student-level, `finding_type` (issue vs strength), `scope`, cooldown (shipped) | prescribes a *harmful* exercise (too hard, or "fixes" a strength) |
| **2. RANK** (soft, weighted) | Of the eligible, which *fits this passage*? | technique/structural match to the passage, multi-dimension coverage, key/register proximity, variety | prescribes a *suboptimal* exercise |
| **3. ADAPT** (transform) | How do I *shape* the chosen drill to this weakness? | severity → tempo + dosage, passage key → transpose, passage length → excerpt span, history → progression | prescribes a *mis-dosed / non-transferring* exercise |

The signals named loosely in discussion map cleanly: "dimension" is FILTER, "techniques for the passage" is RANK, "alterations" is ADAPT.

### Signal inventory (what actually exists vs is aspirational)

**Available at diagnosis time** (`DiagnosisArtifact`, `apps/api/src/harness/artifacts/diagnosis.ts`): `primary_dimension`, `dimensions[]` (plural — multi-dimension, currently unused by `build_briefing`), `severity`, `scope` (stop_moment/passage/session), `bar_range` (student-piece bars), `evidence_refs`, `one_sentence_finding` (free text, rich, underused), `confidence`, `finding_type` (issue/strength/neutral), `piece_id`.

**Derivable from the score now** (`model/data/scores/*.json`, per-bar `notes` with pitch/onset/velocity/duration, `pedal_events`, `key_signature`, `pitch_range`): structural demands of the weak passage — scalar runs, arpeggios, wide leaps, repeated notes, voicing density, register, tempo. This is the **honest source of the "technique" signal** — nameable and debuggable, unlike an opaque embedding.

**Available about the catalog**: dimension tags (shipped), `techniques` free-vocab tags (exist on `TagSet`, currently annotation-only — not load-bearing), implicit difficulty (sources are *graded* series: Burgmüller op.100 "progressive studies", Czerny op.299, Hanon 1-20). The API-side `exercises` table already has `difficulty`, `repertoire_tags`, `variants_json` columns.

**Gated on AMT deploy** (all sessions Tier 3 today): the student's *transcribed performance MIDI* of the passage — the ONLY signal that encodes *how they played* (timing deviations, dynamics). This is the honest version of the rejected score-embedding bridge: embed/analyze what they DID, not what's written.

**Gated on users**: exercise-completion + re-diagnosis outcomes — the *positive* efficacy/progression memory (today's memory is cooldown-only, i.e. negative memory).

### The load-bearing timeline insight

**Within-bucket discrimination — all of RANK, and difficulty's discriminating power — is gated on corpus breadth (#17).** On 22 mostly-Hanon primitives, a difficulty filter or a technique ranker resolves to the same tiny degenerate set; you cannot validate ranking machinery you build now. Three tracks:

- **Track 1 — corpus-independent, buildable now:** the ADAPT layer (transform params operate on the *single chosen* exercise — transposing it to the student's key, slowing by severity, excerpting the passage all deliver visible value with one exercise per dimension) and FILTER *correctness* (skip strengths, handle session scope, multi-dimension) — about right behavior, not discrimination.
- **Track 2 — gated on #17:** difficulty *discrimination*, the technique axis, the soft weighted ranker. Design now, value waits on data. (A soft ranker with untunable weights on a 22-item / 0-user system is just hidden hardcoding — defer it.)
- **Track 3 — gated on AMT + users:** performance-delta matching, efficacy/progression memory.

### Strategic fork to decide

The most transfer-positive "exercise" is often *"loop bars 5-8 of your own piece, slowly"* — the student's own excerpted passage + the existing `excerpt`/`scale_tempo` transforms, needing **no catalog at all**. If the student's piece is the primary substrate and the catalog is supplementary technique-builders, the system's dependence on #17 corpus breadth drops sharply. Decide this before investing further in catalog matching.

### What is shipped vs follow-on (as of 2026-06-09, #36 + #41 merged)

- **Shipped (#36):** FILTER/dimension gating via `technique_tags.toml` + `match_by_dimension`; untagged dimensions raise instead of returning off-dimension drills.
- **Shipped (#41, ADAPT layer):** `build_briefing` now computes passage-driven transform params: `transpose_semitones` + `target_key` (from `diagnosis.bar_range` → score JSON → `key_signature`; uses `keys.py` `parse_key_to_pc` + `transpose_interval`, nearest-octave clamp [-5, 6]); `excerpt_bars` derived from `bar_range` length; tempo-by-severity (`severity` → `scale_tempo`) pre-existing. `TagSet` gained required `key` field; all 22 `technique_tags.toml` entries default to `key = 'C'`. Missing score JSON raises `FileNotFoundError` (strict, no silent fallback); null `key_signature` → `transpose_semitones = None`. New `ExerciseBriefing` fields: `transpose_semitones: int | None`, `target_key: str | None`. The API-side `ExerciseArtifact` has no transform fields — zero ripple to the web/iOS surface.
- **Follow-on issues:** FILTER/difficulty+finding_type+scope (Track 1 correctness + Track 2 discrimination), RANK/dimension∧technique matching (Track 2, gated on #17 corpus breadth), performance-delta matching (Track 3, gated on AMT + users).
- **Built but not yet wired/measured (#103, 2026-06-28):** The RANK stage now has a concrete implementation — **cosine within the dimension bucket** (Track 2 *and* Track 3 combined: the query is an Aria embedding of the student's transcribed weak passage, the honest "what they DID" signal). It lives in `cosine-select.ts` (+ `cosine_select.py` mirror) over the committed 154-drill `exercise_embeddings.json`; `embed_server.py` serves the query embedding. It is NOT yet wired into `corpus-drill.ts`/synthesis, so runtime selection is still the deterministic stable-first. The measurement that decides whether to wire it is **relevance@1** (`relevance.py`, `selector_relevance_at_1` axis): lock the deterministic baseline, then A/B the cosine selection on the same sessions. Both runs are gated on AMT (weak-passage MIDI) + local disk; see the #103 status header. Note this partly overrides the 2026-06-09 "defer the soft ranker on a 22-item corpus" guidance — the corpus is now 154 and the ranker is similarity-based (no untunable weights), so it is testable once the eval can run.
