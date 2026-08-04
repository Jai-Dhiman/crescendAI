# The Harness

Anchor doc for CrescendAI's middle system -- everything between the model outputs and the student's screen. Parallel to `docs/architecture.md` (system-level view) and `docs/model/00-research-timeline.md` (model-level view).

> **Status (2026-08-04):** The runtime-consumed catalog contains 14 atoms, 7 registered molecules, and 4 compounds. Typed Zod artifact contracts and the catalog validator live in `apps/api/src/harness/`. The former Qwen training program is historical; the catalog now serves the provider-agnostic V6 harness and its evaluations.

---

## Why "Harness"

CrescendAI's competitive advantage is not which audio model it calls. The advantage is the infrastructure wrapping the model: how signals accumulate into a session brain, how pedagogical moves are selected, how artifacts are produced, how student memory persists. When practitioners say "the harness is the product," this is the layer they mean.

### Four Systems (Model / Harness / Runtime / Client)

The canonical breakdown of the four systems (Model / Harness / Runtime / Client) lives in `docs/architecture.md`. This doc is the anchor for the Harness system specifically; see architecture.md for how it relates to Model, Runtime, and Client.

---

## Vocabulary

Durable primitives the harness is built from. Each has a precise definition; imprecise use breaks downstream reasoning.

- **Signal** -- An immutable emission from the model system (a MuQ 6-dim vector, an AMT midi_notes frame, a teaching-moment deviation, a score-following alignment). Signals live in the enrichment cache (see below) and never change after emission.
- **Enrichment cache** -- The Layer 1 store of extracted representations over raw audio. Each audio chunk has multiple coexisting cache entries, one per extraction schema (MuQ-quality, AMT-transcription, score-alignment). **Prompt-aware keys**: the same audio processed with different schemas produces different entries that coexist and are cross-queryable. See *How to grep video* (Mahler wiki).
- **Cross-modal query** -- A query that combines cache entries from different extraction schemas, used to catch contradictions no single schema would surface. "MuQ timing high but AMT shows 20ms onset drift" is cross-modal; it is also the highest-signal diagnostic a teacher makes.
- **Entity** -- A canonical resolved identity: a Student, Piece, Movement, Bar, Session, or Exercise. Two references must collapse to the same entity row before any agent reasons about them.
- **Fact** -- A temporal assertion about entities, with `validAt`, `invalidAt`, and an evidence chain back to Signals or Observations. "Student over-pedals in slow movements" is a Fact; its evidence is N Observations, each pointing at N Signals.
- **Skill** -- A markdown file in `docs/harness/skills/` describing one piece of pedagogical logic. Skills come in three tiers: **atoms** (narrow, near-deterministic), **molecules** (2-10 atoms chained for a pedagogical move), **compounds** (orchestrators that run many molecules, one per hook). See `docs/harness/skills/README.md`.
- **Artifact** -- A persistent, addressable output from a skill invocation. Unlike an ephemeral message, an artifact can be consumed by a later skill, rendered by a client, or cited in memory. Exercises, annotations, and score highlights are artifacts.
- **Hook** -- An event-triggered entry point. Two kinds:
  - **Event hooks** fire on external signals: `OnStop`, `OnPieceDetected`, `OnBarRegression`, `OnSessionEnd`, `OnWeeklyReview`.
  - **Middleware hooks** wrap the model call itself: `before_model`, `wrap_model_call`, `wrap_tool_call`, `after_model`. These run inside every invocation (PII redaction, tool-call limits, retries, HITL gates, online review).
- **Contract** -- Pre- and post-conditions on a skill or tool call, expressed in the skill's markdown. Makes silent degradation detectable: when a post-condition fails, the harness knows.
- **Tool** -- A callable the agent loop invokes. Split into **read tools** (fetch context; populate atoms) and **action tools** (change what the student does next: `assign_segment_loop`, `render_annotation`, `schedule_followup_interrupt`). Action tools require permission gating via `wrap_tool_call`.
- **Accumulator** -- The Durable Object-held session state that aggregates signals across a session. Serialized at known boundaries by the runtime.

---

## The Eight Verticals

Bottom-up, model to user. Each vertical has a doc home and a tier.

### V1 -- Model & Signals
MuQ (audio encoder), MoonBeam-839M (symbolic encoder; #138, 2026-08-03; see docs/mirex/track-a-difficulty-prediction.md), AMT (transcription), score follower, piece ID. Populates the enrichment cache with prompt-aware keys. Doc home: `docs/model/`.
**Tier:** NEXT (Phase B/C in flight).

### V2 -- Context Graph (Content / Entity / Fact)
Three-layer store: enrichment cache (immutable signals), resolved entities, temporal facts with evidence chains. Doc home: `docs/harness/entities.md`, `docs/apps/03-memory-system.md`.
**Tier:** DONE. Six entity schemas, EvidenceRef + EntityRef discriminated unions, bi-temporal Fact schema. Shipped 2026-04-26.

### V3 -- Accumulation & Compaction
Session-scoped DO state plus sawtooth compaction (Memento-style) for long sessions and longitudinal history. Doc home: `docs/apps/02-pipeline.md`, `docs/apps/03-memory-system.md`.
**Tier:** NEXT.

### V4 -- Eval Harness
Same code runs prod and eval. Playbook.yaml style injection wired everywhere. Signal ablation is eval #0. Phase 1 dual-judge on 10% sample. Per-tier reliability testing (atoms / molecules / compounds). Production review agent as middleware. Doc home: `docs/apps/07-evaluation.md`.
**Tier:** NOW. P0 beta blocker.

### V5 -- Skills (Atoms / Molecules / Compounds)
Three-tier skill catalog. 14 atoms, 7 registered molecules, 4 compounds. Each has YAML frontmatter, 5 required body sections, and a typed artifact output contract. Three Zod artifact schemas (`DiagnosisArtifact`, `ExerciseArtifact`, `SynthesisArtifact`) live in `apps/api/src/harness/artifacts/`. The validator enforces frontmatter, body sections, and cross-file dependencies. Doc home: `docs/harness/skills/`.
**Tier:** SHIPPED (2026-04-26). Current work is production/eval parity under #28.

### V6 -- Agent Loop & Orchestration
Teacher loop with deferred tool loading, NLAH contracts, event hooks + middleware hooks. Writes stay single-threaded: skills contribute intelligence, one teacher path writes. Providers remain replaceable behind capability and evaluation gates. Doc home: `docs/apps/02-pipeline.md` (Target section).
**Tier:** Core loop SHIPPED (2026-04-29) -- two-phase compound loop, `compound-registry` with `OnSessionEnd` + `OnChatMessage` bindings, V6 is the only synthesis path (flag deleted in #28). NEXT: capability-router across providers, deferred tool loading, remaining hooks (`OnWeeklyReview`, `OnPieceDetected`).

### V7 -- Student Memory / Personalization
Typed per-student memory (baseline, recurring_issue, preference, repertoire, goal, breakthrough) with MIA-style multidim retrieval. `STUDENT.md` index per student. Doc home: `docs/apps/03-memory-system.md`.
**Tier:** LATER. Gated on V2 + V5 + V6.

### V8 -- Action, Artifacts, Client
Artifacts as NLAH durable outputs. Direct-action tools that interrupt playthrough and restructure practice (the answer to the Score Following wiki's 90%-playthrough finding). iOS + web as thin clients over the same harness. Doc home: `docs/apps/04-exercises.md`, `docs/apps/05-ui-system.md`.
**Tier:** V8a SHIPPED (2026-04-29). `assign_segment_loop` is live as the first action atom: `segment_loops` DB table, lifecycle service (accept/decline/dismiss/increment), routes at `/api/segment-loops`, `ASSIGN_SEGMENT_LOOP_TOOL` registered in both OnChatMessage and OnSessionEnd bindings, `PassageLoopDetector` in DO for inbound detection, web `SegmentLoopArtifactCard` with pending/active/completed states. `SynthesisArtifact` carries `assigned_loops` refs; synthesis and chat paths both create durable loop rows. LATER: iOS native loop UI; further action atoms (`render_annotation`, `schedule_followup_interrupt`).

---

## The Two Clocks

Canonical definition lives in `docs/apps/03-memory-system.md` (state clock vs event clock).

---

## Design Principles

Drawn from external sources (Mahler wiki: Agent Harnesses, Natural Language Harnesses, Skill Design, Skill Graphs 2, Context Graphs, Multi-Agent Memory Systems, Multi-Agents What's Actually Working, How to grep video, The runtime behind production deep agents, Music Representation Learning, Music AI Systems, Score Following and Music Education) and from the Claude Code / opencode harness comparison.

1. **Thin runtime, rich primitives.** The agent loop is small. Complexity lives in skill files (markdown), not in a 2,000-line orchestrator. See *Agent Harnesses*.

2. **Natural-language harness.** Skills, contracts, and hook definitions are inspectable, diffable markdown. Changing harness behavior is a markdown edit, not a code deploy. See *Natural Language Harnesses*.

3. **Three-tier skill catalog.** Atoms (near-deterministic), molecules (2-10 atoms chained), compounds (orchestrators run by hooks). Every atom must be solid; every molecule must chain dependably; compounds beyond 10 molecules hit a reliability ceiling. See *Skill Graphs 2*.

4. **Atomic skills, not composite tasks.** Decompose "teaching" into molecules with their own eval signal and reward function. Composite training produces task-specific overfitting; atomic training generalizes. See *Skill Design* (atomic RL).

5. **Writes stay single-threaded.** Skills contribute *intelligence*; one path writes the final artifact. Parallel skills analyze; they do not parallel-speak to the student. See *Multi-Agents: What's Actually Working*.

6. **Context Rot is real.** Attention quality degrades at longer context length. Justifies deferred tool loading (V6), sawtooth compaction (V3), multidim retrieval (V7). A review agent with **no shared context** catches drift that in-context review cannot. See *Multi-Agents: What's Actually Working*.

7. **Enrichment cache with prompt-aware keys.** Raw audio is not grep-able; extraction schemas produce cache entries that make it so. Different skills can produce different extractions over the same chunks, coexisting and cross-queryable. See *How to grep video*.

8. **Cross-modal queries are first-class.** The highest-signal teacher diagnostics combine extractions: MuQ dim-level vs AMT-derived feature vs score-following alignment. Skill `when-to-fire` blocks should express cross-modal patterns, not single-signal thresholds.

9. **Identity resolution before reasoning.** Entity layer must exist before agents walk it, or every trajectory re-fights identity in tokens. See *Context Graphs*.

10. **Facts carry evidence chains.** Every assertion the teacher makes points back to Signals. No unsourced claims in memory. Enables audit and debugging.

11. **Harness vs runtime split.** Harness = prompts + skills + tools + contracts (markdown). Runtime = durable execution + checkpointing + memory storage + multi-tenancy + observability + sandbox (CF Workers + DO + D1 + R2 + AI Gateway). Middleware hooks live in the runtime; they wrap every model call with PII redaction, tool-call limits, retries, HITL gates, online eval. See *The runtime behind production deep agents*.

12. **Provider-agnostic skills; capability-router providers.** Skills are markdown and do not commit the system to one language model. Provider choice is a runtime decision gated by capability and evaluation evidence. See *Multi-Agents: What's Actually Working*.

13. **Signal ablation is non-negotiable.** Periodically substitute MuQ/AMT signals with plausible fakes. If synthesis outputs are unchanged, the harness is doing text-only reasoning and signals are decorative. See *Music AI Systems* (MuChoMusic finding).

14. **Direct action, not just report.** The harness must be able to interrupt playthrough and restructure practice. 90% of home practice is start-to-finish playthrough -- a passive "listen and synthesize" harness reinforces bad practice structure. See *Score Following and Music Education*.

15. **Harness = memory.** Closed harness = surrendered memory. Typed markdown memory format is the prerequisite for long-term ownership of accumulated student knowledge. See *Agent Harnesses* ("your harness, your memory").

16. **Eval code = prod code.** Playbook.yaml drives both `apps/api/src/services/prompts.*` and `apps/evals/teaching_knowledge/run_eval.py`. Drift between them means the eval measures something that is not shipping. Online eval runs as `after_model` middleware in production.

---

## Priority Stack (2026-04-23)

**DONE** -- V2 (entity schema). Shipped 2026-04-26.

**NOW** -- V4 (eval harness: playbook wiring, signal ablation, atomic-skill rubrics, per-tier reliability), V5 (three-tier skill decomposition: atoms / molecules / compounds).

**NEXT** -- V1 continued (Phase B/C with tightened SemiSupCon positive-mining + musically-informed AMT eval), V3 (compaction policy), V6 (agent loop + event hooks + middleware hooks), V8a (direct-action tools + chat tool_use).

**LATER** -- V7 (student memory typed entries + MIA retrieval), V8b (iOS native inference client). The former Qwen 27B fine-tune was closed and is not a standing roadmap item.

---

## Related Docs

- `docs/architecture.md` -- system view (model + harness + runtime + client)
- `docs/model/00-research-timeline.md` -- model system entry point
- `docs/apps/02-pipeline.md` -- current pipeline + Target: Agent Loop
- `docs/apps/03-memory-system.md` -- two clocks + three layers + enrichment cache
- `docs/apps/07-evaluation.md` -- eval harness including signal ablation and per-tier reliability
- `docs/harness/skills/` -- atoms / molecules / compounds catalog (V5)
- `docs/harness/entities.md` -- canonical entity schema (V2, to be written)
