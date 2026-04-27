# The Harness

Anchor doc for CrescendAI's middle system -- everything between the model outputs and the student's screen. Parallel to `docs/architecture.md` (system-level view) and `docs/model/00-research-timeline.md` (model-level view).

> **Status (2026-04-23):** Naming doc. Most components described here already exist in code but have not been named as a coherent layer. This doc formalizes the vocabulary; `docs/apps/` contains implementation detail; `docs/harness/skills/` contains the three-tier skill catalog (atoms / molecules / compounds, populated by V5 work).

---

## Why "Harness"

CrescendAI's competitive advantage is not which audio model it calls. The advantage is the infrastructure wrapping the model: how signals accumulate into a session brain, how pedagogical moves are selected, how artifacts are produced, how student memory persists. When practitioners say "the harness is the product," this is the layer they mean.

### Four Systems (Model / Harness / Runtime / Client)

The architecture has four systems, not two. A recent distinction from the deep-agents-runtime literature: **harness shapes model behavior; runtime handles machinery.** Conflating the two produces doc drift.

- **Model system** (`docs/model/`) -- MuQ, Aria, AMT, encoders, training data, loss discipline. Outputs signals.
- **Harness system** (this doc) -- context graph, skills, agent loop, student memory, contracts, artifacts. Markdown-first. Turns signals into teaching.
- **Runtime system** -- Cloudflare Workers + Durable Objects + D1 + R2 + AI Gateway + Sentry. Handles durable execution, checkpointing, multi-tenancy, observability, sandbox. Invisible to skill authors.
- **Client system** (`apps/ios/`, `apps/web/`) -- capture, playback, UI, local-first sync. Surfaces teaching.

Naming these separately matters because without the split, harness work gets scattered into `apps/` and runtime concerns leak into skills. They do not belong together.

---

## Vocabulary

Durable primitives the harness is built from. Each has a precise definition; imprecise use breaks downstream reasoning.

- **Signal** -- An immutable emission from the model system (a MuQ 6-dim vector, an AMT midi_notes frame, a STOP probability, a score-following alignment). Signals live in the enrichment cache (see below) and never change after emission.
- **Enrichment cache** -- The Layer 1 store of extracted representations over raw audio. Each audio chunk has multiple coexisting cache entries, one per extraction schema (MuQ-quality, AMT-transcription, STOP-moment, score-alignment). **Prompt-aware keys**: the same audio processed with different schemas produces different entries that coexist and are cross-queryable. See *How to grep video* (Mahler wiki).
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
MuQ (audio encoder), Aria (symbolic encoder), AMT (transcription), STOP classifier, score follower, piece ID. Populates the enrichment cache with prompt-aware keys. Doc home: `docs/model/`.
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
Three-tier skill catalog. ~10-15 atoms, 8-12 molecules, 3-5 compounds. Each with YAML triggers, pre/post contracts, artifact specs. Training target for the Qwen finetune. Doc home: `docs/harness/skills/`.
**Tier:** NOW. Gates Qwen data collection and Phase 1 judge design.

### V6 -- Agent Loop & Orchestration
Teacher loop with deferred tool loading, NLAH contracts, event hooks + middleware hooks. Writes stay single-threaded: skills contribute intelligence, one teacher path writes. Capability-router across providers (Groq / Sonnet / eventually Qwen). Doc home: `docs/apps/02-pipeline.md` (Target section).
**Tier:** NEXT.

### V7 -- Student Memory / Personalization
Typed per-student memory (baseline, recurring_issue, preference, repertoire, goal, breakthrough) with MIA-style multidim retrieval. `STUDENT.md` index per student. Doc home: `docs/apps/03-memory-system.md`.
**Tier:** LATER. Gated on V2 + V5 + V6.

### V8 -- Action, Artifacts, Client
Artifacts as NLAH durable outputs. Direct-action tools that interrupt playthrough and restructure practice (the answer to the Score Following wiki's 90%-playthrough finding). iOS + web as thin clients over the same harness. Doc home: `docs/apps/04-exercises.md`, `docs/apps/05-ui-system.md`.
**Tier:** NEXT for direct-action + chat tool_use; LATER for iOS native loop.

---

## The Two Clocks (recap)

From `docs/apps/03-memory-system.md`, retained here because it is foundational vocabulary:

- **State clock** -- what is true right now (baselines per dimension, current level, goals).
- **Event clock** -- what happened, in what order, with what reasoning (observation history, reasoning traces, synthesized facts).

Most systems build only the state clock. Piano teaching requires both: a teacher who has worked with a student for months knows patterns, not just current skill levels. The context graph in V2 is the event clock made addressable.

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

12. **Provider-agnostic skills; capability-router providers.** Skills are markdown; the same file runs under Sonnet today and under the Qwen finetune tomorrow. Provider choice is a runtime decision. The Groq / Anthropic / Qwen mix is a **capability router** (each model handles what it is best at), not a difficulty escalator. See *Multi-Agents: What's Actually Working*.

13. **Signal ablation is non-negotiable.** Periodically substitute MuQ/AMT signals with plausible fakes. If synthesis outputs are unchanged, the harness is doing text-only reasoning and signals are decorative. See *Music AI Systems* (MuChoMusic finding).

14. **Direct action, not just report.** The harness must be able to interrupt playthrough and restructure practice. 90% of home practice is start-to-finish playthrough -- a passive "listen and synthesize" harness reinforces bad practice structure. See *Score Following and Music Education*.

15. **Harness = memory.** Closed harness = surrendered memory. Typed markdown memory format is the prerequisite for long-term ownership of accumulated student knowledge. See *Agent Harnesses* ("your harness, your memory").

16. **Eval code = prod code.** Playbook.yaml drives both `apps/api/src/services/prompts.*` and `apps/evals/teaching_knowledge/run_eval.py`. Drift between them means the eval measures something that is not shipping. Online eval runs as `after_model` middleware in production.

---

## Priority Stack (2026-04-23)

**DONE** -- V2 (entity schema). Shipped 2026-04-26.

**NOW** -- V4 (eval harness: playbook wiring, signal ablation, atomic-skill rubrics, per-tier reliability), V5 (three-tier skill decomposition: atoms / molecules / compounds).

**NEXT** -- V1 continued (Phase B/C with tightened SemiSupCon positive-mining + musically-informed AMT eval), V3 (compaction policy), V6 (agent loop + event hooks + middleware hooks), V8a (direct-action tools + chat tool_use).

**LATER** -- V7 (student memory typed entries + MIA retrieval), V8b (iOS native inference client), Qwen 27B finetune (gated on V4 plateau + V5 molecules locked + 7B probe pass).

---

## Related Docs

- `docs/architecture.md` -- system view (model + harness + runtime + client)
- `docs/model/00-research-timeline.md` -- model system entry point
- `docs/apps/00-status.md` -- implementation dashboard
- `docs/apps/02-pipeline.md` -- current pipeline + Target: Agent Loop
- `docs/apps/03-memory-system.md` -- two clocks + three layers + enrichment cache
- `docs/apps/07-evaluation.md` -- eval harness including signal ablation and per-tier reliability
- `docs/harness/skills/` -- atoms / molecules / compounds catalog (V5)
- `docs/harness/entities.md` -- canonical entity schema (V2, to be written)
