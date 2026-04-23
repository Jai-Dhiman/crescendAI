# Harness Skills (Atoms / Molecules / Compounds)

Three-tier skill catalog. Each tier is a separate directory. The tiering is from *Skill Graphs 2* (Mahler wiki): flat skill graphs lose reliability as depth grows, so reliability-sensitive systems layer skills into near-deterministic atoms, chained molecules, and human-or-hook-driven compounds.

> **Status (2026-04-23):** Three-tier structure established. Catalog files populated by V5 work.

---

## The Three Tiers

### Atoms (`atoms/`)
Single-purpose, near-deterministic building blocks. **Do not call other skills.** Expected target size: ~10-15 files.

Atoms are close to what already exists as tool functions in `apps/api/src/services/`. The V5 work is partly a rename-and-consolidate: name each atom, write a contract for it, guarantee deterministic behavior on a given input.

**Examples:** `compute-velocity-curve`, `compute-pedal-overlap-ratio`, `fetch-student-baseline`, `fetch-reference-percentile`, `align-performance-to-score`, `detect-onset-drift`, `fetch-similar-past-observation`, `classify-stop-moment`, `compute-dimension-delta`.

### Molecules (`molecules/`)
Chain 2-10 atoms with explicit when-to-invoke instructions. **Minimize runtime agent judgment.** Expected target size: 8-12 files.

Each molecule is one pedagogical move -- what we were previously calling an "atomic pedagogical skill." The correction from *Skill Graphs 2*: these are chains of atoms, not atoms themselves. Voicing diagnosis reads baseline, computes velocity balance, compares to reference percentile, fetches similar past observation, proposes finding. That is a chain, not a primitive.

**Examples:** `voicing-diagnosis`, `pedal-triage`, `rubato-coaching`, `phrasing-arc-analysis`, `tempo-stability-triage`, `dynamic-range-audit`, `articulation-clarity-check`, `exercise-proposal`, `cross-modal-contradiction-check`.

### Compounds (`compounds/`)
High-level orchestrators. Run multiple molecules. One compound per hook (event or schedule). Grant the agent meaningful autonomy within a defined scope. Expected target size: 3-5 files.

**Reliability ceiling:** compounds spanning more than 8-10 molecules hit their own reliability failure mode. Keep them tight.

**Examples:** `session-synthesis` (OnSessionEnd), `live-practice-companion` (continuous during recording), `weekly-review` (OnWeeklyReview), `piece-onboarding` (OnPieceDetected first-time).

---

## Why Three Tiers

### Reliability
Every atom must be solid; every molecule must chain dependably; every compound must stay under its ceiling. Testing each tier independently is the only way to catch drift -- testing the whole stack together hides which tier failed. See V4 per-tier reliability plan (`docs/apps/07-evaluation.md`).

### Leverage (operator brain RAM)
The human operator's limiting resource is brain RAM -- the capacity to context-switch across parallel agent threads. Driving atoms wastes that capacity. Driving compounds multiplies output: one compound spans ~10 molecules, spans ~50-100 atomic units of work. For CrescendAI this means the student-facing surface should be at the compound level (sessions, weekly reviews), not at the atom level (raw signals).

### Training (Qwen finetune target)
Molecules are the natural training target. Each molecule has a precise input-output spec and an unambiguous eval signal -- the prerequisite for per-skill reward functions in atomic RL. Composite compound-level training would overfit to task structure; atom-level training is too narrow to capture pedagogical reasoning.

---

## Skill File Shape (applies to all three tiers)

Each skill file has:

1. **YAML frontmatter.** `name`, `tier` (atom/molecule/compound), `description` (5-7 explicit trigger phrases in third person + negative boundaries), `dimensions` (which of the 6 teacher-grounded dimensions this touches), `reads` (input signal contract), `writes` (output artifact spec), `depends_on` (for molecules: which atoms; for compounds: which molecules).
2. **When-to-fire.** Specific signal patterns. Prefer cross-modal patterns (MuQ dim + AMT feature) over single-threshold triggers.
3. **When-NOT-to-fire.** Negative boundaries. Overly broad triggers produce Hijacker skills that fire on unrelated requests.
4. **Procedure.** Testable steps. "Compute velocity curve in bars 12-16, compare against reference percentile" passes; "analyze dynamics" does not.
5. **Concrete example.** One input-signals -> output-artifact example.
6. **Post-conditions.** NLAH contract. What must be true about the output artifact for the skill to have succeeded.

---

## Relationship to the 6 Dimensions

Skills are not 1:1 with the six teacher-grounded dimensions (`docs/model/02-teacher-grounded-taxonomy.md`). A molecule may touch multiple dimensions (rubato-coaching touches timing AND phrasing AND interpretation), and a dimension may be addressed by multiple molecules. Each skill's YAML declares which dimensions it touches so the agent loop can dispatch by signal pattern.

---

## Composition Rules

- **Atoms do not call other skills.** If an atom looks like it needs to call another skill, it is not an atom.
- **Molecules call atoms, not other molecules.** If a molecule looks like it needs to call another molecule, one of them is mis-tiered or the molecule is a compound.
- **Compounds call molecules (and may directly call atoms for utility reads).** Compounds do not call other compounds.
- **Artifacts are the only cross-skill interface.** A molecule's output becomes a compound's input by reference, not by re-reading raw session state. NLAH durable-artifacts primitive.
- **Single write constraint.** A compound may dispatch many molecules for analysis, but the compound writes one teacher-facing artifact. Skills contribute intelligence, not parallel speech.
