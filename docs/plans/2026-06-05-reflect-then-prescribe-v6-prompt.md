# Plan A: Reflect-then-Prescribe — Phase-2 Prompt Extraction & Wording

**Build-agent dispatch note:** This plan is consumed by `/build`. Each task runs in a fresh
Sonnet 4.6 subagent inside an isolated git worktree branched from `issue-11-12-do-chunk-concurrency`.
Tasks execute sequentially (Task 2 depends on Task 1 code). Do not parallelize.

**Goal:** Extract the inline Phase-2 user-prompt assembly in `phase2.ts` into a pure exported
function `buildPhase2Prompt`, then update the wording so the artifact's `headline` is instructed
to be a light reflection ending in exactly one directional question about `dominant_dimension`,
and `proposed_exercises[0]` is instructed to target `dominant_dimension`. No artifact-schema
change. No new molecule, atom, table, or endpoint.

**Spec path:** `docs/specs/2026-06-05-reflect-then-prescribe-design.md`

**Ship-guard note:** Do NOT delete `docs/specs/2026-06-05-reflect-then-prescribe-design.md`
when shipping this plan. The spec is shared across Plan A (this file), the pending-exercise
service plan, and the web plan (`docs/plans/2026-06-05-reflect-then-prescribe-web.md`). Delete
the spec only after the web plan ships.

**Style:** Follow `apps/api/TS_STYLE.md`. No emojis. Explicit exception handling — do not
swallow errors or add silent fallbacks.

**Dependency:** Independent of the staging service plan and the web plan. Recommended merge
order: this plan first, so the in-app reflection wording is correct before the web confirm/deny
gate goes live. `[SHIPS INDEPENDENTLY]`

---

## Task Groups

### Group 1 — Extract `buildPhase2Prompt` (pure refactor, behavior-preserving)

**Goal:** Move the inline user-prompt string assembly out of `runPhase2` into an exported pure
function. `runPhase2` calls it. No behavior change; this is the testable seam that Task 2 will
assert against.

#### Task 1 — Extract pure `buildPhase2Prompt` and lock the current text

**Success criteria:**
1. `buildPhase2Prompt` is exported from `phase2.ts`.
2. `runPhase2` assembles the user prompt by calling `buildPhase2Prompt(ctx.digest, diagnoses, guardrail)`.
3. New tests in `phase2.test.ts` assert the current-text invariants (digest JSON present,
   diagnoses count present, guardrail present when non-empty, `write_synthesis_artifact` tool
   name present) without touching the existing `runPhase2` integration tests.
4. `cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts`
   passes with no skips.

**Step 1 — Write the failing test first.**

Add a new `describe("buildPhase2Prompt")` block to
`apps/api/src/harness/loop/phase2.test.ts`. Import `buildPhase2Prompt` alongside `runPhase2`.
The tests must fail because `buildPhase2Prompt` is not yet exported.

Exact test code to append (after the last `describe` block):

```typescript
import { buildPhase2Prompt } from "./phase2";

describe("buildPhase2Prompt — current text invariants", () => {
  const digest = { dominant_dimension: "phrasing", duration_minutes: 20 };
  const diagnoses = [{ id: "d1" }, { id: "d2" }];
  const guardrail = "This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time.";

  it("contains the digest JSON", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
    expect(prompt).toContain(JSON.stringify(digest, null, 2));
  });

  it("contains the diagnoses count", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
    expect(prompt).toContain(`(${diagnoses.length})`);
  });

  it("contains the guardrail when provided", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
    expect(prompt).toContain(guardrail);
  });

  it("omits the guardrail when empty string", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, "");
    expect(prompt).not.toContain("first session");
  });

  it("contains the write_synthesis_artifact tool name", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
    expect(prompt).toContain("write_synthesis_artifact");
  });
});
```

**Step 2 — Verify the test fails.**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts
```

Expected: import error or "buildPhase2Prompt is not a function" — the new describe block
fails. Existing tests still pass (they do not import `buildPhase2Prompt`). If ALL tests fail,
stop and diagnose.

**Step 3 — Implement.**

Edit `apps/api/src/harness/loop/phase2.ts`:

1. Extract a pure exported function immediately above `runPhase2`:

```typescript
export function buildPhase2Prompt(
  digest: Record<string, unknown>,
  diagnoses: unknown[],
  guardrail: string,
): string {
  return (
    `Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
    `Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
    guardrail +
    `Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
  );
}
```

2. Replace the inline `userPrompt` assembly in `runPhase2` with a call to `buildPhase2Prompt`:

```typescript
const userPrompt = buildPhase2Prompt(ctx.digest, diagnoses, guardrail);
```

The `guardrail` local variable stays as-is (constructed from `FIRST_SESSION_GUARDRAIL` when
`ctx.digest.reference_mode === "within_session"`, otherwise `""`). The `guardrail` string
passed to `buildPhase2Prompt` already contains the trailing `\n\n` when non-empty, or is `""`
when not first-session — match the current behavior exactly.

Note: the existing `BINDING` fixture in the test file uses `artifactToolName: "write_synthesis_artifact"`.
`buildPhase2Prompt` takes a hard-coded tool name string in the final instruction line because the
current inline code also hard-codes the tool name. This is intentional for Task 1 (pure refactor).
Task 2 does not change this.

**Step 4 — Verify the tests pass.**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts
```

All tests — including the pre-existing `runPhase2` integration tests — must pass. Zero skips.

**Step 5 — Commit.**

```bash
git add apps/api/src/harness/loop/phase2.ts apps/api/src/harness/loop/phase2.test.ts
git commit -m "$(cat <<'EOF'
refactor(phase2): extract buildPhase2Prompt pure function as testable seam

Moves inline user-prompt assembly out of runPhase2 into an exported pure
function. runPhase2 delegates to it. No behavior change. New unit tests
lock the current text invariants (digest, diagnoses count, guardrail,
tool name) before wording changes land in Task 2.

Closes part of #reflect-then-prescribe Plan A.
EOF
)"
```

---

### Group 2 — Add reflection + directional-question instructions

**Goal:** Update `buildPhase2Prompt` to additionally instruct the model that (a) `headline`
must be a concise reflection of 2-4 sentences ending in exactly one directional question about
`dominant_dimension`, and (b) `proposed_exercises[0]`, if any, must target `dominant_dimension`.
All Task 1 invariants still hold.

#### Task 2 — Add the new prompt instructions and assert them

**Success criteria:**
1. `buildPhase2Prompt` output contains the reflection+directional-question instruction.
2. `buildPhase2Prompt` output contains the `proposed_exercises[0]` / dominant-dimension
   instruction.
3. All Task 1 assertions still pass (no regression).
4. `cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts`
   passes with no skips.

**Step 1 — Write the failing test first.**

Add a new `describe("buildPhase2Prompt — reflection+prescribe instructions")` block to
`apps/api/src/harness/loop/phase2.test.ts`. These tests must fail because the instructions
are not yet in the prompt text.

```typescript
describe("buildPhase2Prompt — reflection+prescribe instructions", () => {
  const digest = { dominant_dimension: "dynamics", duration_minutes: 30 };
  const diagnoses = [{ id: "d1" }];

  it("instructs headline to be 2-4 sentences ending in one directional question", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, "");
    expect(prompt).toContain("2-4 sentences");
    expect(prompt).toContain("directional question");
    expect(prompt).toContain("dominant_dimension");
  });

  it("instructs proposed_exercises[0] to target dominant_dimension", () => {
    const prompt = buildPhase2Prompt(digest, diagnoses, "");
    expect(prompt).toContain("proposed_exercises[0]");
    expect(prompt).toContain("target the dominant_dimension");
  });

  it("still passes all Task 1 invariants after the new instructions", () => {
    const guardrail = "This is the student's first session -- describe only what happened within this session; do not reference past sessions or claim improvement over time.";
    const prompt = buildPhase2Prompt(digest, diagnoses, guardrail);
    expect(prompt).toContain(JSON.stringify(digest, null, 2));
    expect(prompt).toContain(`(${diagnoses.length})`);
    expect(prompt).toContain(guardrail);
    expect(prompt).toContain("write_synthesis_artifact");
  });
});
```

**Step 2 — Verify the tests fail.**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts
```

Expected: the two new instruction assertions fail. The Task 1 invariant assertion and all
pre-existing tests still pass. If the new assertions pass already, the instruction text was
accidentally added early — stop and diagnose.

**Step 3 — Implement.**

Edit `buildPhase2Prompt` in `apps/api/src/harness/loop/phase2.ts`. Append the new instruction
block between the diagnoses section and the final tool-call line. The full updated function body:

```typescript
export function buildPhase2Prompt(
  digest: Record<string, unknown>,
  diagnoses: unknown[],
  guardrail: string,
): string {
  const reflectionInstruction =
    "Headline instructions: write a light reflection in 2-4 sentences about what happened " +
    "in this session, ending in exactly one directional question about the dominant_dimension " +
    "(e.g. 'Want a drill targeting that?'). The headline must be 300-500 characters total. " +
    "Do not list all dimensions; focus on the one area that matters most.\n\n";

  const exerciseInstruction =
    "Exercise instructions: if proposed_exercises is non-empty, proposed_exercises[0] must " +
    "target the dominant_dimension so that the pre-staged exercise aligns with the question.\n\n";

  return (
    `Session digest:\n${JSON.stringify(digest, null, 2)}\n\n` +
    `Collected diagnoses (${diagnoses.length}):\n${JSON.stringify(diagnoses, null, 2)}\n\n` +
    guardrail +
    reflectionInstruction +
    exerciseInstruction +
    `Write the SynthesisArtifact now using the write_synthesis_artifact tool.`
  );
}
```

**Step 4 — Verify all tests pass.**

```bash
cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts
```

All tests — Task 1 invariants, Task 2 new assertions, and the pre-existing `runPhase2`
integration tests — must pass. Zero skips.

Also run typecheck to confirm no type errors were introduced:

```bash
cd apps/api && bun run typecheck
```

**Step 5 — Commit.**

```bash
git add apps/api/src/harness/loop/phase2.ts apps/api/src/harness/loop/phase2.test.ts
git commit -m "$(cat <<'EOF'
feat(phase2): instruct headline as light reflection + directional question on dominant_dimension

Updates buildPhase2Prompt to include two new instruction blocks:
(1) headline must be a 2-4 sentence reflection ending in exactly one
directional question about dominant_dimension; (2) proposed_exercises[0]
must target dominant_dimension so downstream deterministic staging aligns.
No artifact-schema change. All existing runPhase2 integration tests pass.

Part of reflect-then-prescribe Plan A.
EOF
)"
```

---

## Verification Summary

| Check | Command | Expected |
|-------|---------|----------|
| Phase-2 prompt tests (both tasks) | `cd apps/api && bunx vitest run --config vitest.node.config.ts src/harness/loop/phase2.test.ts` | All pass, zero skips |
| Typecheck | `cd apps/api && bun run typecheck` | No errors |

Manual verification (required, after web plan ships): run `wrangler dev` with
`HARNESS_V6_ENABLED=true`, complete a session, and confirm the synthesis headline reads as a
light reflection ending in a directional question about the session's dominant dimension.
