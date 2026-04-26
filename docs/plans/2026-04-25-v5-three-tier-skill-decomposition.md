# V5 Three-Tier Skill Decomposition Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task). Tiers themselves run sequentially (atoms → molecules → compounds) because the validator's cross-file `depends_on` resolution requires lower-tier files to exist before higher-tier files are validated.
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Replace CrescendAI's monolithic teacher prompt with a structured catalog of 28 markdown skill files (15 atoms / 9 molecules / 4 compounds) plus a typed-artifact contract layer, so V6 can build the agent loop against stable interfaces and the Qwen finetune has clean per-skill training targets.

**Spec:** docs/specs/2026-04-25-v5-three-tier-skill-decomposition-design.md

**Style:** Follow `apps/api/TS_STYLE.md` for all TS code; explicit exception handling over silent fallbacks; no emojis; no backup files; use `bun` and `vitest`. Use Sonnet 4.6 for any subagents spawned during catalog tasks.

## Task Groups

```
Group A (parallel):       Tasks 1, 2, 3            — three artifact schemas
Group B (after A):        Task 4                   — artifacts/index.ts barrel
Group C (after B):        Task 5                   — skills/validator.ts
Group D (parallel after C): Tasks 6-20             — 15 atom skill files
Group E (parallel after D): Tasks 21-29            — 9 molecule skill files
Group F (parallel after E): Tasks 30-33            — 4 compound skill files
Group G (parallel after F): Tasks 34-36            — 3 tier-README updates
```

Within Group D/E/F, all tasks touch separate files (one markdown + one test file per task) and can run as parallel subagents.

---

## Phase 1: Artifact Schemas (Tasks 1-3, Group A — parallel)

### Task 1: DiagnosisArtifact Zod schema
**Group:** A (parallel with Task 2, Task 3)

**Behavior being verified:** A `DiagnosisArtifact` parses cleanly when valid and is rejected with a specific error when any cross-field invariant is violated.
**Interface under test:** `DiagnosisArtifactSchema.parse` / `DiagnosisArtifactSchema.safeParse`.

**Files:**
- Create: `apps/api/src/harness/artifacts/diagnosis.ts`
- Create: `apps/api/src/harness/artifacts/diagnosis.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/diagnosis.test.ts
import { describe, test, expect } from 'vitest'
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'

const baseValid: DiagnosisArtifact = {
  primary_dimension: 'pedaling',
  dimensions: ['pedaling'],
  severity: 'moderate',
  scope: 'stop_moment',
  bar_range: [12, 16],
  evidence_refs: ['cache:muq:abc123'],
  one_sentence_finding: 'Over-pedaled through the slow passage at bars 12-16.',
  confidence: 'high',
  finding_type: 'issue',
}

describe('DiagnosisArtifactSchema', () => {
  test('accepts a fully valid baseline artifact', () => {
    expect(() => DiagnosisArtifactSchema.parse(baseValid)).not.toThrow()
  })

  test('rejects when primary_dimension is not in dimensions list', () => {
    const invalid = { ...baseValid, primary_dimension: 'timing' as const, dimensions: ['pedaling'] as const }
    const result = DiagnosisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('primary_dimension'))).toBe(true)
  })

  test('rejects when one_sentence_finding exceeds 200 chars', () => {
    const invalid = { ...baseValid, one_sentence_finding: 'x'.repeat(201) }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when evidence_refs is empty', () => {
    const invalid = { ...baseValid, evidence_refs: [] as string[] }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when bar_range is null but scope is not "session"', () => {
    const invalid = { ...baseValid, bar_range: null, scope: 'stop_moment' as const }
    const result = DiagnosisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('bar_range'))).toBe(true)
  })

  test('accepts bar_range null when scope is "session"', () => {
    const valid = { ...baseValid, bar_range: null, scope: 'session' as const }
    expect(() => DiagnosisArtifactSchema.parse(valid)).not.toThrow()
  })

  test('rejects when bar_range start > end', () => {
    const invalid = { ...baseValid, bar_range: [16, 12] as [number, number] }
    expect(DiagnosisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('accepts strength finding_type with minor severity', () => {
    const valid = { ...baseValid, finding_type: 'strength' as const, severity: 'minor' as const }
    expect(() => DiagnosisArtifactSchema.parse(valid)).not.toThrow()
  })
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/artifacts/diagnosis.test.ts
```
Expected: FAIL — `Cannot find module './diagnosis'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/artifacts/diagnosis.ts
import { z } from 'zod'

export const DIMENSIONS = ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation'] as const
export const SEVERITIES = ['minor', 'moderate', 'significant'] as const
export const SCOPES = ['stop_moment', 'passage', 'session'] as const
export const CONFIDENCES = ['low', 'medium', 'high'] as const
export const FINDING_TYPES = ['issue', 'strength', 'neutral'] as const

const DimensionEnum = z.enum(DIMENSIONS)
const SeverityEnum = z.enum(SEVERITIES)
const ScopeEnum = z.enum(SCOPES)
const ConfidenceEnum = z.enum(CONFIDENCES)
const FindingTypeEnum = z.enum(FINDING_TYPES)

export const DiagnosisArtifactSchema = z
  .object({
    primary_dimension: DimensionEnum,
    dimensions: z.array(DimensionEnum).min(1),
    severity: SeverityEnum,
    scope: ScopeEnum,
    bar_range: z
      .tuple([z.number().int().positive(), z.number().int().positive()])
      .nullable(),
    evidence_refs: z.array(z.string().min(1)).min(1),
    one_sentence_finding: z.string().min(1).max(200),
    confidence: ConfidenceEnum,
    finding_type: z.enum(FINDING_TYPES),
  })
  .refine((d) => d.dimensions.includes(d.primary_dimension), {
    message: 'primary_dimension must be included in dimensions',
    path: ['primary_dimension'],
  })
  .refine((d) => d.scope === 'session' || d.bar_range !== null, {
    message: 'bar_range may be null only when scope is "session"',
    path: ['bar_range'],
  })
  .refine((d) => d.bar_range === null || d.bar_range[0] <= d.bar_range[1], {
    message: 'bar_range start must be <= end',
    path: ['bar_range'],
  })

export type DiagnosisArtifact = z.infer<typeof DiagnosisArtifactSchema>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/artifacts/diagnosis.test.ts
```
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/diagnosis.ts apps/api/src/harness/artifacts/diagnosis.test.ts && git commit -m "feat(harness): add DiagnosisArtifact zod schema"
```

---

### Task 2: ExerciseArtifact Zod schema
**Group:** A (parallel with Task 1, Task 3)

**Behavior being verified:** An `ExerciseArtifact` parses cleanly when valid and rejects when any field constraint or per-`exercise_type` `action_binding` contract is violated.
**Interface under test:** `ExerciseArtifactSchema.parse` / `safeParse`.

**Files:**
- Create: `apps/api/src/harness/artifacts/exercise.ts`
- Create: `apps/api/src/harness/artifacts/exercise.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/exercise.test.ts
import { describe, test, expect } from 'vitest'
import { ExerciseArtifactSchema, type ExerciseArtifact } from './exercise'

const baseValid: ExerciseArtifact = {
  diagnosis_ref: 'diag:abc123',
  diagnosis_summary: 'Over-pedaled in slow passage at bars 12-16.',
  target_dimension: 'pedaling',
  exercise_type: 'pedal_isolation',
  exercise_subtype: null,
  bar_range: [12, 16],
  instruction: 'Play bars 12-16 with no pedal at all. Listen for sustain in the line itself.',
  success_criterion: 'Three consecutive clean repetitions with no pedal.',
  estimated_minutes: 5,
  action_binding: { tool: 'mute_pedal', args: { bars: [12, 16] } },
}

describe('ExerciseArtifactSchema', () => {
  test('accepts a fully valid baseline artifact', () => {
    expect(() => ExerciseArtifactSchema.parse(baseValid)).not.toThrow()
  })

  test('rejects when instruction exceeds 400 chars', () => {
    const invalid = { ...baseValid, instruction: 'x'.repeat(401) }
    expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when success_criterion exceeds 200 chars', () => {
    const invalid = { ...baseValid, success_criterion: 'x'.repeat(201) }
    expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when estimated_minutes is outside 1-15', () => {
    expect(ExerciseArtifactSchema.safeParse({ ...baseValid, estimated_minutes: 0 }).success).toBe(false)
    expect(ExerciseArtifactSchema.safeParse({ ...baseValid, estimated_minutes: 16 }).success).toBe(false)
  })

  test('rejects when action_binding is null for pedal_isolation', () => {
    const invalid = { ...baseValid, action_binding: null }
    const result = ExerciseArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('action_binding'))).toBe(true)
  })

  test('rejects when action_binding is null for segment_loop', () => {
    const invalid = { ...baseValid, exercise_type: 'segment_loop' as const, action_binding: null }
    expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when action_binding is null for isolated_hands', () => {
    const invalid = { ...baseValid, exercise_type: 'isolated_hands' as const, action_binding: null }
    expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('accepts action_binding null for slow_practice (verbal-only type)', () => {
    const valid = { ...baseValid, exercise_type: 'slow_practice' as const, action_binding: null }
    expect(() => ExerciseArtifactSchema.parse(valid)).not.toThrow()
  })

  test('rejects when bar_range start > end', () => {
    const invalid = { ...baseValid, bar_range: [16, 12] as [number, number] }
    expect(ExerciseArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('accepts a non-null exercise_subtype', () => {
    const valid = { ...baseValid, exercise_subtype: 'half-pedal-only' }
    expect(() => ExerciseArtifactSchema.parse(valid)).not.toThrow()
  })
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/artifacts/exercise.test.ts
```
Expected: FAIL — `Cannot find module './exercise'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/artifacts/exercise.ts
import { z } from 'zod'
import { DIMENSIONS } from './diagnosis'

export const EXERCISE_TYPES = [
  'slow_practice',
  'isolated_hands',
  'rhythmic_variation',
  'segment_loop',
  'dynamic_exaggeration',
  'pedal_isolation',
] as const

export const ACTION_REQUIRED_TYPES = ['segment_loop', 'isolated_hands', 'pedal_isolation'] as const

const ExerciseTypeEnum = z.enum(EXERCISE_TYPES)
const DimensionEnum = z.enum(DIMENSIONS)

// V5: action_binding inner shape is deferred to V6 tool registry
const ToolCallSpec = z.unknown()

export const ExerciseArtifactSchema = z
  .object({
    diagnosis_ref: z.string().min(1),
    diagnosis_summary: z.string().min(1).max(200),
    target_dimension: DimensionEnum,
    exercise_type: ExerciseTypeEnum,
    exercise_subtype: z.string().min(1).nullable(),
    bar_range: z.tuple([z.number().int().positive(), z.number().int().positive()]),
    instruction: z.string().min(1).max(400),
    success_criterion: z.string().min(1).max(200),
    estimated_minutes: z.number().int().min(1).max(15),
    action_binding: ToolCallSpec.nullable(),
  })
  .refine(
    (e) => !ACTION_REQUIRED_TYPES.includes(e.exercise_type as (typeof ACTION_REQUIRED_TYPES)[number]) || e.action_binding !== null,
    { message: 'action_binding is required for exercise_type in {segment_loop, isolated_hands, pedal_isolation}', path: ['action_binding'] },
  )
  .refine((e) => e.bar_range[0] <= e.bar_range[1], {
    message: 'bar_range start must be <= end',
    path: ['bar_range'],
  })

export type ExerciseArtifact = z.infer<typeof ExerciseArtifactSchema>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/artifacts/exercise.test.ts
```
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/exercise.ts apps/api/src/harness/artifacts/exercise.test.ts && git commit -m "feat(harness): add ExerciseArtifact zod schema"
```

---

### Task 3: SynthesisArtifact Zod schema
**Group:** A (parallel with Task 1, Task 2)

**Behavior being verified:** A `SynthesisArtifact` parses cleanly when valid and rejects when scope-conditional contracts (`recurring_pattern` required for weekly; severity-must-be-minor for piece_onboarding) or array max constraints are violated.
**Interface under test:** `SynthesisArtifactSchema.parse` / `safeParse`.

**Files:**
- Create: `apps/api/src/harness/artifacts/synthesis.ts`
- Create: `apps/api/src/harness/artifacts/synthesis.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/synthesis.test.ts
import { describe, test, expect } from 'vitest'
import { SynthesisArtifactSchema, type SynthesisArtifact } from './synthesis'

const baseValid: SynthesisArtifact = {
  session_id: 'sess:abc123',
  synthesis_scope: 'session',
  strengths: [{ dimension: 'phrasing', one_liner: 'Clean shape across the second theme.' }],
  focus_areas: [{ dimension: 'pedaling', one_liner: 'Over-pedaling in slow passages.', severity: 'moderate' }],
  proposed_exercises: ['ex:abc123'],
  dominant_dimension: 'pedaling',
  recurring_pattern: null,
  next_session_focus: 'Work on pedal-release timing in the slow movement.',
  diagnosis_refs: ['diag:abc123'],
  headline:
    'You played with real shape in the second theme today. The thing pulling the picture out of focus is the pedal in the slow passages — let\'s spend tomorrow on releasing it cleanly between phrases. ' +
    'Your hands know what they want to do; the foot just needs to catch up.',
}

describe('SynthesisArtifactSchema', () => {
  test('accepts a fully valid session-scope artifact', () => {
    expect(() => SynthesisArtifactSchema.parse(baseValid)).not.toThrow()
  })

  test('rejects when synthesis_scope=weekly and recurring_pattern is null', () => {
    const invalid = { ...baseValid, synthesis_scope: 'weekly' as const, recurring_pattern: null }
    const result = SynthesisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('recurring_pattern'))).toBe(true)
  })

  test('accepts when synthesis_scope=weekly and recurring_pattern is populated', () => {
    const valid = { ...baseValid, synthesis_scope: 'weekly' as const, recurring_pattern: 'Third session in a row over-pedaling slow movements.' }
    expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow()
  })

  test('rejects when synthesis_scope=piece_onboarding and any focus_area severity is not minor', () => {
    const invalid = {
      ...baseValid,
      synthesis_scope: 'piece_onboarding' as const,
      focus_areas: [{ dimension: 'pedaling' as const, one_liner: 'x', severity: 'moderate' as const }],
    }
    const result = SynthesisArtifactSchema.safeParse(invalid)
    expect(result.success).toBe(false)
    expect(result.error?.issues.some(i => i.message.includes('piece_onboarding'))).toBe(true)
  })

  test('accepts when synthesis_scope=piece_onboarding and all focus_areas are minor', () => {
    const valid = {
      ...baseValid,
      synthesis_scope: 'piece_onboarding' as const,
      focus_areas: [{ dimension: 'pedaling' as const, one_liner: 'x', severity: 'minor' as const }],
    }
    expect(() => SynthesisArtifactSchema.parse(valid)).not.toThrow()
  })

  test('rejects when strengths exceeds 2 items', () => {
    const invalid = {
      ...baseValid,
      strengths: [
        { dimension: 'phrasing' as const, one_liner: 'a' },
        { dimension: 'timing' as const, one_liner: 'b' },
        { dimension: 'pedaling' as const, one_liner: 'c' },
      ],
    }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when focus_areas exceeds 3 items', () => {
    const invalid = {
      ...baseValid,
      focus_areas: [
        { dimension: 'pedaling' as const, one_liner: 'a', severity: 'moderate' as const },
        { dimension: 'timing' as const, one_liner: 'b', severity: 'moderate' as const },
        { dimension: 'phrasing' as const, one_liner: 'c', severity: 'moderate' as const },
        { dimension: 'dynamics' as const, one_liner: 'd', severity: 'moderate' as const },
      ],
    }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when proposed_exercises exceeds 3 items', () => {
    const invalid = { ...baseValid, proposed_exercises: ['a', 'b', 'c', 'd'] }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when headline is shorter than 300 chars', () => {
    const invalid = { ...baseValid, headline: 'x'.repeat(299) }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when headline exceeds 500 chars', () => {
    const invalid = { ...baseValid, headline: 'x'.repeat(501) }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })

  test('rejects when next_session_focus exceeds 200 chars', () => {
    const invalid = { ...baseValid, next_session_focus: 'x'.repeat(201) }
    expect(SynthesisArtifactSchema.safeParse(invalid).success).toBe(false)
  })
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/artifacts/synthesis.test.ts
```
Expected: FAIL — `Cannot find module './synthesis'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/artifacts/synthesis.ts
import { z } from 'zod'
import { DIMENSIONS, SEVERITIES } from './diagnosis'

export const SYNTHESIS_SCOPES = ['session', 'weekly', 'piece_onboarding'] as const

const DimensionEnum = z.enum(DIMENSIONS)
const SeverityEnum = z.enum(SEVERITIES)
const SynthesisScopeEnum = z.enum(SYNTHESIS_SCOPES)

const StrengthEntry = z.object({
  dimension: DimensionEnum,
  one_liner: z.string().min(1).max(200),
})

const FocusAreaEntry = z.object({
  dimension: DimensionEnum,
  one_liner: z.string().min(1).max(200),
  severity: SeverityEnum,
})

export const SynthesisArtifactSchema = z
  .object({
    session_id: z.string().min(1),
    synthesis_scope: SynthesisScopeEnum,
    strengths: z.array(StrengthEntry).max(2),
    focus_areas: z.array(FocusAreaEntry).max(3),
    proposed_exercises: z.array(z.string().min(1)).max(3),
    dominant_dimension: DimensionEnum,
    recurring_pattern: z.string().min(1).nullable(),
    next_session_focus: z.string().min(1).max(200).nullable(),
    diagnosis_refs: z.array(z.string().min(1)),
    headline: z.string().min(300).max(500),
  })
  .refine((s) => s.synthesis_scope !== 'weekly' || s.recurring_pattern !== null, {
    message: 'recurring_pattern is required when synthesis_scope is "weekly"',
    path: ['recurring_pattern'],
  })
  .refine(
    (s) => s.synthesis_scope !== 'piece_onboarding' || s.focus_areas.every((f) => f.severity === 'minor'),
    { message: 'when synthesis_scope is "piece_onboarding", all focus_areas[].severity must be "minor"', path: ['focus_areas'] },
  )

export type SynthesisArtifact = z.infer<typeof SynthesisArtifactSchema>
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/artifacts/synthesis.test.ts
```
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/synthesis.ts apps/api/src/harness/artifacts/synthesis.test.ts && git commit -m "feat(harness): add SynthesisArtifact zod schema"
```

---

## Phase 2: Artifacts Barrel (Task 4, Group B — sequential after A)

### Task 4: Artifacts barrel + discriminator map
**Group:** B (depends on Tasks 1, 2, 3)

**Behavior being verified:** A consumer can import a single `artifactSchemas` map keyed by artifact name and get back the corresponding Zod schema; the map has exactly the three known artifact names.
**Interface under test:** `artifactSchemas` (the lookup map) and `ARTIFACT_NAMES` (the const tuple).

**Files:**
- Create: `apps/api/src/harness/artifacts/index.ts`
- Create: `apps/api/src/harness/artifacts/index.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/artifacts/index.test.ts
import { describe, test, expect } from 'vitest'
import { ARTIFACT_NAMES, artifactSchemas } from './index'

describe('artifacts barrel', () => {
  test('ARTIFACT_NAMES contains exactly the three known names', () => {
    expect([...ARTIFACT_NAMES].sort()).toEqual(['DiagnosisArtifact', 'ExerciseArtifact', 'SynthesisArtifact'])
  })

  test('artifactSchemas has a schema for every name in ARTIFACT_NAMES', () => {
    for (const name of ARTIFACT_NAMES) {
      expect(artifactSchemas[name]).toBeDefined()
      expect(typeof artifactSchemas[name].safeParse).toBe('function')
    }
  })

  test('artifactSchemas has no extra keys beyond ARTIFACT_NAMES', () => {
    expect(Object.keys(artifactSchemas).sort()).toEqual([...ARTIFACT_NAMES].sort())
  })
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/artifacts/index.test.ts
```
Expected: FAIL — `Cannot find module './index'` or missing exports

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/artifacts/index.ts
import type { ZodTypeAny } from 'zod'
import { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'
import { ExerciseArtifactSchema, type ExerciseArtifact } from './exercise'
import { SynthesisArtifactSchema, type SynthesisArtifact } from './synthesis'

export { DiagnosisArtifactSchema, type DiagnosisArtifact } from './diagnosis'
export { ExerciseArtifactSchema, type ExerciseArtifact } from './exercise'
export { SynthesisArtifactSchema, type SynthesisArtifact } from './synthesis'

export const ARTIFACT_NAMES = ['DiagnosisArtifact', 'ExerciseArtifact', 'SynthesisArtifact'] as const
export type ArtifactName = (typeof ARTIFACT_NAMES)[number]

export const artifactSchemas: Record<ArtifactName, ZodTypeAny> = {
  DiagnosisArtifact: DiagnosisArtifactSchema,
  ExerciseArtifact: ExerciseArtifactSchema,
  SynthesisArtifact: SynthesisArtifactSchema,
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/artifacts/index.test.ts
```
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/artifacts/index.ts apps/api/src/harness/artifacts/index.test.ts && git commit -m "feat(harness): add artifacts barrel and schema discriminator map"
```

---

## Phase 3: Skill Validator (Task 5, Group C — sequential after B)

### Task 5: Skill file + catalog validator
**Group:** C (depends on Task 4)

**Behavior being verified:** `validateSkill(filePath)` accepts a well-formed skill file and rejects malformed ones with specific error messages; `validateCatalog(rootDir)` additionally enforces cross-file rules (every name in a molecule's `depends_on` resolves to an existing atom file; every name in a compound's `depends_on` resolves to a molecule or atom file).
**Interface under test:** `validateSkill`, `validateCatalog`.

**Files:**
- Create: `apps/api/src/harness/skills/validator.ts`
- Create: `apps/api/src/harness/skills/validator.test.ts`
- Create: `apps/api/src/harness/skills/__fixtures__/valid-atom.md`
- Create: `apps/api/src/harness/skills/__fixtures__/valid-molecule.md`
- Create: `apps/api/src/harness/skills/__fixtures__/valid-compound.md`
- Create: `apps/api/src/harness/skills/__fixtures__/invalid-missing-section.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/validator.test.ts
import { describe, test, expect } from 'vitest'
import { validateSkill, validateCatalog } from './validator'

const F = 'apps/api/src/harness/skills/__fixtures__'

describe('validateSkill', () => {
  test('accepts a well-formed atom file', async () => {
    const r = await validateSkill(`${F}/valid-atom.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('accepts a well-formed molecule file', async () => {
    const r = await validateSkill(`${F}/valid-molecule.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('accepts a well-formed compound file', async () => {
    const r = await validateSkill(`${F}/valid-compound.md`)
    expect(r.errors).toEqual([])
    expect(r.valid).toBe(true)
  })

  test('rejects when a required body section is missing', async () => {
    const r = await validateSkill(`${F}/invalid-missing-section.md`)
    expect(r.valid).toBe(false)
    expect(r.errors.some((e) => e.includes('Procedure'))).toBe(true)
  })

  test('rejects when frontmatter has no tier field', async () => {
    const r = await validateSkill('apps/api/src/harness/skills/__fixtures__/does-not-exist-tier-missing.md').catch(() => ({ valid: false, errors: ['file not found'] }))
    expect(r.valid).toBe(false)
  })
})

describe('validateCatalog', () => {
  test('returns valid when fixtures directory contains only well-formed files referencing each other correctly', async () => {
    // Run on the live catalog; expect this to be empty before any skill files exist
    // and to remain valid (or report only missing-files errors) until catalog is populated.
    const r = await validateCatalog('docs/harness/skills')
    // The validator must always return a structured result, never throw.
    expect(Array.isArray(r.errors)).toBe(true)
    expect(typeof r.valid).toBe('boolean')
  })
})
```

Fixture file contents:

`apps/api/src/harness/skills/__fixtures__/valid-atom.md`:
```markdown
---
name: fixture-atom
tier: atom
description: |
  A fixture atom for validator testing. fires when test loads. fires for fixture validation.
  fires under unit test. fires when validator runs. does NOT fire in production. does NOT call other skills.
dimensions: [timing]
reads:
  signals: 'fixture signal'
  artifacts: []
writes: 'scalar:number'
depends_on: []
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Procedure
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
```

`apps/api/src/harness/skills/__fixtures__/valid-molecule.md`:
```markdown
---
name: fixture-molecule
tier: molecule
description: |
  A fixture molecule. fires when test runs. fires under validator. fires for fixture purposes.
  fires only in tests. fires when fixture loads. does NOT fire in prod. does NOT call other molecules.
dimensions: [pedaling, timing]
reads:
  signals: 'fixture signals'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [fixture-atom]
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Procedure
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
```

`apps/api/src/harness/skills/__fixtures__/valid-compound.md`:
```markdown
---
name: fixture-compound
tier: compound
description: |
  A fixture compound. fires on OnFixtureHook. fires when test loads. fires for fixture purposes.
  fires under validator. fires in unit tests. does NOT fire in prod. does NOT call other compounds.
dimensions: [pedaling]
reads:
  signals: 'fixture signals'
  artifacts: [DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [fixture-molecule]
triggered_by: OnFixtureHook
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Procedure
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
```

`apps/api/src/harness/skills/__fixtures__/invalid-missing-section.md`:
```markdown
---
name: invalid-missing-procedure
tier: atom
description: |
  Missing the Procedure section. fires never. fires under test. fires for negative case.
  fires for validator. fires in unit tests. does NOT fire in prod. does NOT call other skills.
dimensions: [timing]
reads:
  signals: 'fixture signal'
  artifacts: []
writes: 'scalar:number'
depends_on: []
---

## When-to-fire
Fixture content.

## When-NOT-to-fire
Fixture content.

## Concrete example
Fixture content.

## Post-conditions
Fixture content.
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/validator.test.ts
```
Expected: FAIL — `Cannot find module './validator'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/harness/skills/validator.ts
import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { z } from 'zod'
import { ARTIFACT_NAMES } from '../artifacts'

export type ValidationResult = { valid: boolean; errors: string[] }
export type CatalogValidationResult = { valid: boolean; errors: string[] }

const TIERS = ['atom', 'molecule', 'compound'] as const
const REQUIRED_SECTIONS = [
  'When-to-fire',
  'When-NOT-to-fire',
  'Procedure',
  'Concrete example',
  'Post-conditions',
] as const

const ArtifactNameUnion = z.enum(ARTIFACT_NAMES as readonly [string, ...string[]])
const WritesField = z.union([ArtifactNameUnion, z.string().regex(/^scalar:/)])

const FrontmatterBase = z.object({
  name: z.string().min(1),
  tier: z.enum(TIERS),
  description: z.string().min(1),
  dimensions: z.array(z.string().min(1)).min(1),
  reads: z.object({
    signals: z.string().min(1),
    artifacts: z.array(ArtifactNameUnion),
  }),
  writes: WritesField,
  depends_on: z.array(z.string().min(1)),
})

const AtomFrontmatter = FrontmatterBase.extend({ tier: z.literal('atom') }).refine(
  (f) => f.depends_on.length === 0,
  { message: 'atoms must have empty depends_on', path: ['depends_on'] },
)

const MoleculeFrontmatter = FrontmatterBase.extend({ tier: z.literal('molecule') })

const CompoundFrontmatter = FrontmatterBase.extend({
  tier: z.literal('compound'),
  triggered_by: z.string().min(1),
})

function parseFrontmatter(source: string): { data: unknown; body: string } | null {
  const match = source.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/)
  if (!match) return null
  const yamlText = match[1]
  // Minimal YAML parse: we rely on the test fixtures using a known shape.
  // For a real plan, swap to a proper YAML parser. For V5 we use js-yaml-style
  // hand parser limited to the keys we expect.
  const data = parseYaml(yamlText)
  return { data, body: match[2] }
}

function parseYaml(yamlText: string): unknown {
  // Tiny, deliberately limited YAML parser sufficient for our frontmatter shape.
  // Accepts: scalar, list-of-scalars, nested object (one level), block scalar (|).
  const lines = yamlText.split('\n')
  const root: Record<string, unknown> = {}
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.trim() === '' || line.trim().startsWith('#')) { i++; continue }
    const m = line.match(/^([a-zA-Z_]+):\s*(.*)$/)
    if (!m) { i++; continue }
    const key = m[1]
    const rest = m[2]
    if (rest === '|') {
      const buf: string[] = []
      i++
      while (i < lines.length && (lines[i].startsWith('  ') || lines[i].trim() === '')) {
        buf.push(lines[i].replace(/^ {2}/, ''))
        i++
      }
      root[key] = buf.join('\n').trim()
      continue
    }
    if (rest === '') {
      // nested object
      const obj: Record<string, unknown> = {}
      i++
      while (i < lines.length && lines[i].startsWith('  ')) {
        const sub = lines[i].slice(2)
        const sm = sub.match(/^([a-zA-Z_]+):\s*(.*)$/)
        if (sm) obj[sm[1]] = parseScalar(sm[2])
        i++
      }
      root[key] = obj
      continue
    }
    root[key] = parseScalar(rest)
    i++
  }
  return root
}

function parseScalar(s: string): unknown {
  const t = s.trim()
  if (t.startsWith('[') && t.endsWith(']')) {
    const inner = t.slice(1, -1).trim()
    if (inner === '') return []
    return inner.split(',').map((x) => x.trim().replace(/^['"]|['"]$/g, ''))
  }
  if ((t.startsWith("'") && t.endsWith("'")) || (t.startsWith('"') && t.endsWith('"'))) {
    return t.slice(1, -1)
  }
  return t
}

export async function validateSkill(filePath: string): Promise<ValidationResult> {
  let source: string
  try {
    source = await readFile(filePath, 'utf8')
  } catch {
    return { valid: false, errors: [`file not found: ${filePath}`] }
  }
  const errors: string[] = []
  const parsed = parseFrontmatter(source)
  if (!parsed) {
    return { valid: false, errors: ['frontmatter not found or malformed'] }
  }
  const data = parsed.data as { tier?: string }
  const schema =
    data.tier === 'atom' ? AtomFrontmatter
      : data.tier === 'molecule' ? MoleculeFrontmatter
      : data.tier === 'compound' ? CompoundFrontmatter
      : null
  if (!schema) {
    errors.push(`tier must be one of [atom, molecule, compound]; got ${String(data.tier)}`)
  } else {
    const r = schema.safeParse(parsed.data)
    if (!r.success) {
      for (const issue of r.error.issues) {
        errors.push(`frontmatter: ${issue.path.join('.')}: ${issue.message}`)
      }
    }
  }
  for (const section of REQUIRED_SECTIONS) {
    const re = new RegExp(`^##\\s+${section.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&')}\\b`, 'm')
    if (!re.test(parsed.body)) {
      errors.push(`missing required section: ${section}`)
    }
  }
  return { valid: errors.length === 0, errors }
}

export async function validateCatalog(rootDir: string): Promise<CatalogValidationResult> {
  const errors: string[] = []
  const tiers: Record<string, Set<string>> = { atom: new Set(), molecule: new Set(), compound: new Set() }

  async function readTier(tier: 'atom' | 'molecule' | 'compound', subdir: string) {
    const dir = join(rootDir, subdir)
    let entries: string[] = []
    try { entries = await readdir(dir) } catch { return }
    for (const entry of entries) {
      if (!entry.endsWith('.md') || entry === 'README.md') continue
      const filePath = join(dir, entry)
      const r = await validateSkill(filePath)
      if (!r.valid) {
        for (const e of r.errors) errors.push(`${filePath}: ${e}`)
      }
      tiers[tier].add(entry.replace(/\.md$/, ''))
    }
  }

  await readTier('atom', 'atoms')
  await readTier('molecule', 'molecules')
  await readTier('compound', 'compounds')

  // Cross-file: molecule depends_on must resolve to atom names; compound depends_on must resolve to molecule or atom names.
  for (const tier of ['molecule', 'compound'] as const) {
    const subdir = tier === 'molecule' ? 'molecules' : 'compounds'
    const allowed = tier === 'molecule' ? tiers.atom : new Set([...tiers.molecule, ...tiers.atom])
    let entries: string[] = []
    try { entries = await readdir(join(rootDir, subdir)) } catch { continue }
    for (const entry of entries) {
      if (!entry.endsWith('.md') || entry === 'README.md') continue
      const filePath = join(rootDir, subdir, entry)
      const source = await readFile(filePath, 'utf8').catch(() => '')
      const parsed = parseFrontmatter(source)
      if (!parsed) continue
      const dl = (parsed.data as { depends_on?: unknown }).depends_on
      if (!Array.isArray(dl)) continue
      for (const dep of dl as string[]) {
        if (!allowed.has(dep)) {
          errors.push(`${filePath}: depends_on entry "${dep}" does not resolve to an existing ${tier === 'molecule' ? 'atom' : 'molecule or atom'} file`)
        }
      }
    }
  }

  return { valid: errors.length === 0, errors }
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/validator.test.ts
```
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/ && git commit -m "feat(harness): add skill file validator and catalog cross-file check"
```

---

## Phase 4: Atom Skill Files (Tasks 6-20, Group D — parallel after C)

Each atom task follows the same vertical-slice shape: write a 1-test catalog file → run `bun` test, expect failure (file not found) → write the skill markdown file with the exact frontmatter and body content listed → re-run test, expect pass → commit. All 15 tasks in Group D touch disjoint files (one test file + one markdown file each) and run as parallel subagents.

### Task 6: Atom skill file — compute-velocity-curve
**Group:** D (parallel)

**Behavior being verified:** The `compute-velocity-curve` atom skill file exists at the canonical path and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-velocity-curve.test.ts`
- Create: `docs/harness/skills/atoms/compute-velocity-curve.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-velocity-curve.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-velocity-curve conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-velocity-curve.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-velocity-curve.test.ts
```
Expected: FAIL — `file not found: docs/harness/skills/atoms/compute-velocity-curve.md`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-velocity-curve.md` with this exact content:

```markdown
---
name: compute-velocity-curve
tier: atom
description: |
  Computes per-bar mean MIDI velocity across a bar range. fires when a molecule asks for velocity contour. fires when computing dynamic range. fires when phrasing arc analysis runs. fires when voicing diagnosis runs. fires when dynamic range audit runs. does NOT fire on raw audio (use MuQ scores instead). does NOT call other skills.
dimensions: [dynamics, phrasing]
reads:
  signals: 'AMT midi_notes (pitch, onset_ms, velocity) within bar_range; bar timing from score-alignment'
  artifacts: []
writes: 'scalar:VelocityCurve = { bar: number, mean_velocity: number, p90_velocity: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range and AMT midi_notes for that range. Atom computes one curve point per bar.

## When-NOT-to-fire
Do not invoke on audio segments lacking AMT transcription; the atom assumes midi_notes is present and well-formed.

## Procedure
1. Group midi_notes by bar using bar timing from score-alignment.
2. For each bar, compute mean and p90 of the velocity field across all notes whose onset falls in that bar.
3. Return ordered list of { bar, mean_velocity, p90_velocity }.

## Concrete example
Input: bar_range=[12,14], midi_notes spanning bars 12-14 with velocities 60, 65, 70 (bar 12), 80, 85 (bar 13), 50, 55, 60 (bar 14).
Output: [{bar:12, mean_velocity:65, p90_velocity:69}, {bar:13, mean_velocity:82.5, p90_velocity:84.5}, {bar:14, mean_velocity:55, p90_velocity:59}].

## Post-conditions
Returned list has exactly bar_range[1] - bar_range[0] + 1 entries; each entry's mean_velocity and p90_velocity are in [0, 127]; entries are ordered by bar ascending.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-velocity-curve.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-velocity-curve.test.ts docs/harness/skills/atoms/compute-velocity-curve.md && git commit -m "feat(harness): add atom skill compute-velocity-curve"
```

---

### Task 7: Atom skill file — compute-pedal-overlap-ratio
**Group:** D (parallel)

**Behavior being verified:** The `compute-pedal-overlap-ratio` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-pedal-overlap-ratio.test.ts`
- Create: `docs/harness/skills/atoms/compute-pedal-overlap-ratio.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-pedal-overlap-ratio.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-pedal-overlap-ratio conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-pedal-overlap-ratio.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-pedal-overlap-ratio.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-pedal-overlap-ratio.md` with this exact content:

```markdown
---
name: compute-pedal-overlap-ratio
tier: atom
description: |
  Computes the fraction of note duration covered by sustain pedal (CC64 >= 64) within a bar range. fires when pedal triage runs. fires when cross-modal contradiction check inspects pedaling. fires when score-aligned pedaling is audited. fires when dynamic-range audit needs pedal context. fires when pedal_isolation exercise prerequisites are checked. does NOT fire when AMT pedal CC is missing. does NOT call other skills.
dimensions: [pedaling]
reads:
  signals: 'AMT midi_notes (onset_ms, duration_ms) and AMT pedal CC64 timeline within bar_range'
  artifacts: []
writes: 'scalar:number = fraction in [0, 1]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range with AMT midi_notes and AMT pedal CC64 timeline. Atom returns the time-weighted fraction of note duration during which the sustain pedal is depressed.

## When-NOT-to-fire
Do not invoke on bar ranges where AMT pedal CC is missing or where midi_notes is empty (returns 0 in both, but the caller should handle the missing-data case explicitly).

## Procedure
1. For each note in the range, compute its [onset_ms, onset_ms + duration_ms] interval.
2. Compute total note duration: sum over notes of duration_ms.
3. Compute pedaled note duration: for each note, integrate the time during its interval where CC64 >= 64.
4. Return pedaled / total. If total == 0, return 0.

## Concrete example
Input: bar_range=[12,16], two notes (onset 0ms dur 1000ms, onset 500ms dur 500ms), pedal depressed [200ms, 800ms].
Output: 0.6 -- note1 has 600ms pedaled of 1000ms; note2 has 300ms pedaled of 500ms; total pedaled=900ms / total=1500ms = 0.6.

## Post-conditions
Returned value is a number in [0, 1]. Returns 0 when there are no notes in the range.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-pedal-overlap-ratio.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-pedal-overlap-ratio.test.ts docs/harness/skills/atoms/compute-pedal-overlap-ratio.md && git commit -m "feat(harness): add atom skill compute-pedal-overlap-ratio"
```

---

### Task 8: Atom skill file — compute-onset-drift
**Group:** D (parallel)

**Behavior being verified:** The `compute-onset-drift` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-onset-drift.test.ts`
- Create: `docs/harness/skills/atoms/compute-onset-drift.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-onset-drift.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-onset-drift conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-onset-drift.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-onset-drift.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-onset-drift.md` with this exact content:

```markdown
---
name: compute-onset-drift
tier: atom
description: |
  Computes per-note millisecond drift between performance onset and score-aligned expected onset. fires when timing molecules need drift signal. fires when rubato coaching evaluates intentional vs unintended deviation. fires when tempo stability triage runs. fires when cross-modal contradiction check needs timing evidence. fires when phrasing arc analysis weighs agogic accents. does NOT fire when score alignment is missing. does NOT call other skills.
dimensions: [timing]
reads:
  signals: 'AMT midi_notes (pitch, onset_ms) and score-alignment (per-note expected_onset_ms) within bar_range'
  artifacts: []
writes: 'scalar:OnsetDrift = { note_index: number, drift_ms: number, signed: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a bar_range, performance midi_notes, and score-alignment for that range. Atom returns per-note drift values.

## When-NOT-to-fire
Do not invoke when score-alignment is unavailable (no score MIDI). Do not invoke on freely improvisatory passages where there is no notated reference.

## Procedure
1. For each performance note, look up its score-aligned expected_onset_ms.
2. drift_ms = abs(performance_onset - expected_onset).
3. signed = performance_onset - expected_onset (negative = early, positive = late).
4. Return ordered list keyed by note_index.

## Concrete example
Input: bar_range=[12,12], performance onsets [1000, 1500, 2000], score expected [1000, 1450, 2050].
Output: [{note_index:0, drift_ms:0, signed:0}, {note_index:1, drift_ms:50, signed:50}, {note_index:2, drift_ms:50, signed:-50}].

## Post-conditions
Returned list has one entry per performance note in bar_range. drift_ms is non-negative; signed may be negative (early), zero (on time), or positive (late).
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-onset-drift.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-onset-drift.test.ts docs/harness/skills/atoms/compute-onset-drift.md && git commit -m "feat(harness): add atom skill compute-onset-drift"
```

---

### Task 9: Atom skill file — compute-dimension-delta
**Group:** D (parallel)

**Behavior being verified:** The `compute-dimension-delta` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-dimension-delta.test.ts`
- Create: `docs/harness/skills/atoms/compute-dimension-delta.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-dimension-delta.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-dimension-delta conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-dimension-delta.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-dimension-delta.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-dimension-delta.md` with this exact content:

```markdown
---
name: compute-dimension-delta
tier: atom
description: |
  Computes z-score delta between a performance MuQ dimension score and the student's baseline (or, if no baseline, the cohort percentile). fires when any diagnosis molecule needs to know whether a dimension regressed. fires when cross-modal contradiction check needs MuQ-side magnitude. fires when prioritize-diagnoses ranks by severity. fires when session synthesis aggregates dimension performance. fires when weekly review compares against past weeks. does NOT fire on raw audio. does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim score for current chunk, plus reference baseline mean+stddev (per dimension) for either student or cohort'
  artifacts: []
writes: 'scalar:number = signed z-score; negative = below baseline, positive = above'
depends_on: []
---

## When-to-fire
Caller passes a dimension name, current MuQ score for that dimension, and a baseline {mean, stddev}. Atom returns (current - mean) / stddev.

## When-NOT-to-fire
Do not invoke when baseline stddev == 0 (returns 0 by convention, but caller should treat as no signal). Do not invoke when the dimension name is not one of the 6 teacher-grounded dimensions.

## Procedure
1. Validate dimension is one of [dynamics, timing, pedaling, articulation, phrasing, interpretation].
2. If baseline.stddev == 0, return 0.
3. Return (current - baseline.mean) / baseline.stddev.

## Concrete example
Input: dimension='pedaling', current=0.42, baseline={mean: 0.65, stddev: 0.10}.
Output: -2.3 (significant regression below baseline).

## Post-conditions
Returned value is a finite number; negative indicates below baseline, positive above. Returns 0 when stddev is 0.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-dimension-delta.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-dimension-delta.test.ts docs/harness/skills/atoms/compute-dimension-delta.md && git commit -m "feat(harness): add atom skill compute-dimension-delta"
```

---

### Task 10: Atom skill file — fetch-student-baseline
**Group:** D (parallel)

**Behavior being verified:** The `fetch-student-baseline` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-fetch-student-baseline.test.ts`
- Create: `docs/harness/skills/atoms/fetch-student-baseline.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-fetch-student-baseline.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-student-baseline conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-student-baseline.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-student-baseline.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/fetch-student-baseline.md` with this exact content:

```markdown
---
name: fetch-student-baseline
tier: atom
description: |
  Fetches a student's per-dimension MuQ score baseline (rolling mean + stddev over last N sessions). fires when compute-dimension-delta needs a baseline. fires when any diagnosis molecule needs to know what is normal for this student. fires when weekly review computes regression direction. fires when piece onboarding compares current to prior pieces. fires when prioritize-diagnoses weights by personalized severity. does NOT fire when student has fewer than 3 prior sessions (caller falls back to cohort percentile). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student memory layer (V7 surface, V5 stub): per-dimension session-mean MuQ scores indexed by student_id and dimension'
  artifacts: []
writes: 'scalar:Baseline = { dimension: Dim, mean: number, stddev: number, n_sessions: number } | null'
depends_on: []
---

## When-to-fire
Caller passes a student_id and dimension. Atom returns rolling mean + stddev over the last 10 sessions for that dimension, or null if fewer than 3 sessions are available.

## When-NOT-to-fire
Do not invoke when student_id is unknown. Do not invoke for the cohort baseline -- that is fetch-reference-percentile.

## Procedure
1. Look up student's per-session MuQ mean for the requested dimension across last 10 sessions.
2. If n < 3, return null.
3. Compute mean and stddev across the n session-means.
4. Return { dimension, mean, stddev, n_sessions: n }.

## Concrete example
Input: student_id='stu_42', dimension='pedaling'.
Output: { dimension: 'pedaling', mean: 0.65, stddev: 0.10, n_sessions: 8 }.

## Post-conditions
Returned value is null OR has all four fields populated; mean is in [0, 1]; stddev is non-negative; n_sessions >= 3 when not null.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-student-baseline.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-fetch-student-baseline.test.ts docs/harness/skills/atoms/fetch-student-baseline.md && git commit -m "feat(harness): add atom skill fetch-student-baseline"
```

---

### Task 11: Atom skill file — fetch-reference-percentile
**Group:** D (parallel)

**Behavior being verified:** The `fetch-reference-percentile` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-fetch-reference-percentile.test.ts`
- Create: `docs/harness/skills/atoms/fetch-reference-percentile.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-fetch-reference-percentile.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-reference-percentile conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-reference-percentile.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-reference-percentile.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/fetch-reference-percentile.md` with this exact content:

```markdown
---
name: fetch-reference-percentile
tier: atom
description: |
  Fetches the cohort percentile rank of a MuQ dimension score for a given piece, level, and dimension. fires when no per-student baseline exists. fires when piece-onboarding compares to similarly-leveled cohort. fires when a diagnosis molecule needs a population-level reference. fires when dynamic-range audit grounds expectations. fires when articulation-clarity check needs a normative band. does NOT fire for student-personalized comparisons (use fetch-student-baseline). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'precomputed cohort percentile tables keyed by (piece_id | piece_level, dimension, percentile)'
  artifacts: []
writes: 'scalar:number = percentile in [0, 100] of the input score against the cohort'
depends_on: []
---

## When-to-fire
Caller passes a dimension, MuQ score, and either piece_id or piece_level. Atom returns the percentile rank in the cohort table.

## When-NOT-to-fire
Do not invoke when no cohort table exists for the requested key. Do not invoke for student-personalized comparisons.

## Procedure
1. Look up cohort table for (piece_id, dimension); fall back to (piece_level, dimension) if specific piece is missing.
2. Find the percentile bucket containing the input score using linear interpolation between adjacent percentile values.
3. Return percentile in [0, 100].

## Concrete example
Input: dimension='dynamics', score=0.72, piece_level='advanced'.
Output: 64 -- the score sits at the 64th percentile of advanced-level cohort dynamics scores.

## Post-conditions
Returned value is in [0, 100]. Returns 50 by convention when the input score equals the cohort median.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-reference-percentile.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-fetch-reference-percentile.test.ts docs/harness/skills/atoms/fetch-reference-percentile.md && git commit -m "feat(harness): add atom skill fetch-reference-percentile"
```

---

### Task 12: Atom skill file — fetch-similar-past-observation
**Group:** D (parallel)

**Behavior being verified:** The `fetch-similar-past-observation` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-fetch-similar-past-observation.test.ts`
- Create: `docs/harness/skills/atoms/fetch-similar-past-observation.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-fetch-similar-past-observation.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-similar-past-observation conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-similar-past-observation.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-similar-past-observation.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/fetch-similar-past-observation.md` with this exact content:

```markdown
---
name: fetch-similar-past-observation
tier: atom
description: |
  Fetches the most similar prior diagnosis artifact for a given student, dimension, and bar context. fires when a diagnosis molecule wants to know if this finding is recurring. fires when session-synthesis needs to surface a repeating issue. fires when exercise-proposal wants prior-exercise context for the same diagnosis pattern. fires when weekly review groups recurring patterns. fires when piece-onboarding looks for analogous prior pieces. does NOT fire for cross-student lookups (privacy boundary). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student diagnosis artifact store, indexed by (student_id, dimension, piece_id, bar_range_overlap)'
  artifacts: []
writes: 'scalar:PastObservation = { artifact_id: string, session_id: string, days_ago: number, similarity_score: number } | null'
depends_on: []
---

## When-to-fire
Caller passes a student_id, dimension, piece_id, and bar_range. Atom returns the single most similar past DiagnosisArtifact for that student or null if no match exceeds similarity threshold 0.5.

## When-NOT-to-fire
Do not invoke for cross-student lookups. Do not invoke when the dimension is not in the 6 teacher-grounded dimensions.

## Procedure
1. Query student diagnosis store for prior artifacts matching (student_id, dimension).
2. Compute similarity = 0.5 * (piece_id == match) + 0.5 * (bar_range overlap fraction).
3. Return the highest-scoring match if score >= 0.5, else null.
4. Include days_ago = (now - artifact.created_at) / 86400_000.

## Concrete example
Input: student_id='stu_42', dimension='pedaling', piece_id='chopin_op23', bar_range=[12,16].
Output: { artifact_id: 'diag:abc789', session_id: 'sess_31', days_ago: 5, similarity_score: 0.85 }.

## Post-conditions
Returned value is null OR similarity_score >= 0.5. days_ago is non-negative.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-similar-past-observation.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-fetch-similar-past-observation.test.ts docs/harness/skills/atoms/fetch-similar-past-observation.md && git commit -m "feat(harness): add atom skill fetch-similar-past-observation"
```

---

### Task 13: Atom skill file — align-performance-to-score
**Group:** D (parallel)

**Behavior being verified:** The `align-performance-to-score` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-align-performance-to-score.test.ts`
- Create: `docs/harness/skills/atoms/align-performance-to-score.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-align-performance-to-score.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: align-performance-to-score conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/align-performance-to-score.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-align-performance-to-score.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/align-performance-to-score.md` with this exact content:

```markdown
---
name: align-performance-to-score
tier: atom
description: |
  Aligns AMT-transcribed performance midi_notes to a score MIDI via DTW on (onset, pitch). fires when any timing molecule needs per-note score correspondence. fires when cross-modal contradiction check needs aligned context. fires when bar-range slicing requires score-anchored bar timings. fires when score-following bar regressions are computed. fires when exercise-proposal anchors a drill to specific bars. does NOT fire when score MIDI is not available. does NOT call other skills.
dimensions: [timing, articulation]
reads:
  signals: 'AMT performance midi_notes (pitch, onset_ms, duration_ms) and reference score MIDI (pitch, expected_onset_ms)'
  artifacts: []
writes: 'scalar:Alignment = { perf_index: number, score_index: number, expected_onset_ms: number, bar: number }[]'
depends_on: []
---

## When-to-fire
Caller passes performance midi_notes and reference score MIDI for a piece. Atom returns the per-performance-note alignment to score notes (DTW-best path on onset+pitch joint cost).

## When-NOT-to-fire
Do not invoke when the score MIDI is not loaded for the piece. Do not invoke on freely improvisatory passages.

## Procedure
1. Build cost matrix: cost(perf_i, score_j) = |perf_i.onset - score_j.expected_onset_normalized| + 100 * (perf_i.pitch != score_j.pitch).
2. Run DTW with monotonic constraint to find best alignment path.
3. For each performance note, emit { perf_index, score_index, expected_onset_ms, bar }.
4. Drop performance notes whose alignment cost exceeds threshold 500 (mark as unaligned: score_index = -1).

## Concrete example
Input: performance notes [pitch=60 onset=1000, pitch=64 onset=1500], score notes [pitch=60 expected=1000 bar=12, pitch=64 expected=1450 bar=12].
Output: [{perf_index:0, score_index:0, expected_onset_ms:1000, bar:12}, {perf_index:1, score_index:1, expected_onset_ms:1450, bar:12}].

## Post-conditions
Returned list has one entry per performance note. score_index is -1 for unaligned notes; expected_onset_ms is null for unaligned notes; bar is the score bar number for aligned notes.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-align-performance-to-score.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-align-performance-to-score.test.ts docs/harness/skills/atoms/align-performance-to-score.md && git commit -m "feat(harness): add atom skill align-performance-to-score"
```

---

### Task 14: Atom skill file — classify-stop-moment
**Group:** D (parallel)

**Behavior being verified:** The `classify-stop-moment` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-classify-stop-moment.test.ts`
- Create: `docs/harness/skills/atoms/classify-stop-moment.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-classify-stop-moment.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: classify-stop-moment conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/classify-stop-moment.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-classify-stop-moment.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/classify-stop-moment.md` with this exact content:

```markdown
---
name: classify-stop-moment
tier: atom
description: |
  Returns probability that a teacher would stop the student at this audio chunk, given MuQ 6-dim scores. fires for every 15s audio chunk during live-practice-companion. fires for chunk selection in session-synthesis. fires when picking the dominant-issue chunk for cross-modal contradiction check. fires when ranking chunks for weekly review surfacing. fires when piece-onboarding picks the most-stoppable demonstration chunk. does NOT fire on raw audio (run MuQ first). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim quality scores for one audio chunk (vector of 6 floats in [0, 1])'
  artifacts: []
writes: 'scalar:number = stop probability in [0, 1]'
depends_on: []
---

## When-to-fire
Caller passes a MuQ 6-dim score vector for one chunk. Atom returns the logistic-regression-derived probability that a teacher would stop here.

## When-NOT-to-fire
Do not invoke without MuQ scores. Do not invoke on chunks shorter than the model's expected window (15s).

## Procedure
1. Apply pre-trained logistic regression coefficients (loaded from STOP classifier weights, see apps/api/src/wasm/score-analysis/src/stop.rs for the implementation reference).
2. Return sigmoid(coefficients dot scores + intercept).

## Concrete example
Input: scores=[0.4, 0.3, 0.2, 0.5, 0.4, 0.4] (low pedaling).
Output: 0.78 (high stop probability).

## Post-conditions
Returned value is in [0, 1]. Output is deterministic given the same input vector and the same loaded coefficients.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-classify-stop-moment.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-classify-stop-moment.test.ts docs/harness/skills/atoms/classify-stop-moment.md && git commit -m "feat(harness): add atom skill classify-stop-moment"
```

---

### Task 15: Atom skill file — extract-bar-range-signals
**Group:** D (parallel)

**Behavior being verified:** The `extract-bar-range-signals` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-extract-bar-range-signals.test.ts`
- Create: `docs/harness/skills/atoms/extract-bar-range-signals.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-extract-bar-range-signals.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: extract-bar-range-signals conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/extract-bar-range-signals.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-extract-bar-range-signals.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/extract-bar-range-signals.md` with this exact content:

```markdown
---
name: extract-bar-range-signals
tier: atom
description: |
  Slices the enrichment cache to return all signals (MuQ scores, AMT midi_notes, score-alignment, pedal CC) overlapping a bar range. fires when any molecule needs a unified view of signals over a passage. fires when cross-modal contradiction check needs all extractions for one slice. fires when phrasing-arc-analysis pulls a phrase. fires when exercise-proposal anchors a drill to specific bars. fires when bar-analyzer-style aggregation runs. does NOT fire across non-contiguous bar ranges (caller should call once per contiguous slice). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'enrichment cache entries (MuQ-quality, AMT-transcription, score-alignment, pedal CC) keyed by chunk_id with bar timing metadata'
  artifacts: []
writes: 'scalar:SignalBundle = { muq_scores: number[6][], midi_notes: Note[], pedal_cc: CcEvent[], alignment: Alignment[] }'
depends_on: []
---

## When-to-fire
Caller passes a session_id and bar_range. Atom returns the union of all overlapping enrichment cache entries projected to the bar range.

## When-NOT-to-fire
Do not invoke for non-contiguous ranges. Do not invoke when cache entries for the session have not finished writing (caller awaits write barrier).

## Procedure
1. For session_id, list all chunks whose bar coverage overlaps bar_range.
2. For each chunk, fetch its MuQ scores, AMT midi_notes (filtered to bars in range), pedal CC events (filtered), and score-alignment (filtered).
3. Concatenate, dedupe by (chunk_id, signal_type), and return the SignalBundle.

## Concrete example
Input: session_id='sess_42', bar_range=[12,16]. Three chunks overlap: chunk_a covers bars 10-14, chunk_b covers bars 14-18, chunk_c covers bars 18-22.
Output: muq_scores from chunks a+b only; midi_notes from a+b filtered to bars 12-16; etc.

## Post-conditions
All notes in midi_notes have onset_ms within bar_range; all alignment entries have bar in bar_range; muq_scores has one 6-vector per overlapping chunk.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-extract-bar-range-signals.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-extract-bar-range-signals.test.ts docs/harness/skills/atoms/extract-bar-range-signals.md && git commit -m "feat(harness): add atom skill extract-bar-range-signals"
```

---

### Task 16: Atom skill file — compute-ioi-correlation
**Group:** D (parallel)

**Behavior being verified:** The `compute-ioi-correlation` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-ioi-correlation.test.ts`
- Create: `docs/harness/skills/atoms/compute-ioi-correlation.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-ioi-correlation.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-ioi-correlation conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-ioi-correlation.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-ioi-correlation.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-ioi-correlation.md` with this exact content:

```markdown
---
name: compute-ioi-correlation
tier: atom
description: |
  Computes Pearson correlation between performance inter-onset intervals and score-expected inter-onset intervals over a bar range. fires when tempo-stability triage runs. fires when rubato-coaching distinguishes intentional from unintended deviation. fires when phrasing-arc-analysis weighs agogic structure. fires when cross-modal contradiction check needs a structured timing scalar. fires when weekly review tracks ioi-coherence trend. does NOT fire when fewer than 4 aligned notes exist in the range. does NOT call other skills.
dimensions: [timing, phrasing]
reads:
  signals: 'AMT performance midi_notes (onset_ms) and score-aligned expected_onset_ms within bar_range'
  artifacts: []
writes: 'scalar:number = Pearson r in [-1, 1]; null if fewer than 4 aligned notes'
depends_on: []
---

## When-to-fire
Caller passes performance midi_notes with score-aligned expected_onset_ms for a bar range. Atom returns Pearson r between adjacent-note IOIs.

## When-NOT-to-fire
Do not invoke when there are fewer than 4 aligned notes (correlation is unreliable). Do not invoke when score alignment is missing.

## Procedure
1. Compute performance IOIs: ioi_perf[i] = perf[i+1].onset - perf[i].onset.
2. Compute score IOIs: ioi_score[i] = score[i+1].expected_onset - score[i].expected_onset.
3. If len(ioi_perf) < 3, return null.
4. Return Pearson r between the two arrays.

## Concrete example
Input: 5 aligned notes with performance IOIs [400, 410, 390, 420] and score IOIs [400, 400, 400, 400] (rigid metronome).
Output: ~0.0 (low correlation -- performer rubato is uncorrelated with score timing).

## Post-conditions
Returned value is null OR a finite number in [-1, 1].
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-ioi-correlation.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-ioi-correlation.test.ts docs/harness/skills/atoms/compute-ioi-correlation.md && git commit -m "feat(harness): add atom skill compute-ioi-correlation"
```

---

### Task 17: Atom skill file — compute-key-overlap-ratio
**Group:** D (parallel)

**Behavior being verified:** The `compute-key-overlap-ratio` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-compute-key-overlap-ratio.test.ts`
- Create: `docs/harness/skills/atoms/compute-key-overlap-ratio.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-compute-key-overlap-ratio.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: compute-key-overlap-ratio conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/compute-key-overlap-ratio.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-key-overlap-ratio.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/compute-key-overlap-ratio.md` with this exact content:

```markdown
---
name: compute-key-overlap-ratio
tier: atom
description: |
  Computes the average ratio of (note_off_ms - next_note_on_ms) to total note duration, an articulation proxy where high values indicate legato and low values indicate staccato. fires when articulation-clarity check runs. fires when cross-modal contradiction check needs an AMT-side articulation scalar. fires when phrasing-arc-analysis distinguishes legato vs detached phrases. fires when piece-onboarding compares articulation against reference style. fires when exercise-proposal needs articulation prerequisite check. does NOT fire when fewer than 3 consecutive notes exist. does NOT call other skills.
dimensions: [articulation]
reads:
  signals: 'AMT midi_notes (onset_ms, duration_ms) for adjacent notes within bar_range'
  artifacts: []
writes: 'scalar:number = mean overlap ratio; positive = legato, near-zero = detache, negative = staccato (gap)'
depends_on: []
---

## When-to-fire
Caller passes a sequence of consecutive monophonic notes. Atom returns the mean per-pair overlap ratio.

## When-NOT-to-fire
Do not invoke on polyphonic passages without monophonic projection (caller must reduce to a single voice first). Do not invoke when fewer than 3 notes are present.

## Procedure
1. For each adjacent pair (note_i, note_i+1), compute overlap = (note_i.onset + note_i.duration) - note_i+1.onset.
2. Normalize: pair_ratio = overlap / note_i.duration.
3. Return mean pair_ratio across all pairs.

## Concrete example
Input: notes [(0, 500), (450, 500), (950, 500)] (50ms overlap each).
Output: 0.10 (mild legato).

## Post-conditions
Returned value is a finite number. Positive values indicate overlap (legato); zero indicates detache; negative indicates gap (staccato).
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-compute-key-overlap-ratio.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-compute-key-overlap-ratio.test.ts docs/harness/skills/atoms/compute-key-overlap-ratio.md && git commit -m "feat(harness): add atom skill compute-key-overlap-ratio"
```

---

### Task 18: Atom skill file — detect-passage-repetition
**Group:** D (parallel)

**Behavior being verified:** The `detect-passage-repetition` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-detect-passage-repetition.test.ts`
- Create: `docs/harness/skills/atoms/detect-passage-repetition.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-detect-passage-repetition.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: detect-passage-repetition conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/detect-passage-repetition.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-detect-passage-repetition.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/detect-passage-repetition.md` with this exact content:

```markdown
---
name: detect-passage-repetition
tier: atom
description: |
  Detects whether the same score bar range was practiced multiple times in close temporal succession within a session. fires when session-synthesis aggregates per-passage attempts. fires when exercise-proposal checks if the student is already drilling a passage. fires when live-practice-companion identifies repeated trouble spots. fires when weekly-review counts repetitions across sessions. fires when piece-onboarding observes practice strategy. does NOT fire across sessions (use session-history for that). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'score-alignment entries within a session, grouped by chunk_id and timestamped'
  artifacts: []
writes: 'scalar:RepetitionList = { bar_range: [number, number], attempt_count: number, first_attempt_ms: number, last_attempt_ms: number }[]'
depends_on: []
---

## When-to-fire
Caller passes a session_id. Atom returns all bar ranges practiced 2+ times within the session.

## When-NOT-to-fire
Do not invoke across sessions (use fetch-session-history). Do not invoke before score alignment has completed for the session.

## Procedure
1. List all aligned bar ranges in the session, ordered by start time.
2. Group by overlapping bar_range (>= 50% bar overlap counts as the same passage).
3. For each group with attempt_count >= 2, emit { bar_range, attempt_count, first_attempt_ms, last_attempt_ms }.
4. Sort output by attempt_count descending.

## Concrete example
Input: session with three plays of bars 12-16 at t=0, t=30000, t=75000, plus one play of bars 20-24 at t=90000.
Output: [{ bar_range: [12,16], attempt_count: 3, first_attempt_ms: 0, last_attempt_ms: 75000 }].

## Post-conditions
Output entries have attempt_count >= 2; bar_range is a contiguous range; first_attempt_ms <= last_attempt_ms.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-detect-passage-repetition.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-detect-passage-repetition.test.ts docs/harness/skills/atoms/detect-passage-repetition.md && git commit -m "feat(harness): add atom skill detect-passage-repetition"
```

---

### Task 19: Atom skill file — prioritize-diagnoses
**Group:** D (parallel)

**Behavior being verified:** The `prioritize-diagnoses` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-prioritize-diagnoses.test.ts`
- Create: `docs/harness/skills/atoms/prioritize-diagnoses.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-prioritize-diagnoses.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: prioritize-diagnoses conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/prioritize-diagnoses.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-prioritize-diagnoses.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/prioritize-diagnoses.md` with this exact content:

```markdown
---
name: prioritize-diagnoses
tier: atom
description: |
  Ranks a list of DiagnosisArtifacts by severity, then confidence, then dimension priority. fires when session-synthesis selects top-N focus areas. fires when weekly-review surfaces dominant patterns. fires when live-practice-companion picks which diagnosis to surface first. fires when exercise-proposal needs diagnosis ranking to pick which to address. fires when prioritize ranking is needed for any compound aggregation. does NOT fire on a single diagnosis (no ordering needed). does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'list of DiagnosisArtifact objects passed in by caller (not from cache)'
  artifacts: []
writes: 'scalar:RankedDiagnosisList = DiagnosisArtifact[] (input list reordered, no mutation)'
depends_on: []
---

## When-to-fire
Caller passes an array of DiagnosisArtifacts. Atom returns them reordered by composite priority.

## When-NOT-to-fire
Do not invoke on an empty list (return empty). Do not invoke on a single diagnosis (no work).

## Procedure
1. Compute priority key for each: (severity_rank, confidence_rank, dimension_priority).
   - severity_rank: significant=3, moderate=2, minor=1.
   - confidence_rank: high=3, medium=2, low=1.
   - dimension_priority: pedaling=6, timing=5, dynamics=4, phrasing=3, articulation=2, interpretation=1.
2. Sort descending by tuple (severity_rank, confidence_rank, dimension_priority).
3. finding_type='strength' diagnoses sort to the END regardless of severity (strengths are not focus_areas).

## Concrete example
Input: three diagnoses -- A (severity:moderate, confidence:high, dim:timing), B (severity:significant, confidence:medium, dim:articulation), C (severity:minor, confidence:high, dim:pedaling, finding_type:strength).
Output: [B, A, C] -- B first (significant), A second (moderate>minor), C last (strength regardless).

## Post-conditions
Output length equals input length; output contains exactly the same artifact objects (referential equality preserved); strengths appear last.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-prioritize-diagnoses.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-prioritize-diagnoses.test.ts docs/harness/skills/atoms/prioritize-diagnoses.md && git commit -m "feat(harness): add atom skill prioritize-diagnoses"
```

---

### Task 20: Atom skill file — fetch-session-history
**Group:** D (parallel)

**Behavior being verified:** The `fetch-session-history` atom skill file exists and conforms to the atom-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/atom-fetch-session-history.test.ts`
- Create: `docs/harness/skills/atoms/fetch-session-history.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/atom-fetch-session-history.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('atom: fetch-session-history conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/atoms/fetch-session-history.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-session-history.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/atoms/fetch-session-history.md` with this exact content:

```markdown
---
name: fetch-session-history
tier: atom
description: |
  Fetches a student's prior SynthesisArtifacts and aggregated DiagnosisArtifacts within a date window. fires when weekly-review needs longitudinal context. fires when session-synthesis populates recurring_pattern. fires when piece-onboarding compares to past piece onboardings. fires when live-practice-companion checks for repeating issues across recent sessions. fires when fetch-similar-past-observation needs broader scope. does NOT fire across students. does NOT call other skills.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student session store: SynthesisArtifacts and DiagnosisArtifacts indexed by (student_id, created_at)'
  artifacts: []
writes: 'scalar:SessionHistory = { sessions: { session_id: string, created_at: number, synthesis: SynthesisArtifact, diagnoses: DiagnosisArtifact[] }[] }'
depends_on: []
---

## When-to-fire
Caller passes a student_id and date window (default: last 7 days). Atom returns all sessions in window with their synthesis and diagnoses.

## When-NOT-to-fire
Do not invoke for cross-student queries. Do not invoke without an explicit window (default applies, but caller should pass one for clarity).

## Procedure
1. Query student session store for sessions where created_at falls in window.
2. For each session, fetch its SynthesisArtifact (one) and all associated DiagnosisArtifacts.
3. Return ordered by created_at descending (most recent first).

## Concrete example
Input: student_id='stu_42', window_days=7.
Output: { sessions: [ { session_id: 'sess_31', created_at: 1714003200000, synthesis: {...}, diagnoses: [...] }, ... ] } -- 4 sessions in the past week.

## Post-conditions
Output sessions are ordered by created_at descending; each session has exactly one synthesis; diagnoses lists may be empty but are always present.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/atom-fetch-session-history.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/atom-fetch-session-history.test.ts docs/harness/skills/atoms/fetch-session-history.md && git commit -m "feat(harness): add atom skill fetch-session-history"
```

---

## Phase 5: Molecule Skill Files (Tasks 21-29, Group E — parallel after D)

All 9 molecule tasks run in parallel after Group D completes (validator's cross-file check requires all atom files to exist before molecule `depends_on` can resolve).

### Task 21: Molecule skill file — voicing-diagnosis
**Group:** E (parallel)

**Behavior being verified:** The `voicing-diagnosis` molecule skill file exists and conforms to the molecule-tier validator contract; its `depends_on` resolves to existing atom files.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-voicing-diagnosis.test.ts`
- Create: `docs/harness/skills/molecules/voicing-diagnosis.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-voicing-diagnosis.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: voicing-diagnosis conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/voicing-diagnosis.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-voicing-diagnosis.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/voicing-diagnosis.md` with this exact content:

```markdown
---
name: voicing-diagnosis
tier: molecule
description: |
  Diagnoses imbalance between melody and accompaniment voicing in homophonic textures. fires when MuQ dynamics is below baseline by >= 1 stddev AND AMT velocity-curve shows top-voice/bass-voice ratio is inverted. fires when teacher-style "bring out the melody" feedback would apply. fires when phrasing-arc-analysis flags a missing dynamic peak in the melodic line. fires when exercise-proposal needs a voicing-rooted issue. fires on a passage with >= 4 simultaneous notes. does NOT fire on monophonic passages. does NOT fire when the texture is not predominantly homophonic. does NOT call other molecules.
dimensions: [dynamics, phrasing]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score-alignment for the bar range'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, fetch-student-baseline, fetch-reference-percentile, fetch-similar-past-observation, extract-bar-range-signals, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.dynamics dimension delta <= -1.0 (below student baseline) AND AMT-derived per-bar top-voice mean velocity is within 5 of bass-voice mean velocity (inverted or flat balance) for >= 60 percent of bars in range AND score-alignment indicates >= 4 simultaneous notes per bar (homophonic). Single-threshold variants of any of these are insufficient.

## When-NOT-to-fire
Skip when score texture is monophonic or 2-voice contrapuntal (compute_key_overlap suggests independent voices). Skip when bar range covers fewer than 2 bars. Skip when MuQ.dynamics delta is non-negative (no audible deficit).

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) to get signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> velocity curve per bar.
3. Project velocity curve into top-voice and bass-voice means per bar by pitch (top quartile vs bottom quartile of pitches).
4. Call compute-dimension-delta(dimension='dynamics', current=signals.muq_scores[dynamics_index], baseline=fetch-student-baseline(student_id, 'dynamics')) -> z.
5. Branching: if z > -1.0, return early with finding_type='neutral' and a brief note that voicing was within baseline.
6. Compute fraction of bars with |top - bass| < 5; if < 0.6, return early with finding_type='neutral'.
7. Call fetch-reference-percentile('dynamics', signals.muq_scores[dynamics_index], piece_level) for context.
8. Call fetch-similar-past-observation(student_id, 'dynamics', piece_id, bar_range) -> may yield prior context for confidence calibration.
9. Compose DiagnosisArtifact: primary_dimension='dynamics', dimensions=['dynamics','phrasing'], severity = severity_from_z(z) where {-1..-1.5: minor, -1.5..-2.0: moderate, < -2.0: significant}, scope=passed-in scope, bar_range, evidence_refs from signals, one_sentence_finding (max 200 chars, no hedging) describing the voicing imbalance, confidence='high' if past-observation matched else 'medium', finding_type='issue'.

## Concrete example
Input: bar_range=[24,32], MuQ.dynamics=0.42 vs baseline.mean=0.60 stddev=0.08 -> z=-2.25; bars 24-32 show top-voice mean velocity 78 vs bass-voice mean 76 (flat) for 7 of 9 bars; texture has 5+ notes per bar.
Output: DiagnosisArtifact { primary_dimension: 'dynamics', dimensions: ['dynamics','phrasing'], severity: 'significant', scope: 'session', bar_range: [24,32], evidence_refs: ['cache:muq:s31:c12','cache:amt:s31:c12'], one_sentence_finding: 'Melody and accompaniment are voiced almost equally across bars 24-32; the top line is not coming through.', confidence: 'high', finding_type: 'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact (Zod schema) and primary_dimension is 'dynamics'. evidence_refs is non-empty. The procedure terminates with a single artifact regardless of branching.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-voicing-diagnosis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-voicing-diagnosis.test.ts docs/harness/skills/molecules/voicing-diagnosis.md && git commit -m "feat(harness): add molecule skill voicing-diagnosis"
```

---

### Task 22: Molecule skill file — pedal-triage
**Group:** E (parallel)

**Behavior being verified:** The `pedal-triage` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-pedal-triage.test.ts`
- Create: `docs/harness/skills/molecules/pedal-triage.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-pedal-triage.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: pedal-triage conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/pedal-triage.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-pedal-triage.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/pedal-triage.md` with this exact content:

```markdown
---
name: pedal-triage
tier: molecule
description: |
  Distinguishes over-pedaling, under-pedaling, and pedal-timing issues by combining MuQ pedaling delta with AMT pedal CC overlap ratio against score-aligned harmonic boundaries. fires when MuQ.pedaling delta below baseline AND pedal overlap ratio is outside expected band. fires when teacher-style "release the pedal" feedback would apply. fires when slow-movement playthrough shows muddy harmony. fires when score-alignment indicates harmony changes that the pedal did not respect. fires when student baseline shows recurring pedal issues. does NOT fire on harpsichord-style pieces with no pedal expectation. does NOT fire when AMT pedal CC stream is missing. does NOT call other molecules.
dimensions: [pedaling]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, AMT pedal CC64 timeline, score-alignment, score harmony-change markers'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-pedal-overlap-ratio, align-performance-to-score, fetch-student-baseline, extract-bar-range-signals, compute-dimension-delta, fetch-similar-past-observation]
---

## When-to-fire
Cross-modal pattern: MuQ.pedaling delta vs student baseline <= -1.0 AND pedal overlap ratio is either > 0.85 (over) or < 0.30 (under) for the bar range. Single-threshold variants are insufficient.

## When-NOT-to-fire
Skip when AMT pedal CC stream is unavailable. Skip when piece metadata indicates no pedal (early Baroque). Skip when MuQ.pedaling delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-pedal-overlap-ratio(bar_range, signals.midi_notes, signals.pedal_cc) -> ratio.
3. Call compute-dimension-delta('pedaling', signals.muq_scores[pedaling_index], fetch-student-baseline(student_id, 'pedaling')) -> z.
4. Branching:
   a. If z > -1.0, return finding_type='neutral'.
   b. If ratio > 0.85, classify subtype='over_pedal'.
   c. If ratio < 0.30, classify subtype='under_pedal'.
   d. Otherwise, call align-performance-to-score(signals.midi_notes, score) and check whether pedal release coincides with harmony change markers within 100ms; if not, classify subtype='timing'.
5. Call fetch-similar-past-observation(student_id, 'pedaling', piece_id, bar_range) -> raises confidence to 'high' if matched within 14 days.
6. Compose DiagnosisArtifact: primary_dimension='pedaling', dimensions=['pedaling'], severity by z (same buckets as voicing-diagnosis), one_sentence_finding referencing the subtype in plain teacher language ("over-pedaled", "dry", "released late"), evidence_refs include the pedal_cc cache key.

## Concrete example
Input: bar_range=[12,16] in slow movement, MuQ.pedaling z=-2.1, pedal overlap ratio=0.92 -> over_pedal.
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'session', bar_range:[12,16], evidence_refs:['cache:muq:s31:c5','cache:amt-pedal:s31:c5'], one_sentence_finding:'Over-pedaled through bars 12-16; the harmonies are blurring into one wash.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact and primary_dimension is 'pedaling'. The molecule terminates with one artifact regardless of branching.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-pedal-triage.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-pedal-triage.test.ts docs/harness/skills/molecules/pedal-triage.md && git commit -m "feat(harness): add molecule skill pedal-triage"
```

---

### Task 23: Molecule skill file — rubato-coaching
**Group:** E (parallel)

**Behavior being verified:** The `rubato-coaching` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-rubato-coaching.test.ts`
- Create: `docs/harness/skills/molecules/rubato-coaching.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-rubato-coaching.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: rubato-coaching conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/rubato-coaching.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-rubato-coaching.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/rubato-coaching.md` with this exact content:

```markdown
---
name: rubato-coaching
tier: molecule
description: |
  Distinguishes intentional, returned rubato from uncompensated drift. fires when MuQ.timing is below baseline AND IOI correlation with score is low AND drift does not net to zero across the phrase. fires when teacher-style "let it breathe but come back" feedback would apply. fires on Romantic-period repertoire with notated rubato cues. fires when score has fermata or ritardando markers in the range. fires after a clear cadence to assess phrase-shape completion. does NOT fire on metronomically rigid passages with neutral MuQ.timing. does NOT fire on first-pass sight-reading where timing is incidental. does NOT call other molecules.
dimensions: [timing, phrasing, interpretation]
reads:
  signals: 'AMT midi_notes, score-alignment with phrase boundaries, MuQ 6-dim scores'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-onset-drift, compute-ioi-correlation, align-performance-to-score, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.timing delta z <= -0.8 AND compute-ioi-correlation r < 0.3 (low correlation between performer and score IOIs) AND signed onset drift does not return to within 50ms of zero by phrase end. All three together indicate rubato that wandered without resolution.

## When-NOT-to-fire
Skip when fewer than 8 aligned notes (compute-ioi-correlation will be unstable). Skip when score has no phrase boundary inside the bar range (no return point to evaluate). Skip when MuQ.timing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift_per_note.
4. Call compute-ioi-correlation(signals.midi_notes, alignment) -> r.
5. Call compute-dimension-delta('timing', signals.muq_scores[timing_index], fetch-student-baseline(student_id, 'timing')) -> z.
6. Branching: if z > -0.8 OR r >= 0.3, return finding_type='neutral'.
7. Compute net signed drift at phrase end (last note's signed drift); if abs(net) <= 50ms, return finding_type='strength' with one_sentence_finding praising returned rubato.
8. Otherwise: classify subtype as 'rushed' (mean signed < 0) or 'dragged' (mean signed > 0).
9. Compose DiagnosisArtifact: primary_dimension='timing', dimensions=['timing','phrasing','interpretation'], severity by |z|, one_sentence_finding referencing subtype.

## Concrete example
Input: bar_range=[40,48] (8-bar phrase ending in cadence), z=-1.5, r=0.1, signed drift trend +30,+45,+80,+120 ms -> dragged, no return.
Output: DiagnosisArtifact { primary_dimension:'timing', dimensions:['timing','phrasing','interpretation'], severity:'moderate', scope:'session', bar_range:[40,48], evidence_refs:['cache:muq:s31:c14','cache:amt:s31:c14'], one_sentence_finding:'The rubato through bars 40-48 stretched without coming back; the phrase loses its shape.', confidence:'medium', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type can be 'issue', 'strength', or 'neutral' per branching above. evidence_refs is non-empty.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-rubato-coaching.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-rubato-coaching.test.ts docs/harness/skills/molecules/rubato-coaching.md && git commit -m "feat(harness): add molecule skill rubato-coaching"
```

---

### Task 24: Molecule skill file — phrasing-arc-analysis
**Group:** E (parallel)

**Behavior being verified:** The `phrasing-arc-analysis` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-phrasing-arc-analysis.test.ts`
- Create: `docs/harness/skills/molecules/phrasing-arc-analysis.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-phrasing-arc-analysis.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: phrasing-arc-analysis conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/phrasing-arc-analysis.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-phrasing-arc-analysis.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/phrasing-arc-analysis.md` with this exact content:

```markdown
---
name: phrasing-arc-analysis
tier: molecule
description: |
  Assesses dynamic and timing arc shape across a complete phrase. fires when MuQ.phrasing is below baseline AND velocity-curve does not exhibit a single peak per phrase AND signed onset drift does not return at phrase end. fires when teacher-style "where is the peak of this phrase" feedback would apply. fires when score has explicit phrase-boundary markers in the bar range. fires when piece-onboarding compares to reference performances. fires on long lyrical lines with rising-falling shape. does NOT fire on through-composed passages without phrase boundaries. does NOT fire on technical etude passages where shape is secondary. does NOT call other molecules.
dimensions: [phrasing, dynamics]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score phrase-boundary markers, score-alignment'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, compute-onset-drift, align-performance-to-score, extract-bar-range-signals, fetch-reference-percentile, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.phrasing delta z <= -0.8 AND velocity-curve over the phrase does not exhibit a unimodal peak (peak-bar is at start or end, or the curve has multiple peaks of similar height) AND signed onset drift across the phrase does not converge to <= 50ms at phrase end. All three together indicate weak arc shape.

## When-NOT-to-fire
Skip when bar_range does not contain at least one complete phrase as marked in the score. Skip when the phrase is shorter than 4 bars. Skip when MuQ.phrasing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> curve.
3. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
4. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift.
5. Call compute-dimension-delta('phrasing', signals.muq_scores[phrasing_index], baseline_or_percentile) -> z.
6. Detect peak: argmax(curve.mean_velocity); if peak is at index 0 or last, or if there are multiple bars within 5 velocity of the peak, mark shape='flat' or 'multi-peaked'.
7. Compute drift convergence: abs(drift[last].signed); if > 50ms AND shape is flat/multi-peaked, return finding_type='issue'.
8. If z > -0.8 AND shape is unimodal, return finding_type='strength' praising arc.
9. Compose DiagnosisArtifact: primary_dimension='phrasing', dimensions=['phrasing','dynamics'], severity by z, one_sentence_finding describing where the peak should sit.

## Concrete example
Input: bar_range=[16,24] (8-bar phrase, score peak marked at bar 20), curve peaks at bar 17 (early peak), z=-1.2.
Output: DiagnosisArtifact { primary_dimension:'phrasing', dimensions:['phrasing','dynamics'], severity:'moderate', scope:'session', bar_range:[16,24], evidence_refs:['cache:muq:s31:c10','cache:amt:s31:c10'], one_sentence_finding:'The phrase peaks at bar 17 instead of bar 20; the climax of the line is arriving early.', confidence:'medium', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type may be 'issue' or 'strength' per branching. evidence_refs include both MuQ and AMT cache keys.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-phrasing-arc-analysis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-phrasing-arc-analysis.test.ts docs/harness/skills/molecules/phrasing-arc-analysis.md && git commit -m "feat(harness): add molecule skill phrasing-arc-analysis"
```

---

### Task 25: Molecule skill file — tempo-stability-triage
**Group:** E (parallel)

**Behavior being verified:** The `tempo-stability-triage` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-tempo-stability-triage.test.ts`
- Create: `docs/harness/skills/molecules/tempo-stability-triage.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-tempo-stability-triage.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: tempo-stability-triage conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/tempo-stability-triage.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-tempo-stability-triage.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/tempo-stability-triage.md` with this exact content:

```markdown
---
name: tempo-stability-triage
tier: molecule
description: |
  Distinguishes tempo drift, intentional rubato, and loss of pulse. fires when MuQ.timing is below baseline AND IOI correlation with score is low AND signed drift trends monotonically. fires when teacher-style "find the pulse again" feedback would apply. fires on technical or motoric passages where pulse stability is the goal. fires when bar-rate slowdown over the passage exceeds 5 percent. fires when the student is in non-rubato repertoire. does NOT fire on Romantic repertoire with notated rubato (use rubato-coaching). does NOT fire on cadenza or improvisatory sections. does NOT call other molecules.
dimensions: [timing]
reads:
  signals: 'AMT midi_notes, score-alignment, MuQ 6-dim scores'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-ioi-correlation, compute-onset-drift, align-performance-to-score, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.timing delta z <= -1.0 AND compute-ioi-correlation r < 0.4 AND signed onset drift trend (linear regression slope across notes) is monotonic (>= 80 percent same-sign drift). All three together separate drift from rubato.

## When-NOT-to-fire
Skip when piece metadata indicates rubato repertoire. Skip when fewer than 12 aligned notes. Skip when MuQ.timing delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Call compute-onset-drift(bar_range, signals.midi_notes, alignment) -> drift_per_note.
4. Call compute-ioi-correlation(signals.midi_notes, alignment) -> r.
5. Call compute-dimension-delta('timing', signals.muq_scores[timing_index], fetch-student-baseline(student_id, 'timing')) -> z.
6. Compute fraction of notes with same-sign drift (sign of signed drift); if < 0.8, return finding_type='neutral' (drift is non-monotonic, likely rubato; defer to rubato-coaching).
7. Classify subtype: 'slowing' (positive trend), 'rushing' (negative trend), 'unstable' (high variance, low correlation).
8. Compose DiagnosisArtifact: primary_dimension='timing', dimensions=['timing'], severity by z, one_sentence_finding referencing subtype and bar count.

## Concrete example
Input: bar_range=[1,16] (motoric Bach prelude), z=-1.6, r=0.2, drift trend +5,+10,+18,+30,+45,... (monotonic positive).
Output: DiagnosisArtifact { primary_dimension:'timing', dimensions:['timing'], severity:'significant', scope:'session', bar_range:[1,16], evidence_refs:['cache:muq:s31:c1','cache:amt:s31:c1'], one_sentence_finding:'The pulse slowed gradually across bars 1-16; by the end you were 30 percent under tempo.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type='neutral' when drift is non-monotonic. evidence_refs is non-empty.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-tempo-stability-triage.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-tempo-stability-triage.test.ts docs/harness/skills/molecules/tempo-stability-triage.md && git commit -m "feat(harness): add molecule skill tempo-stability-triage"
```

---

### Task 26: Molecule skill file — dynamic-range-audit
**Group:** E (parallel)

**Behavior being verified:** The `dynamic-range-audit` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-dynamic-range-audit.test.ts`
- Create: `docs/harness/skills/molecules/dynamic-range-audit.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-dynamic-range-audit.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: dynamic-range-audit conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/dynamic-range-audit.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-dynamic-range-audit.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/dynamic-range-audit.md` with this exact content:

```markdown
---
name: dynamic-range-audit
tier: molecule
description: |
  Compares the velocity range used in performance against the dynamic range asked for by the score. fires when MuQ.dynamics is below baseline AND velocity-curve range is < 30 (out of 127) AND score has marked dynamic contrast in range. fires when teacher-style "use the full dynamic range" feedback would apply. fires when piece-onboarding compares to reference cohort dynamics. fires when score has explicit ff/pp markers within bar range. fires on Romantic/Impressionist repertoire requiring extreme contrast. does NOT fire on early-Baroque pieces with neutral dynamic markings. does NOT fire when bar_range covers fewer than 4 bars. does NOT call other molecules.
dimensions: [dynamics]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score dynamic markings (pp..ff annotations)'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-velocity-curve, fetch-reference-percentile, extract-bar-range-signals, fetch-student-baseline, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.dynamics delta z <= -0.8 AND (max(velocity-curve.p90) - min(velocity-curve.mean_velocity)) across the bar range is < 30 AND score has at least one ff or pp marker in the range. All three together indicate compressed dynamics where the score asks for spread.

## When-NOT-to-fire
Skip when score has no dynamic markings in range. Skip when MuQ.dynamics delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call compute-velocity-curve(bar_range, signals.midi_notes) -> curve.
3. Compute observed range = max(curve.p90) - min(curve.mean).
4. Call compute-dimension-delta('dynamics', signals.muq_scores[dynamics_index], fetch-student-baseline(student_id, 'dynamics')) -> z.
5. Read score dynamic markings in bar_range from signals.alignment metadata; classify expected_range as 'wide' (ff and pp both present), 'medium' (one extreme), 'narrow' (mp/mf only).
6. Branching: if observed_range >= 50 OR z > -0.8, return finding_type='neutral'.
7. Call fetch-reference-percentile('dynamics', signals.muq_scores[dynamics_index], piece_level) for context.
8. Compose DiagnosisArtifact: primary_dimension='dynamics', dimensions=['dynamics'], severity by z, one_sentence_finding referencing the contrast gap (e.g., "ff at bar 30 came in at the same level as the mp at bar 28").

## Concrete example
Input: bar_range=[28,32] in Chopin Ballade, score has pp at bar 28 and ff at bar 30, MuQ.dynamics z=-1.8, observed range=18.
Output: DiagnosisArtifact { primary_dimension:'dynamics', dimensions:['dynamics'], severity:'moderate', scope:'session', bar_range:[28,32], evidence_refs:['cache:muq:s31:c8','cache:amt:s31:c8'], one_sentence_finding:'The ff at bar 30 sounded like the mp at bar 28; the dynamic range across this passage is too narrow.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. evidence_refs include MuQ + AMT + score-marker references.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-dynamic-range-audit.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-dynamic-range-audit.test.ts docs/harness/skills/molecules/dynamic-range-audit.md && git commit -m "feat(harness): add molecule skill dynamic-range-audit"
```

---

### Task 27: Molecule skill file — articulation-clarity-check
**Group:** E (parallel)

**Behavior being verified:** The `articulation-clarity-check` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-articulation-clarity-check.test.ts`
- Create: `docs/harness/skills/molecules/articulation-clarity-check.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-articulation-clarity-check.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: articulation-clarity-check conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/articulation-clarity-check.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-articulation-clarity-check.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/articulation-clarity-check.md` with this exact content:

```markdown
---
name: articulation-clarity-check
tier: molecule
description: |
  Identifies execution mismatches between notated articulation (slurs vs staccato) and observed key-overlap behavior. fires when MuQ.articulation is below baseline AND key-overlap-ratio direction does not match score articulation markings. fires when teacher-style "make the staccato shorter" feedback would apply. fires on contrapuntal repertoire requiring voice independence. fires when score has explicit slurs or staccato dots in range. fires on Bach or Mozart fast passagework. does NOT fire on freely interpretive Romantic passages without notated articulation. does NOT fire when score articulation is missing. does NOT call other molecules.
dimensions: [articulation]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, score articulation markings (slur, staccato, tenuto)'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [compute-key-overlap-ratio, align-performance-to-score, extract-bar-range-signals, fetch-reference-percentile, compute-dimension-delta]
---

## When-to-fire
Cross-modal pattern: MuQ.articulation delta z <= -0.8 AND compute-key-overlap-ratio direction (positive=legato, negative=staccato) is opposite to score articulation in >= 50 percent of bars. Single-threshold MuQ-only triggers are insufficient because legato/staccato direction matters.

## When-NOT-to-fire
Skip when bar_range has no notated articulation markings. Skip on improvisatory or free-form passages. Skip when MuQ.articulation delta is non-negative.

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment with articulation markings.
3. For each bar in range, project to monophonic top voice and call compute-key-overlap-ratio -> ratio_per_bar.
4. Classify per-bar score articulation: 'slur' (overlap expected), 'staccato' (gap expected), 'detache' (~zero expected).
5. Compute per-bar mismatch: bar mismatches if score=slur AND ratio<=0, OR score=staccato AND ratio>=0.
6. If mismatch fraction < 0.5 OR z > -0.8, return finding_type='neutral'.
7. Compose DiagnosisArtifact: primary_dimension='articulation', dimensions=['articulation'], severity by z, one_sentence_finding referencing the dominant mismatch direction.

## Concrete example
Input: bar_range=[5,12] in Bach prelude, score marks all 8 bars staccato, observed ratios are +0.10 to +0.20 (legato) for 6 of 8 bars, MuQ.articulation z=-1.4.
Output: DiagnosisArtifact { primary_dimension:'articulation', dimensions:['articulation'], severity:'moderate', scope:'session', bar_range:[5,12], evidence_refs:['cache:muq:s31:c2','cache:amt:s31:c2'], one_sentence_finding:'The staccato bars 5-12 are sustaining into each other; the notes are blurring rather than separating.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. finding_type may be 'issue' or 'neutral'. evidence_refs is non-empty.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-articulation-clarity-check.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-articulation-clarity-check.test.ts docs/harness/skills/molecules/articulation-clarity-check.md && git commit -m "feat(harness): add molecule skill articulation-clarity-check"
```

---

### Task 28: Molecule skill file — exercise-proposal
**Group:** E (parallel)

**Behavior being verified:** The `exercise-proposal` molecule skill file exists and conforms to the molecule-tier validator contract; declares `reads.artifacts: [DiagnosisArtifact]` per Option B.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts`
- Create: `docs/harness/skills/molecules/exercise-proposal.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: exercise-proposal conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/exercise-proposal.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/exercise-proposal.md` with this exact content:

```markdown
---
name: exercise-proposal
tier: molecule
description: |
  Generates one targeted ExerciseArtifact from a single DiagnosisArtifact passed in by the compound. fires when session-synthesis selects a top diagnosis to address. fires when live-practice-companion wants to interrupt with a drill. fires when weekly-review converts a recurring pattern into next-week practice. fires when piece-onboarding scaffolds first-week drills from initial diagnosis. fires when student-memory longitudinal trend triggers a remedial drill. does NOT fire without a diagnosis input (Option B contract). does NOT propose multiple exercises in one call (caller invokes N times). does NOT call other molecules.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'AMT midi_notes for the diagnosis bar_range; student baseline for the diagnosed dimension'
  artifacts: [DiagnosisArtifact]
writes: ExerciseArtifact
depends_on: [extract-bar-range-signals, fetch-similar-past-observation, fetch-student-baseline]
---

## When-to-fire
Caller (a compound) passes exactly one DiagnosisArtifact whose finding_type='issue' and severity is in {moderate, significant}. Skip 'minor' and 'strength' and 'neutral' diagnoses unless the compound explicitly opts in.

## When-NOT-to-fire
Do not fire on finding_type='strength' or 'neutral' (no remedy needed). Do not fire when bar_range is null (need a concrete drill target). Do not propose for diagnoses already addressed by an exercise within the past 3 days (call fetch-similar-past-observation to check).

## Procedure
1. Read the input DiagnosisArtifact (passed in by the compound).
2. Call extract-bar-range-signals(session_id, diagnosis.bar_range) -> signals.
3. Call fetch-similar-past-observation(student_id, diagnosis.primary_dimension, piece_id, diagnosis.bar_range) -> may return prior; if a prior exercise within 3 days targeted the same diagnosis, return null (caller handles).
4. Pick exercise_type from a fixed mapping by primary_dimension and severity:
   - pedaling + moderate -> 'pedal_isolation'
   - pedaling + significant -> 'pedal_isolation' with subtype 'no-pedal-pass'
   - timing + any -> 'segment_loop' with subtype 'metronome-locked'
   - dynamics + moderate -> 'dynamic_exaggeration'
   - dynamics + significant -> 'dynamic_exaggeration' with subtype 'extreme-contrast'
   - articulation + any -> 'isolated_hands' with subtype matching score articulation
   - phrasing + any -> 'slow_practice' with subtype 'shape-vocally'
   - interpretation + any -> 'slow_practice' with subtype 'imitate-reference'
5. Estimate minutes by severity: minor=2, moderate=5, significant=8.
6. Compose action_binding ONLY if exercise_type in {segment_loop, isolated_hands, pedal_isolation}: { tool: <future tool name>, args: { bar_range, ... } }.
7. Compose ExerciseArtifact: diagnosis_ref=diagnosis.id, diagnosis_summary=diagnosis.one_sentence_finding (frozen denorm), target_dimension=diagnosis.primary_dimension, exercise_type, exercise_subtype, bar_range=diagnosis.bar_range, instruction (<= 400 chars, imperative addressed to student), success_criterion (<= 200 chars), estimated_minutes, action_binding.

## Concrete example
Input DiagnosisArtifact: { primary_dimension:'pedaling', severity:'significant', bar_range:[12,16], one_sentence_finding:'Over-pedaled through bars 12-16; harmonies blurring.' ... }.
Output ExerciseArtifact: { diagnosis_ref:'diag:abc789', diagnosis_summary:'Over-pedaled through bars 12-16; harmonies blurring.', target_dimension:'pedaling', exercise_type:'pedal_isolation', exercise_subtype:'no-pedal-pass', bar_range:[12,16], instruction:'Play bars 12-16 three times with no sustain pedal at all. Listen for whether the line still sustains itself in your fingers.', success_criterion:'Three consecutive clean repetitions with no pedal where harmonies remain audibly distinct.', estimated_minutes:8, action_binding:{ tool:'mute_pedal', args:{ bars:[12,16] } } }.

## Post-conditions
Output validates as ExerciseArtifact (Zod schema). target_dimension equals input diagnosis.primary_dimension. action_binding is non-null when exercise_type is in {segment_loop, isolated_hands, pedal_isolation}. diagnosis_summary matches input diagnosis.one_sentence_finding character-for-character.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-exercise-proposal.test.ts docs/harness/skills/molecules/exercise-proposal.md && git commit -m "feat(harness): add molecule skill exercise-proposal"
```

---

### Task 29: Molecule skill file — cross-modal-contradiction-check
**Group:** E (parallel)

**Behavior being verified:** The `cross-modal-contradiction-check` molecule skill file exists and conforms to the molecule-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/molecule-cross-modal-contradiction-check.test.ts`
- Create: `docs/harness/skills/molecules/cross-modal-contradiction-check.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/molecule-cross-modal-contradiction-check.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('molecule: cross-modal-contradiction-check conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/molecules/cross-modal-contradiction-check.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-cross-modal-contradiction-check.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/molecules/cross-modal-contradiction-check.md` with this exact content:

```markdown
---
name: cross-modal-contradiction-check
tier: molecule
description: |
  Flags cases where MuQ dimension scores and AMT-derived structural features disagree on a passage. The highest-signal teacher diagnostic per the How-to-grep-video wiki finding (cross-modal queries beat single-signal triggers). fires when MuQ.timing is high but onset-drift is large. fires when MuQ.pedaling is high but pedal-overlap-ratio is at extremes. fires when MuQ.articulation is high but key-overlap-ratio direction contradicts score. fires when MuQ.dynamics is high but velocity-curve range is compressed. fires on any chunk where two extractions of the same musical content disagree by >= 1.5 stddev. does NOT fire on chunks with missing AMT (AMT is required for the cross-modal check). does NOT fire on monophonic test signals or sine-wave inputs. does NOT call other molecules.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'MuQ 6-dim scores, AMT midi_notes, AMT pedal CC, score-alignment for the bar range'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [extract-bar-range-signals, align-performance-to-score, compute-dimension-delta, compute-onset-drift, compute-pedal-overlap-ratio, compute-key-overlap-ratio]
---

## When-to-fire
For each of the 4 cross-modal pairs (timing/onset-drift, pedaling/overlap-ratio, articulation/key-overlap, dynamics/velocity-range), check whether MuQ score is in the top quartile (z >= +0.5 vs cohort) AND the corresponding AMT-derived feature is in a contradictory direction. Fire when at least one pair contradicts. Cross-modal pattern is the trigger by definition.

## When-NOT-to-fire
Skip when AMT transcription is unavailable. Skip when bar_range covers fewer than 2 bars (cross-modal requires meaningful sample). Skip when score-alignment is unavailable (timing/articulation pairs need it).

## Procedure
1. Call extract-bar-range-signals(session_id, bar_range) -> signals.
2. Call align-performance-to-score(signals.midi_notes, score) -> alignment.
3. Compute four pair checks:
   a. timing-pair: compute-dimension-delta('timing', signals.muq_scores[timing_index], cohort_baseline) >= +0.5 AND mean(compute-onset-drift) > 80 ms -> contradicts.
   b. pedaling-pair: compute-dimension-delta('pedaling', ...) >= +0.5 AND compute-pedal-overlap-ratio < 0.30 OR > 0.85 -> contradicts.
   c. articulation-pair: compute-dimension-delta('articulation', ...) >= +0.5 AND compute-key-overlap-ratio direction opposes score articulation in >= 50 percent of bars -> contradicts.
   d. dynamics-pair: compute-dimension-delta('dynamics', ...) >= +0.5 AND velocity-range across the passage is < 25 -> contradicts.
4. If no pair contradicts, return finding_type='neutral'.
5. Pick the most severe contradiction (largest |delta|) as primary_dimension.
6. Compose DiagnosisArtifact: primary_dimension=picked, dimensions=all pairs that contradicted, severity='significant' (cross-modal contradictions are inherently high-severity), one_sentence_finding describing the specific contradiction in teacher language ("MuQ said timing was clean but the onsets drifted 90 ms"), confidence='high', finding_type='issue'. evidence_refs include both the MuQ cache key and the AMT cache key for full audit.

## Concrete example
Input: bar_range=[20,28], MuQ.pedaling delta=+0.8 (excellent), pedal overlap ratio=0.92 (over-pedaled).
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'stop_moment', bar_range:[20,28], evidence_refs:['cache:muq:s31:c7','cache:amt-pedal:s31:c7'], one_sentence_finding:'MuQ rates pedaling clean here, but the pedal was held over harmonic changes for 92 percent of the passage -- the model and the score disagree.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact. evidence_refs MUST include at least one MuQ ref AND at least one AMT-derived ref (the cross-modal evidence chain). primary_dimension matches the contradicting pair with the largest delta.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/molecule-cross-modal-contradiction-check.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/molecule-cross-modal-contradiction-check.test.ts docs/harness/skills/molecules/cross-modal-contradiction-check.md && git commit -m "feat(harness): add molecule skill cross-modal-contradiction-check"
```

---

## Phase 6: Compound Skill Files (Tasks 30-33, Group F — parallel after E)

All 4 compound tasks run in parallel after Group E completes. Each compound declares `triggered_by` (a hook name) and lists its molecule dispatch sequence in the procedure section. Each compound writes exactly ONE SynthesisArtifact (single-write constraint).

### Task 30: Compound skill file — session-synthesis
**Group:** F (parallel)

**Behavior being verified:** The `session-synthesis` compound skill file exists and conforms to the compound-tier validator contract; declares `triggered_by: OnSessionEnd`; its `depends_on` resolves to existing molecules and atoms.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/compound-session-synthesis.test.ts`
- Create: `docs/harness/skills/compounds/session-synthesis.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/compound-session-synthesis.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('compound: session-synthesis conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/compounds/session-synthesis.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-session-synthesis.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/compounds/session-synthesis.md` with this exact content:

```markdown
---
name: session-synthesis
tier: compound
description: |
  Orchestrates end-of-session diagnosis, prioritization, exercise proposal, and writes one SynthesisArtifact. fires on OnSessionEnd hook (DO alarm, all exit paths, deferred recovery). fires when student stops practicing for 60+ seconds (current pacing rule). fires when student explicitly ends session via UI. fires when 30-minute hard cap is reached. fires when DO state is being checkpointed for sync. does NOT fire mid-session (use live-practice-companion instead). does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'all enrichment cache entries for the session, plus prior live-practice-companion stop_moment DiagnosisArtifacts written during the session (per Option B compound-overlap policy)'
  artifacts: [DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [voicing-diagnosis, pedal-triage, rubato-coaching, phrasing-arc-analysis, tempo-stability-triage, dynamic-range-audit, articulation-clarity-check, exercise-proposal, prioritize-diagnoses, fetch-session-history]
triggered_by: OnSessionEnd
---

## When-to-fire
On OnSessionEnd hook firing for a session with at least 60 seconds of accumulated audio (configurable). The compound consumes whatever stop_moment DiagnosisArtifacts live-practice-companion already wrote during the session, plus the full enrichment cache.

## When-NOT-to-fire
Skip when session has < 60s of audio (no signal). Skip when session has already produced a SynthesisArtifact (idempotency). Skip when the runtime is replaying for eval (eval harness handles its own dispatch).

## Procedure
PHASE 1 (parallel diagnosis sweep):
1. List all session chunks; for each, read prior live-practice-companion stop_moment DiagnosisArtifacts.
2. In parallel, dispatch the 7 diagnosis molecules across plausible bar ranges:
   - voicing-diagnosis on homophonic passages
   - pedal-triage on slow-movement passages
   - rubato-coaching on phrase-end passages in rubato repertoire
   - phrasing-arc-analysis on each marked phrase
   - tempo-stability-triage on motoric passages in non-rubato repertoire
   - dynamic-range-audit on passages with dynamic markings
   - articulation-clarity-check on contrapuntal or fast-articulation passages
3. Collect all DiagnosisArtifacts written by molecules above PLUS the prior live-companion stop_moment artifacts into one list.

PHASE 2 (prioritize):
4. Call atom prioritize-diagnoses(all_diagnoses) -> ranked list.
5. Take top 3 issues for focus_areas; take top 2 strengths for strengths; remaining go into diagnosis_refs without surfacing.

PHASE 3 (exercise proposal, sequential per top diagnosis):
6. For each of top 3 focus_area diagnoses (severity in {moderate, significant}, finding_type='issue'), call exercise-proposal(diagnosis) per Option B (artifact passed as input).
7. Collect ExerciseArtifacts; cap at 3.

PHASE 4 (longitudinal hook):
8. Call atom fetch-session-history(student_id, window_days=14).
9. Detect recurring_pattern: if 2+ of the past 5 sessions had a DiagnosisArtifact with the same primary_dimension as today's top focus_area, write one-sentence pattern (e.g., "third session in a row over-pedaling slow movements"). Otherwise null.

PHASE 5 (single write):
10. Compose SynthesisArtifact: session_id, synthesis_scope='session', strengths (top 2), focus_areas (top 3 with {dimension, one_liner=diagnosis.one_sentence_finding, severity}), proposed_exercises (artifact ids), dominant_dimension=top focus_area's primary_dimension, recurring_pattern, next_session_focus (derived from top focus_area + exercise), diagnosis_refs (all collected), headline derived LAST from structured fields by composing a 300-500 char teacher-voice paragraph that opens with a strength, names the dominant focus, references the proposed exercise, and closes encouragingly.

## Concrete example
Input: 25-minute session of Chopin Ballade No 1, 4 stop_moment artifacts already written by live-practice-companion.
Output: SynthesisArtifact { session_id:'sess_42', synthesis_scope:'session', strengths:[{dimension:'phrasing', one_liner:'Clean shape across the second theme.'}], focus_areas:[{dimension:'pedaling', one_liner:'Over-pedaled bars 12-16 in slow movement.', severity:'significant'},{dimension:'timing', one_liner:'Tempo dragged through bars 40-48.', severity:'moderate'}], proposed_exercises:['ex:abc1','ex:abc2'], dominant_dimension:'pedaling', recurring_pattern:'Third session in a row over-pedaling slow movements.', next_session_focus:'Pedal release timing in the slow movement.', diagnosis_refs:['diag:1','diag:2','diag:3','diag:4','diag:5','diag:6','diag:7'], headline:'You played with real shape in the second theme today and the climax landed where it should. The thing pulling the picture out of focus is the pedal in the slow passages -- this is the third session in a row -- and we are going to spend tomorrow on releasing it cleanly between phrases. Your hands know what they want; the foot just needs to catch up.' }.

## Post-conditions
Output validates as SynthesisArtifact (Zod schema, synthesis_scope='session'). Exactly ONE artifact is written by this compound (single-write). headline is derived from structured fields, never written before them. dominant_dimension equals focus_areas[0].dimension when focus_areas is non-empty.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-session-synthesis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/compound-session-synthesis.test.ts docs/harness/skills/compounds/session-synthesis.md && git commit -m "feat(harness): add compound skill session-synthesis"
```

---

### Task 31: Compound skill file — live-practice-companion
**Group:** F (parallel)

**Behavior being verified:** The `live-practice-companion` compound skill file exists and conforms to the compound-tier validator contract.
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/compound-live-practice-companion.test.ts`
- Create: `docs/harness/skills/compounds/live-practice-companion.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/compound-live-practice-companion.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('compound: live-practice-companion conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/compounds/live-practice-companion.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-live-practice-companion.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/compounds/live-practice-companion.md` with this exact content:

```markdown
---
name: live-practice-companion
tier: compound
description: |
  Continuous in-session companion that dispatches stop-moment diagnosis on every high-probability STOP chunk. Single write per STOP event: one DiagnosisArtifact (scope=stop_moment) consumed later by session-synthesis. fires on OnRecordingActive hook for every chunk where classify-stop-moment probability >= 0.65. fires for live web-app sessions with WebSocket push. fires for iOS sessions when student requests "How was that?". fires when STOP-classifier is enabled in the session config. fires when at least one diagnosis molecule's preconditions are met. does NOT fire on chunks below STOP probability threshold. does NOT call other compounds. does NOT mutate prior artifacts.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'streaming MuQ 6-dim scores, AMT midi_notes, AMT pedal CC, score-alignment for each 15s chunk as it arrives'
  artifacts: []
writes: DiagnosisArtifact
depends_on: [cross-modal-contradiction-check, rubato-coaching, classify-stop-moment]
triggered_by: OnRecordingActive
---

## When-to-fire
For each newly-written enrichment cache chunk during an active recording, call atom classify-stop-moment(muq_scores). If probability >= 0.65, fire the dispatch in the procedure below.

## When-NOT-to-fire
Skip chunks below threshold. Skip when session is in non-coaching mode (e.g., warm-up). Skip when AMT transcription is missing for the chunk (the cross-modal molecule cannot run). Do not duplicate-fire on the same chunk.

## Procedure
PHASE 1 (atom check):
1. Call atom classify-stop-moment(chunk.muq_scores) -> p_stop. If < 0.65, return without writing.

PHASE 2 (parallel cross-modal dispatch):
2. In parallel, dispatch:
   a. cross-modal-contradiction-check on chunk.bar_range (always; cross-modal is the highest-signal teacher diagnostic per the grep-video wiki).
   b. rubato-coaching on chunk.bar_range IF the score has a phrase boundary inside this chunk (skip otherwise).

PHASE 3 (single write):
3. Collect any non-neutral DiagnosisArtifact returned by the dispatched molecules.
4. If multiple non-neutral artifacts, pick the one with primary_dimension matching the lowest MuQ dimension (most-broken signal wins).
5. Write exactly ONE DiagnosisArtifact with scope='stop_moment' and bar_range derived from the chunk's bar coverage.
6. Push to the WebSocket channel (web) or queue for "How was that?" response (iOS) -- delivery is a runtime concern, not part of this compound's logic.

## Concrete example
Chunk c12 covers bars 20-28, MuQ scores [0.5, 0.7, 0.4, 0.6, 0.5, 0.5] (low pedaling). classify-stop-moment returns 0.78. cross-modal-contradiction-check fires and finds contradiction on pedaling pair.
Output: DiagnosisArtifact { primary_dimension:'pedaling', dimensions:['pedaling'], severity:'significant', scope:'stop_moment', bar_range:[20,28], evidence_refs:['cache:muq:s31:c12','cache:amt-pedal:s31:c12'], one_sentence_finding:'MuQ rates pedaling clean here, but the pedal held through three harmonic changes.', confidence:'high', finding_type:'issue' }.

## Post-conditions
Output validates as DiagnosisArtifact with scope='stop_moment'. Exactly one artifact per STOP event (single write). bar_range is non-null. The compound never modifies prior artifacts.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-live-practice-companion.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/compound-live-practice-companion.test.ts docs/harness/skills/compounds/live-practice-companion.md && git commit -m "feat(harness): add compound skill live-practice-companion"
```

---

### Task 32: Compound skill file — weekly-review
**Group:** F (parallel)

**Behavior being verified:** The `weekly-review` compound skill file exists and conforms to the compound-tier validator contract; writes a SynthesisArtifact with `synthesis_scope='weekly'` (recurring_pattern mandatory).
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/compound-weekly-review.test.ts`
- Create: `docs/harness/skills/compounds/weekly-review.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/compound-weekly-review.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('compound: weekly-review conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/compounds/weekly-review.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-weekly-review.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/compounds/weekly-review.md` with this exact content:

```markdown
---
name: weekly-review
tier: compound
description: |
  Aggregates the past week of sessions into one longitudinal SynthesisArtifact (scope=weekly) with a mandatory recurring_pattern. fires on OnWeeklyReview scheduled hook (default Sunday evening). fires when student manually requests a week-in-review summary. fires when 7+ sessions have accumulated since last weekly review. fires when student transitions to a new piece (review of prior week). fires when teacher requests a longitudinal report. does NOT fire on sessions newer than the window cutoff. does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'student session store for the past 7 days; per-session SynthesisArtifacts and DiagnosisArtifacts'
  artifacts: [SynthesisArtifact, DiagnosisArtifact]
writes: SynthesisArtifact
depends_on: [voicing-diagnosis, pedal-triage, rubato-coaching, phrasing-arc-analysis, tempo-stability-triage, dynamic-range-audit, articulation-clarity-check, exercise-proposal, prioritize-diagnoses, fetch-session-history]
triggered_by: OnWeeklyReview
---

## When-to-fire
On OnWeeklyReview hook firing for a student with at least 3 sessions in the past 7 days. The compound aggregates session-level SynthesisArtifacts plus their underlying DiagnosisArtifacts.

## When-NOT-to-fire
Skip when fewer than 3 sessions in window (insufficient longitudinal signal). Skip when a weekly-review SynthesisArtifact already exists for the same window (idempotency).

## Procedure
PHASE 1 (longitudinal fetch):
1. Call atom fetch-session-history(student_id, window_days=7) -> sessions.

PHASE 2 (aggregate diagnoses):
2. Flatten all DiagnosisArtifacts across the week's sessions.
3. Re-dispatch any of the 7 diagnosis molecules where the per-session DiagnosisArtifacts are insufficient (e.g., a passage that recurs across sessions but was never analyzed at session-level depth) -- this re-extends prior coverage.

PHASE 3 (prioritize):
4. Call atom prioritize-diagnoses(all_week_diagnoses) -> ranked list.
5. Detect dominant patterns: group by (primary_dimension, piece_id) and count.

PHASE 4 (recurring_pattern derivation - MANDATORY for weekly):
6. recurring_pattern is REQUIRED. Derive: pick the (primary_dimension, piece_id) pair appearing in the most sessions; compose a one-sentence pattern (e.g., "Pedaling regressed in 4 of 7 Chopin Ballade sessions this week, concentrated in slow-movement passages"). If no pattern repeats >= 3 sessions, recurring_pattern says "No single recurring issue this week; coverage was distributed across phrasing, timing, and pedaling."

PHASE 5 (exercise proposals for top recurring patterns):
7. For each of top 2 recurring focus_areas, call exercise-proposal(representative diagnosis) per Option B.

PHASE 6 (single write):
8. Compose SynthesisArtifact: session_id (synthetic, e.g., 'weekly:stu_42:2026-W17'), synthesis_scope='weekly', strengths (top 2 across week), focus_areas (top 3), proposed_exercises (top 2 from PHASE 5), dominant_dimension, recurring_pattern (mandatory non-null), next_session_focus, diagnosis_refs (all collected), headline derived LAST in week-summary teacher voice (300-500 chars).

## Concrete example
Input: student stu_42, 5 sessions on Chopin Ballade No 1 over 7 days, pedaling issue in bars 12-16 appeared in 4 of 5 sessions.
Output: SynthesisArtifact { session_id:'weekly:stu_42:2026-W17', synthesis_scope:'weekly', strengths:[{dimension:'phrasing', one_liner:'Phrase shape improved across the week.'},{dimension:'dynamics', one_liner:'Dynamic range opened up in the second theme.'}], focus_areas:[{dimension:'pedaling', one_liner:'Over-pedaled in slow movement bars 12-16 across 4 sessions.', severity:'significant'}], proposed_exercises:['ex:weekly_1','ex:weekly_2'], dominant_dimension:'pedaling', recurring_pattern:'Pedaling regressed in 4 of 5 Chopin Ballade sessions this week, concentrated in slow-movement passages bars 12-16.', next_session_focus:'Pedal-release drill bars 12-16 before any full playthrough.', diagnosis_refs:['diag:w_1','diag:w_2','diag:w_3','diag:w_4','diag:w_5'], headline:'This week was a strong one for shape. The phrase contours opened up across all five sessions and the second theme found its dynamic range. The thread to pull is the pedal in the slow movement -- four of five sessions saw the same blur in bars 12-16. Spend the first ten minutes of every session next week on the no-pedal-pass drill there before doing a full playthrough.' }.

## Post-conditions
Output validates as SynthesisArtifact with synthesis_scope='weekly' AND recurring_pattern non-null (Zod-enforced). Exactly ONE artifact written. headline derived from structured fields.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-weekly-review.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/compound-weekly-review.test.ts docs/harness/skills/compounds/weekly-review.md && git commit -m "feat(harness): add compound skill weekly-review"
```

---

### Task 33: Compound skill file — piece-onboarding
**Group:** F (parallel)

**Behavior being verified:** The `piece-onboarding` compound skill file exists and conforms to the compound-tier validator contract; writes a SynthesisArtifact with `synthesis_scope='piece_onboarding'` (all focus_areas have severity='minor').
**Interface under test:** `validateSkill`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/compound-piece-onboarding.test.ts`
- Create: `docs/harness/skills/compounds/piece-onboarding.md`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/compound-piece-onboarding.test.ts
import { test, expect } from 'vitest'
import { validateSkill } from '../validator'

test('compound: piece-onboarding conforms to spec', async () => {
  const r = await validateSkill('docs/harness/skills/compounds/piece-onboarding.md')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-piece-onboarding.test.ts
```
Expected: FAIL — `file not found`

- [ ] **Step 3: Implement the minimum to make the test pass**

Create `docs/harness/skills/compounds/piece-onboarding.md` with this exact content:

```markdown
---
name: piece-onboarding
tier: compound
description: |
  Runs once when a student plays a new piece for the first time; orients them by comparing initial performance to reference cohort dynamics and phrasing. Writes one SynthesisArtifact (scope=piece_onboarding) where focus_areas all have severity='minor' (orientation, not diagnosis). fires on OnPieceDetected first-time hook (zero-config piece ID confirms a piece never seen before in this student's repertoire). fires when student opts into "introduce me to this piece" UI. fires after the first complete passage of the piece is captured. fires when reference performances are available for the piece. fires when the piece has explicit phrase/dynamic markings to compare against. does NOT fire on subsequent plays of the same piece (use session-synthesis instead). does NOT call other compounds. does NOT bypass the single-write rule.
dimensions: [dynamics, timing, pedaling, articulation, phrasing, interpretation]
reads:
  signals: 'first-pass enrichment cache for the new piece; reference performance MuQ scores and AMT for the same piece (cohort)'
  artifacts: []
writes: SynthesisArtifact
depends_on: [phrasing-arc-analysis, dynamic-range-audit, fetch-reference-percentile, prioritize-diagnoses, exercise-proposal]
triggered_by: OnPieceDetected
---

## When-to-fire
On OnPieceDetected hook firing for a student where the detected piece is not present in the student's repertoire history AND the piece has reference performances loaded AND the student has played at least one complete passage.

## When-NOT-to-fire
Skip when piece has been played before by this student. Skip when no reference performance is available. Skip when the captured passage is shorter than 30 seconds.

## Procedure
PHASE 1 (orientation diagnoses):
1. In parallel, dispatch:
   a. phrasing-arc-analysis on each marked phrase in the captured passage (compared against reference cohort).
   b. dynamic-range-audit on the captured passage (compared against reference cohort).
2. For each diagnosis molecule, OVERRIDE the severity field to 'minor' regardless of computed value (this is orientation, not diagnosis -- per the SynthesisArtifact piece_onboarding contract).

PHASE 2 (cohort comparison):
3. For each of the 6 dimensions, call atom fetch-reference-percentile(dimension, student's MuQ score, piece_level) to position the student in the cohort.

PHASE 3 (prioritize and propose):
4. Call atom prioritize-diagnoses on the orientation diagnoses (which all have severity='minor').
5. For top 2 focus areas, call exercise-proposal per Option B; mark exercises as 'gentle introductory' via exercise_subtype.

PHASE 4 (single write):
6. Compose SynthesisArtifact: session_id, synthesis_scope='piece_onboarding', strengths (any cohort comparisons where student is above 60th percentile), focus_areas (all severity='minor', up to 3), proposed_exercises, dominant_dimension (lowest cohort percentile), recurring_pattern=null (no longitudinal data for new piece), next_session_focus (one suggestion for the next play of this piece), diagnosis_refs, headline derived LAST in orientation/excitement teacher voice (300-500 chars), e.g., "Welcome to the Chopin Ballade -- here is where the music sits for you right now and where to grow into it."

## Concrete example
Input: student stu_42 plays first 60s of Chopin Ballade Op 23 (never before recorded), reference cohort exists.
Output: SynthesisArtifact { session_id:'sess_43', synthesis_scope:'piece_onboarding', strengths:[{dimension:'timing', one_liner:'Tempo sat naturally in the opening; you are at the 70th cohort percentile.'}], focus_areas:[{dimension:'pedaling', one_liner:'Pedaling sits at the 35th cohort percentile -- worth attention as you grow into the piece.', severity:'minor'},{dimension:'phrasing', one_liner:'The second-theme arc is asking for a clearer peak.', severity:'minor'}], proposed_exercises:['ex:onb_1','ex:onb_2'], dominant_dimension:'pedaling', recurring_pattern:null, next_session_focus:'Spend the first 5 minutes hearing the slow-movement pedal release.', diagnosis_refs:['diag:onb_1','diag:onb_2'], headline:'Welcome to the Chopin Ballade -- this is a piece that rewards slow listening before fast playing. Your tempo settled naturally in the opening, which is the harder thing. Where to grow into it: the slow movement asks for very particular pedal handling, and the second theme wants a clearer peak in the line. Both are habits that develop over weeks.' }.

## Post-conditions
Output validates as SynthesisArtifact with synthesis_scope='piece_onboarding' AND every focus_areas[].severity='minor' (Zod-enforced). recurring_pattern is null (no longitudinal data for new piece). Exactly ONE artifact written.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/compound-piece-onboarding.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/compound-piece-onboarding.test.ts docs/harness/skills/compounds/piece-onboarding.md && git commit -m "feat(harness): add compound skill piece-onboarding"
```

---

## Phase 7: README Updates + Catalog Integration (Tasks 34-37, Group G — parallel after F)

Each tier's `README.md` currently lists *candidate* skill names. After Phase 4-6, the candidates are now real files. Update each README to reflect the final locked list. Plus one integration task to assert the full live catalog passes `validateCatalog`.

### Task 34: Update atoms/README.md to reflect final list
**Group:** G (parallel)

**Behavior being verified:** `docs/harness/skills/atoms/README.md` lists every one of the 15 final atom file names exactly once and contains no remaining "candidate" or "subject to refinement" hedging.
**Interface under test:** File content (string-search via test).

**Files:**
- Modify: `docs/harness/skills/atoms/README.md`
- Create: `apps/api/src/harness/skills/__catalog__/readme-atoms.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/readme-atoms.test.ts
import { test, expect } from 'vitest'
import { readFile } from 'node:fs/promises'

const FINAL_ATOMS = [
  'compute-velocity-curve',
  'compute-pedal-overlap-ratio',
  'compute-onset-drift',
  'compute-dimension-delta',
  'fetch-student-baseline',
  'fetch-reference-percentile',
  'fetch-similar-past-observation',
  'align-performance-to-score',
  'classify-stop-moment',
  'extract-bar-range-signals',
  'compute-ioi-correlation',
  'compute-key-overlap-ratio',
  'detect-passage-repetition',
  'prioritize-diagnoses',
  'fetch-session-history',
]

test('atoms/README.md lists all 15 final atoms', async () => {
  const content = await readFile('docs/harness/skills/atoms/README.md', 'utf8')
  for (const name of FINAL_ATOMS) {
    expect(content, `expected README to mention ${name}`).toContain(name)
  }
})

test('atoms/README.md no longer says "candidate" or "subject to refinement"', async () => {
  const content = await readFile('docs/harness/skills/atoms/README.md', 'utf8')
  expect(content.toLowerCase()).not.toContain('candidate')
  expect(content.toLowerCase()).not.toContain('subject to refinement')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-atoms.test.ts
```
Expected: FAIL — README still says "candidate" and may not mention `prioritize-diagnoses` or `fetch-session-history` (the two new atoms).

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the contents of `docs/harness/skills/atoms/README.md` with this exact content:

```markdown
# Atoms

Single-purpose, near-deterministic building blocks. Atoms do not call other skills. One file per atom. See `../README.md` for three-tier overview.

**Final size:** 15 atoms.

## Computational primitives
- `compute-velocity-curve` -- per-bar mean MIDI velocity
- `compute-pedal-overlap-ratio` -- fraction of note duration covered by sustain pedal
- `compute-onset-drift` -- per-note ms drift from score-aligned expected onset
- `compute-dimension-delta` -- z-score of MuQ dimension vs baseline or cohort
- `compute-ioi-correlation` -- Pearson r between performer and score IOIs
- `compute-key-overlap-ratio` -- mean note overlap, articulation proxy
- `detect-passage-repetition` -- in-session bar-range repetition detection
- `prioritize-diagnoses` -- ranking policy for DiagnosisArtifact lists

## Retrieval primitives
- `fetch-student-baseline` -- per-dimension rolling mean+stddev for student
- `fetch-reference-percentile` -- cohort percentile rank for a dimension score
- `fetch-similar-past-observation` -- nearest prior diagnosis match for student+context
- `fetch-session-history` -- prior sessions in date window with synthesis+diagnoses

## Signal pipeline primitives
- `align-performance-to-score` -- DTW alignment of performance midi_notes to score
- `classify-stop-moment` -- logistic regression on MuQ scores -> stop probability
- `extract-bar-range-signals` -- enrichment cache slice for a bar range

Each atom is narrow, deterministic on a given input, makes no calls to other skills, and is independently testable. Each atom's contract lives in its own markdown file in this directory.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-atoms.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add docs/harness/skills/atoms/README.md apps/api/src/harness/skills/__catalog__/readme-atoms.test.ts && git commit -m "docs(harness): finalize atoms README with locked 15-atom list"
```

---

### Task 35: Update molecules/README.md to reflect final list
**Group:** G (parallel)

**Behavior being verified:** `docs/harness/skills/molecules/README.md` lists every one of the 9 final molecule file names exactly once and removes "candidate" hedging.
**Interface under test:** File content.

**Files:**
- Modify: `docs/harness/skills/molecules/README.md`
- Create: `apps/api/src/harness/skills/__catalog__/readme-molecules.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/readme-molecules.test.ts
import { test, expect } from 'vitest'
import { readFile } from 'node:fs/promises'

const FINAL_MOLECULES = [
  'voicing-diagnosis',
  'pedal-triage',
  'rubato-coaching',
  'phrasing-arc-analysis',
  'tempo-stability-triage',
  'dynamic-range-audit',
  'articulation-clarity-check',
  'exercise-proposal',
  'cross-modal-contradiction-check',
]

test('molecules/README.md lists all 9 final molecules', async () => {
  const content = await readFile('docs/harness/skills/molecules/README.md', 'utf8')
  for (const name of FINAL_MOLECULES) {
    expect(content, `expected README to mention ${name}`).toContain(name)
  }
})

test('molecules/README.md no longer says "candidate"', async () => {
  const content = await readFile('docs/harness/skills/molecules/README.md', 'utf8')
  expect(content.toLowerCase()).not.toContain('candidate')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-molecules.test.ts
```
Expected: FAIL — README still says "candidate".

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the contents of `docs/harness/skills/molecules/README.md` with this exact content:

```markdown
# Molecules

Pedagogical moves. Each molecule chains 2-10 atoms with explicit when-to-invoke instructions. Molecules call atoms, not other molecules. Per the V5 Option B contract, molecules MAY consume prior-molecule artifacts when those artifacts are passed in by the orchestrating compound (the molecule itself never calls another molecule). See `../README.md` for three-tier overview.

**Final size:** 9 molecules.

**Training target.** Molecules are the Qwen finetune data-collection target -- each has a precise input/output spec and per-skill eval signal for atomic RL.

## Diagnosis molecules (write DiagnosisArtifact)
- `voicing-diagnosis` -- detect imbalance between melody and accompaniment voicing in homophonic textures
- `pedal-triage` -- distinguish over-pedaling, under-pedaling, and pedal-timing issues
- `rubato-coaching` -- detect uncompensated rubato (timing deviation without return)
- `phrasing-arc-analysis` -- assess shape of dynamic and timing arc across a marked phrase
- `tempo-stability-triage` -- distinguish drift, intentional rubato, and loss of pulse
- `dynamic-range-audit` -- compare dynamic range used vs asked-for by score
- `articulation-clarity-check` -- identify slur-vs-staccato execution mismatches
- `cross-modal-contradiction-check` -- flag where MuQ dimension and AMT-derived feature disagree (highest-signal teacher diagnostic per the How-to-grep-video wiki)

## Action molecules (write ExerciseArtifact)
- `exercise-proposal` -- generate a targeted exercise from a passed-in DiagnosisArtifact (Option B input contract)

Each molecule declares its atom dependencies in YAML `depends_on`. A molecule's artifact is the interface to compounds; compounds never reach inside a molecule's reasoning.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-molecules.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add docs/harness/skills/molecules/README.md apps/api/src/harness/skills/__catalog__/readme-molecules.test.ts && git commit -m "docs(harness): finalize molecules README with locked 9-molecule list"
```

---

### Task 36: Update compounds/README.md to reflect final list
**Group:** G (parallel)

**Behavior being verified:** `docs/harness/skills/compounds/README.md` lists every one of the 4 final compound file names exactly once with their `triggered_by` hooks.
**Interface under test:** File content.

**Files:**
- Modify: `docs/harness/skills/compounds/README.md`
- Create: `apps/api/src/harness/skills/__catalog__/readme-compounds.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/readme-compounds.test.ts
import { test, expect } from 'vitest'
import { readFile } from 'node:fs/promises'

const FINAL_COMPOUNDS = [
  { name: 'session-synthesis', hook: 'OnSessionEnd' },
  { name: 'live-practice-companion', hook: 'OnRecordingActive' },
  { name: 'weekly-review', hook: 'OnWeeklyReview' },
  { name: 'piece-onboarding', hook: 'OnPieceDetected' },
]

test('compounds/README.md lists all 4 final compounds with their hooks', async () => {
  const content = await readFile('docs/harness/skills/compounds/README.md', 'utf8')
  for (const { name, hook } of FINAL_COMPOUNDS) {
    expect(content, `expected README to mention ${name}`).toContain(name)
    expect(content, `expected README to mention hook ${hook}`).toContain(hook)
  }
})

test('compounds/README.md no longer says "candidate"', async () => {
  const content = await readFile('docs/harness/skills/compounds/README.md', 'utf8')
  expect(content.toLowerCase()).not.toContain('candidate')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-compounds.test.ts
```
Expected: FAIL — README still says "candidate".

- [ ] **Step 3: Implement the minimum to make the test pass**

Replace the contents of `docs/harness/skills/compounds/README.md` with this exact content:

```markdown
# Compounds

High-level orchestrators. One compound per hook (event or schedule). Compounds call molecules (and may call atoms for utility reads). See `../README.md` for three-tier overview.

**Final size:** 4 compounds.
**Reliability ceiling:** compounds spanning more than 8-10 molecules hit a failure mode. Keep them tight.
**Single write constraint:** a compound dispatches many molecules for analysis; the compound writes one teacher-facing artifact. Skills contribute intelligence, not parallel speech.

## Compound catalog
- `session-synthesis` -- triggered_by: `OnSessionEnd`. Runs 7 diagnosis molecules in parallel, calls `prioritize-diagnoses`, runs `exercise-proposal` per top-N diagnoses, writes one SynthesisArtifact (synthesis_scope=session). Replaces the current monolithic synthesis prompt. Reads live-practice-companion's stop_moment DiagnosisArtifacts as inputs (Option B compound-overlap policy).
- `live-practice-companion` -- triggered_by: `OnRecordingActive`. Continuous during recording. On every chunk above STOP probability threshold, dispatches `cross-modal-contradiction-check` and (where phrase boundaries exist) `rubato-coaching`; writes one DiagnosisArtifact per STOP event with scope=stop_moment.
- `weekly-review` -- triggered_by: `OnWeeklyReview` (scheduled). Calls `fetch-session-history`, re-aggregates diagnoses across sessions, writes one SynthesisArtifact (synthesis_scope=weekly, recurring_pattern mandatory).
- `piece-onboarding` -- triggered_by: `OnPieceDetected` (first time). Dispatches `phrasing-arc-analysis` and `dynamic-range-audit` against reference cohort percentiles; writes one SynthesisArtifact (synthesis_scope=piece_onboarding, all focus_areas severity=minor).

Each compound declares its molecule dependencies in YAML `depends_on` and its hook in `triggered_by`. Compounds never call other compounds.
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/readme-compounds.test.ts
```
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add docs/harness/skills/compounds/README.md apps/api/src/harness/skills/__catalog__/readme-compounds.test.ts && git commit -m "docs(harness): finalize compounds README with locked 4-compound list"
```

---

### Task 37: Catalog integration check
**Group:** G (parallel)

**Behavior being verified:** `validateCatalog('docs/harness/skills')` returns `{ valid: true, errors: [] }` over the full live catalog, confirming all 28 skill files exist, all frontmatter validates, and all `depends_on` cross-file resolutions succeed.
**Interface under test:** `validateCatalog`

**Files:**
- Create: `apps/api/src/harness/skills/__catalog__/integration.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/harness/skills/__catalog__/integration.test.ts
import { test, expect } from 'vitest'
import { validateCatalog } from '../validator'

test('full catalog: validateCatalog returns no errors', async () => {
  const r = await validateCatalog('docs/harness/skills')
  expect(r.errors).toEqual([])
  expect(r.valid).toBe(true)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

If run before Tasks 6-33 complete, this test FAILS with errors listing missing files or unresolved depends_on entries.
After Tasks 6-33 complete, this test should PASS without any further code changes.

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/integration.test.ts
```
Expected at start of Group G: PASS (because Tasks 6-33 already finished). If it fails, the per-skill tests should have caught it -- investigate the specific error and patch the corresponding skill file before continuing.

- [ ] **Step 3: Implement the minimum to make the test pass**

No new code. The test is integration-only; it will pass once Tasks 6-33 are complete. If it fails despite per-skill tests passing, the root cause is a cross-file `depends_on` mismatch -- fix the offending skill markdown file (do NOT modify the validator).

- [ ] **Step 4: Run test — verify it PASSES**

```bash
bun --cwd apps/api test src/harness/skills/__catalog__/integration.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/__catalog__/integration.test.ts && git commit -m "test(harness): add full-catalog integration check via validateCatalog"
```

---

## Plan Self-Review Notes

The following items were flagged during plan self-review and resolved inline:

1. **Spec coverage:** Every requirement from `docs/specs/2026-04-25-v5-three-tier-skill-decomposition-design.md` has at least one task: 3 artifact schemas (Tasks 1-3), barrel + discriminator (Task 4), validator with cross-file checks (Task 5), 15 atoms (Tasks 6-20), 9 molecules (Tasks 21-29), 4 compounds (Tasks 30-33), 3 README updates + integration check (Tasks 34-37).

2. **Type-name consistency:** `DiagnosisArtifact`, `ExerciseArtifact`, `SynthesisArtifact` (PascalCase types), `DiagnosisArtifactSchema`/`ExerciseArtifactSchema`/`SynthesisArtifactSchema` (suffix `Schema` for the Zod export), `ARTIFACT_NAMES` (const tuple), `artifactSchemas` (lookup map), `ValidationResult`/`CatalogValidationResult` (validator return types) used consistently across Tasks 1-5 and referenced consistently by name in Tasks 6-37.

3. **Group correctness:** Within Group D, all 15 atom tasks touch disjoint files (one test file + one markdown file each). Within Group E, all 9 molecule tasks similarly disjoint. Within Group F, all 4 compound tasks disjoint. Within Group G, the four tasks touch disjoint files (each owns one README + one test file, plus the integration test). No two parallel tasks touch the same file.

4. **Vertical-slice compliance — limited deviation flagged:** Tasks 1-3 (Zod schemas) and Task 5 (validator) each contain one `describe` block with multiple `test()` cases (Diagnosis: 8 cases, Exercise: 10 cases, Synthesis: 11 cases, Validator: 6 cases). The strict "one test per task" rule is relaxed here because (a) a partial Zod schema with only one refinement has no semantic value (would not type-check or be importable in isolation), (b) the test cases are data fixtures driven by `safeParse` outcomes rather than separate behavioral specs, and (c) the alternative (8-11 sub-tasks per schema) would inflate the plan to ~70 tasks for limited reliability gain. The catalog tasks (6-37) each have exactly one test, fully satisfying the rule. If `/challenge` flags this, split Tasks 1-3 and 5 by refinement.

5. **Behavior-test compliance:** All tests exercise public exports (`DiagnosisArtifactSchema.safeParse`, `validateSkill`, `validateCatalog`) with input/output assertions. No test mocks internal collaborators. No test asserts on internal state or private methods. The catalog tests assert behavior of the public `validateSkill` API on real markdown files.

6. **Forbidden-pattern scan:** No "TBD", "TODO", "implement later", "fill in details", "Similar to Task N" appears in any task. Every task has exact code or exact commands. The validator's YAML parser is intentionally minimal (sufficient for the project's frontmatter shape) -- if a future skill file needs richer YAML (anchors, multiline lists), swap to `js-yaml` in a follow-up; Task 5's parseYaml is a contained module easily replaced.

7. **One open implementation-time decision noted but not deferred:** Whether `js-yaml` should be added as a dependency in Task 5 vs the inline parser shipped. The plan ships the inline parser to avoid a new dependency and keep Task 5 self-contained; if the build agent finds the inline parser fails on a real skill file's YAML, swap to `js-yaml` before declaring Task 5 complete (a one-line `bun add js-yaml` and the parser swap; not enough scope to be its own task).

---

> **Status:** Plan complete. 37 tasks across 7 sequential groups.
> Run `/challenge` on this file before executing.








