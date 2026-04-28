# V6 Molecules Implementation Plan

> **For the build agent:** Dispatch each task group in parallel (one subagent per task).
> Do NOT start execution until /challenge returns VERDICT: PROCEED.

**Goal:** Implement all 9 molecule ToolDefinition objects so the V6 harness loop's Phase 1 tool registry has concrete diagnosis and exercise skills that chain atom functions deterministically.
**Spec:** docs/specs/2026-04-28-v6-molecules-design.md
**Style:** Follow apps/api/TS_STYLE.md. No `any`. No network calls. Explicit exceptions over silent fallbacks.

## Shared conventions (read before writing any molecule)

- Dimension index: `const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const`
- Severity from z: `|z| >= 2.0 → 'significant'`, `>= 1.5 → 'moderate'`, else `'minor'`
- Atoms are called as regular async functions: `await computeVelocityCurve.invoke({ ... }) as VelocityCurve[]`
- Import types from atom source files (e.g. `import type { VelocityCurve } from '../atoms/compute-velocity-curve'`)
- All molecules validate output with `DiagnosisArtifactSchema.parse(...)` or `ExerciseArtifactSchema.parse(...)`

## Task Groups

Group A (parallel): Tasks 1–9 (all molecules, independent of each other)
Group B (sequential, depends on Group A): Task 10 (index barrel)

---

## Task 1: voicing-diagnosis molecule

**Group:** A (parallel with Tasks 2–9)

**Behavior being verified:** Given midi_notes where top-voice and bass-voice velocities are within 5 for both bars, muq dynamics z ≈ −2.2 vs student baseline, the molecule returns a DiagnosisArtifact with primary_dimension='dynamics', finding_type='issue', severity='significant'.
**Interface under test:** `voicingDiagnosis.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/voicing-diagnosis.ts`
- Test: `apps/api/src/harness/skills/molecules/voicing-diagnosis.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { voicingDiagnosis } from './voicing-diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('voicingDiagnosis: flat top/bass voicing with z=-2.2 returns issue/significant', async () => {
  // bar 1 and bar 2: top pitch (72,70) vel=76, bass pitch (45,40) vel=74 → |76-74|=2 < 5 for 2/2 bars
  // session_means=[0.50,0.60,0.70] → baseline mean=0.60 stddev≈0.082 → z=(0.42-0.60)/0.082≈-2.2
  const input = {
    bar_range: [1, 2] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1', 'cache:amt:s1:c1'],
    muq_scores: [0.42, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 72, onset_ms: 0,    duration_ms: 500, velocity: 76, bar: 1 },
      { pitch: 70, onset_ms: 250,  duration_ms: 500, velocity: 76, bar: 1 },
      { pitch: 45, onset_ms: 0,    duration_ms: 500, velocity: 74, bar: 1 },
      { pitch: 40, onset_ms: 250,  duration_ms: 500, velocity: 74, bar: 1 },
      { pitch: 71, onset_ms: 1000, duration_ms: 500, velocity: 76, bar: 2 },
      { pitch: 69, onset_ms: 1250, duration_ms: 500, velocity: 76, bar: 2 },
      { pitch: 44, onset_ms: 1000, duration_ms: 500, velocity: 74, bar: 2 },
      { pitch: 41, onset_ms: 1250, duration_ms: 500, velocity: 74, bar: 2 },
    ],
    session_means_dynamics: [0.50, 0.60, 0.70],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await voicingDiagnosis.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.dimensions).toContain('phrasing')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/voicing-diagnosis.test.ts
```
Expected: FAIL — `Cannot find module './voicing-diagnosis'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type VoicingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  session_means_dynamics: number[]
  cohort_table_dynamics: { p: number; value: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

function projectVoices(notes: { pitch: number; velocity: number; bar: number }[], bar: number) {
  const bn = notes.filter(n => n.bar === bar)
  if (bn.length === 0) return null
  const sorted = [...bn].sort((a, b) => b.pitch - a.pitch)
  const k = Math.max(1, Math.floor(sorted.length * 0.25))
  const topMean = sorted.slice(0, k).reduce((s, n) => s + n.velocity, 0) / k
  const bassMean = sorted.slice(-k).reduce((s, n) => s + n.velocity, 0) / k
  return { topMean, bassMean }
}

export const voicingDiagnosis: ToolDefinition = {
  name: 'voicing-diagnosis',
  description: 'Diagnoses melody/accompaniment voicing imbalance in homophonic textures. Returns DiagnosisArtifact with primary_dimension "dynamics".',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      session_means_dynamics: { type: 'array', items: { type: 'number' } },
      cohort_table_dynamics: { type: 'array', items: { type: 'object' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'session_means_dynamics', 'cohort_table_dynamics', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as VoicingInput
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'dynamics', session_means: i.session_means_dynamics }) as Baseline | null
    if (!baseline) throw new Error('voicing-diagnosis: insufficient session history for dynamics baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'dynamics', current: i.muq_scores[DIM.dynamics], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Voicing balance is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0) return neutral
    const bars = [...new Set(i.midi_notes.map(n => n.bar))].sort((a, b) => a - b)
    let flatCount = 0
    for (const bar of bars) {
      const p = projectVoices(i.midi_notes, bar)
      if (p && Math.abs(p.topMean - p.bassMean) < 5) flatCount++
    }
    if (bars.length === 0 || flatCount / bars.length < 0.6) return neutral
    await fetchReferencePercentile.invoke({ dimension: 'dynamics', score: i.muq_scores[DIM.dynamics], cohort_table: i.cohort_table_dynamics })
    const past = await fetchSimilarPastObservation.invoke({ dimension: 'dynamics', piece_id: i.piece_id, bar_range: i.bar_range, past_diagnoses: i.past_diagnoses, now_ms: i.now_ms }) as PastObservation | null
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics', 'phrasing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Melody and accompaniment are voiced almost equally; the top line is not coming through.',
      confidence: past ? 'high' : 'medium',
      finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/voicing-diagnosis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/voicing-diagnosis.ts apps/api/src/harness/skills/molecules/voicing-diagnosis.test.ts && git commit -m "feat(harness): implement voicing-diagnosis molecule"
```

---

## Task 2: pedal-triage molecule

**Group:** A (parallel with Tasks 1, 3–9)

**Behavior being verified:** Given pedal_cc held at value 127 throughout note durations (overlap ratio > 0.85) and dynamics z ≈ −3.0 vs baseline, the molecule returns primary_dimension='pedaling', finding_type='issue', severity='significant'.
**Interface under test:** `pedalTriage.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/pedal-triage.ts`
- Test: `apps/api/src/harness/skills/molecules/pedal-triage.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { pedalTriage } from './pedal-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('pedalTriage: pedal held throughout (ratio>0.85) with z=-3.0 returns issue/significant/over_pedal', async () => {
  // 4 notes spanning 0-4000ms; pedal on at time_ms=0 value=127 → ratio=1.0 > 0.85
  // session_means_pedaling=[0.50,0.60,0.70] → mean=0.60 stddev≈0.082 → z=(0.35-0.60)/0.082≈-3.05
  const input = {
    bar_range: [12, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c5', 'cache:amt-pedal:s1:c5'],
    muq_scores: [0.54, 0.48, 0.35, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 12 },
      { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 13 },
      { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 14 },
      { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 15 },
    ],
    pedal_cc: [{ time_ms: 0, value: 127 }],
    alignment: [
      { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 12 },
      { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 13 },
      { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 14 },
      { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 15 },
    ],
    harmony_changes: [],
    session_means_pedaling: [0.50, 0.60, 0.70],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await pedalTriage.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/pedal-triage.test.ts
```
Expected: FAIL — `Cannot find module './pedal-triage'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PedalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  harmony_changes: { bar: number; time_ms: number }[]
  session_means_pedaling: number[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

export const pedalTriage: ToolDefinition = {
  name: 'pedal-triage',
  description: 'Distinguishes over-pedaling, under-pedaling, and pedal-timing issues by combining MuQ pedaling delta with AMT pedal CC overlap ratio.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      pedal_cc: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      harmony_changes: { type: 'array', items: { type: 'object' } },
      session_means_pedaling: { type: 'array', items: { type: 'number' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'pedal_cc', 'alignment', 'harmony_changes', 'session_means_pedaling', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as PedalInput
    const ratio = await computePedalOverlapRatio.invoke({ notes: i.midi_notes, pedal_cc: i.pedal_cc }) as number
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'pedaling', session_means: i.session_means_pedaling }) as Baseline | null
    if (!baseline) throw new Error('pedal-triage: insufficient session history for pedaling baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'pedaling', current: i.muq_scores[DIM.pedaling], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'pedaling', dimensions: ['pedaling'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Pedaling is within student baseline.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0) return neutral
    let subtype: string
    let finding: string
    if (ratio > 0.85) {
      subtype = 'over_pedal'
      finding = `Over-pedaled through bars ${i.bar_range[0]}-${i.bar_range[1]}; the harmonies are blurring into one wash.`
    } else if (ratio < 0.30) {
      subtype = 'under_pedal'
      finding = `Under-pedaled through bars ${i.bar_range[0]}-${i.bar_range[1]}; the tone sounds dry and disconnected.`
    } else {
      // timing subtype: check pedal release vs harmony changes
      const aligned = i.alignment.filter(a => a.expected_onset_ms !== null)
      const lateReleases = i.harmony_changes.filter(hc => {
        return aligned.some(a => a.expected_onset_ms !== null && Math.abs((a.expected_onset_ms as number) - hc.time_ms) < 100)
      })
      subtype = lateReleases.length > 0 ? 'timing' : 'timing'
      finding = `Pedal not released at harmony changes in bars ${i.bar_range[0]}-${i.bar_range[1]}; notes from adjacent harmonies are blurring.`
    }
    const past = await fetchSimilarPastObservation.invoke({ dimension: 'pedaling', piece_id: i.piece_id, bar_range: i.bar_range, past_diagnoses: i.past_diagnoses, now_ms: i.now_ms }) as PastObservation | null
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'pedaling', dimensions: ['pedaling'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: past ? 'high' : 'medium',
      finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/pedal-triage.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/pedal-triage.ts apps/api/src/harness/skills/molecules/pedal-triage.test.ts && git commit -m "feat(harness): implement pedal-triage molecule"
```

---

## Task 3: rubato-coaching molecule

**Group:** A (parallel with Tasks 1–2, 4–9)

**Behavior being verified:** Given 8 notes where perf IOIs constantly stretch (score IOIs constant → Pearson r=0) and signed drift grows to 5600ms at phrase end (>> 50ms), with timing z ≈ −3.3, the molecule returns primary_dimension='timing', finding_type='issue', severity='significant'.
**Interface under test:** `rubatoCoaching.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/rubato-coaching.ts`
- Test: `apps/api/src/harness/skills/molecules/rubato-coaching.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { rubatoCoaching } from './rubato-coaching'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('rubatoCoaching: constant score IOIs vs stretching perf IOIs with large drift returns issue/significant/dragged', async () => {
  // score IOIs: all 500ms → variance=0 → Pearson r=0 < 0.3
  // perf onset: each note 100ms later than previous drift → signed drift grows to 5600ms >> 50ms
  // session_means_timing=[0.48,0.54,0.60] → mean=0.54 stddev≈0.049 → z=(0.38-0.54)/0.049≈-3.3
  const midi_notes = [
    { pitch: 60, onset_ms: 0,    duration_ms: 400, velocity: 70, bar: 40 },
    { pitch: 62, onset_ms: 700,  duration_ms: 400, velocity: 70, bar: 40 },
    { pitch: 64, onset_ms: 1600, duration_ms: 400, velocity: 70, bar: 41 },
    { pitch: 65, onset_ms: 2700, duration_ms: 400, velocity: 70, bar: 42 },
    { pitch: 67, onset_ms: 4000, duration_ms: 400, velocity: 70, bar: 43 },
    { pitch: 69, onset_ms: 5500, duration_ms: 400, velocity: 70, bar: 44 },
    { pitch: 71, onset_ms: 7200, duration_ms: 400, velocity: 70, bar: 45 },
    { pitch: 72, onset_ms: 9100, duration_ms: 400, velocity: 70, bar: 46 },
  ]
  const alignment = [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 40 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 500,  bar: 40 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 1000, bar: 41 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 1500, bar: 42 },
    { perf_index: 4, score_index: 4, expected_onset_ms: 2000, bar: 43 },
    { perf_index: 5, score_index: 5, expected_onset_ms: 2500, bar: 44 },
    { perf_index: 6, score_index: 6, expected_onset_ms: 3000, bar: 45 },
    { perf_index: 7, score_index: 7, expected_onset_ms: 3500, bar: 46 },
  ]
  const input = {
    bar_range: [40, 48] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c14', 'cache:amt:s1:c14'],
    muq_scores: [0.54, 0.38, 0.46, 0.54, 0.52, 0.51],
    midi_notes,
    alignment,
    session_means_timing: [0.48, 0.54, 0.60],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await rubatoCoaching.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.dimensions).toContain('phrasing')
  expect(result.dimensions).toContain('interpretation')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/rubato-coaching.test.ts
```
Expected: FAIL — `Cannot find module './rubato-coaching'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type RubatoInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  session_means_timing: number[]
  piece_id: string; now_ms: number
}

export const rubatoCoaching: ToolDefinition = {
  name: 'rubato-coaching',
  description: 'Distinguishes intentional returned rubato from uncompensated drift using IOI correlation and net phrase-end drift.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      session_means_timing: { type: 'array', items: { type: 'number' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'alignment', 'session_means_timing', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as RubatoInput
    const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
    const correlationNotes = i.midi_notes.map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) ?? null }))
    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'timing', session_means: i.session_means_timing }) as Baseline | null
    if (!baseline) throw new Error('rubato-coaching: insufficient session history for timing baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'timing', current: i.muq_scores[DIM.timing], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Timing is within baseline or rubato resolves cleanly.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8 || (r !== null && r >= 0.3)) return neutral
    const alignedNotes = i.midi_notes
      .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null)
    const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral
    const lastSigned = drift[drift.length - 1].signed
    if (Math.abs(lastSigned) <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase stretched and came back; rubato is well-shaped.',
        confidence: 'medium', finding_type: 'strength',
      })
    }
    const meanSigned = drift.reduce((s, d) => s + d.signed, 0) / drift.length
    const subtype = meanSigned < 0 ? 'rushed' : 'dragged'
    const finding = subtype === 'dragged'
      ? `The rubato through bars ${i.bar_range[0]}-${i.bar_range[1]} stretched without coming back; the phrase loses its shape.`
      : `The rubato through bars ${i.bar_range[0]}-${i.bar_range[1]} rushed without settling; the phrase ends too abruptly.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing', 'phrasing', 'interpretation'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/rubato-coaching.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/rubato-coaching.ts apps/api/src/harness/skills/molecules/rubato-coaching.test.ts && git commit -m "feat(harness): implement rubato-coaching molecule"
```

---

## Task 4: phrasing-arc-analysis molecule

**Group:** A (parallel with Tasks 1–3, 5–9)

**Behavior being verified:** Given 8 bars where bar 1 has the highest velocity (early peak at index 0) and drift grows to 5600ms at phrase end, with phrasing z ≈ −1.75 vs cohort, the molecule returns primary_dimension='phrasing', finding_type='issue', severity='moderate'.
**Interface under test:** `phrasingArcAnalysis.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/phrasing-arc-analysis.ts`
- Test: `apps/api/src/harness/skills/molecules/phrasing-arc-analysis.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('phrasingArcAnalysis: early velocity peak + no drift convergence with z=-1.75 returns issue/moderate', async () => {
  // bar 1 velocity=90 (highest, index 0), then decreasing → peak at index 0 → early/flat shape
  // same stretching alignment as rubato test → last signed drift=5600ms >> 50ms → no convergence
  // cohort_table_phrasing p50=0.52, p84=0.60 → stddev=0.08 → z=(0.38-0.52)/0.08=-1.75 → moderate
  const midi_notes = [
    { pitch: 65, onset_ms: 0,    duration_ms: 400, velocity: 90, bar: 1 },
    { pitch: 65, onset_ms: 1000, duration_ms: 400, velocity: 70, bar: 2 },
    { pitch: 65, onset_ms: 2000, duration_ms: 400, velocity: 65, bar: 3 },
    { pitch: 65, onset_ms: 3000, duration_ms: 400, velocity: 60, bar: 4 },
    { pitch: 65, onset_ms: 4000, duration_ms: 400, velocity: 55, bar: 5 },
    { pitch: 65, onset_ms: 5000, duration_ms: 400, velocity: 50, bar: 6 },
    { pitch: 65, onset_ms: 6000, duration_ms: 400, velocity: 45, bar: 7 },
    { pitch: 65, onset_ms: 7000, duration_ms: 400, velocity: 40, bar: 8 },
  ]
  const alignment = [
    { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 1 },
    { perf_index: 1, score_index: 1, expected_onset_ms: 500,  bar: 2 },
    { perf_index: 2, score_index: 2, expected_onset_ms: 1000, bar: 3 },
    { perf_index: 3, score_index: 3, expected_onset_ms: 1500, bar: 4 },
    { perf_index: 4, score_index: 4, expected_onset_ms: 2000, bar: 5 },
    { perf_index: 5, score_index: 5, expected_onset_ms: 2500, bar: 6 },
    { perf_index: 6, score_index: 6, expected_onset_ms: 3000, bar: 7 },
    { perf_index: 7, score_index: 7, expected_onset_ms: 3500, bar: 8 },
  ]
  const input = {
    bar_range: [1, 8] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c10', 'cache:amt:s1:c10'],
    muq_scores: [0.54, 0.48, 0.46, 0.54, 0.38, 0.51],
    midi_notes,
    alignment,
    cohort_table_phrasing: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await phrasingArcAnalysis.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('phrasing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('moderate')
  expect(result.dimensions).toContain('dynamics')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/phrasing-arc-analysis.test.ts
```
Expected: FAIL — `Cannot find module './phrasing-arc-analysis'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type PhrasingInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  cohort_table_phrasing: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const phrasingArcAnalysis: ToolDefinition = {
  name: 'phrasing-arc-analysis',
  description: 'Assesses dynamic and timing arc shape across a complete phrase by detecting peak position and drift convergence.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      cohort_table_phrasing: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'alignment', 'cohort_table_phrasing', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as PhrasingInput
    const curve = await computeVelocityCurve.invoke({ bar_range: i.bar_range, notes: i.midi_notes }) as VelocityCurve[]
    const p50 = i.cohort_table_phrasing.find(e => e.p === 50)?.value ?? 0.52
    const p84 = i.cohort_table_phrasing.find(e => e.p === 84)?.value ?? 0.60
    const cohortBaseline = { mean: p50, stddev: Math.max(0.01, p84 - p50) }
    const z = await computeDimensionDelta.invoke({ dimension: 'phrasing', current: i.muq_scores[DIM.phrasing], baseline: cohortBaseline }) as number
    await fetchReferencePercentile.invoke({ dimension: 'phrasing', score: i.muq_scores[DIM.phrasing], cohort_table: i.cohort_table_phrasing })
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Phrase shape is within cohort norms.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8) return neutral
    // Peak detection: find argmax of mean_velocity; check if at index 0 or last, or multi-peaked
    const maxVel = Math.max(...curve.map(c => c.mean_velocity))
    const peakIndex = curve.findIndex(c => c.mean_velocity === maxVel)
    const nearPeak = curve.filter(c => maxVel - c.mean_velocity < 5)
    const flatOrMulti = peakIndex === 0 || peakIndex === curve.length - 1 || nearPeak.length > 1
    // Drift convergence: last aligned note's signed drift
    const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
    const alignedNotes = i.midi_notes
      .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
      .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null)
    const drift = await computeOnsetDrift.invoke({ notes: alignedNotes }) as OnsetDrift[]
    const lastDrift = drift.length > 0 ? Math.abs(drift[drift.length - 1].signed) : 0
    if (!flatOrMulti && lastDrift <= 50) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
        severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'The phrase has a clear arc with a well-placed peak.',
        confidence: 'medium', finding_type: 'strength',
      })
    }
    const peakBar = curve[peakIndex]?.bar ?? i.bar_range[0]
    const finding = `The phrase peaks at bar ${peakBar} instead of the expected middle; the climax of the line is arriving too early.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'phrasing', dimensions: ['phrasing', 'dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'medium', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/phrasing-arc-analysis.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/phrasing-arc-analysis.ts apps/api/src/harness/skills/molecules/phrasing-arc-analysis.test.ts && git commit -m "feat(harness): implement phrasing-arc-analysis molecule"
```

---

## Task 5: tempo-stability-triage molecule

**Group:** A (parallel with Tasks 1–4, 6–9)

**Behavior being verified:** Given 14 notes with monotonically increasing positive drift (all same-sign → fraction=1.0 ≥ 0.8), constant score IOIs (Pearson r=0 < 0.4), and timing z ≈ −3.3, the molecule returns primary_dimension='timing', finding_type='issue', severity='significant'.
**Interface under test:** `tempoStabilityTriage.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/tempo-stability-triage.ts`
- Test: `apps/api/src/harness/skills/molecules/tempo-stability-triage.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { tempoStabilityTriage } from './tempo-stability-triage'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('tempoStabilityTriage: monotonic positive drift 14 notes, r=0, z=-3.3 returns issue/significant/slowing', async () => {
  // score: 14 notes at 500ms each; perf: monotonically slower
  // all signed drifts positive → fraction=1.0 >= 0.8 → monotonic → slowing
  // session_means_timing=[0.48,0.54,0.60] → z=(0.38-0.54)/0.049≈-3.3 < -1.0
  const notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[] = []
  const alignment: { perf_index: number; score_index: number; expected_onset_ms: number; bar: number }[] = []
  for (let idx = 0; idx < 14; idx++) {
    const expected = idx * 500
    const drift = idx * 120  // monotonically growing positive drift
    notes.push({ pitch: 60 + idx, onset_ms: expected + drift, duration_ms: 400, velocity: 70, bar: Math.floor(idx / 2) + 1 })
    alignment.push({ perf_index: idx, score_index: idx, expected_onset_ms: expected, bar: Math.floor(idx / 2) + 1 })
  }
  const input = {
    bar_range: [1, 16] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c1', 'cache:amt:s1:c1'],
    muq_scores: [0.54, 0.38, 0.46, 0.54, 0.52, 0.51],
    midi_notes: notes,
    alignment,
    session_means_timing: [0.48, 0.54, 0.60],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await tempoStabilityTriage.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('timing')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/tempo-stability-triage.test.ts
```
Expected: FAIL — `Cannot find module './tempo-stability-triage'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeIoiCorrelation } from '../atoms/compute-ioi-correlation'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type TempoInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number; bar: number }[]
  session_means_timing: number[]
  piece_id: string; now_ms: number
}

export const tempoStabilityTriage: ToolDefinition = {
  name: 'tempo-stability-triage',
  description: 'Distinguishes tempo drift, intentional rubato, and loss of pulse by checking monotonic drift and IOI correlation.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      session_means_timing: { type: 'array', items: { type: 'number' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'alignment', 'session_means_timing', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as TempoInput
    const correlationNotes = i.midi_notes.map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: i.alignment[idx]?.expected_onset_ms ?? null }))
    const r = await computeIoiCorrelation.invoke({ notes: correlationNotes }) as number | null
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'timing', session_means: i.session_means_timing }) as Baseline | null
    if (!baseline) throw new Error('tempo-stability-triage: insufficient session history for timing baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'timing', current: i.muq_scores[DIM.timing], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Tempo is stable or within acceptable rubato range.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -1.0 || (r !== null && r >= 0.4)) return neutral
    const driftNotes = i.alignment.map((a, idx) => ({ onset_ms: i.midi_notes[idx]?.onset_ms ?? a.expected_onset_ms, expected_onset_ms: a.expected_onset_ms }))
    const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
    if (drift.length === 0) return neutral
    const positiveCount = drift.filter(d => d.signed > 0).length
    const negativeCount = drift.filter(d => d.signed < 0).length
    const dominantFraction = Math.max(positiveCount, negativeCount) / drift.length
    if (dominantFraction < 0.8) return neutral  // non-monotonic → defer to rubato-coaching
    const subtype: 'slowing' | 'rushing' | 'unstable' = positiveCount > negativeCount ? 'slowing' : 'rushing'
    const finding = subtype === 'slowing'
      ? `The pulse slowed gradually across bars ${i.bar_range[0]}-${i.bar_range[1]}; the tempo is drifting under the beat.`
      : `The pulse rushed across bars ${i.bar_range[0]}-${i.bar_range[1]}; the tempo is running ahead of the beat.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'timing', dimensions: ['timing'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/tempo-stability-triage.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/tempo-stability-triage.ts apps/api/src/harness/skills/molecules/tempo-stability-triage.test.ts && git commit -m "feat(harness): implement tempo-stability-triage molecule"
```

---

## Task 6: dynamic-range-audit molecule

**Group:** A (parallel with Tasks 1–5, 7–9)

**Behavior being verified:** Given midi_notes with velocities all in [60,65] (observed range < 30) and score_marking_type='wide', with dynamics z ≈ −3.3, the molecule returns primary_dimension='dynamics', finding_type='issue', severity='significant'.
**Interface under test:** `dynamicRangeAudit.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/dynamic-range-audit.ts`
- Test: `apps/api/src/harness/skills/molecules/dynamic-range-audit.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { dynamicRangeAudit } from './dynamic-range-audit'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('dynamicRangeAudit: compressed velocities [60-65] with wide score markings and z=-3.3 returns issue/significant', async () => {
  // 5 bars, velocities 60-65 → max p90=65, min mean≈60 → range=5 < 30
  // score_marking_type='wide' (ff and pp both present) → fire
  // session_means_dynamics=[0.48,0.54,0.60] → mean=0.54 stddev≈0.049 → z=(0.38-0.54)/0.049≈-3.3
  const input = {
    bar_range: [28, 32] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c8', 'cache:amt:s1:c8'],
    muq_scores: [0.38, 0.48, 0.46, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 500, velocity: 60, bar: 28 },
      { pitch: 62, onset_ms: 500,  duration_ms: 500, velocity: 62, bar: 29 },
      { pitch: 64, onset_ms: 1000, duration_ms: 500, velocity: 63, bar: 30 },
      { pitch: 65, onset_ms: 1500, duration_ms: 500, velocity: 65, bar: 31 },
      { pitch: 67, onset_ms: 2000, duration_ms: 500, velocity: 61, bar: 32 },
    ],
    score_marking_type: 'wide' as const,
    session_means_dynamics: [0.48, 0.54, 0.60],
    cohort_table_dynamics: [{ p: 50, value: 0.55 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await dynamicRangeAudit.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('dynamics')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
  expect(result.evidence_refs.length).toBeGreaterThan(0)
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/dynamic-range-audit.test.ts
```
Expected: FAIL — `Cannot find module './dynamic-range-audit'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeVelocityCurve } from '../atoms/compute-velocity-curve'
import type { VelocityCurve } from '../atoms/compute-velocity-curve'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchStudentBaseline } from '../atoms/fetch-student-baseline'
import type { Baseline } from '../atoms/fetch-student-baseline'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type DynamicInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  score_marking_type: 'wide' | 'medium' | 'narrow' | 'none'
  session_means_dynamics: number[]; cohort_table_dynamics: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const dynamicRangeAudit: ToolDefinition = {
  name: 'dynamic-range-audit',
  description: 'Compares velocity range used in performance against the dynamic range asked for by the score.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      score_marking_type: { type: 'string', enum: ['wide', 'medium', 'narrow', 'none'] },
      session_means_dynamics: { type: 'array', items: { type: 'number' } },
      cohort_table_dynamics: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'score_marking_type', 'session_means_dynamics', 'cohort_table_dynamics', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as DynamicInput
    const curve = await computeVelocityCurve.invoke({ bar_range: i.bar_range, notes: i.midi_notes }) as VelocityCurve[]
    const baseline = await fetchStudentBaseline.invoke({ dimension: 'dynamics', session_means: i.session_means_dynamics }) as Baseline | null
    if (!baseline) throw new Error('dynamic-range-audit: insufficient session history for dynamics baseline (need >= 3 sessions)')
    const z = await computeDimensionDelta.invoke({ dimension: 'dynamics', current: i.muq_scores[DIM.dynamics], baseline }) as number
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Dynamic range is adequate for the score markings.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8 || i.score_marking_type === 'none') return neutral
    const maxP90 = Math.max(...curve.map(c => c.p90_velocity))
    const minMean = Math.min(...curve.map(c => c.mean_velocity))
    const observedRange = maxP90 - minMean
    if (observedRange >= 30) return neutral
    await fetchReferencePercentile.invoke({ dimension: 'dynamics', score: i.muq_scores[DIM.dynamics], cohort_table: i.cohort_table_dynamics })
    const finding = `The dynamic range across bars ${i.bar_range[0]}-${i.bar_range[1]} is only ${Math.round(observedRange)} velocity points; the score asks for much more contrast here.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'dynamics', dimensions: ['dynamics'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/dynamic-range-audit.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/dynamic-range-audit.ts apps/api/src/harness/skills/molecules/dynamic-range-audit.test.ts && git commit -m "feat(harness): implement dynamic-range-audit molecule"
```

---

## Task 7: articulation-clarity-check molecule

**Group:** A (parallel with Tasks 1–6, 8–9)

**Behavior being verified:** Given 8 bars all marked staccato in the score but mono_notes with positive overlap ratios (legato), with articulation z ≈ −1.75 vs cohort, the molecule returns primary_dimension='articulation', finding_type='issue', severity='moderate'.
**Interface under test:** `articulationClarityCheck.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/articulation-clarity-check.ts`
- Test: `apps/api/src/harness/skills/molecules/articulation-clarity-check.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { articulationClarityCheck } from './articulation-clarity-check'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('articulationClarityCheck: staccato score with legato execution (mismatch=8/8) and z=-1.75 returns issue/moderate', async () => {
  // 8 bars, each with 3 notes where note duration > gap → overlap ratio > 0 (legato)
  // score marks all bars 'staccato' → mismatch for all 8 bars → fraction=1.0 >= 0.5
  // cohort_table_articulation p50=0.52, p84=0.60 → z=(0.38-0.52)/0.08=-1.75 → moderate
  const barsData = Array.from({ length: 8 }, (_, barIdx) => ({
    bar: 5 + barIdx,
    notes: [
      { onset_ms: barIdx * 1000,       duration_ms: 450 },  // overlaps with next (onset+450 > onset+333)
      { onset_ms: barIdx * 1000 + 333, duration_ms: 450 },
      { onset_ms: barIdx * 1000 + 666, duration_ms: 450 },
    ],
  }))
  const input = {
    bar_range: [5, 12] as [number, number],
    scope: 'session' as const,
    evidence_refs: ['cache:muq:s1:c2', 'cache:amt:s1:c2'],
    muq_scores: [0.54, 0.48, 0.46, 0.38, 0.52, 0.51],
    mono_notes_per_bar: barsData,
    score_articulation_per_bar: barsData.map(b => ({ bar: b.bar, articulation: 'staccato' as const })),
    cohort_table_articulation: [{ p: 50, value: 0.52 }, { p: 84, value: 0.60 }],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await articulationClarityCheck.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('articulation')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('moderate')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/articulation-clarity-check.test.ts
```
Expected: FAIL — `Cannot find module './articulation-clarity-check'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeKeyOverlapRatio } from '../atoms/compute-key-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'
import { fetchReferencePercentile } from '../atoms/fetch-reference-percentile'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const

function severityFromZ(z: number): 'minor' | 'moderate' | 'significant' {
  const a = Math.abs(z)
  return a >= 2.0 ? 'significant' : a >= 1.5 ? 'moderate' : 'minor'
}

type ArticulationInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]; muq_scores: number[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_table_articulation: { p: number; value: number }[]
  piece_id: string; now_ms: number
}

export const articulationClarityCheck: ToolDefinition = {
  name: 'articulation-clarity-check',
  description: 'Identifies execution mismatches between notated articulation (slurs vs staccato) and observed key-overlap behavior.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      mono_notes_per_bar: { type: 'array', items: { type: 'object' } },
      score_articulation_per_bar: { type: 'array', items: { type: 'object' } },
      cohort_table_articulation: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'mono_notes_per_bar', 'score_articulation_per_bar', 'cohort_table_articulation', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as ArticulationInput
    const p50 = i.cohort_table_articulation.find(e => e.p === 50)?.value ?? 0.52
    const p84 = i.cohort_table_articulation.find(e => e.p === 84)?.value ?? 0.60
    const cohortBaseline = { mean: p50, stddev: Math.max(0.01, p84 - p50) }
    const z = await computeDimensionDelta.invoke({ dimension: 'articulation', current: i.muq_scores[DIM.articulation], baseline: cohortBaseline }) as number
    await fetchReferencePercentile.invoke({ dimension: 'articulation', score: i.muq_scores[DIM.articulation], cohort_table: i.cohort_table_articulation })
    const neutral = DiagnosisArtifactSchema.parse({
      primary_dimension: 'articulation', dimensions: ['articulation'],
      severity: 'minor', scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: 'Articulation execution matches score markings.',
      confidence: 'low', finding_type: 'neutral',
    })
    if (z > -0.8) return neutral
    const articMap = new Map(i.score_articulation_per_bar.map(a => [a.bar, a.articulation]))
    let mismatchCount = 0; let totalBars = 0
    let dominantMismatch: 'legato_as_staccato' | 'staccato_as_legato' = 'staccato_as_legato'
    for (const barData of i.mono_notes_per_bar) {
      if (barData.notes.length < 3) continue
      totalBars++
      const ratio = await computeKeyOverlapRatio.invoke({ notes: barData.notes }) as number
      const scoreArt = articMap.get(barData.bar) ?? 'detache'
      const isMismatch = (scoreArt === 'staccato' && ratio >= 0) || (scoreArt === 'slur' && ratio <= 0)
      if (isMismatch) {
        mismatchCount++
        dominantMismatch = scoreArt === 'staccato' ? 'staccato_as_legato' : 'legato_as_staccato'
      }
    }
    if (totalBars === 0 || mismatchCount / totalBars < 0.5) return neutral
    const finding = dominantMismatch === 'staccato_as_legato'
      ? `The staccato bars ${i.bar_range[0]}-${i.bar_range[1]} are sustaining into each other; the notes are blurring rather than separating.`
      : `The slurred bars ${i.bar_range[0]}-${i.bar_range[1]} are detaching between notes; the legato line is breaking up.`
    return DiagnosisArtifactSchema.parse({
      primary_dimension: 'articulation', dimensions: ['articulation'],
      severity: severityFromZ(z), scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/articulation-clarity-check.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/articulation-clarity-check.ts apps/api/src/harness/skills/molecules/articulation-clarity-check.test.ts && git commit -m "feat(harness): implement articulation-clarity-check molecule"
```

---

## Task 8: cross-modal-contradiction-check molecule

**Group:** A (parallel with Tasks 1–7, 9)

**Behavior being verified:** Given MuQ pedaling z ≈ +3.3 vs cohort (excellent) but pedal_cc held throughout (overlap ratio 1.0 > 0.85), the molecule detects the pedaling contradiction and returns primary_dimension='pedaling', finding_type='issue', severity='significant'.
**Interface under test:** `crossModalContradictionCheck.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.ts`
- Test: `apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'

test('crossModalContradictionCheck: MuQ pedaling high (z=+3.3) but overlap ratio=1.0 returns issue/significant/pedaling', async () => {
  // cohort_baselines.pedaling = { mean: 0.46, stddev: 0.08 } → z=(0.72-0.46)/0.08=3.25 >= +0.5
  // pedal_cc held at 127 → overlap ratio=1.0 > 0.85 → contradiction fires
  const input = {
    bar_range: [20, 28] as [number, number],
    scope: 'stop_moment' as const,
    evidence_refs: ['cache:muq:s1:c7', 'cache:amt-pedal:s1:c7'],
    muq_scores: [0.54, 0.48, 0.72, 0.54, 0.52, 0.51],
    midi_notes: [
      { pitch: 60, onset_ms: 0,    duration_ms: 1000, velocity: 70, bar: 20 },
      { pitch: 62, onset_ms: 1000, duration_ms: 1000, velocity: 70, bar: 22 },
      { pitch: 64, onset_ms: 2000, duration_ms: 1000, velocity: 70, bar: 24 },
      { pitch: 65, onset_ms: 3000, duration_ms: 1000, velocity: 70, bar: 26 },
    ],
    pedal_cc: [{ time_ms: 0, value: 127 }],
    alignment: [
      { perf_index: 0, score_index: 0, expected_onset_ms: 0,    bar: 20 },
      { perf_index: 1, score_index: 1, expected_onset_ms: 1000, bar: 22 },
      { perf_index: 2, score_index: 2, expected_onset_ms: 2000, bar: 24 },
      { perf_index: 3, score_index: 3, expected_onset_ms: 3000, bar: 26 },
    ],
    mono_notes_per_bar: [],
    score_articulation_per_bar: [],
    cohort_baselines: {
      dynamics:        { mean: 0.54, stddev: 0.07 },
      timing:          { mean: 0.48, stddev: 0.04 },
      pedaling:        { mean: 0.46, stddev: 0.08 },
      articulation:    { mean: 0.54, stddev: 0.02 },
    },
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await crossModalContradictionCheck.invoke(input) as DiagnosisArtifact
  expect(result.primary_dimension).toBe('pedaling')
  expect(result.finding_type).toBe('issue')
  expect(result.severity).toBe('significant')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/cross-modal-contradiction-check.test.ts
```
Expected: FAIL — `Cannot find module './cross-modal-contradiction-check'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { computeOnsetDrift } from '../atoms/compute-onset-drift'
import type { OnsetDrift } from '../atoms/compute-onset-drift'
import { computePedalOverlapRatio } from '../atoms/compute-pedal-overlap-ratio'
import { computeKeyOverlapRatio } from '../atoms/compute-key-overlap-ratio'
import { computeDimensionDelta } from '../atoms/compute-dimension-delta'

const DIM = { dynamics: 0, timing: 1, pedaling: 2, articulation: 3, phrasing: 4, interpretation: 5 } as const
const DIM_NAMES = ['dynamics', 'timing', 'pedaling', 'articulation', 'phrasing', 'interpretation'] as const

type CrossModalInput = {
  bar_range: [number, number]; scope: 'stop_moment' | 'passage' | 'session'
  evidence_refs: string[]
  muq_scores: number[]
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  pedal_cc: { time_ms: number; value: number }[]
  alignment: { perf_index: number; score_index: number; expected_onset_ms: number | null; bar: number }[]
  mono_notes_per_bar: { bar: number; notes: { onset_ms: number; duration_ms: number }[] }[]
  score_articulation_per_bar: { bar: number; articulation: 'slur' | 'staccato' | 'detache' }[]
  cohort_baselines: { [dim: string]: { mean: number; stddev: number } }
  piece_id: string; now_ms: number
}

export const crossModalContradictionCheck: ToolDefinition = {
  name: 'cross-modal-contradiction-check',
  description: 'Flags cases where MuQ dimension scores and AMT-derived structural features disagree on a passage.',
  input_schema: {
    type: 'object',
    properties: {
      bar_range: { type: 'array', items: { type: 'number' }, minItems: 2, maxItems: 2 },
      scope: { type: 'string', enum: ['stop_moment', 'passage', 'session'] },
      evidence_refs: { type: 'array', items: { type: 'string' } },
      muq_scores: { type: 'array', items: { type: 'number' }, minItems: 6, maxItems: 6 },
      midi_notes: { type: 'array', items: { type: 'object' } },
      pedal_cc: { type: 'array', items: { type: 'object' } },
      alignment: { type: 'array', items: { type: 'object' } },
      mono_notes_per_bar: { type: 'array', items: { type: 'object' } },
      score_articulation_per_bar: { type: 'array', items: { type: 'object' } },
      cohort_baselines: { type: 'object' },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['bar_range', 'scope', 'evidence_refs', 'muq_scores', 'midi_notes', 'pedal_cc', 'alignment', 'mono_notes_per_bar', 'score_articulation_per_bar', 'cohort_baselines', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<DiagnosisArtifact> => {
    const i = input as CrossModalInput

    async function dimZ(dimName: string): Promise<number> {
      const bl = i.cohort_baselines[dimName]
      if (!bl) return 0
      return await computeDimensionDelta.invoke({ dimension: dimName, current: i.muq_scores[DIM[dimName as keyof typeof DIM]], baseline: bl }) as number
    }

    const contradictions: { dimension: string; delta: number; finding: string }[] = []

    // timing pair: z >= +0.5 AND mean onset drift > 80ms
    const timingZ = await dimZ('timing')
    if (timingZ >= 0.5) {
      const alignMap = new Map(i.alignment.map(a => [a.perf_index, a.expected_onset_ms]))
      const driftNotes = i.midi_notes
        .map((n, idx) => ({ onset_ms: n.onset_ms, expected_onset_ms: alignMap.get(idx) }))
        .filter((n): n is { onset_ms: number; expected_onset_ms: number } => n.expected_onset_ms !== null)
      if (driftNotes.length >= 2) {
        const drift = await computeOnsetDrift.invoke({ notes: driftNotes }) as OnsetDrift[]
        const meanDrift = drift.reduce((s, d) => s + d.drift_ms, 0) / drift.length
        if (meanDrift > 80) {
          contradictions.push({ dimension: 'timing', delta: Math.abs(timingZ), finding: `MuQ rates timing clean here, but the onsets drifted ${Math.round(meanDrift)}ms on average.` })
        }
      }
    }

    // pedaling pair: z >= +0.5 AND overlap ratio < 0.30 OR > 0.85
    const pedalZ = await dimZ('pedaling')
    if (pedalZ >= 0.5) {
      const ratio = await computePedalOverlapRatio.invoke({ notes: i.midi_notes, pedal_cc: i.pedal_cc }) as number
      if (ratio < 0.30 || ratio > 0.85) {
        contradictions.push({ dimension: 'pedaling', delta: Math.abs(pedalZ), finding: `MuQ rates pedaling clean here, but the pedal overlap ratio was ${(ratio * 100).toFixed(0)}% -- the model and the score disagree.` })
      }
    }

    // articulation pair: z >= +0.5 AND key overlap direction opposes score in >= 50% bars
    const articulationZ = await dimZ('articulation')
    if (articulationZ >= 0.5 && i.mono_notes_per_bar.length >= 2) {
      const articMap = new Map(i.score_articulation_per_bar.map(a => [a.bar, a.articulation]))
      let artMismatch = 0; let artTotal = 0
      for (const bd of i.mono_notes_per_bar) {
        if (bd.notes.length < 3) continue
        artTotal++
        const ratio = await computeKeyOverlapRatio.invoke({ notes: bd.notes }) as number
        const sa = articMap.get(bd.bar) ?? 'detache'
        if ((sa === 'staccato' && ratio >= 0) || (sa === 'slur' && ratio <= 0)) artMismatch++
      }
      if (artTotal > 0 && artMismatch / artTotal >= 0.5) {
        contradictions.push({ dimension: 'articulation', delta: Math.abs(articulationZ), finding: `MuQ rates articulation clean here, but the key-overlap direction contradicts the score markings in ${artMismatch}/${artTotal} bars.` })
      }
    }

    // dynamics pair: z >= +0.5 AND velocity range < 25
    const dynamicsZ = await dimZ('dynamics')
    if (dynamicsZ >= 0.5 && i.midi_notes.length > 0) {
      const vels = i.midi_notes.map(n => n.velocity)
      const velRange = Math.max(...vels) - Math.min(...vels)
      if (velRange < 25) {
        contradictions.push({ dimension: 'dynamics', delta: Math.abs(dynamicsZ), finding: `MuQ rates dynamics clean here, but the velocity range across the passage is only ${velRange} -- compressed.` })
      }
    }

    if (contradictions.length === 0) {
      return DiagnosisArtifactSchema.parse({
        primary_dimension: 'dynamics', dimensions: ['dynamics'],
        severity: 'minor', scope: i.scope, bar_range: i.bar_range,
        evidence_refs: i.evidence_refs,
        one_sentence_finding: 'No cross-modal contradictions detected.',
        confidence: 'high', finding_type: 'neutral',
      })
    }

    const winner = contradictions.reduce((best, c) => c.delta > best.delta ? c : best)
    const dims = contradictions.map(c => c.dimension) as typeof DIM_NAMES[number][]
    return DiagnosisArtifactSchema.parse({
      primary_dimension: winner.dimension,
      dimensions: [...new Set(dims)],
      severity: 'significant',
      scope: i.scope, bar_range: i.bar_range,
      evidence_refs: i.evidence_refs,
      one_sentence_finding: winner.finding,
      confidence: 'high', finding_type: 'issue',
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/cross-modal-contradiction-check.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.ts apps/api/src/harness/skills/molecules/cross-modal-contradiction-check.test.ts && git commit -m "feat(harness): implement cross-modal-contradiction-check molecule"
```

---

## Task 9: exercise-proposal molecule

**Group:** A (parallel with Tasks 1–8)

**Behavior being verified:** Given a DiagnosisArtifact with primary_dimension='pedaling', severity='significant', finding_type='issue', and no prior exercise within 3 days, the molecule returns an ExerciseArtifact with exercise_type='pedal_isolation', exercise_subtype='no-pedal-pass', and non-null action_binding.
**Interface under test:** `exerciseProposal.invoke(input)`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/exercise-proposal.ts`
- Test: `apps/api/src/harness/skills/molecules/exercise-proposal.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { exerciseProposal } from './exercise-proposal'
import { DiagnosisArtifactSchema } from '../../artifacts/diagnosis'
import type { ExerciseArtifact } from '../../artifacts/exercise'

const pedaling_diagnosis = DiagnosisArtifactSchema.parse({
  primary_dimension: 'pedaling',
  dimensions: ['pedaling'],
  severity: 'significant',
  scope: 'session',
  bar_range: [12, 16],
  evidence_refs: ['cache:muq:s1:c5'],
  one_sentence_finding: 'Over-pedaled through bars 12-16; harmonies blurring.',
  confidence: 'high',
  finding_type: 'issue',
})

test('exerciseProposal: pedaling+significant with no prior exercise returns pedal_isolation/no-pedal-pass', async () => {
  const input = {
    diagnosis: pedaling_diagnosis,
    diagnosis_ref: 'diag:abc789',
    midi_notes: [
      { pitch: 60, onset_ms: 0, duration_ms: 500, velocity: 70, bar: 12 },
    ],
    past_diagnoses: [],
    piece_id: 'test-piece',
    now_ms: 1000,
  }
  const result = await exerciseProposal.invoke(input) as ExerciseArtifact
  expect(result.exercise_type).toBe('pedal_isolation')
  expect(result.exercise_subtype).toBe('no-pedal-pass')
  expect(result.target_dimension).toBe('pedaling')
  expect(result.bar_range).toEqual([12, 16])
  expect(result.action_binding).not.toBeNull()
  expect(result.estimated_minutes).toBe(8)
  expect(result.diagnosis_summary).toBe('Over-pedaled through bars 12-16; harmonies blurring.')
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/exercise-proposal.test.ts
```
Expected: FAIL — `Cannot find module './exercise-proposal'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { ExerciseArtifactSchema } from '../../artifacts/exercise'
import type { ExerciseArtifact } from '../../artifacts/exercise'
import type { DiagnosisArtifact } from '../../artifacts/diagnosis'
import { fetchSimilarPastObservation } from '../atoms/fetch-similar-past-observation'
import type { PastObservation } from '../atoms/fetch-similar-past-observation'

type ExerciseInput = {
  diagnosis: DiagnosisArtifact
  diagnosis_ref: string
  midi_notes: { pitch: number; onset_ms: number; duration_ms: number; velocity: number; bar: number }[]
  past_diagnoses: { artifact_id: string; session_id: string; created_at: number; primary_dimension: string; bar_range: [number,number]|null; piece_id: string }[]
  piece_id: string; now_ms: number
}

const EXERCISE_MAP: Record<string, Record<string, { type: string; subtype: string | null }>> = {
  pedaling: {
    moderate:    { type: 'pedal_isolation', subtype: null },
    significant: { type: 'pedal_isolation', subtype: 'no-pedal-pass' },
    minor:       { type: 'pedal_isolation', subtype: null },
  },
  timing: {
    moderate:    { type: 'segment_loop', subtype: 'metronome-locked' },
    significant: { type: 'segment_loop', subtype: 'metronome-locked' },
    minor:       { type: 'segment_loop', subtype: 'metronome-locked' },
  },
  dynamics: {
    moderate:    { type: 'dynamic_exaggeration', subtype: null },
    significant: { type: 'dynamic_exaggeration', subtype: 'extreme-contrast' },
    minor:       { type: 'dynamic_exaggeration', subtype: null },
  },
  articulation: {
    moderate:    { type: 'isolated_hands', subtype: null },
    significant: { type: 'isolated_hands', subtype: null },
    minor:       { type: 'isolated_hands', subtype: null },
  },
  phrasing: {
    moderate:    { type: 'slow_practice', subtype: 'shape-vocally' },
    significant: { type: 'slow_practice', subtype: 'shape-vocally' },
    minor:       { type: 'slow_practice', subtype: 'shape-vocally' },
  },
  interpretation: {
    moderate:    { type: 'slow_practice', subtype: 'imitate-reference' },
    significant: { type: 'slow_practice', subtype: 'imitate-reference' },
    minor:       { type: 'slow_practice', subtype: 'imitate-reference' },
  },
}

const MINUTES_MAP: Record<string, number> = { minor: 2, moderate: 5, significant: 8 }

export const exerciseProposal: ToolDefinition = {
  name: 'exercise-proposal',
  description: 'Generates one targeted ExerciseArtifact from a single DiagnosisArtifact. Requires finding_type="issue" and severity in {moderate, significant}.',
  input_schema: {
    type: 'object',
    properties: {
      diagnosis: { type: 'object' },
      diagnosis_ref: { type: 'string' },
      midi_notes: { type: 'array', items: { type: 'object' } },
      past_diagnoses: { type: 'array', items: { type: 'object' } },
      piece_id: { type: 'string' },
      now_ms: { type: 'number' },
    },
    required: ['diagnosis', 'diagnosis_ref', 'midi_notes', 'past_diagnoses', 'piece_id', 'now_ms'],
  },
  invoke: async (input: unknown): Promise<ExerciseArtifact> => {
    const i = input as ExerciseInput
    const d = i.diagnosis
    if (d.finding_type !== 'issue') {
      throw new Error(`exercise-proposal: diagnosis must have finding_type "issue", got "${d.finding_type}"`)
    }
    if (d.bar_range === null) {
      throw new Error('exercise-proposal: diagnosis bar_range must not be null')
    }
    const prior = await fetchSimilarPastObservation.invoke({
      dimension: d.primary_dimension,
      piece_id: i.piece_id,
      bar_range: d.bar_range,
      past_diagnoses: i.past_diagnoses,
      now_ms: i.now_ms,
    }) as PastObservation | null
    if (prior && prior.days_ago < 3) {
      throw new Error(`exercise-proposal: diagnosis already addressed by exercise ${prior.artifact_id} ${prior.days_ago} day(s) ago`)
    }
    const mapping = EXERCISE_MAP[d.primary_dimension]?.[d.severity] ?? { type: 'slow_practice', subtype: null }
    const estimatedMinutes = MINUTES_MAP[d.severity] ?? 5
    const action_binding = ['segment_loop', 'isolated_hands', 'pedal_isolation'].includes(mapping.type)
      ? { tool: mapping.type === 'pedal_isolation' ? 'mute_pedal' : mapping.type, args: { bars: d.bar_range } }
      : null
    const instructionMap: Record<string, string> = {
      pedal_isolation: `Play bars ${d.bar_range[0]}-${d.bar_range[1]} three times with no sustain pedal. Listen for whether the line still sustains itself in your fingers.`,
      segment_loop: `Loop bars ${d.bar_range[0]}-${d.bar_range[1]} with a metronome. Match the click exactly. Do not accelerate or slow down.`,
      dynamic_exaggeration: `Play bars ${d.bar_range[0]}-${d.bar_range[1]} with exaggerated dynamics. Make the loud parts louder and the soft parts softer than you think is right.`,
      isolated_hands: `Play bars ${d.bar_range[0]}-${d.bar_range[1]} hands separately. Focus on even articulation between fingers.`,
      slow_practice: `Play bars ${d.bar_range[0]}-${d.bar_range[1]} at half tempo. Listen to the shape of the phrase as you play.`,
    }
    const successMap: Record<string, string> = {
      pedal_isolation: 'Three consecutive clean repetitions with no pedal where harmonies remain audibly distinct.',
      segment_loop: 'Five consecutive repetitions matching the metronome within 20ms.',
      dynamic_exaggeration: 'The loudest and softest moments are clearly audible as different from each other.',
      isolated_hands: 'Each hand plays with consistent articulation for three repetitions.',
      slow_practice: 'The phrase shape is clear and intentional at half tempo.',
    }
    const instruction = (instructionMap[mapping.type] ?? instructionMap.slow_practice).slice(0, 400)
    const success_criterion = (successMap[mapping.type] ?? successMap.slow_practice).slice(0, 200)
    return ExerciseArtifactSchema.parse({
      diagnosis_ref: i.diagnosis_ref,
      diagnosis_summary: d.one_sentence_finding,
      target_dimension: d.primary_dimension,
      exercise_type: mapping.type,
      exercise_subtype: mapping.type === 'pedal_isolation' && d.severity === 'significant' ? 'no-pedal-pass' : mapping.subtype,
      bar_range: d.bar_range,
      instruction,
      success_criterion,
      estimated_minutes: estimatedMinutes,
      action_binding,
    })
  },
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/exercise-proposal.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/exercise-proposal.ts apps/api/src/harness/skills/molecules/exercise-proposal.test.ts && git commit -m "feat(harness): implement exercise-proposal molecule"
```

---

## Task 10: ALL_MOLECULES index barrel

**Group:** B (sequential, depends on Group A)

**Behavior being verified:** `ALL_MOLECULES` is an array of 9 ToolDefinition objects, one per molecule, each with a unique non-empty `name`.
**Interface under test:** `ALL_MOLECULES` exported from `molecules/index.ts`

**Files:**
- Create: `apps/api/src/harness/skills/molecules/index.ts`
- Test: `apps/api/src/harness/skills/molecules/index.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
import { test, expect } from 'vitest'
import { ALL_MOLECULES } from './index'

test('ALL_MOLECULES contains 9 ToolDefinition objects with unique names', () => {
  expect(ALL_MOLECULES).toHaveLength(9)
  const names = ALL_MOLECULES.map(m => m.name)
  expect(new Set(names).size).toBe(9)
  for (const mol of ALL_MOLECULES) {
    expect(typeof mol.name).toBe('string')
    expect(mol.name.length).toBeGreaterThan(0)
    expect(typeof mol.description).toBe('string')
    expect(typeof mol.invoke).toBe('function')
    expect(typeof mol.input_schema).toBe('object')
  }
})

test('ALL_MOLECULES includes all 9 named molecules', () => {
  const names = new Set(ALL_MOLECULES.map(m => m.name))
  const expected = [
    'voicing-diagnosis', 'pedal-triage', 'rubato-coaching', 'phrasing-arc-analysis',
    'tempo-stability-triage', 'dynamic-range-audit', 'articulation-clarity-check',
    'cross-modal-contradiction-check', 'exercise-proposal',
  ]
  for (const name of expected) {
    expect(names.has(name), `missing molecule: ${name}`).toBe(true)
  }
})
```

- [ ] **Step 2: Run test — verify it FAILS**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/index.test.ts
```
Expected: FAIL — `Cannot find module './index'`

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
import type { ToolDefinition } from '../../loop/types'
import { voicingDiagnosis } from './voicing-diagnosis'
import { pedalTriage } from './pedal-triage'
import { rubatoCoaching } from './rubato-coaching'
import { phrasingArcAnalysis } from './phrasing-arc-analysis'
import { tempoStabilityTriage } from './tempo-stability-triage'
import { dynamicRangeAudit } from './dynamic-range-audit'
import { articulationClarityCheck } from './articulation-clarity-check'
import { crossModalContradictionCheck } from './cross-modal-contradiction-check'
import { exerciseProposal } from './exercise-proposal'

export const ALL_MOLECULES: ToolDefinition[] = [
  voicingDiagnosis,
  pedalTriage,
  rubatoCoaching,
  phrasingArcAnalysis,
  tempoStabilityTriage,
  dynamicRangeAudit,
  articulationClarityCheck,
  crossModalContradictionCheck,
  exerciseProposal,
]
```

- [ ] **Step 4: Run test — verify it PASSES**

```bash
cd apps/api && bun run test -- --run src/harness/skills/molecules/index.test.ts
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add apps/api/src/harness/skills/molecules/index.ts apps/api/src/harness/skills/molecules/index.test.ts && git commit -m "feat(harness): add ALL_MOLECULES barrel"
```

---

## Challenge Review

### CEO Pass

**Premise:** The problem is real. `compound-registry.ts` has `tools: []` in the `OnSessionEnd` binding — without molecules, the compound dispatches nothing in Phase 1 and must hallucinate pedagogical reasoning inline. This is the correct next step after Plan 2 (atoms).

**Direct path:** Yes. Each molecule is a pure deterministic transform from a signal bundle to a validated artifact. The plan scope exactly matches the spec (9 molecules + 1 barrel = 10 tasks). No scope drift identified.

**Existing coverage:** The `__catalog__` tests at `apps/api/src/harness/skills/__catalog__/molecule-*.test.ts` already exist and validate the 9 markdown spec files in `docs/harness/skills/molecules/`. These tests are NOT broken by this plan (they validate specs, not TS implementations) and should already pass today.

**12-Month alignment:**
```
CURRENT STATE                     THIS PLAN                        12-MONTH IDEAL
compound-registry has tools:[]  → 9 molecules as ToolDefinitions → Qwen finetune data collection
LLM must reason over raw atoms  → compound dispatches molecules   → per-skill RL signal per molecule
no structured DiagnosisArtifact → validated artifacts in Phase 1  → trainable, reproducible scoring
```
Plan moves squarely toward the ideal.

**Alternatives:** The spec documents the flat-signal-bundle vs session-ID decision explicitly. No alternatives gap.

---

### Engineering Pass

#### Architecture

Molecules follow the flat-signal-bundle contract established in the molecules spec: callers materialize all data from `HookContext.digest` before invoking. Atoms accept pre-loaded data directly (confirmed by checking `2026-04-27-v6-atoms.md`). No network, DB, or LLM calls inside molecules. Data flow is:

```
compound (Phase 1)
  → extract signals from digest
  → molecule.invoke({ flat bundle })
      → atom1.invoke(...)  → scalar
      → atom2.invoke(...)  → scalar
      → branch logic
      → DiagnosisArtifactSchema.parse(...)
  ← DiagnosisArtifact
```

No security concerns (no user input flows to LLM/SQL/storage — all inputs are already-validated MuQ scores and AMT MIDI arrays).

#### Module Depth Audit

All 9 diagnosis/action molecules expose 1 exported symbol (`invoke`) and hide 50–100 LOC of branching logic, atom chaining, and Zod validation. Each is DEEP by Ousterhout's definition.

The index barrel (Task 10) is SHALLOW (9 re-exports, ~15 LOC). Acknowledged as intentional in the spec: "same pattern as atoms/index.ts".

#### Findings

[BLOCKER] (confidence: 10/10) — **Atoms directory is empty.** `apps/api/src/harness/skills/atoms/` contains zero TypeScript files. Every molecule imports from atoms (e.g., `import { computeVelocityCurve } from '../atoms/compute-velocity-curve'`). When the build agent writes each molecule test file (Step 1) and runs it (Step 2), the test will fail with `Cannot find module '../atoms/compute-velocity-curve'` — NOT the expected `Cannot find module './voicing-diagnosis'`. The TDD watch-it-fail discipline breaks because the atom import error fires before the molecule-not-found error. The atoms plan (`docs/plans/2026-04-27-v6-atoms.md`) covers this but has not been executed. **Required change: execute the v6-atoms plan (Task Groups A + B) to completion before dispatching any molecule task. Add a prerequisite note at the top of this plan.**

[RISK] (confidence: 9/10) — **`DIM` constant and `severityFromZ` duplicated across all 9 molecule files.** The plan's shared-conventions section acknowledges this but keeps them inline. A single `_shared.ts` in the molecules directory (or a reexport from the atoms barrel) would eliminate the 9-copy duplication. If severity thresholds change, 9 files require edits. Fallback: add a comment in each file pointing to the shared convention so future editors know it's intentional. Not a correctness bug today, but will compound as molecules grow.

[RISK] (confidence: 8/10) — **Task 5 (tempo-stability-triage): silent out-of-bounds in drift computation.** Line:
```typescript
const driftNotes = i.alignment.map((a, idx) => ({ onset_ms: i.midi_notes[idx]?.onset_ms ?? a.expected_onset_ms, ... }))
```
If `i.alignment.length > i.midi_notes.length`, the missing notes fall back to `expected_onset_ms`, producing zero drift for those alignment entries. The test guards against this (14 notes, 14 alignments), but production inputs may diverge. Fallback: add an explicit `if (i.alignment.length !== i.midi_notes.length) throw new Error(...)` guard before this line.

[RISK] (confidence: 7/10) — **Task 4 (phrasing-arc-analysis): empty velocity curve crashes silently with wrong result.** If `computeVelocityCurve` returns `[]` (empty notes), `Math.max(...[])` = `-Infinity`, `findIndex` returns `-1`, and `-1 === curve.length - 1` evaluates to `-1 === -1` = `true`, setting `flatOrMulti = true` and producing a spurious issue finding. The atom spec guarantees a non-empty curve for valid bar ranges, but an explicit guard (`if (curve.length === 0) return neutral`) is defensive and cheap. Fallback: add the guard before the peak detection block.

[RISK] (confidence: 7/10) — **Task 2 (pedal-triage): dead code in timing subtype branch.** In the `else` branch:
```typescript
subtype = lateReleases.length > 0 ? 'timing' : 'timing'
```
Both ternary arms assign `'timing'`. The `lateReleases` computation is wasted, and the subtype logic for distinguishing timing subtypes is not differentiated. Fallback: simplify to `subtype = 'timing'`. If finer subtype logic is intended, it needs to be implemented here.

[OBS] — The `__catalog__` molecule tests (`apps/api/src/harness/skills/__catalog__/molecule-*.test.ts`) validate the 9 markdown spec files at `docs/harness/skills/molecules/*.md`. These files already exist and appear well-formed. The catalog tests should pass today without any changes from this plan. The build agent does not need to create or modify any `docs/harness/skills/molecules/*.md` files.

[OBS] — All 9 molecule tests use `scope: 'session'`. The `DiagnosisArtifactSchema` refinement `scope === 'session' || bar_range !== null` allows `bar_range` to be null only for session scope. The non-session paths (stop_moment, passage with non-null bar_range) are untested. These are valid paths per the schema but no test verifies them. Not a blocker for this plan's scope.

[OBS] — `crossModalContradictionCheck` hardcodes `primary_dimension: 'dynamics'` in the neutral path (no contradictions detected). This is semantically odd since a neutral finding has no natural primary dimension, but Zod requires one. The neutral finding_type means consumers should ignore the content anyway. Acceptable as-is.

---

### Presumption Inventory

| Assumption | Verdict | Reason |
|---|---|---|
| Atoms have TS implementations at `../atoms/*` | **RISKY** | atoms directory is empty; v6-atoms plan not yet executed |
| `fetch-student-baseline` takes `{ dimension, session_means }` | SAFE | confirmed in `2026-04-27-v6-atoms.md` Task 14 |
| `fetch-similar-past-observation` takes `{ dimension, piece_id, bar_range, past_diagnoses, now_ms }` | SAFE | confirmed in `2026-04-27-v6-atoms.md` Task 13 |
| `DiagnosisArtifact.evidence_refs` allows any non-empty string array | SAFE | schema validates `z.array(z.string().min(1)).min(1)` |
| `computeVelocityCurve` returns non-empty array for valid input | VALIDATE | atom spec guarantees this but no TS implementation to verify |
| `ExerciseArtifactSchema` accepts `action_binding = null` when `exercise_type` is not in ACTION_REQUIRED_TYPES | SAFE | schema refine confirms `!ACTION_REQUIRED_TYPES.includes(...) || binding !== null` |
| `catalog/__catalog__` tests don't test molecule TS implementations | SAFE | verified: all catalog tests call `validateSkill` on markdown files only |

---

### Summary

[BLOCKER] count: 1
[RISK]    count: 4
[QUESTION] count: 0

VERDICT: NEEDS_REWORK — atoms plan (`docs/plans/2026-04-27-v6-atoms.md`) must complete first. Add an explicit prerequisite note at the plan header stating this. Once atoms are in place, the 4 risks are manageable: the critical one (Task 5 out-of-bounds) and two defensive guards (Task 4 empty-curve, Task 2 dead code) should be fixed before committing. The DRY risk is acceptable to defer.
