# Tasks 11-15 (Group B — TS `teacher_style.ts` track, sequential)

All tasks modify `apps/api/src/services/teacher_style.ts` and its test file.

---

## Task 11: TS DSL evaluator — arithmetic + signal lookup
**Group:** B (depends on Task 1)

**Behavior:** TS `evaluate(formula, signals)` returns expected float for arithmetic + signal lookup.

**Interface under test:** `apps/api/src/services/teacher_style.ts::evaluate`.

**Files:**
- Create: `apps/api/src/services/teacher_style.ts`
- Create: `apps/api/src/services/teacher_style.test.ts`

- [ ] **Step 1: Write the failing test**

```typescript
// apps/api/src/services/teacher_style.test.ts
import { describe, expect, it } from "vitest";
import { evaluate } from "./teacher_style";

const SIGNALS = {
  max_neg_dev: 0.2, max_pos_dev: 0.0, n_significant: 2,
  drilling_present: false, drilling_improved: false,
  duration_min: 15.0, mode_count: 1, has_piece: true,
};

describe("teacher_style.evaluate", () => {
  it("evaluates arithmetic over signals", () => {
    expect(evaluate("1.5 * max_neg_dev + 0.3 * n_significant", SIGNALS)).toBeCloseTo(0.9, 6);
  });

  it("looks up a single signal", () => {
    expect(evaluate("max_neg_dev", SIGNALS)).toBeCloseTo(0.2, 6);
  });

  it("rejects unknown signal names", () => {
    expect(() => evaluate("max_neg_dev + bogus", SIGNALS)).toThrow(/unknown signal/);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: FAIL — `Cannot find module './teacher_style'`.

- [ ] **Step 3: Implement the minimum to make the test pass**

```typescript
// apps/api/src/services/teacher_style.ts
const ALLOWED_SIGNALS = new Set([
  "max_neg_dev", "max_pos_dev", "n_significant",
  "drilling_present", "drilling_improved",
  "duration_min", "mode_count", "has_piece",
]);

export type Signals = Record<string, number | boolean>;

type Tok = { kind: "number" | "ident" | "op"; text: string };

function tokenize(text: string): Tok[] {
  const tokens: Tok[] = [];
  const re = /(\d+(?:\.\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(<=|>=|==|!=|[+\-*/<>()])/g;
  for (const m of text.matchAll(re)) {
    if (m[1] !== undefined) tokens.push({ kind: "number", text: m[1] });
    else if (m[2] !== undefined) tokens.push({ kind: "ident", text: m[2] });
    else if (m[3] !== undefined) tokens.push({ kind: "op", text: m[3] });
  }
  return tokens;
}

class Cursor {
  i = 0;
  constructor(public toks: Tok[]) {}
  peek(): Tok | null { return this.toks[this.i] ?? null; }
  take(): Tok { return this.toks[this.i++]; }
}

export function evaluate(formula: string, signals: Signals): number {
  const cur = new Cursor(tokenize(formula));
  const v = expr(cur, signals);
  if (cur.peek() !== null) throw new Error(`trailing tokens in: ${formula}`);
  return v;
}

function expr(c: Cursor, s: Signals): number { return arith(c, s); }

function arith(c: Cursor, s: Signals): number {
  let left = term(c, s);
  while (c.peek() && (c.peek()!.text === "+" || c.peek()!.text === "-")) {
    const op = c.take().text;
    const right = term(c, s);
    left = op === "+" ? left + right : left - right;
  }
  return left;
}

function term(c: Cursor, s: Signals): number {
  let left = factor(c, s);
  while (c.peek() && (c.peek()!.text === "*" || c.peek()!.text === "/")) {
    const op = c.take().text;
    const right = factor(c, s);
    left = op === "*" ? left * right : left / right;
  }
  return left;
}

function factor(c: Cursor, s: Signals): number {
  const nxt = c.peek();
  if (!nxt) throw new Error("unexpected end of formula");
  if (nxt.kind === "number") { c.take(); return parseFloat(nxt.text); }
  if (nxt.kind === "ident") {
    if (!ALLOWED_SIGNALS.has(nxt.text)) throw new Error(`unknown signal: ${nxt.text}`);
    c.take();
    const v = s[nxt.text];
    return typeof v === "boolean" ? (v ? 1 : 0) : Number(v);
  }
  if (nxt.text === "(") {
    c.take();
    const v = expr(c, s);
    if (!c.peek() || c.peek()!.text !== ")") throw new Error("missing close paren");
    c.take();
    return v;
  }
  throw new Error(`unexpected token: ${JSON.stringify(nxt)}`);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/teacher_style.ts apps/api/src/services/teacher_style.test.ts
git commit -m "feat(teacher-style): TS DSL evaluator (arithmetic + signals)"
```

---

## Task 12: TS DSL — boolean conditionals
**Group:** B (depends on Task 11)

**Behavior:** TS `evaluate("1 if drilling_improved else 0", signals)` handles conditionals.

**Files:** modify `teacher_style.ts` and `teacher_style.test.ts`.

- [ ] **Step 1: Write the failing test** — append:

```typescript
describe("teacher_style.evaluate conditionals", () => {
  it("returns then-branch when condition is true", () => {
    const sig = { ...SIGNALS, drilling_improved: true };
    expect(evaluate("1.5 if drilling_improved else 0", sig)).toBeCloseTo(1.5, 6);
  });

  it("returns else-branch when condition is false", () => {
    const sig = { ...SIGNALS, drilling_improved: false };
    expect(evaluate("1.5 if drilling_improved else 0.5", sig)).toBeCloseTo(0.5, 6);
  });

  it("evaluates the technical-corrective formula", () => {
    const sig = { ...SIGNALS, max_neg_dev: 0.2, n_significant: 2, drilling_improved: false };
    const formula = "1.5 * max_neg_dev + 0.3 * n_significant - 0.5 * (1 if drilling_improved else 0)";
    expect(evaluate(formula, sig)).toBeCloseTo(0.9, 6);
  });

  it("supports compound boolean: a < x and b < x", () => {
    const sig = { ...SIGNALS, max_neg_dev: 0.05, max_pos_dev: 0.05 };
    expect(evaluate("1 if max_neg_dev < 0.1 and max_pos_dev < 0.1 else 0", sig)).toBeCloseTo(1, 6);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: FAIL — first conditional test throws on unknown signal "if".

- [ ] **Step 3: Implement the minimum to make the test pass** — replace `expr` and add boolean helpers:

```typescript
function expr(c: Cursor, s: Signals): number {
  const value = arith(c, s);
  const nxt = c.peek();
  if (nxt && nxt.kind === "ident" && nxt.text === "if") {
    c.take();
    const cond = boolExpr(c, s);
    const kw = c.peek();
    if (!kw || kw.text !== "else") throw new Error("expected 'else' in conditional");
    c.take();
    const elseValue = arith(c, s);
    return cond ? value : elseValue;
  }
  return value;
}

function boolExpr(c: Cursor, s: Signals): boolean {
  let left = boolAnd(c, s);
  while (c.peek() && c.peek()!.kind === "ident" && c.peek()!.text === "or") {
    c.take();
    const right = boolAnd(c, s);
    left = left || right;
  }
  return left;
}

function boolAnd(c: Cursor, s: Signals): boolean {
  let left = boolNot(c, s);
  while (c.peek() && c.peek()!.kind === "ident" && c.peek()!.text === "and") {
    c.take();
    const right = boolNot(c, s);
    left = left && right;
  }
  return left;
}

function boolNot(c: Cursor, s: Signals): boolean {
  const nxt = c.peek();
  if (nxt && nxt.kind === "ident" && nxt.text === "not") { c.take(); return !cmp(c, s); }
  return cmp(c, s);
}

function cmp(c: Cursor, s: Signals): boolean {
  const left = arith(c, s);
  const nxt = c.peek();
  if (nxt && ["<", "<=", ">", ">=", "==", "!="].includes(nxt.text)) {
    const op = c.take().text;
    const right = arith(c, s);
    switch (op) {
      case "<":  return left < right;
      case "<=": return left <= right;
      case ">":  return left > right;
      case ">=": return left >= right;
      case "==": return left === right;
      case "!=": return left !== right;
    }
  }
  return Boolean(left);
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/teacher_style.ts apps/api/src/services/teacher_style.test.ts
git commit -m "feat(teacher-style): TS DSL conditionals + boolean ops"
```

---

## Task 13: TS `selectClusters` — top-2 from compiled playbook
**Group:** B (depends on Task 12 + Task 16; cannot run before Task 16 lands `playbook.json`)

**Behavior:** `selectClusters(signals)` reads `apps/api/src/lib/playbook.json`, evaluates each cluster's score, returns top-2 with priority tie-break and confidence-floor fallback.

**Files:** modify `teacher_style.ts` and `teacher_style.test.ts`. Verify `apps/api/tsconfig.json` has `"resolveJsonModule": true` under `compilerOptions`; if missing, add it.

- [ ] **Step 1: Write the failing test** — append:

```typescript
import { selectClusters } from "./teacher_style";
import fixtures from "../../../../shared/teacher-style/test_fixtures.json";

describe("teacher_style.selectClusters parity fixtures", () => {
  for (const f of fixtures) {
    it(`fixture ${f.name}: primary contains ${f.expected_primary_substring}`, () => {
      const sel = selectClusters(f.signals);
      expect(sel.primary.name.toLowerCase()).toContain(f.expected_primary_substring.toLowerCase());
      expect(sel.secondary.name.toLowerCase()).toContain(f.expected_secondary_substring.toLowerCase());
    });
  }

  it("fallback when all scores low", () => {
    const sel = selectClusters({
      max_neg_dev: 0, max_pos_dev: 0, n_significant: 0,
      drilling_present: false, drilling_improved: false,
      duration_min: 15, mode_count: 1, has_piece: false,
    });
    expect(sel.primary.name.toLowerCase()).toContain("technical");
    expect(sel.secondary.name.toLowerCase()).toMatch(/(positive|praise)/);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: FAIL — module-resolution error on `playbook.json` import or `selectClusters is not exported`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append:

```typescript
import playbook from "../lib/playbook.json";

const PRIORITY_ORDER = [
  "Technical-corrective", "Positive-encouragement", "Artifact-based",
  "Guided-discovery", "Motivational",
] as const;
const FALLBACK_PRIMARY_KEY = "Technical-corrective";
const FALLBACK_SECONDARY_KEY = "Positive-encouragement";
const CONFIDENCE_FLOOR = 0.3;

type Cluster = {
  name: string;
  language_patterns?: { register?: string; tone?: string };
  good_examples?: { text?: string; source_id?: string }[];
  when_to_use?: string[];
  triggers: { score: string };
};
type Playbook = { teaching_playbook: { clusters: Cluster[] } };

export type ClusterRef = { name: string; score: number; raw: Cluster };
export type ClusterSelection = { primary: ClusterRef; secondary: ClusterRef };

function priorityIndex(name: string): number {
  for (let i = 0; i < PRIORITY_ORDER.length; i += 1) {
    if (name.toLowerCase().includes(PRIORITY_ORDER[i].toLowerCase())) return i;
  }
  return PRIORITY_ORDER.length;
}

function findCluster(substring: string): Cluster {
  const clusters = (playbook as unknown as Playbook).teaching_playbook.clusters;
  const found = clusters.find((c) => c.name.toLowerCase().includes(substring.toLowerCase()));
  if (!found) throw new Error(`no cluster matching ${substring}`);
  return found;
}

export function selectClusters(signals: Signals): ClusterSelection {
  const clusters = (playbook as unknown as Playbook).teaching_playbook.clusters;
  const scored: ClusterRef[] = clusters.map((c) => ({
    name: c.name,
    score: evaluate(c.triggers.score, signals),
    raw: c,
  }));
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return priorityIndex(a.name) - priorityIndex(b.name);
  });
  let primary = scored[0];
  let secondary = scored[1];
  if (primary.score < CONFIDENCE_FLOOR && secondary.score < CONFIDENCE_FLOOR) {
    const fp = findCluster(FALLBACK_PRIMARY_KEY);
    const fs = findCluster(FALLBACK_SECONDARY_KEY);
    primary = { name: fp.name, score: 0, raw: fp };
    secondary = { name: fs.name, score: 0, raw: fs };
  }
  return { primary, secondary };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: PASS (12 tests total).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/teacher_style.ts apps/api/src/services/teacher_style.test.ts apps/api/tsconfig.json
git commit -m "feat(teacher-style): TS selectClusters with parity fixtures"
```

---

## Task 14: TS `formatTeacherVoiceBlocks`
**Group:** B (depends on Task 13)

**Behavior:** `formatTeacherVoiceBlocks(selection)` returns the canonical two-block string.

**Files:** modify `teacher_style.ts` and `teacher_style.test.ts`.

- [ ] **Step 1: Write the failing test** — append:

```typescript
import { formatTeacherVoiceBlocks } from "./teacher_style";

describe("teacher_style.formatTeacherVoiceBlocks", () => {
  const sig = {
    max_neg_dev: 0.25, max_pos_dev: 0, n_significant: 3,
    drilling_present: false, drilling_improved: false,
    duration_min: 15, mode_count: 1, has_piece: true,
  };

  it("emits both teacher_voice and also_consider blocks", () => {
    const out = formatTeacherVoiceBlocks(selectClusters(sig));
    expect(out).toContain("<teacher_voice");
    expect(out).toContain("<also_consider");
    expect(out).toContain("Register:");
    expect(out).toContain("Tone:");
  });

  it("includes a normalized cluster id in the attribute", () => {
    const out = formatTeacherVoiceBlocks(selectClusters(sig));
    expect(out).toMatch(/cluster="[a-z][a-z0-9-]+"/);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: FAIL — `formatTeacherVoiceBlocks is not exported`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append:

```typescript
function normalizeClusterId(name: string): string {
  return name.toLowerCase()
    .replace(/‑|–|—/g, "-")
    .replace(/[“”]/g, "")
    .replace(/ \/ /g, "-")
    .replace(/ /g, "-")
    .trim();
}

function firstExemplar(cluster: Cluster): string {
  for (const ex of cluster.good_examples ?? []) {
    if (ex?.text) return ex.text;
  }
  return "";
}

export function formatTeacherVoiceBlocks(selection: ClusterSelection): string {
  const lines: string[] = [];
  const p = selection.primary;
  const s = selection.secondary;

  const pId = normalizeClusterId(p.name);
  const pReg = p.raw.language_patterns?.register ?? "";
  const pTone = p.raw.language_patterns?.tone ?? "";
  const pEx = firstExemplar(p.raw);
  lines.push(`<teacher_voice cluster="${pId}">`);
  lines.push(`Register: ${pReg}`);
  lines.push(`Tone: ${pTone}`);
  if (pEx) lines.push(`Exemplar: ${pEx}`);
  lines.push("</teacher_voice>");

  const sId = normalizeClusterId(s.name);
  const sWhen = (s.raw.when_to_use ?? []).join("; ");
  const sEx = firstExemplar(s.raw);
  lines.push("");
  lines.push(`<also_consider cluster="${sId}">`);
  if (sWhen) lines.push(`Apply when: ${sWhen}`);
  if (sEx) lines.push(`Exemplar: ${sEx}`);
  lines.push("</also_consider>");
  return lines.join("\n");
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/teacher_style.ts apps/api/src/services/teacher_style.test.ts
git commit -m "feat(teacher-style): TS formatTeacherVoiceBlocks"
```

---

## Task 15: TS `deriveSignals` helper
**Group:** B (depends on Task 14)

**Behavior:** `deriveSignals(topMoments, drillingRecords, sessionDurationMs, pieceMetadata, practicePattern)` returns the canonical `Signals` object.

**Files:** modify `teacher_style.ts` and `teacher_style.test.ts`.

- [ ] **Step 1: Write the failing test** — append:

```typescript
import { deriveSignals } from "./teacher_style";

describe("teacher_style.deriveSignals", () => {
  it("produces the documented signal vector", () => {
    const sig = deriveSignals(
      [
        { dimension: "dynamics", score: 0.8, deviation_from_mean: 0.25, direction: "above_average" },
        { dimension: "timing", score: 0.3, deviation_from_mean: -0.18, direction: "below_average" },
      ],
      [],
      900_000,
      { title: "Prelude", composer: "Bach", skill_level: 3 },
      "continuous_play",
    );
    expect(sig.max_neg_dev).toBeCloseTo(0.18, 3);
    expect(sig.max_pos_dev).toBeCloseTo(0.25, 3);
    expect(sig.n_significant).toBe(2);
    expect(sig.drilling_present).toBe(false);
    expect(sig.drilling_improved).toBe(false);
    expect(sig.duration_min).toBeCloseTo(15, 1);
    expect(sig.mode_count).toBe(1);
    expect(sig.has_piece).toBe(true);
  });

  it("derives drilling_improved from first vs final score", () => {
    const sig = deriveSignals(
      [],
      [{ first_score: 0.5, final_score: 0.8 }],
      600_000,
      { title: "X", composer: "Bach", skill_level: 1 },
      "continuous_play",
    );
    expect(sig.drilling_present).toBe(true);
    expect(sig.drilling_improved).toBe(true);
  });
});
```

- [ ] **Step 2: Run test — verify it FAILS**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: FAIL — `deriveSignals is not exported`.

- [ ] **Step 3: Implement the minimum to make the test pass** — append:

```typescript
type TopMoment = { dimension?: string; deviation_from_mean?: number };
type DrillingRecord = { first_score?: number; final_score?: number };

export function deriveSignals(
  topMoments: TopMoment[] | unknown,
  drillingRecords: DrillingRecord[] | unknown,
  sessionDurationMs: number,
  pieceMetadata: { title?: string } | unknown,
  practicePattern: unknown,
): Signals {
  const moments = Array.isArray(topMoments) ? (topMoments as TopMoment[]) : [];
  const drilling = Array.isArray(drillingRecords) ? (drillingRecords as DrillingRecord[]) : [];

  const devs = moments.map((m) => Number(m.deviation_from_mean)).filter((v) => Number.isFinite(v));
  const negs = devs.filter((v) => v < 0).map((v) => -v);
  const poss = devs.filter((v) => v > 0);
  const max_neg_dev = negs.length ? Math.max(...negs) : 0;
  const max_pos_dev = poss.length ? Math.max(...poss) : 0;
  const n_significant = devs.filter((v) => Math.abs(v) >= 0.1).length;

  const drilling_present = drilling.length > 0;
  const drilling_improved = drilling_present
    && Number(drilling[drilling.length - 1].final_score) > Number(drilling[0].first_score) + 0.15;

  const duration_min = sessionDurationMs / 60_000;

  let mode_count = 1;
  if (Array.isArray(practicePattern)) mode_count = new Set(practicePattern).size;
  else if (practicePattern && typeof practicePattern === "object") {
    const modes = (practicePattern as { modes?: unknown }).modes;
    if (Array.isArray(modes)) mode_count = new Set(modes).size;
  }

  const piece = pieceMetadata as { title?: string } | undefined;
  const has_piece = !!piece?.title && piece.title !== "Unknown";

  return { max_neg_dev, max_pos_dev, n_significant, drilling_present, drilling_improved, duration_min, mode_count, has_piece };
}
```

- [ ] **Step 4: Run test — verify it PASSES**

```
cd apps/api && bun run vitest run src/services/teacher_style.test.ts
```
Expected: PASS (16 tests).

- [ ] **Step 5: Commit**

```
git add apps/api/src/services/teacher_style.ts apps/api/src/services/teacher_style.test.ts
git commit -m "feat(teacher-style): TS deriveSignals helper"
```
