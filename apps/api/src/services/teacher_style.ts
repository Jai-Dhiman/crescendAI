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

// ---- Task 13: selectClusters ----

import playbookRaw from "../lib/playbook.json";

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
  good_examples?: Array<{ text?: string; source_id?: string } | undefined>;
  when_to_use?: string[];
  triggers: { score: string };
};
type Playbook = { teaching_playbook: { clusters: Cluster[] } };

export type ClusterRef = { name: string; score: number; raw: Cluster };
export type ClusterSelection = { primary: ClusterRef; secondary: ClusterRef };

const CURLY_QUOTE_RE = /^[“”‘’„‟«»]+|[“”‘’„‟«»]+$/g;

function cleanFormula(s: string): string {
  return s.replace(CURLY_QUOTE_RE, "").trim();
}

function priorityIndex(name: string): number {
  const nameNorm = normalizeForMatch(name);
  for (let i = 0; i < PRIORITY_ORDER.length; i += 1) {
    if (nameNorm.includes(normalizeForMatch(PRIORITY_ORDER[i]))) return i;
  }
  return PRIORITY_ORDER.length;
}

function normalizeForMatch(s: string): string {
  return s.toLowerCase().replace(/-/g, "").replace(/‑/g, "");
}

function findCluster(substring: string): Cluster {
  const clusters = (playbookRaw as unknown as Playbook).teaching_playbook.clusters;
  const subNorm = normalizeForMatch(substring);
  const found = clusters.find((c) => normalizeForMatch(cleanFormula(c.name)).includes(subNorm));
  if (!found) throw new Error(`no cluster matching ${substring}`);
  return found;
}

export function selectClusters(signals: Signals): ClusterSelection {
  const clusters = (playbookRaw as unknown as Playbook).teaching_playbook.clusters;
  const scored: ClusterRef[] = clusters.map((c) => ({
    name: cleanFormula(c.name),
    score: evaluate(cleanFormula(c.triggers.score), signals),
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
    primary = { name: cleanFormula(fp.name), score: 0, raw: fp };
    secondary = { name: cleanFormula(fs.name), score: 0, raw: fs };
  }
  return { primary, secondary };
}
