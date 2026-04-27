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
