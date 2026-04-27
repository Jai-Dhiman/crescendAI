# apps/evals/shared/teacher_style.py
"""Cluster selection + teacher-voice prompt formatting.

Mirrors apps/api/src/services/teacher_style.ts. Both implementations evaluate
the same DSL formulas against the same signals.
"""
from __future__ import annotations
import re
from typing import Any

ALLOWED_SIGNALS = {
    "max_neg_dev", "max_pos_dev", "n_significant",
    "drilling_present", "drilling_improved",
    "duration_min", "mode_count", "has_piece",
}


class _Tokenizer:
    _TOKEN_RE = re.compile(
        r"\s*(?:"
        r"(?P<number>\d+(?:\.\d+)?)"
        r"|(?P<ident>[A-Za-z_][A-Za-z0-9_]*)"
        r"|(?P<op><=|>=|==|!=|[+\-*/<>()])"
        r")"
    )

    def __init__(self, text: str) -> None:
        self.tokens: list[tuple[str, str]] = []
        pos = 0
        while pos < len(text):
            m = self._TOKEN_RE.match(text, pos)
            if not m or m.end() == pos:
                if text[pos].isspace():
                    pos += 1
                    continue
                raise ValueError(f"unexpected character at {pos}: {text[pos]!r}")
            kind = m.lastgroup
            self.tokens.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self):
        tok = self.tokens[self.i]
        self.i += 1
        return tok


def evaluate(formula: str, signals: dict[str, Any]) -> float:
    tok = _Tokenizer(formula)
    value = _expr(tok, signals)
    if tok.peek() is not None:
        raise ValueError(f"unexpected trailing tokens in: {formula!r}")
    return float(value)


def _expr(tok, sig):
    return _arith(tok, sig)


def _arith(tok, sig):
    left = _term(tok, sig)
    while tok.peek() and tok.peek()[1] in ("+", "-"):
        op = tok.take()[1]
        right = _term(tok, sig)
        left = left + right if op == "+" else left - right
    return left


def _term(tok, sig):
    left = _factor(tok, sig)
    while tok.peek() and tok.peek()[1] in ("*", "/"):
        op = tok.take()[1]
        right = _factor(tok, sig)
        left = left * right if op == "*" else left / right
    return left


def _factor(tok, sig):
    nxt = tok.peek()
    if nxt is None:
        raise ValueError("unexpected end of formula")
    kind, text = nxt
    if kind == "number":
        tok.take()
        return float(text)
    if kind == "ident":
        if text not in ALLOWED_SIGNALS:
            raise ValueError(f"unknown signal: {text}")
        tok.take()
        v = sig[text]
        return float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
    if kind == "op" and text == "(":
        tok.take()
        v = _expr(tok, sig)
        if not tok.peek() or tok.peek()[1] != ")":
            raise ValueError("missing close paren")
        tok.take()
        return v
    raise ValueError(f"unexpected token: {nxt}")
