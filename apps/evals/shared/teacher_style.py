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
    value = _arith(tok, sig)
    nxt = tok.peek()
    if nxt and nxt[0] == "ident" and nxt[1] == "if":
        tok.take()
        cond = _bool_expr(tok, sig)
        kw = tok.peek()
        if not kw or kw[1] != "else":
            raise ValueError("expected 'else' in conditional")
        tok.take()
        else_value = _arith(tok, sig)
        return value if cond else else_value
    return value


def _bool_expr(tok, sig):
    left = _bool_and(tok, sig)
    while tok.peek() and tok.peek()[0] == "ident" and tok.peek()[1] == "or":
        tok.take()
        right = _bool_and(tok, sig)
        left = left or right
    return left


def _bool_and(tok, sig):
    left = _bool_not(tok, sig)
    while tok.peek() and tok.peek()[0] == "ident" and tok.peek()[1] == "and":
        tok.take()
        right = _bool_not(tok, sig)
        left = left and right
    return left


def _bool_not(tok, sig):
    nxt = tok.peek()
    if nxt and nxt[0] == "ident" and nxt[1] == "not":
        tok.take()
        return not _cmp(tok, sig)
    return _cmp(tok, sig)


def _cmp(tok, sig):
    left = _arith(tok, sig)
    nxt = tok.peek()
    if nxt and nxt[1] in ("<", "<=", ">", ">=", "==", "!="):
        op = tok.take()[1]
        right = _arith(tok, sig)
        return {
            "<": left < right, "<=": left <= right,
            ">": left > right, ">=": left >= right,
            "==": left == right, "!=": left != right,
        }[op]
    return bool(left)


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
        _KEYWORDS = {"if", "else", "and", "or", "not"}
        if text in _KEYWORDS:
            raise ValueError(f"unexpected keyword in arithmetic position: {text!r}")
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


from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import yaml

PLAYBOOK_PATH = Path(__file__).resolve().parents[3] / "shared" / "teacher-style" / "playbook.yaml"

PRIORITY_ORDER = [
    "Technical-corrective", "Positive-encouragement", "Artifact-based",
    "Guided-discovery", "Motivational",
]
FALLBACK_PRIMARY_KEY = "Technical-corrective"
FALLBACK_SECONDARY_KEY = "Positive-encouragement"
CONFIDENCE_FLOOR = 0.3


@dataclass(frozen=True)
class ClusterRef:
    name: str
    score: float
    raw: dict


@dataclass(frozen=True)
class ClusterSelection:
    primary: ClusterRef
    secondary: ClusterRef


@lru_cache(maxsize=1)
def _load_playbook():
    return yaml.safe_load(PLAYBOOK_PATH.read_text())["teaching_playbook"]["clusters"]


def _priority_index(name: str) -> int:
    for i, key in enumerate(PRIORITY_ORDER):
        if key.lower() in name.lower():
            return i
    return len(PRIORITY_ORDER)


_QUOTE_CHARS = '"\'' + chr(0x201C) + chr(0x201D) + chr(0x2018) + chr(0x2019)


def _cluster_name(cluster: dict) -> str:
    return cluster["name"].strip(_QUOTE_CHARS)


def _normalize_name(s: str) -> str:
    return s.lower().replace("-", "").replace("‑", "")


def _find_cluster(substring: str) -> dict:
    sub_norm = _normalize_name(substring)
    for cluster in _load_playbook():
        name_norm = _normalize_name(_cluster_name(cluster))
        if sub_norm in name_norm:
            return cluster
    raise ValueError(f"no cluster matching {substring!r}")


def _formula(cluster: dict) -> str:
    raw = cluster["triggers"]["score"]
    if not isinstance(raw, str):
        return str(raw)
    # Strip ASCII and Unicode curly quotes that YAML may preserve
    return raw.strip(_QUOTE_CHARS)


def select_clusters(signals: dict[str, Any]) -> ClusterSelection:
    scored = [
        ClusterRef(name=_cluster_name(c), score=evaluate(_formula(c), signals), raw=c)
        for c in _load_playbook()
    ]
    scored.sort(key=lambda c: (-c.score, _priority_index(c.name)))
    primary, secondary = scored[0], scored[1]
    if primary.score < CONFIDENCE_FLOOR and secondary.score < CONFIDENCE_FLOOR:
        fp = _find_cluster(FALLBACK_PRIMARY_KEY)
        fs = _find_cluster(FALLBACK_SECONDARY_KEY)
        primary = ClusterRef(name=_cluster_name(fp), score=0.0, raw=fp)
        secondary = ClusterRef(name=_cluster_name(fs), score=0.0, raw=fs)
    return ClusterSelection(primary=primary, secondary=secondary)
