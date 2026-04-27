# apps/evals/shared/judge_atomic.py
"""Single-judge atomic-skill matrix scoring (8 moves x 5 binary criteria)."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

RUBRICS_PATH = Path(__file__).parent / "prompts" / "atomic_skill_rubrics.json"


class JudgeClient(Protocol):
    def complete(self, *, user: str, system: str, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class MoveResult:
    move_id: str
    attempted: bool
    criteria: list[bool] | None


@dataclass(frozen=True)
class AtomicMatrixResult:
    moves: list[MoveResult]


SYSTEM = (
    "You are a careful evaluator. Given a piano teacher's session synthesis, "
    "judge for each of 8 pedagogical moves whether the synthesis attempted that move "
    "(observable from the text), and if attempted, whether each of 5 binary criteria is satisfied. "
    "Output strict JSON with the documented schema. No prose."
)


def _build_user(synthesis_text: str, context: dict[str, Any]) -> str:
    rubrics = json.loads(RUBRICS_PATH.read_text())
    parts = [
        f"<context>{json.dumps(context)}</context>",
        "<synthesis>", synthesis_text, "</synthesis>",
        "<rubrics>", json.dumps(rubrics, indent=2), "</rubrics>",
        ('Output JSON: {"moves": [{"move_id": "...", "attempted": true|false, '
         '"criteria": [true|false, ...] | null}, ...]} with all 8 move_ids in order.'),
    ]
    return "\n".join(parts)


def judge_atomic_matrix(*, synthesis_text: str, context: dict, client: JudgeClient) -> AtomicMatrixResult:
    user = _build_user(synthesis_text, context)
    raw = client.complete(user=user, system=SYSTEM, max_tokens=2048)
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip().rstrip("`").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"atomic judge returned non-JSON: {exc}; raw={raw[:200]!r}") from exc
    if not isinstance(data, dict) or "moves" not in data:
        raise ValueError(f"atomic judge response missing 'moves' key: raw={raw[:200]!r}")
    moves_raw = data["moves"]
    moves = [
        MoveResult(
            move_id=m["move_id"],
            attempted=bool(m["attempted"]),
            criteria=[bool(c) for c in m["criteria"]] if m.get("criteria") is not None else None,
        )
        for m in moves_raw
    ]
    return AtomicMatrixResult(moves=moves)
