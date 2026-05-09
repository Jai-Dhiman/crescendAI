# apps/evals/teacher_model/stage0/tests/test_cli.py
"""CLI shape: subcommand registry + sample subcommand happy path."""
from __future__ import annotations

import json
from pathlib import Path

from teacher_model.stage0.cli import _build_parser, sample_main


def test_parser_lists_seven_subcommands() -> None:
    parser = _build_parser()
    # Pull subparser choices via a bit of introspection.
    subparsers_action = next(
        a for a in parser._subparsers._group_actions  # type: ignore[attr-defined]
        if a.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers_action.choices.keys()) == {
        "pin-tokenizer", "sample", "synthesis", "tool",
        "continuation", "mcq", "aggregate",
    }


def test_sample_subcommand_writes_holdout_file(tmp_path: Path, monkeypatch) -> None:
    # Build a tiny fake briefings dir + manifest.
    briefings = tmp_path / "briefings"
    briefings.mkdir()
    for rid, comp, sk in [
        ("r1", "Chopin", 3),
        ("r2", "Bach", 2),
        ("r3", "Mozart", 4),
        ("r4", "Debussy", 5),
        ("r5", "Chopin", 1),
        ("r6", "Bach", 4),
    ]:
        (briefings / f"{rid}.json").write_text(json.dumps({"recording_id": rid}))

    fake_manifests = {
        "r1": {"piece_slug": "p1", "title": "p1", "composer": "Chopin", "skill_bucket": 3},
        "r2": {"piece_slug": "p2", "title": "p2", "composer": "Bach", "skill_bucket": 2},
        "r3": {"piece_slug": "p3", "title": "p3", "composer": "Mozart", "skill_bucket": 4},
        "r4": {"piece_slug": "p4", "title": "p4", "composer": "Debussy", "skill_bucket": 5},
        "r5": {"piece_slug": "p5", "title": "p5", "composer": "Chopin", "skill_bucket": 1},
        "r6": {"piece_slug": "p6", "title": "p6", "composer": "Bach", "skill_bucket": 4},
    }

    out = tmp_path / "holdout.jsonl"
    sample_main(briefings_dir=briefings, manifests=fake_manifests, n=4, seed=42, out_path=out)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert 1 <= len(rows) <= 6
    for r in rows:
        assert "recording_id" in r and "stratum" in r
