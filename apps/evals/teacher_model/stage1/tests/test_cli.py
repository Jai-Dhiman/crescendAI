import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_holdout_writes_manifest(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i, composer in enumerate(["Chopin"] * 5 + ["Bach"] * 5):
        (cache / f"rec_{i:03d}.json").write_text(
            json.dumps(
                {
                    "briefing_id": f"rec_{i:03d}",
                    "framing_text": "stub",
                    "composer": composer,
                    "skill_bucket": "intermediate",
                }
            )
        )
    out = tmp_path / "holdout.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "teacher_model.stage1",
            "holdout",
            "--cache-dir",
            str(cache),
            "--out",
            str(out),
            "--frac",
            "0.20",
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    # 10 briefings, frac 0.20, stratified by composer (5 per stratum) -> 1 per stratum -> 2 total
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert all("briefing_id" in p for p in parsed)


@pytest.mark.parametrize("subcmd", ["holdout", "distill", "coverage", "render", "harness"])
def test_cli_subcommand_help(subcmd):
    result = subprocess.run(
        [sys.executable, "-m", "teacher_model.stage1", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert subcmd in result.stdout.lower() or "usage" in result.stdout.lower()
