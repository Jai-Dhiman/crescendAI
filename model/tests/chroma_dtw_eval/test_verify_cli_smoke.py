import json
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[2] / "data" / "evals" / "chroma_dtw_fixtures"


def test_verify_cli_returns_one_float_and_exits_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0},
    }))
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline), "--fixtures", str(FIXTURES)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {result.stdout!r}"
    value = float(lines[0])
    assert 0.0 <= value <= 100.0
