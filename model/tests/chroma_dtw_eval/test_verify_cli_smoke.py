import json
import subprocess
import sys
from pathlib import Path


FIXTURES = Path(__file__).resolve().parents[2] / "data" / "evals" / "chroma_dtw_fixtures"
MANIFEST = FIXTURES / "manifest.json"
PSEUDO_TRUTH = Path(__file__).resolve().parents[2] / "data" / "evals" / "pseudo_truth"
MODEL_DIR = Path(__file__).resolve().parents[2]


def test_verify_cli_returns_one_float_and_exits_zero(tmp_path):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({
        "primary": 0.0,
        "guards": {"g1": 100.0, "g2": 0.0, "g3": 100.0, "g4": 0.0, "g5": 100.0},
    }))
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(MANIFEST),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one stdout line, got {result.stdout!r}"
    value = float(lines[0])
    assert 0.0 <= value <= 100.0


def test_verify_cli_exits_nonzero_when_baseline_above_current(tmp_path):
    baseline = tmp_path / "baseline.json"
    # primary=100.0 means current run (which produces primary <= 100) will regress
    baseline.write_text(json.dumps({
        "primary": 100.0,
        "guards": {"g1": 0.0, "g2": 0.99, "g3": 0.0, "g4": 100.0, "g5": 0.0},
    }))
    sidecar_path = tmp_path / "sidecar.json"
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.verify",
         "--baseline", str(baseline),
         "--manifest", str(MANIFEST),
         "--sidecar", str(sidecar_path),
         "--skip-dtw"],
        capture_output=True, text=True, timeout=120,
        cwd=str(MODEL_DIR),
    )
    assert result.returncode != 0, (
        f"expected non-zero, got {result.returncode}; stdout={result.stdout}; stderr={result.stderr}"
    )
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["regressed"], "sidecar must list regressed guards"
