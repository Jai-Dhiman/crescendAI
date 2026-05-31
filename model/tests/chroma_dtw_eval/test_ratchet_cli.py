import json
import subprocess
import sys


def test_ratchet_writes_baseline_when_no_regression(tmp_path):
    sidecar = tmp_path / "side.json"
    sidecar.write_text(json.dumps({
        "primary": 73.5,
        "guards": {"g1": 12.0, "g2": 0.78, "g3": 80.0, "g4": 65.0, "g5": 8.0},
        "regressed": [],
    }))
    baseline = tmp_path / "base.json"
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.ratchet",
         "--from", str(sidecar), "--to", str(baseline)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(baseline.read_text())
    assert data["primary"] == 73.5
    assert data["guards"]["g2"] == 0.78


def test_ratchet_refuses_when_regressed_nonempty(tmp_path):
    sidecar = tmp_path / "side.json"
    sidecar.write_text(json.dumps({
        "primary": 50.0,
        "guards": {"g1": 0.0, "g2": 0.5, "g3": 0.0, "g4": 0.0, "g5": 0.0},
        "regressed": ["g2"],
    }))
    baseline = tmp_path / "base.json"
    result = subprocess.run(
        [sys.executable, "-m", "chroma_dtw_eval.ratchet",
         "--from", str(sidecar), "--to", str(baseline)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert not baseline.exists()
