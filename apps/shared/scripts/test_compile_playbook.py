# scripts/test_compile_playbook.py
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "compile_playbook.py"
OUT = REPO_ROOT / "apps" / "api" / "src" / "lib" / "playbook.json"


def test_compile_writes_valid_json():
    res = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert OUT.exists()
    data = json.loads(OUT.read_text())
    assert "teaching_playbook" in data
    assert len(data["teaching_playbook"]["clusters"]) == 5


def test_check_mode_exits_zero_when_synced():
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_check_mode_exits_nonzero_on_drift():
    OUT.write_text(json.dumps({"stale": "value"}))
    res = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True)
    assert res.returncode != 0
    subprocess.run([sys.executable, str(SCRIPT)], check=True)
